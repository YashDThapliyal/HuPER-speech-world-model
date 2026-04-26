"""
Shared phone-CTC components for training, inference, and evaluation.

This module provides:
- PhoneCTCHead: trainable phone classifier on top of E_t features
- PhoneTeacher: frozen pseudo-label generator from a pretrained phone CTC model
- Unified token normalization/decoding helpers
- Lightweight sequence metrics (edit distance, PER)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

EXCLUDE_TOKENS = {
    "[PAD]", "|", "[UNK]", "<s>", "</s>", " ",
}


class PhoneCTCHead(nn.Module):
    """CTC phone head: E_t (T, d) -> logits (T, vocab)."""

    def __init__(
        self,
        d_in: int,
        vocab_size: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(d_in, vocab_size)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, vocab_size),
            )

    def forward(self, E_t: torch.Tensor) -> torch.Tensor:
        return self.net(E_t)


@dataclass
class PhoneVocabulary:
    """Vocabulary metadata shared between train/infer/eval."""

    labels: list[str]
    blank_id: int

    @property
    def vocab_size(self) -> int:
        return len(self.labels)


@dataclass
class PseudoLabelBundle:
    """Pseudo labels aligned to a target frame count."""

    frame_ids: np.ndarray  # (T_target,)
    target_ids: np.ndarray  # (U,) collapsed non-blank IDs for CTC targets
    vocab: PhoneVocabulary


def normalize_token(token: str) -> str:
    return token.strip()


def collapse_ctc_ids(frame_ids: list[int] | np.ndarray, blank_id: int) -> list[int]:
    """CTC collapse: merge repeats and remove blank."""
    collapsed: list[int] = []
    last = -1
    for pid in frame_ids:
        pid_i = int(pid)
        if pid_i == last:
            continue
        last = pid_i
        if pid_i == blank_id:
            continue
        collapsed.append(pid_i)
    return collapsed


def decode_logits_greedy(
    logits: torch.Tensor,
    blank_id: int,
    blank_bias: float = 0.0,
    min_phone_conf: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode logits (T, V) to frame IDs and collapsed CTC sequence IDs.
    """
    tuned_logits = logits
    if blank_bias != 0.0:
        tuned_logits = tuned_logits.clone()
        tuned_logits[:, blank_id] = tuned_logits[:, blank_id] + float(blank_bias)

    probs = torch.softmax(tuned_logits, dim=-1)
    pred_ids_t = probs.argmax(dim=-1)  # (T,)

    if min_phone_conf is not None and min_phone_conf > 0.0:
        max_conf = probs.max(dim=-1).values
        low_conf = max_conf < float(min_phone_conf)
        if low_conf.any():
            pred_ids_t = pred_ids_t.clone()
            pred_ids_t[low_conf] = int(blank_id)

    pred_ids = pred_ids_t.detach().cpu().numpy().astype(np.int64)
    seq_ids = np.array(collapse_ctc_ids(pred_ids, blank_id), dtype=np.int64)
    return pred_ids, seq_ids


def ids_to_tokens(ids: np.ndarray | list[int], labels: list[str]) -> list[str]:
    out: list[str] = []
    for pid in ids:
        i = int(pid)
        if 0 <= i < len(labels):
            out.append(normalize_token(labels[i]))
        else:
            out.append("[OOB]")
    return out


def confidence_from_logits(logits: torch.Tensor) -> float:
    """Mean max-softmax confidence across frames."""
    probs = torch.softmax(logits, dim=-1)
    conf = probs.max(dim=-1).values.mean()
    return float(conf.detach().cpu().item())


def edit_distance(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> int:
    a_list = [int(x) for x in a]
    b_list = [int(x) for x in b]
    n, m = len(a_list), len(b_list)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a_list[i - 1] == b_list[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def phone_error_rate(pred_ids: np.ndarray, ref_ids: np.ndarray) -> float:
    denom = max(int(len(ref_ids)), 1)
    return float(edit_distance(pred_ids.tolist(), ref_ids.tolist()) / denom)


class PhoneTeacher:
    """Frozen phone CTC model used for pseudo-label generation."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)  # type: ignore[arg-type]
        self.model.eval()

        vocab = self.processor.tokenizer.get_vocab()  # type: ignore[attr-defined]
        id_to_token = {v: k for k, v in vocab.items()}
        labels = [normalize_token(id_to_token[i]) for i in range(len(vocab))]

        pad_id = vocab.get("[PAD]")
        if pad_id is None:
            pad_id = self.processor.tokenizer.pad_token_id  # type: ignore[attr-defined]
        if pad_id is None:
            pad_id = 0

        self.vocab = PhoneVocabulary(labels=labels, blank_id=int(pad_id))

    def _frame_ids_raw(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits
        return logits[0].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    def pseudo_labels(self, waveform: np.ndarray, sr: int, target_T: int) -> PseudoLabelBundle:
        """
        Build teacher pseudo labels aligned to target frame count target_T.
        """
        raw_ids = self._frame_ids_raw(waveform, sr)

        if len(raw_ids) == target_T:
            frame_ids = raw_ids
        else:
            idx = (np.arange(target_T) * len(raw_ids) / max(target_T, 1)).astype(int)
            idx = np.clip(idx, 0, max(len(raw_ids) - 1, 0))
            frame_ids = raw_ids[idx]

        target_ids = np.array(collapse_ctc_ids(frame_ids, self.vocab.blank_id), dtype=np.int64)
        return PseudoLabelBundle(
            frame_ids=frame_ids,
            target_ids=target_ids,
            vocab=self.vocab,
        )


def phone_cache_path(cache_root: Path, utt_id: str) -> Path:
    out = cache_root / "phone_labels"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{utt_id}_phones.pkl"


def save_pseudo_label_cache(path: Path, bundle: PseudoLabelBundle) -> None:
    payload = {
        "frame_ids": bundle.frame_ids,
        "target_ids": bundle.target_ids,
        "labels": bundle.vocab.labels,
        "blank_id": bundle.vocab.blank_id,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_pseudo_label_cache(path: Path) -> PseudoLabelBundle:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    vocab = PhoneVocabulary(
        labels=[normalize_token(x) for x in payload["labels"]],
        blank_id=int(payload["blank_id"]),
    )
    return PseudoLabelBundle(
        frame_ids=np.asarray(payload["frame_ids"], dtype=np.int64),
        target_ids=np.asarray(payload["target_ids"], dtype=np.int64),
        vocab=vocab,
    )


def ctc_loss_from_logits(
    logits: torch.Tensor,   # (T, V)
    target_ids: np.ndarray, # (U,)
    blank_id: int,
) -> torch.Tensor:
    """Compute CTC loss for a single utterance."""
    T = int(logits.shape[0])
    ctc = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    # CTC requires non-empty targets; skip with zero loss if empty.
    if int(target_ids.size) == 0:
        return logits.sum() * 0.0

    # MPS backend currently lacks aten::_ctc_loss. Fall back to CPU for this op.
    # Autograd still propagates through the device copy.
    if logits.device.type == "mps":
        log_probs_cpu = F.log_softmax(logits.float(), dim=-1).unsqueeze(1).cpu()  # (T,1,V)
        targets_cpu = torch.from_numpy(target_ids.astype(np.int64)).cpu()
        input_lengths_cpu = torch.tensor([T], dtype=torch.long)
        target_lengths_cpu = torch.tensor([int(targets_cpu.numel())], dtype=torch.long)
        loss_cpu = ctc(log_probs_cpu, targets_cpu, input_lengths_cpu, target_lengths_cpu)
        return loss_cpu.to(logits.device)

    log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1)  # (T, 1, V)
    input_lengths = torch.tensor([T], dtype=torch.long, device=logits.device)
    targets = torch.from_numpy(target_ids.astype(np.int64)).to(logits.device)
    target_lengths = torch.tensor([int(targets.numel())], dtype=torch.long, device=logits.device)
    return ctc(log_probs, targets, input_lengths, target_lengths)
