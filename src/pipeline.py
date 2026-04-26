"""
Module: pipeline
Phase: 7
Goal: End-to-end SpeechWorldModelPipeline.

Chains Phases 3–6 into a single reusable object:
  audio  →  WavLM layer-24  →  EvidenceProjector  →  Slotizer  →  BeliefTransitionGRU
         (cached to disk)      (trainable, 256-dim)  (mean/attn)  (trainable, GRU)

What this teaches:
- All previous phases live in a single coherent forward pass
- Caching WavLM features avoids redundant computation across training epochs
- A structured PipelineResult keeps tensor shapes explicit and inspectable
"""

# --- Imports ---
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMModel

# Import canonical model classes from their phase modules
from huper_features import EvidenceProjector                    # Phase 3
from syllable_clock import detect_syllables, HOP_LENGTH         # Phase 4
from slotizer import MeanPoolSlotizer, AttentionSlotizer        # Phase 5
from belief_model import BeliefTransitionGRU                    # Phase 6
from feature_sources import extract_streaming_features, FeatureBundle  # Phase 8 — streaming wiring
from phone_ctc import (
    PhoneCTCHead,
    confidence_from_logits,
    decode_logits_greedy,
    ids_to_tokens,
)

matplotlib.use("Agg")

# --- Constants ---
TARGET_SR   = 16_000
WAVLM_ID    = "microsoft/wavlm-large"
D_RAW       = 1024      # WavLM layer-24 width
D_PROJ      = 256       # EvidenceProjector output / world-model dim
LAYER_IDX   = 24        # which WavLM transformer layer to use

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"

ROOT_DIR   = Path(__file__).parent.parent
CACHE_DIR  = ROOT_DIR / "data" / "cache"
FIG_DIR    = ROOT_DIR / "data" / "figures_phase7"


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult — structured output from a single utterance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """All intermediate and final tensors for one utterance.

    Every field has an explicit shape comment so callers never have to guess.
    """
    utt_id:      str
    waveform:    np.ndarray     # (N_samples,)
    sample_rate: int
    layer24:     np.ndarray | None  # (T, 1024) frozen WavLM features — None in streaming mode
    E_t:         np.ndarray     # (T, 256)   projected evidence
    T:           int
    boundaries:  list[tuple[int, int]]   # K pairs of (start_frame, end_frame)
    K:           int
    slots:       np.ndarray     # (K, 256)   syllable-pooled evidence
    beliefs:     np.ndarray     # (K, 256)   GRU belief states
    predictions: np.ndarray     # (K-1, 256) next-slot predictions
    mismatch:    np.ndarray     # (K-1,)     1 − cosine(pred, target)
    durations_ms: np.ndarray    # (K,)       milliseconds per syllable slot
    pooling:     str            # "mean" | "attn"
    phone_logits: np.ndarray | None = None      # (T, V) if phone mode enabled
    phone_ids: np.ndarray | None = None         # (T,) argmax IDs per frame
    phone_sequence_ids: np.ndarray | None = None  # (U,) CTC-collapsed IDs
    phone_tokens: list[str] | None = None       # (T,) frame token strings
    phone_sequence_tokens: list[str] | None = None  # (U,) collapsed token strings
    phone_confidence: float | None = None
    metadata:    dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def log_shape(name: str, x: torch.Tensor | np.ndarray) -> None:
    if isinstance(x, torch.Tensor):
        print(f"  {name:40s}: {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    else:
        print(f"  {name:40s}: {x.shape}  dtype={x.dtype}")


def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:  # type: ignore[name-defined]
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Pure-numpy 2-component PCA.  X: (N, D) → (N, 2)"""
    Xc = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(Xc.T))
    top2 = vecs[:, np.argsort(vals)[::-1][:2]]
    return Xc @ top2


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load any audio file → mono float32 numpy array at target_sr."""
    waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return waveform.astype(np.float32), int(sr)


def _utt_hash(utt_id: str) -> str:
    """Short deterministic hash for cache filenames."""
    return hashlib.md5(utt_id.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# WavLM feature extractor — loaded once, reused across all utterances
# ─────────────────────────────────────────────────────────────────────────────

class WavLMExtractor:
    """Loads WavLM-Large once and extracts layer-24 hidden states on demand.

    Results are cached to disk so subsequent calls skip WavLM entirely.
    """

    def __init__(self, device: torch.device, cache_dir: Path = CACHE_DIR) -> None:
        self.device    = device
        self.cache_dir = cache_dir / "features"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load: model is loaded on first call to extract()
        self._feat_extractor = None
        self._wavlm: WavLMModel | None = None

    def _ensure_loaded(self) -> None:
        if self._wavlm is None:
            print(f"  Loading {WAVLM_ID} … (one-time)")
            self._feat_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
            self._wavlm = WavLMModel.from_pretrained(WAVLM_ID).to(self.device)   # type: ignore[arg-type]
            self._wavlm.eval()
            n = sum(p.numel() for p in self._wavlm.parameters())
            print(f"  WavLM-Large loaded  ({n / 1e6:.0f}M params)  device={self.device}")

    def extract(self, waveform: np.ndarray, sr: int, utt_id: str) -> np.ndarray:
        """
        Return layer-24 hidden states for waveform.

        Parameters
        ----------
        waveform : (N_samples,)
        sr       : sample rate (should be 16kHz)
        utt_id   : unique utterance identifier for caching

        Returns
        -------
        layer24 : (T, 1024)  frozen WavLM features
        """
        cache_path = self.cache_dir / f"{_utt_hash(utt_id)}_layer24.pt"

        if cache_path.exists():
            layer24 = torch.load(cache_path, weights_only=True).numpy()
            return layer24  # (T, 1024)

        self._ensure_loaded()
        fe  = self._feat_extractor
        mdl = self._wavlm

        inputs = fe(waveform, sampling_rate=sr, return_tensors="pt", padding=True)  # type: ignore[operator]
        with torch.no_grad():
            out = mdl(inputs.input_values.to(self.device), output_hidden_states=True)  # type: ignore[union-attr,operator]

        # hidden_states: tuple[25]  index 0=CNN, 1-24=transformer
        layer24 = out.hidden_states[LAYER_IDX].squeeze(0).cpu()   # type: ignore[index]  # (T, 1024)
        torch.save(layer24, cache_path)
        return layer24.numpy()   # (T, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary cache — syllable boundaries are computed once from raw audio
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryCache:
    """Detect and cache syllable boundaries for each utterance."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir / "boundaries"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, waveform: np.ndarray, sr: int, utt_id: str) -> list[tuple[int, int]]:
        """Return (start_frame, end_frame) list at 50Hz, from cache or computed."""
        cache_path = self.cache_dir / f"{_utt_hash(utt_id)}_boundaries.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        boundaries, method = detect_syllables(waveform, sr)
        print(f"  [{utt_id[:20]}]  K={len(boundaries)}  method={method}")
        with open(cache_path, "wb") as fh:
            pickle.dump(boundaries, fh)
        return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# SpeechWorldModelPipeline — orchestrates all phases into one forward pass
# ─────────────────────────────────────────────────────────────────────────────

class SpeechWorldModelPipeline:
    """
    End-to-end pipeline:
        audio  →  layer-24 (cached)  →  E_t (256)  →  S_k (256)  →  B_k (256)

    All model components are passed in at construction time so the pipeline
    can be used with both randomly-initialised (demo) and trained (verify) models.

    Parameters
    ----------
    projector    : EvidenceProjector — Linear(1024→256)+GELU
    slotizer     : MeanPoolSlotizer | AttentionSlotizer
    belief_model : BeliefTransitionGRU
    pooling      : "mean" or "attn"
    device       : torch.device
    cache_dir    : where to store layer-24 features and boundaries
    """

    def __init__(
        self,
        projector:    EvidenceProjector,
        slotizer:     nn.Module,
        belief_model: BeliefTransitionGRU,
        phone_head:   PhoneCTCHead | None       = None,
        phone_meta:   dict | None               = None,
        pooling:      Literal["mean", "attn"] = "mean",
        device:       torch.device            = torch.device("cpu"),
        cache_dir:    Path                    = CACHE_DIR,
    ) -> None:
        self.projector    = projector.to(device)
        self.slotizer     = slotizer.to(device)
        self.belief_model = belief_model.to(device)
        self.phone_head   = phone_head.to(device) if phone_head is not None else None
        self.phone_meta   = phone_meta or {}
        self.pooling      = pooling
        self.device       = device

        self._wavlm_extractor = WavLMExtractor(device=device, cache_dir=cache_dir)
        self._boundary_cache  = BoundaryCache(cache_dir=cache_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Step-by-step public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(
        self,
        waveform: np.ndarray,
        sr:       int,
        utt_id:   str,
    ) -> torch.Tensor:
        """
        WavLM layer-24 (cached) → EvidenceProjector → E_t.

        Returns
        -------
        E_t : (T, 256)  on self.device
        """
        layer24 = self._wavlm_extractor.extract(waveform, sr, utt_id)
        # layer24 : (T, 1024)  numpy

        x = torch.from_numpy(layer24).to(self.device)   # (T, 1024)
        self.projector.eval()
        with torch.no_grad():
            E_t = self.projector(x)   # (T, 256)
        return E_t   # (T, 256)

    def extract_features_train(
        self,
        layer24_np: np.ndarray,
    ) -> torch.Tensor:
        """
        Project pre-loaded layer-24 features (for use during training where
        gradients must flow through the projector).

        Parameters
        ----------
        layer24_np : (T, 1024)  numpy, already loaded from cache

        Returns
        -------
        E_t : (T, 256)  on self.device  (with grad)
        """
        x = torch.from_numpy(layer24_np).to(self.device)   # (T, 1024)
        return self.projector(x)   # (T, 256)  — keeps gradient graph alive

    def segment_syllables(
        self,
        waveform: np.ndarray,
        sr:       int,
        utt_id:   str,
    ) -> list[tuple[int, int]]:
        """Return (start_frame, end_frame) boundaries at 50Hz from cache."""
        return self._boundary_cache.get(waveform, sr, utt_id)

    def build_slots(
        self,
        E_t:        torch.Tensor,            # (T, 256)
        boundaries: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Pool E_t frames into syllable slots.

        Returns
        -------
        S : (K, 256)
        """
        if self.pooling == "mean":
            return self.slotizer(E_t, boundaries)          # (K, 256)
        else:
            S, _ = self.slotizer(E_t, boundaries)          # (K, 256)
            return S

    def infer_beliefs(
        self,
        S: torch.Tensor,   # (K, 256)  or (1, K, 256)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run BeliefTransitionGRU on slot sequence.

        Returns
        -------
        B    : (K, 256)    belief states
        pred : (K-1, 256)  next-slot predictions
        L    : (K, 256)    language prior (auxiliary)
        """
        if S.dim() == 2:
            S = S.unsqueeze(0)   # → (1, K, 256)

        self.belief_model.eval()
        with torch.no_grad():
            B, pred, L = self.belief_model(S)

        return (
            B.squeeze(0).cpu().numpy(),      # (K, 256)
            pred.squeeze(0).cpu().numpy(),   # (K-1, 256)
            L.squeeze(0).cpu().numpy(),      # (K, 256)
        )

    def run_full(
        self,
        audio_path:      str | Path,
        utt_id:          str | None = None,
        pooling:         str | None = None,
        mode:            Literal["offline", "streaming"] = "offline",
        chunk_ms:        int = 320,
        left_context_ms: int = 640,
        lookahead_ms:    int = 40,
        enable_phone_mode: bool = False,
        phone_blank_bias: float = 0.0,
        phone_min_conf: float = 0.0,
    ) -> PipelineResult:
        """
        Complete pipeline: audio path → PipelineResult.

        Parameters
        ----------
        audio_path      : path to a 16kHz mono wav file
        utt_id          : unique id used for caching (defaults to filename stem)
        pooling         : "mean" | "attn"  (overrides self.pooling if given)
        mode            : "offline" (default) — cached full-context WavLM + projector;
                          "streaming" — pseudo-streaming windowed WavLM via
                          feature_sources.extract_streaming_features; bypasses
                          WavLMExtractor cache; returns layer24=None.
        chunk_ms        : (streaming only) audio chunk size in ms (default 320)
        left_context_ms : (streaming only) history window in ms (default 640)
        lookahead_ms    : (streaming only) right-lookahead in ms (default 40)
        enable_phone_mode: if True, decode frame-level phones from E_t using
                           self.phone_head + self.phone_meta.
        phone_blank_bias: add bias to blank logit before greedy decoding.
        phone_min_conf  : if >0, force low-confidence frames to blank.
        """
        audio_path = Path(audio_path)
        if utt_id is None:
            utt_id = audio_path.stem
        if pooling is not None:
            self.pooling = pooling  # type: ignore[assignment]

        # 1. Load audio
        waveform, sr = load_audio(audio_path)
        print(f"\n[pipeline] {utt_id}")
        print(f"  audio        : {waveform.shape[0]/sr:.2f}s  ({waveform.shape[0]} samples @ {sr}Hz)")

        # 2. Acoustic evidence extraction (branches on mode)
        if mode == "offline":
            # Offline path: cached WavLM layer-24 → EvidenceProjector (unchanged)
            layer24 = self._wavlm_extractor.extract(waveform, sr, utt_id)
            T = layer24.shape[0]
            log_shape("layer24", layer24)

            x = torch.from_numpy(layer24).to(self.device)
            self.projector.eval()
            with torch.no_grad():
                E_t_tensor = self.projector(x)   # (T, 256)
            E_t = E_t_tensor.cpu().numpy()
            log_shape("E_t", E_t_tensor)

            source_latency_ms = 0.0
            source_wall_s     = 0.0
            source_config: dict = {"mode": "offline"}
        elif mode == "streaming":
            # Streaming path: extract_streaming_features applies the projector
            # internally, so we must NOT re-project here.
            self._wavlm_extractor._ensure_loaded()   # force HF objects into memory
            assert self._wavlm_extractor._wavlm is not None
            assert self._wavlm_extractor._feat_extractor is not None
            bundle: FeatureBundle = extract_streaming_features(
                waveform           = waveform,
                sr                 = sr,
                projector          = self.projector,
                wavlm_model        = self._wavlm_extractor._wavlm,
                feat_extractor     = self._wavlm_extractor._feat_extractor,
                device             = self.device,
                chunk_ms           = chunk_ms,
                left_context_ms    = left_context_ms,
                right_lookahead_ms = lookahead_ms,
            )
            layer24    = None          # streaming never materialises (T, 1024)
            E_t        = bundle.E_t   # (T, 256) numpy — already projected
            E_t_tensor = torch.from_numpy(E_t).to(self.device)
            T          = bundle.T
            log_shape("E_t (streaming)", E_t_tensor)
            source_latency_ms = bundle.latency_ms
            source_wall_s     = bundle.wall_s
            source_config     = bundle.config
        else:
            raise ValueError(
                f"run_full(mode=...) must be 'offline' or 'streaming', got {mode!r}"
            )

        # 4. Syllable boundaries (cached) — clamp to actual T so streaming
        # T_stream (±1-3 frames vs T_offline) never causes out-of-bounds slicing
        raw_boundaries = self._boundary_cache.get(waveform, sr, utt_id)
        boundaries = [
            (s, min(e, T - 1))
            for s, e in raw_boundaries
            if s < T
        ]
        K = len(boundaries)
        print(f"  boundaries   : K={K}  ({K / (waveform.shape[0]/sr):.1f} Hz)  mode={mode}")

        # 5. Slotizer
        slots_tensor = self.build_slots(E_t_tensor, boundaries)   # (K, 256)
        slots = slots_tensor.cpu().numpy()
        log_shape("slots", slots_tensor)

        # 5b. Optional phone mode: E_t -> phone logits -> greedy CTC decode
        phone_logits_np: np.ndarray | None = None
        phone_ids_np: np.ndarray | None = None
        phone_seq_ids_np: np.ndarray | None = None
        phone_tokens: list[str] | None = None
        phone_seq_tokens: list[str] | None = None
        phone_conf: float | None = None

        if enable_phone_mode:
            if self.phone_head is None:
                raise ValueError(
                    "enable_phone_mode=True but pipeline has no phone_head. "
                    "Rebuild with build_pipeline(..., enable_phone_mode=True, phone_vocab_size=...)."
                )
            if "blank_id" not in self.phone_meta or "labels" not in self.phone_meta:
                raise ValueError(
                    "enable_phone_mode=True but phone_meta missing blank_id/labels."
                )
            blank_id = int(self.phone_meta["blank_id"])
            labels = list(self.phone_meta["labels"])

            self.phone_head.eval()
            with torch.no_grad():
                logits_phone = self.phone_head(E_t_tensor)  # (T, V)
            phone_logits_np = logits_phone.cpu().numpy()
            phone_ids_np, phone_seq_ids_np = decode_logits_greedy(
                logits_phone,
                blank_id=blank_id,
                blank_bias=phone_blank_bias,
                min_phone_conf=phone_min_conf if phone_min_conf > 0.0 else None,
            )
            phone_tokens = ids_to_tokens(phone_ids_np, labels)
            phone_seq_tokens = ids_to_tokens(phone_seq_ids_np, labels)
            phone_conf = confidence_from_logits(logits_phone)
            print(
                f"  phone mode   : T={len(phone_ids_np)}  U={len(phone_seq_ids_np)}  "
                f"conf={phone_conf:.3f}"
            )

        # 6. BeliefTransitionGRU
        B_np, pred_np, L_np = self.infer_beliefs(slots_tensor)
        log_shape("beliefs", B_np)
        log_shape("predictions", pred_np)

        # 7. Mismatch signal
        pred_t = torch.from_numpy(pred_np)
        tgt_t  = torch.from_numpy(slots[1:])   # (K-1, 256)
        cos    = F.cosine_similarity(pred_t, tgt_t, dim=-1).numpy()
        mismatch = (1.0 - cos).astype(np.float32)   # (K-1,)

        # 8. Durations
        durations_ms = np.array(
            [(e - s + 1) * HOP_LENGTH / sr * 1000 for s, e in boundaries],
            dtype=np.float32,
        )

        print(f"  mismatch     : mean={mismatch.mean():.4f}  max={mismatch.max():.4f}")
        print(f"  belief cos   : mean={float(F.cosine_similarity(torch.from_numpy(B_np[:-1]), torch.from_numpy(B_np[1:]), dim=-1).mean()):.4f}")

        return PipelineResult(
            utt_id       = utt_id,
            waveform     = waveform,
            sample_rate  = sr,
            layer24      = layer24,
            E_t          = E_t,
            T            = T,
            boundaries   = boundaries,
            K            = K,
            slots        = slots,
            beliefs      = B_np,
            predictions  = pred_np,
            mismatch     = mismatch,
            durations_ms = durations_ms,
            pooling      = self.pooling,
            phone_logits = phone_logits_np,
            phone_ids = phone_ids_np,
            phone_sequence_ids = phone_seq_ids_np,
            phone_tokens = phone_tokens,
            phone_sequence_tokens = phone_seq_tokens,
            phone_confidence = phone_conf,
            metadata     = {
                "duration_s":    waveform.shape[0] / sr,
                "T":             T,
                "K":             K,
                "sr":            sr,
                "utt_id":        utt_id,
                "mode":          mode,
                "enable_phone_mode": enable_phone_mode,
                "phone_blank_bias": phone_blank_bias,
                "phone_min_conf": phone_min_conf,
                "latency_ms":    source_latency_ms,
                "source_wall_s": source_wall_s,
                "source_config": source_config,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build a pipeline with freshly-initialised models
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    pooling:    Literal["mean", "attn"] = "mean",
    device:     torch.device | None     = None,
    cache_dir:  Path                    = CACHE_DIR,
    enable_phone_mode: bool             = False,
    phone_vocab_size: int | None        = None,
    phone_meta: dict | None             = None,
    phone_head_hidden_dim: int          = 0,
) -> SpeechWorldModelPipeline:
    """
    Instantiate all model components with random init weights and wrap in pipeline.
    Used for demos and for initialising models before training.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    projector = EvidenceProjector(d_in=D_RAW, d_out=D_PROJ)
    slotizer: nn.Module = MeanPoolSlotizer() if pooling == "mean" else AttentionSlotizer(D_PROJ)
    belief_model = BeliefTransitionGRU(d=D_PROJ)
    phone_head: PhoneCTCHead | None = None
    if enable_phone_mode:
        if phone_vocab_size is None:
            raise ValueError("phone_vocab_size is required when enable_phone_mode=True")
        hidden_dim = phone_head_hidden_dim if phone_head_hidden_dim > 0 else None
        phone_head = PhoneCTCHead(
            d_in=D_PROJ,
            vocab_size=phone_vocab_size,
            hidden_dim=hidden_dim,
        )

    return SpeechWorldModelPipeline(
        projector    = projector,
        slotizer     = slotizer,
        belief_model = belief_model,
        phone_head   = phone_head,
        phone_meta   = phone_meta,
        pooling      = pooling,
        device       = device,
        cache_dir    = cache_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — Viz 7-1: full story in one image
# ─────────────────────────────────────────────────────────────────────────────

def viz7_1_pipeline_overview(result: PipelineResult) -> Path:
    """
    Viz 7-1 — End-to-end pipeline overview for one utterance.

    Panels (top to bottom):
      1. Waveform
      2. Log-mel spectrogram
      3. Syllable boundary markers
      4. Slot timeline (mean E_t per slot)
      5. Mismatch signal
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    waveform = result.waveform
    sr       = result.sample_rate
    bounds   = result.boundaries
    slots    = result.slots       # (K, 256)
    mismatch = result.mismatch    # (K-1,)
    K        = result.K

    hop = HOP_LENGTH
    t_frames = np.arange(result.T) * hop / sr
    t_audio  = np.arange(len(waveform)) / sr
    duration = t_audio[-1]

    # Slot time axis: midpoint of each boundary in seconds
    slot_times = np.array([
        (s + e) / 2 * hop / sr for s, e in bounds
    ])

    fig, axes = plt.subplots(
        5, 1, figsize=(16, 12), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.5, "height_ratios": [2, 2.5, 0.8, 2, 1.5]},
    )
    fig.suptitle(
        f"Viz 7-1 — Full Pipeline  |  {result.utt_id}  |  "
        f"T={result.T}  K={K}  pooling={result.pooling}",
        color="white", fontsize=12, fontweight="bold",
    )

    # Panel 1: Waveform
    ax = axes[0]
    ax.plot(t_audio, waveform, color="#4fc3f7", lw=0.4, alpha=0.85)
    ax.set_xlim(0, duration)
    _style_ax(ax, title="Waveform", xlabel="", ylabel="Amplitude")

    # Panel 2: Log-mel spectrogram
    ax = axes[1]
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, hop_length=hop,
                                          n_mels=80, fmin=80, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (80, T)
    ax.imshow(mel_db, origin="lower", aspect="auto",
              extent=[0, duration, 0, 80],
              cmap="magma", vmin=-80, vmax=0, interpolation="nearest")
    _style_ax(ax, title="Log-mel spectrogram  (80 bands)", ylabel="Mel band")

    # Panel 3: Syllable boundaries as tick marks
    ax = axes[2]
    ax.set_facecolor(PANEL_BG)
    for s, e in bounds:
        mid = (s + e) / 2 * hop / sr
        ax.axvline(mid, color="#ffd740", lw=0.8, alpha=0.7)
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.spines[:].set_color(SPINE_C)
    ax.set_title(f"Syllable clock  K={K}  "
                 f"({K / duration:.1f} Hz)",
                 color="white", fontsize=9, pad=4)

    # Panel 4: Slot timeline — first principal component of slots
    ax = axes[3]
    if slots.shape[0] > 2:
        slot_pc1 = pca_2d(slots)[:, 0]   # (K,)
    else:
        slot_pc1 = slots.mean(axis=1)     # fallback
    ax.bar(np.arange(K), slot_pc1, color="#81c784", width=0.7)
    ax.set_xlim(-0.5, K - 0.5)
    _style_ax(ax, title="Slot evidence  S_k  (PC-1 of 256-dim)",
              xlabel="Slot k", ylabel="PC-1")

    # Panel 5: Mismatch signal
    ax = axes[4]
    k1_axis = np.arange(len(mismatch))
    safe_max = max(float(mismatch.max()), 1e-4)
    ax.bar(k1_axis, mismatch, color="#ff7043", width=0.7)
    ax.axhline(float(mismatch.mean()), color="white", lw=0.8, linestyle=":",
               label=f"mean={mismatch.mean():.4f}")
    ax.set_xlim(-0.5, len(mismatch) - 0.5)
    ax.set_ylim(-safe_max * 0.05, safe_max * 1.35)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=SPINE_C)
    _style_ax(ax, title="Mismatch r_k = 1 − cos(Ŝ_{k+1}, S_{k+1})",
              xlabel="Slot k", ylabel="Mismatch")

    out = FIG_DIR / "viz7_1_pipeline_overview.png"
    plt.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Demo — __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("█" * 60)
    print("  PHASE 7 — pipeline.py DEMO")
    print("  Speech World Model: end-to-end on one utterance")
    print("█" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    sample_path = ROOT_DIR / "data" / "samples" / "librispeech_sample.wav"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    print("\n--- Building pipeline (random-init weights) ---")
    pipeline = build_pipeline(pooling="mean", device=device)

    print("\n--- Running full pipeline ---")
    result = pipeline.run_full(sample_path, utt_id="librispeech_demo")

    print("\n--- Pipeline Result Summary ---")
    print(f"  utt_id       : {result.utt_id}")
    print(f"  duration     : {result.metadata['duration_s']:.2f}s")
    print(f"  T (frames)   : {result.T}  @ 50Hz")
    print(f"  K (slots)    : {result.K}  @ {result.K / result.metadata['duration_s']:.1f}Hz")
    print(f"  pooling      : {result.pooling}")
    print(f"  E_t shape    : {result.E_t.shape}")
    print(f"  slots shape  : {result.slots.shape}")
    print(f"  beliefs shape: {result.beliefs.shape}")
    print(f"  pred shape   : {result.predictions.shape}")
    print(f"  mismatch     : mean={result.mismatch.mean():.4f}  max={result.mismatch.max():.4f}")

    print("\n--- Viz 7-1: pipeline overview ---")
    viz7_1_pipeline_overview(result)

    print("\n  Pipeline demo complete.")
    print(f"  Figures → {FIG_DIR}")
