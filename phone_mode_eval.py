"""
phone_mode_eval.py
Evaluate phone mode with:
1) Frame token agreement vs cached pseudo labels
2) Sequence PER-style metric vs pseudo labels
3) Offline-vs-streaming phone agreement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from phone_ctc import (  # noqa: E402
    PhoneTeacher,
    edit_distance,
    load_pseudo_label_cache,
    phone_cache_path,
    phone_error_rate,
    save_pseudo_label_cache,
)
from pipeline import TARGET_SR, build_pipeline  # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "data" / "checkpoints" / "belief_model_best.pt"
DEFAULT_OUT = ROOT / "data" / "figures_phase8_final" / "phone_mode_eval.json"


def load_utterances(n: int) -> list[dict]:
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    out = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        audio = item["audio"]
        wav = np.array(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        out.append(
            {
                "id": f"val_{i:04d}",
                "waveform": wav,
                "sr": sr,
                "duration_s": len(wav) / sr,
            }
        )
    return out


def load_pipeline(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "phone_head" not in ckpt or "phone_meta" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain phone mode artifacts. "
            "Train with --enable_phone_mode first."
        )

    cfg = ckpt.get("cfg", {})
    phone_meta = ckpt["phone_meta"]
    pipeline = build_pipeline(
        pooling=cfg.get("pooling", "mean"),
        device=device,
        enable_phone_mode=True,
        phone_vocab_size=len(phone_meta["labels"]),
        phone_meta=phone_meta,
        phone_head_hidden_dim=int(cfg.get("phone_head_hidden_dim", 0)),
    )
    pipeline.projector.load_state_dict(ckpt["projector"])
    pipeline.slotizer.load_state_dict(ckpt["slotizer"])
    pipeline.belief_model.load_state_dict(ckpt["belief_model"])
    pipeline.phone_head.load_state_dict(ckpt["phone_head"])

    pipeline.projector.eval()
    pipeline.slotizer.eval()
    pipeline.belief_model.eval()
    pipeline.phone_head.eval()

    return pipeline, ckpt


def teacher_bundle_cached(
    teacher: PhoneTeacher,
    cache_root: Path,
    utt_id: str,
    waveform: np.ndarray,
    sr: int,
    T: int,
):
    key = f"{utt_id}_T{T}"
    p = phone_cache_path(cache_root, key)
    if p.exists():
        return load_pseudo_label_cache(p)
    bundle = teacher.pseudo_labels(waveform, sr, target_T=T)
    save_pseudo_label_cache(p, bundle)
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate phone mode")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--n_utt", type=int, default=2)
    ap.add_argument(
        "--preset",
        type=str,
        default="best_w20",
        choices=["none", "best_w20"],
        help="Decode preset; best_w20 applies the current best operating point.",
    )
    ap.add_argument("--lookahead_ms", type=int, default=80)
    ap.add_argument("--blank_bias", type=float, default=0.2, help="Bias added to blank logit during decode")
    ap.add_argument("--min_phone_conf", type=float, default=0.35, help="Frames below this confidence forced to blank")
    ap.add_argument("--out_json", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.preset == "best_w20":
        args.lookahead_ms = 80
        args.blank_bias = 0.2
        args.min_phone_conf = 0.35

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline, ckpt = load_pipeline(args.ckpt, device)

    cfg = ckpt.get("cfg", {})
    teacher_model_id = cfg.get("teacher_model_id", "vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
    teacher = PhoneTeacher(teacher_model_id, device)

    cache_root = ROOT / "data" / "cache"
    utterances = load_utterances(args.n_utt)

    rows = []
    for utt in utterances:
        utt_id = utt["id"]
        wav = utt["waveform"]
        sr = utt["sr"]

        # Save temp wav because pipeline.run_full currently takes file path.
        wav_path = cache_root / "phone_eval_wavs" / f"{utt_id}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(str(wav_path), wav, sr)

        off = pipeline.run_full(
            wav_path,
            utt_id=f"{utt_id}_off",
            mode="offline",
            enable_phone_mode=True,
            phone_blank_bias=args.blank_bias,
            phone_min_conf=args.min_phone_conf,
        )
        st = pipeline.run_full(
            wav_path,
            utt_id=f"{utt_id}_str",
            mode="streaming",
            lookahead_ms=args.lookahead_ms,
            enable_phone_mode=True,
            phone_blank_bias=args.blank_bias,
            phone_min_conf=args.min_phone_conf,
        )

        off_bundle = teacher_bundle_cached(teacher, cache_root, f"{utt_id}_off", wav, sr, off.T)
        st_bundle = teacher_bundle_cached(teacher, cache_root, f"{utt_id}_str", wav, sr, st.T)

        off_ids = np.asarray(off.phone_ids if off.phone_ids is not None else np.array([], dtype=np.int64))
        st_ids = np.asarray(st.phone_ids if st.phone_ids is not None else np.array([], dtype=np.int64))
        off_seq = np.asarray(off.phone_sequence_ids if off.phone_sequence_ids is not None else np.array([], dtype=np.int64))
        st_seq = np.asarray(st.phone_sequence_ids if st.phone_sequence_ids is not None else np.array([], dtype=np.int64))

        off_frame_T = min(len(off_ids), len(off_bundle.frame_ids))
        st_frame_T = min(len(st_ids), len(st_bundle.frame_ids))

        off_frame_acc = float((off_ids[:off_frame_T] == off_bundle.frame_ids[:off_frame_T]).mean()) if off_frame_T > 0 else float("nan")
        st_frame_acc = float((st_ids[:st_frame_T] == st_bundle.frame_ids[:st_frame_T]).mean()) if st_frame_T > 0 else float("nan")

        off_per = phone_error_rate(off_seq, off_bundle.target_ids)
        st_per = phone_error_rate(st_seq, st_bundle.target_ids)

        # Agreement score in [0, 1]:
        # 1.0 = identical sequences, 0.0 = completely mismatched by normalized edit distance.
        off_stream_den = max(len(off_seq), len(st_seq), 1)
        off_stream_agree = float(
            1.0
            - edit_distance(
                off_seq.tolist(),
                st_seq.tolist(),
            )
            / off_stream_den
        )
        off_stream_agree = float(np.clip(off_stream_agree, 0.0, 1.0))

        seq_len_ratio = float(len(st_seq) / max(len(off_seq), 1))

        row = {
            "utt_id": utt_id,
            "duration_s": utt["duration_s"],
            "lookahead_ms": args.lookahead_ms,
            "offline": {
                "T": off.T,
                "K": off.K,
                "frame_acc_vs_teacher": off_frame_acc,
                "per_vs_teacher": off_per,
                "seq_len": len(off_seq),
                "confidence": off.phone_confidence,
            },
            "streaming": {
                "T": st.T,
                "K": st.K,
                "frame_acc_vs_teacher": st_frame_acc,
                "per_vs_teacher": st_per,
                "seq_len": len(st_seq),
                "confidence": st.phone_confidence,
            },
            "offline_vs_streaming_seq_agreement": off_stream_agree,
            "offline_vs_streaming_seq_len_ratio": seq_len_ratio,
        }
        rows.append(row)

        print(
            f"{utt_id:10s}  off_acc={off_frame_acc:.4f}  str_acc={st_frame_acc:.4f}  "
            f"off_per={off_per:.4f}  str_per={st_per:.4f}  off-vs-str={off_stream_agree:.4f}"
        )

    def mean_of(path: list[str]) -> float:
        vals = rows
        for key in path[:-1]:
            vals = [x[key] for x in vals]
        leaf = path[-1]
        arr = [x[leaf] for x in vals if not np.isnan(float(x[leaf]))]
        return float(np.mean(arr)) if arr else float("nan")

    aggregate = {
        "n_utt": len(rows),
        "lookahead_ms": args.lookahead_ms,
        "offline_frame_acc_vs_teacher": mean_of(["offline", "frame_acc_vs_teacher"]),
        "streaming_frame_acc_vs_teacher": mean_of(["streaming", "frame_acc_vs_teacher"]),
        "offline_per_vs_teacher": mean_of(["offline", "per_vs_teacher"]),
        "streaming_per_vs_teacher": mean_of(["streaming", "per_vs_teacher"]),
        "offline_vs_streaming_seq_agreement": float(
            np.mean([r["offline_vs_streaming_seq_agreement"] for r in rows])
        ) if rows else float("nan"),
        "offline_vs_streaming_seq_len_ratio": float(
            np.mean([r["offline_vs_streaming_seq_len_ratio"] for r in rows])
        ) if rows else float("nan"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "ckpt": str(args.ckpt),
            "teacher_model_id": teacher_model_id,
            "n_utt": args.n_utt,
            "lookahead_ms": args.lookahead_ms,
            "blank_bias": args.blank_bias,
            "min_phone_conf": args.min_phone_conf,
            "device": str(device),
        },
        "per_utterance": rows,
        "aggregate": aggregate,
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\nAggregate:")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
