"""
phone_infer.py
Run phone-mode inference from a trained checkpoint in offline and/or streaming mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import build_pipeline  # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "data" / "checkpoints" / "belief_model_best.pt"
DEFAULT_AUDIO = ROOT / "data" / "samples" / "librispeech_sample.wav"


def load_pipeline_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "phone_head" not in ckpt or "phone_meta" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain phone mode weights/meta. "
            "Train with --enable_phone_mode first."
        )

    cfg = ckpt.get("cfg", {})
    pooling = cfg.get("pooling", "mean")

    phone_meta = ckpt["phone_meta"]
    pipeline = build_pipeline(
        pooling=pooling,
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

    return pipeline


def run_one(
    pipeline,
    audio_path: Path,
    mode: str,
    lookahead_ms: int,
    blank_bias: float,
    min_phone_conf: float,
) -> dict:
    res = pipeline.run_full(
        audio_path,
        mode=mode,
        lookahead_ms=lookahead_ms,
        enable_phone_mode=True,
        phone_blank_bias=blank_bias,
        phone_min_conf=min_phone_conf,
    )

    return {
        "mode": mode,
        "lookahead_ms": lookahead_ms,
        "T": res.T,
        "K": res.K,
        "phone_confidence": res.phone_confidence,
        "phone_seq_len": int(len(res.phone_sequence_tokens or [])),
        "phone_sequence": " ".join(res.phone_sequence_tokens or []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phone-mode inference from checkpoint")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--mode", choices=["offline", "streaming", "both"], default="both")
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
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    if args.preset == "best_w20":
        args.lookahead_ms = 80
        args.blank_bias = 0.2
        args.min_phone_conf = 0.35

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline = load_pipeline_from_ckpt(args.ckpt, device)

    records: list[dict] = []
    if args.mode in ("offline", "both"):
        records.append(
            run_one(
                pipeline,
                args.audio,
                mode="offline",
                lookahead_ms=args.lookahead_ms,
                blank_bias=args.blank_bias,
                min_phone_conf=args.min_phone_conf,
            )
        )
    if args.mode in ("streaming", "both"):
        records.append(
            run_one(
                pipeline,
                args.audio,
                mode="streaming",
                lookahead_ms=args.lookahead_ms,
                blank_bias=args.blank_bias,
                min_phone_conf=args.min_phone_conf,
            )
        )

    for rec in records:
        print("-" * 72)
        print(
            f"mode={rec['mode']}  T={rec['T']}  K={rec['K']}  "
            f"seq_len={rec['phone_seq_len']}  conf={rec['phone_confidence']:.3f}"
        )
        print(rec["phone_sequence"])

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump({"records": records}, fh, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
