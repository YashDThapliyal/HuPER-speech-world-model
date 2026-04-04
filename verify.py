"""
verify.py
Phase 7 — Scientific verification of the trained Speech World Model.

Loads the best checkpoint from train.py and answers:
  1. Does next-slot prediction improve over training?
  2. Are belief states smooth but not constant?
  3. Where is the model most surprised?
  4. Do different utterances produce coherent belief trajectories?
  5. Does the model generalise beyond single-utterance overfit?

Generates 6 visualizations to data/figures_phase7/.

Usage
─────
  python verify.py                   # uses default checkpoint
  python verify.py --ckpt data/checkpoints/belief_model_best.pt
  python verify.py --demo            # use single local sample (no training needed)
"""

# --- Imports ---
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))

from huper_features import EvidenceProjector
from slotizer import MeanPoolSlotizer, AttentionSlotizer
from belief_model import BeliefTransitionGRU
from pipeline import (
    build_pipeline, pca_2d,
    D_RAW, D_PROJ, CACHE_DIR, FIG_DIR,
    DARK_BG, PANEL_BG, SPINE_C, TICK_C, TEXT_C,
)
from train import build_dataset, Config

matplotlib.use("Agg")

ROOT_DIR   = Path(__file__).parent
CKPT_PATH  = ROOT_DIR / "data" / "checkpoints" / "belief_model_best.pt"
HIST_PATH  = ROOT_DIR / "data" / "training_history.json"
SAMPLE_WAV = ROOT_DIR / "data" / "samples" / "librispeech_sample.wav"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:  # type: ignore[name-defined]
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


def _save(fig: plt.Figure, name: str) -> None:  # type: ignore[name-defined]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_models(
    ckpt_path: Path,
    device:    torch.device,
    cfg:       Config,
) -> tuple[EvidenceProjector, nn.Module, BeliefTransitionGRU]:
    """Load EvidenceProjector, Slotizer, BeliefTransitionGRU from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    projector = EvidenceProjector(d_in=D_RAW, d_out=cfg.d_proj).to(device)
    projector.load_state_dict(ckpt["projector"])

    if cfg.pooling == "mean":
        slotizer: nn.Module = MeanPoolSlotizer().to(device)
    else:
        slotizer = AttentionSlotizer(cfg.d_proj).to(device)
    slotizer.load_state_dict(ckpt["slotizer"])

    belief_model = BeliefTransitionGRU(d=cfg.d_gru, n_layers=cfg.gru_layers).to(device)
    belief_model.load_state_dict(ckpt["belief_model"])

    epoch     = ckpt.get("epoch", "?")
    val_loss  = ckpt.get("val_loss", float("nan"))
    print(f"  Checkpoint loaded  (epoch={epoch}  val_loss={val_loss:.6f})")
    return projector, slotizer, belief_model


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_utterance(
    item:         dict,
    projector:    EvidenceProjector,
    slotizer:     nn.Module,
    belief_model: BeliefTransitionGRU,
    device:       torch.device,
    pooling:      str,
) -> dict:
    """
    Run inference on one utterance and return a diagnostics dict.

    Keys
    ----
    utt_id, K, duration_s, loss, cos_mean, belief_cos_mean, belief_cos_min,
    mismatch_mean, mismatch_max, mismatch_array (K-1,),
    beliefs (K, 256), predictions (K-1, 256), slots (K, 256)
    """
    layer24    = item["layer24"]
    boundaries = item["boundaries"]
    K          = item["K"]

    projector.eval()
    slotizer.eval()
    belief_model.eval()

    with torch.no_grad():
        x   = torch.from_numpy(layer24).to(device)
        E_t = projector(x)                         # (T, 256)

        if pooling == "mean":
            S = slotizer(E_t, boundaries)          # (K, 256)
        else:
            S, _ = slotizer(E_t, boundaries)

        S_batch = S.unsqueeze(0)                   # (1, K, 256)
        B, pred, _ = belief_model(S_batch)

    B_np   = B.squeeze(0).cpu().numpy()            # (K, 256)
    S_np   = S.cpu().numpy()                       # (K, 256)
    pred_np = pred.squeeze(0).cpu().numpy()        # (K-1, 256)
    tgt_np  = S_np[1:]                             # (K-1, 256)

    # Per-slot cosine
    pred_t = torch.from_numpy(pred_np)
    tgt_t  = torch.from_numpy(tgt_np)
    cos_ps = F.cosine_similarity(pred_t, tgt_t, dim=-1).numpy()   # (K-1,)
    mismatch = 1.0 - cos_ps                                        # (K-1,)

    # Belief evolution
    B_t  = torch.from_numpy(B_np)
    bel_cos = F.cosine_similarity(B_t[:-1], B_t[1:], dim=-1).numpy()   # (K-1,)

    loss = float(F.mse_loss(pred_t, tgt_t))

    return {
        "utt_id":          item["id"],
        "K":               K,
        "duration_s":      item["duration_s"],
        "loss":            loss,
        "cos_mean":        float(cos_ps.mean()),
        "belief_cos_mean": float(bel_cos.mean()),
        "belief_cos_min":  float(bel_cos.min()),
        "mismatch_mean":   float(mismatch.mean()),
        "mismatch_max":    float(mismatch.max()),
        "mismatch_array":  mismatch,      # (K-1,)
        "beliefs":         B_np,          # (K, 256)
        "predictions":     pred_np,       # (K-1, 256)
        "slots":           S_np,          # (K, 256)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verification questions
# ─────────────────────────────────────────────────────────────────────────────

def verify_questions(
    train_diags: list[dict],
    val_diags:   list[dict],
    history:     dict,
) -> None:
    """Print answers to the 5 verification questions."""
    all_diags = train_diags + val_diags

    print("\n  ┌─ Verification Questions ────────────────────────────────────┐")

    # Q1: Does prediction improve?
    if history:
        start_cos = np.mean(history["val_cos"][:3]) if len(history["val_cos"]) >= 3 else history["val_cos"][0]
        end_cos   = np.mean(history["val_cos"][-3:])
        improved  = end_cos > start_cos + 0.01
        print(f"  │ Q1. Prediction improves?  "
              f"val cos: {start_cos:.4f} → {end_cos:.4f}  "
              f"{'✓ YES' if improved else '⚠ MARGINAL'}")
    else:
        print("  │ Q1. No training history available.")

    # Q2: Belief smoothness
    mean_bel = np.mean([d["belief_cos_mean"] for d in val_diags])
    healthy  = 0.6 < mean_bel < 0.98
    print(f"  │ Q2. Beliefs smooth not constant?  "
          f"mean cos(B_k,B_k+1)={mean_bel:.4f}  "
          f"{'✓ healthy' if healthy else '⚠ check'}")

    # Q3: Surprise concentration
    all_mismatch = np.concatenate([d["mismatch_array"] for d in all_diags])
    top_pct = np.percentile(all_mismatch, 95)
    print(f"  │ Q3. Surprise concentrated?  "
          f"95th percentile mismatch={top_pct:.4f}  "
          f"(surprise peaks at specific transition slots)")

    # Q4: Coherent trajectories
    mean_cos = np.mean([d["cos_mean"] for d in val_diags])
    std_cos  = np.std([d["cos_mean"] for d in val_diags])
    print(f"  │ Q4. Coherent trajectories?  "
          f"val pred cos: mean={mean_cos:.4f}  std={std_cos:.4f}  "
          f"({'✓ consistent' if std_cos < 0.15 else '⚠ high variance'})")

    # Q5: Generalization
    tr_cos  = np.mean([d["cos_mean"] for d in train_diags])
    val_cos = np.mean([d["cos_mean"] for d in val_diags])
    gap     = tr_cos - val_cos
    print(f"  │ Q5. Generalises beyond train?  "
          f"train cos={tr_cos:.4f}  val cos={val_cos:.4f}  gap={gap:.4f}  "
          f"({'✓ ok' if gap < 0.1 else '⚠ some overfit'})")

    print("  └─────────────────────────────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def viz7_2_training_curves(history: dict) -> None:
    """Viz 7-2 — Multi-utterance training curves (4 panels)."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle("Viz 7-2 — Training Curves  (multi-utterance)",
                 color="white", fontsize=13, fontweight="bold")

    pairs = [
        (axes[0, 0], "train_loss", "val_loss",
         "Loss (MSE)", "Train Loss vs Epoch", "#4fc3f7", "#ff7043"),
        (axes[0, 1], "train_cos",  "val_cos",
         "Mean Cosine", "Prediction Cosine vs Epoch", "#81c784", "#ffd740"),
    ]
    for ax, tr_key, va_key, ylabel, title, c_tr, c_va in pairs:
        ax.plot(epochs, history[tr_key], color=c_tr,  lw=1.2, label="train")
        ax.plot(epochs, history[va_key], color=c_va,  lw=1.2, label="val", linestyle="--")
        ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
                  labelcolor="white", edgecolor=SPINE_C)
        ax.set_xlim(1, len(epochs))
        _style_ax(ax, title=title, xlabel="Epoch", ylabel=ylabel)

    # LR schedule
    ax = axes[1, 0]
    ax.plot(epochs, history["lr"], color="#ce93d8", lw=1.2)
    ax.set_xlim(1, len(epochs))
    ax.set_yscale("log")
    _style_ax(ax, title="Learning Rate Schedule", xlabel="Epoch", ylabel="LR")

    # Val loss vs val cosine scatter
    ax = axes[1, 1]
    ax.scatter(history["val_loss"], history["val_cos"],
               c=epochs, cmap="cool", s=8, alpha=0.7)
    _style_ax(ax, title="Val Loss vs Val Cosine (each point = 1 epoch)",
              xlabel="Val Loss", ylabel="Val Cosine")

    _save(fig, "viz7_2_training_curves.png")


def viz7_3_prediction_quality(val_diags: list[dict]) -> None:
    """Viz 7-3 — Per-utterance prediction cosine, sorted best → worst."""
    sorted_diags = sorted(val_diags, key=lambda d: d["cos_mean"], reverse=True)
    labels = [d["utt_id"].split("_")[-1] for d in sorted_diags]   # short label
    cosines = [d["cos_mean"] for d in sorted_diags]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    fig.suptitle(f"Viz 7-3 — Prediction Quality  (val, n={len(val_diags)})",
                 color="white", fontsize=13, fontweight="bold")

    colors = ["#81c784" if c > 0.5 else "#ff7043" for c in cosines]
    ax.bar(np.arange(len(cosines)), cosines, color=colors, width=0.7)
    ax.axhline(float(np.mean(cosines)), color="white", lw=1.0, linestyle=":",
               label=f"mean={np.mean(cosines):.4f}")
    ax.axhline(0.0, color=SPINE_C, lw=0.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=TEXT_C)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=SPINE_C)
    _style_ax(ax, title="mean cos(pred, target)  per utterance  (green ≥ 0.5)",
              xlabel="Utterance (sorted best→worst)", ylabel="Prediction Cosine")

    _save(fig, "viz7_3_prediction_quality.png")


def viz7_4_belief_smoothness(all_diags: list[dict], train_n: int) -> None:
    """Viz 7-4 — Belief smoothness per utterance with healthy range."""
    bel_means = [d["belief_cos_mean"] for d in all_diags]
    bel_mins  = [d["belief_cos_min"]  for d in all_diags]
    labels    = [d["utt_id"].split("_")[-1] for d in all_diags]
    n         = len(all_diags)
    x         = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n // 2), 5), facecolor=DARK_BG)
    fig.suptitle("Viz 7-4 — Belief Smoothness  cos(B_k, B_{k+1})  per utterance",
                 color="white", fontsize=13, fontweight="bold")

    ax.bar(x[:train_n], bel_means[:train_n], color="#4fc3f7", width=0.6, label="train")
    if n > train_n:
        ax.bar(x[train_n:], bel_means[train_n:], color="#ffd740", width=0.6, label="val")

    ax.scatter(x, bel_mins, color="#ff7043", s=25, zorder=4,
               label="min per utterance")

    # Healthy range shading
    ax.axhspan(0.6, 0.95, alpha=0.1, color="#81c784", label="healthy 0.6–0.95")
    ax.axhline(0.6,  color="#81c784", lw=0.7, linestyle="--", alpha=0.6)
    ax.axhline(0.95, color="#81c784", lw=0.7, linestyle="--", alpha=0.6)
    ax.axhline(float(np.mean(bel_means)), color="white", lw=0.8,
               linestyle=":", label=f"overall mean={np.mean(bel_means):.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=TEXT_C)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=SPINE_C, ncol=3)
    _style_ax(ax, xlabel="Utterance", ylabel="cos(B_k, B_{k+1})")

    _save(fig, "viz7_4_belief_smoothness.png")


def viz7_5_mismatch_heatmap(all_diags: list[dict]) -> None:
    """Viz 7-5 — Mismatch heatmap: utterances × slot index."""
    # Pad all mismatch arrays to the same length
    max_K = max(len(d["mismatch_array"]) for d in all_diags)
    mat   = np.full((len(all_diags), max_K), fill_value=np.nan)
    for i, d in enumerate(all_diags):
        arr = d["mismatch_array"]
        mat[i, :len(arr)] = arr

    labels = [d["utt_id"].split("_")[-1] for d in all_diags]

    fig, ax = plt.subplots(figsize=(max(12, max_K // 2), max(6, len(all_diags) // 3)),
                           facecolor=DARK_BG)
    fig.suptitle("Viz 7-5 — Mismatch Heatmap  (utterance × slot)",
                 color="white", fontsize=13, fontweight="bold")

    # Mask NaN (padding) as transparent
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", origin="upper",
                   cmap="YlOrRd", vmin=0.0, vmax=float(np.nanpercentile(mat, 98)),
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, pad=0.01,
                 label="Mismatch").ax.tick_params(colors=TICK_C)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, color=TEXT_C)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    ax.set_title("Red = high surprise  |  White = low surprise  |  Grey = no data",
                 color="white", fontsize=9, pad=6)
    ax.set_xlabel("Slot index k", color=TEXT_C, fontsize=9)

    _save(fig, "viz7_5_mismatch_heatmap.png")


def viz7_6_belief_gallery(diags: list[dict]) -> None:
    """Viz 7-6 — PCA belief trajectories for up to 4 utterances."""
    n_show = min(4, len(diags))
    chosen = diags[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5), facecolor=DARK_BG,
                             gridspec_kw={"wspace": 0.35})
    if n_show == 1:
        axes = [axes]   # type: ignore[assignment]

    fig.suptitle("Viz 7-6 — Belief Trajectory Gallery  (PCA of B_k, each dot = one syllable tick)",
                 color="white", fontsize=12, fontweight="bold")

    for ax, d in zip(axes, chosen):
        B   = d["beliefs"]           # (K, 256)
        K   = B.shape[0]
        if K < 3:
            ax.text(0.5, 0.5, "K < 3\nno PCA", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            _style_ax(ax, title=d["utt_id"].split("_")[-1])
            continue

        Z   = pca_2d(B)              # (K, 2)

        ax.plot(Z[:, 0], Z[:, 1], color="#444444", lw=0.8, zorder=1)
        ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(K),
                   cmap="cool", s=30, zorder=2)

        # Label every 5th tick
        for ki in range(0, K, max(1, K // 8)):
            ax.text(Z[ki, 0] + 0.005, Z[ki, 1] + 0.005,
                    f"{ki}", color="white", fontsize=6)

        ax.text(Z[0, 0],  Z[0, 1],  "start", color="#4fc3f7", fontsize=7,
                ha="center", va="top")
        ax.text(Z[-1, 0], Z[-1, 1], "end",   color="#ff7043", fontsize=7,
                ha="center", va="bottom")

        short_id = d["utt_id"].split("_")[-1]
        _style_ax(ax,
                  title=f"{short_id}  (K={K}  cos={d['cos_mean']:.3f})",
                  xlabel="PC-1", ylabel="PC-2")

    _save(fig, "viz7_6_belief_gallery.png")


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — single local sample, no training needed
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(device: torch.device) -> None:
    """Run verify in demo mode using the local librispeech_sample.wav."""
    print("\n  [DEMO MODE] — using local sample, random-init weights")
    print("  (Metrics will not be meaningful — this just tests the pipeline.)")

    pipeline = build_pipeline(pooling="mean", device=device)
    result   = pipeline.run_full(SAMPLE_WAV, utt_id="demo_sample")

    # Build a fake diag from the result
    diag = {
        "utt_id":          result.utt_id,
        "K":               result.K,
        "duration_s":      result.metadata["duration_s"],
        "loss":            float(np.mean(result.mismatch)),
        "cos_mean":        float(1.0 - np.mean(result.mismatch)),
        "belief_cos_mean": float(F.cosine_similarity(
            torch.from_numpy(result.beliefs[:-1]),
            torch.from_numpy(result.beliefs[1:]), dim=-1).mean()),
        "belief_cos_min":  float(F.cosine_similarity(
            torch.from_numpy(result.beliefs[:-1]),
            torch.from_numpy(result.beliefs[1:]), dim=-1).min()),
        "mismatch_mean":   float(result.mismatch.mean()),
        "mismatch_max":    float(result.mismatch.max()),
        "mismatch_array":  result.mismatch,
        "beliefs":         result.beliefs,
        "predictions":     result.predictions,
        "slots":           result.slots,
    }

    # Fake history (no real training)
    fake_history = {
        "train_loss": [0.05, 0.03, 0.02], "val_loss": [0.06, 0.04, 0.03],
        "train_cos":  [0.1,  0.3,  0.5],  "val_cos":  [0.1,  0.2,  0.4],
        "lr":         [1e-3, 1e-3, 5e-4],
    }

    all_diags = [diag]
    verify_questions([diag], [diag], fake_history)
    viz7_2_training_curves(fake_history)
    viz7_3_prediction_quality([diag])
    viz7_4_belief_smoothness(all_diags, train_n=0)
    viz7_5_mismatch_heatmap(all_diags)
    viz7_6_belief_gallery(all_diags)


# ─────────────────────────────────────────────────────────────────────────────
# Full verify — load trained checkpoint + cached data
# ─────────────────────────────────────────────────────────────────────────────

def run_verify(
    ckpt_path: Path,
    device:    torch.device,
) -> None:
    """Full verification pass using a trained checkpoint."""

    # Load checkpoint config
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_cfg = ckpt.get("cfg", {})
    cfg = Config(
        train_n   = saved_cfg.get("train_n",    32),
        val_n     = saved_cfg.get("val_n",       8),
        pooling   = saved_cfg.get("pooling",  "mean"),
        d_proj    = saved_cfg.get("d_proj",    D_PROJ),
        d_gru     = saved_cfg.get("d_gru",     D_PROJ),
        gru_layers= saved_cfg.get("gru_layers",   1),
    )

    # Load models
    projector, slotizer, belief_model = load_models(ckpt_path, device, cfg)

    # Load training history
    history = {}
    if HIST_PATH.exists():
        with open(HIST_PATH) as fh:
            history = json.load(fh)
        print(f"  History loaded  ({len(history['val_loss'])} epochs)")
    else:
        print("  No history file found — skipping Q1.")

    # Rebuild dataset (features are cached, fast)
    print("\n--- Loading dataset (from cache) ---")
    train_items, val_items = build_dataset(cfg, device)

    # Run diagnostics
    print("\n--- Computing diagnostics ---")
    print("  Train set:")
    train_diags = [
        diagnose_utterance(it, projector, slotizer, belief_model, device, cfg.pooling)
        for it in train_items
    ]
    print("  Val set:")
    val_diags = [
        diagnose_utterance(it, projector, slotizer, belief_model, device, cfg.pooling)
        for it in val_items
    ]
    all_diags = train_diags + val_diags

    # Print per-utterance table
    print(f"\n  {'ID':>20}  {'split':>5}  {'K':>4}  {'cos':>7}  {'bel_cos':>8}  {'mismatch':>9}")
    print("  " + "-" * 60)
    for d in train_diags[:5]:   # show first 5
        print(f"  {d['utt_id'][-20:]:>20}  train  {d['K']:>4}  "
              f"{d['cos_mean']:>7.4f}  {d['belief_cos_mean']:>8.4f}  {d['mismatch_mean']:>9.4f}")
    if len(train_diags) > 5:
        print(f"  ... ({len(train_diags)-5} more train utterances)")
    for d in val_diags:
        print(f"  {d['utt_id'][-20:]:>20}  val    {d['K']:>4}  "
              f"{d['cos_mean']:>7.4f}  {d['belief_cos_mean']:>8.4f}  {d['mismatch_mean']:>9.4f}")

    # Verification questions
    verify_questions(train_diags, val_diags, history)

    # Rankings
    val_sorted = sorted(val_diags, key=lambda d: d["cos_mean"])
    print(f"\n  Hardest val utterances:  {[d['utt_id'] for d in val_sorted[:3]]}")
    print(f"  Easiest val utterances:  {[d['utt_id'] for d in val_sorted[-3:]]}")

    # Visualizations
    print("\n--- Generating visualizations ---")
    if history:
        viz7_2_training_curves(history)
    viz7_3_prediction_quality(val_diags)
    viz7_4_belief_smoothness(all_diags, train_n=len(train_diags))
    viz7_5_mismatch_heatmap(all_diags)
    viz7_6_belief_gallery(val_diags)

    # Phase 7 summary
    print()
    print("─" * 52)
    print("  Phase 7 Verification Summary")
    print("─" * 52)
    print(f"  ├── Train utterances    : {len(train_diags)}")
    print(f"  ├── Val utterances      : {len(val_diags)}")
    print(f"  ├── Mean slots / utt    : {np.mean([d['K'] for d in all_diags]):.1f}")
    print(f"  ├── Best val loss       : {min(history['val_loss']):.6f}" if history else "  ├── Best val loss       : n/a")
    print(f"  ├── Mean val pred cos   : {np.mean([d['cos_mean'] for d in val_diags]):.4f}")
    print(f"  ├── Mean belief smooth  : {np.mean([d['belief_cos_mean'] for d in val_diags]):.4f}")
    print(f"  ├── Mean val mismatch   : {np.mean([d['mismatch_mean'] for d in val_diags]):.4f}")
    print(f"  ├── Cached artifacts    : {CACHE_DIR}")
    print(f"  ├── Figures             : {FIG_DIR}")
    print(f"  │")
    print(f"  ├── WORKING             : full pipeline + caching + training + all 6 vizs")
    print(f"  ├── PLACEHOLDER         : phone CTC loss, ASR CTC loss (need labels)")
    print(f"  └── NEXT STEP           : Phase 8 — add phone/ASR CTC heads with LibriSpeech labels")
    print("─" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("█" * 60)
    print("  PHASE 7 — VERIFICATION")
    print("  Speech World Model: scientific diagnostics")
    print("█" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode: single local sample, no training needed")
    args = ap.parse_args()

    if args.demo:
        run_demo(device)
    else:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            print(f"\n  Checkpoint not found: {ckpt}")
            print("  Run train.py first, or use --demo for a quick sanity check.")
            sys.exit(1)
        run_verify(ckpt, device)

    print(f"\n  Figures saved → {FIG_DIR}")
