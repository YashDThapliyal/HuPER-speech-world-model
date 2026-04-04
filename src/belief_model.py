"""
Module: belief_model
Phase: 6
Goal: BeliefTransitionGRU — the core world-model loop.

Given syllable-level slot evidence S_k (K, 256), maintain a running
belief state B_k and predict the next slot Ŝ_{k+1} from that belief.

This is the first true world-model objective:
    "Given everything heard so far, what should come next?"

The architecture is intentionally minimal for Phase 6:
- No contrastive losses, no transformers, no language model.
- One GRU, two MLPs, one cosine prediction loss.
- We OVERFIT one utterance first. Prove it works, then scale.

Architecture at a glance:

    S_k (K, 256)  — syllable slots from Phase 5
          │
     GRU (256→256, 1 layer)
          │
     LayerNorm
          │
    B_k (K, 256)  — belief states  ← what the model "believes" after tick k
     │         │
     │    MLP_prior           L_k (K, 256)  — language prior auxiliary
     │
    MLP_pred (B_k[:K-1] → Ŝ_{k+1})
          │
    Ŝ_{k+1} (K-1, 256)  — next-slot prediction
          │
    Loss: MSE(Ŝ, S[1:]) + monitor cosine(Ŝ, S[1:])
"""

# --- Imports ---
from __future__ import annotations

import pickle
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

matplotlib.use("Agg")

# --- Constants ---
D          = 256         # slot / belief dimension
GRU_LAYERS = 1
LR         = 1e-3
MAX_EPOCHS = 1500
PATIENCE   = 150         # early-stop patience (no improvement in loss)
LOG_EVERY  = 25

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"

VIZ_DIR  = Path(__file__).parent.parent / "data" / "visualizations" / "phase6"
DATA_DIR = Path(__file__).parent.parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_shape(name: str, x: torch.Tensor | np.ndarray) -> None:
    if isinstance(x, torch.Tensor):
        print(f"  {name:45s}: {tuple(x.shape)}  dtype={x.dtype}")
    else:
        print(f"  {name:45s}: shape={x.shape}  dtype={x.dtype}")


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Pure-numpy 2-component PCA.  X: (N, D) → (N, 2)"""
    Xc = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(Xc.T))
    top2 = vecs[:, np.argsort(vals)[::-1][:2]]
    return Xc @ top2


def _style_ax(
    ax: plt.Axes,   # type: ignore[name-defined]
    title: str = "", xlabel: str = "", ylabel: str = "",
) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — BeliefTransitionGRU
# ─────────────────────────────────────────────────────────────────────────────

class NextSlotPredictor(nn.Module):
    """Linear → GELU → Linear:  B_k (256) → Ŝ_{k+1} (256)"""
    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguagePriorHead(nn.Module):
    """Linear → GELU → Linear:  B_k (256) → L_k (256)
    Auxiliary learned state — future phases will use this as top-down expectation."""
    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeliefTransitionGRU(nn.Module):
    """
    Core world-model loop.

    forward(S) where S: (1, K, 256)

    Returns
    -------
    B            : (1, K, 256)    belief state after each syllable tick
    pred         : (1, K-1, 256)  next-slot prediction from B_0..B_{K-2}
    L            : (1, K, 256)    language prior auxiliary state
    """

    def __init__(self, d: int = D, n_layers: int = GRU_LAYERS) -> None:
        super().__init__()
        self.d = d

        # GRU encoder
        self.gru = nn.GRU(
            input_size  = d,
            hidden_size = d,
            num_layers  = n_layers,
            batch_first = True,
        )

        # Belief head: normalise GRU outputs
        self.belief_norm = nn.LayerNorm(d)

        # Next-slot predictor
        self.predictor = NextSlotPredictor(d)

        # Language prior head (auxiliary)
        self.prior_head = LanguagePriorHead(d)

    def forward(
        self, S: torch.Tensor          # (1, K, 256)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass.

        The GRU reads ALL K slots and produces a hidden state at each tick.
        Next-slot prediction uses B_k (k=0..K-2) to predict S_{k+1}.
        """
        # GRU over all K slots
        gru_out, _ = self.gru(S)         # (1, K, 256)

        # Belief states: normalise
        B = self.belief_norm(gru_out)    # (1, K, 256)

        # Next-slot prediction: B_k → Ŝ_{k+1}
        pred = self.predictor(B[:, :-1, :])   # (1, K-1, 256)

        # Language prior auxiliary
        L = self.prior_head(B)           # (1, K, 256)

        return B, pred, L


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Prepare training data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    slots_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    """
    Load S_mean, add batch dim, split into input / target.

    Returns
    -------
    S       : (1, K, 256)    full slot sequence with batch dim
    inp     : (1, K-1, 256)  slots 0..K-2  (teacher-forced input)  — unused:
              we feed ALL K slots to the GRU; the split happens inside forward()
    target  : (1, K-1, 256)  slots 1..K-1  (prediction targets)
    K       : int
    duration_s : float
    """
    with open(slots_path, "rb") as fh:
        d = pickle.load(fh)

    S_np = d["S"]                                     # (K, 256)
    S    = torch.from_numpy(S_np).unsqueeze(0)        # (1, K, 256)
    S    = S.to(device)

    target = S[:, 1:, :].clone()                      # (1, K-1, 256)

    log_shape("S  (full sequence)", S)
    log_shape("target  S[1:]", target)

    return S, target, S, int(d["K"]), float(d["duration_s"])


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: BeliefTransitionGRU,
    S: torch.Tensor,           # (1, K, 256)
    target: torch.Tensor,      # (1, K-1, 256)
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int   = PATIENCE,
    lr: float       = LR,
) -> dict[str, list[float]]:
    """
    Train on one utterance.  Returns history dict with 'loss' and 'cosine' lists.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=60,
    )

    history: dict[str, list[float]] = {"loss": [], "cosine": []}
    best_loss   = float("inf")
    stale_count = 0

    print(f"\n  Training on {device}  |  lr={lr}  |  max_epochs={max_epochs}"
          f"  |  patience={patience}")
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'MeanCos':>10}  {'LR':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        B, pred, L = model(S)             # pred: (1, K-1, 256)

        # Primary loss: MSE between predicted and actual next slots
        loss = F.mse_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        # Cosine similarity (no grad needed)
        with torch.no_grad():
            cos = F.cosine_similarity(
                pred.squeeze(0), target.squeeze(0), dim=-1
            ).mean().item()                # scalar

        history["loss"].append(float(loss.item()))
        history["cosine"].append(float(cos))

        # Early stopping
        if float(loss.item()) < best_loss - 1e-7:
            best_loss   = float(loss.item())
            stale_count = 0
        else:
            stale_count += 1

        if epoch % LOG_EVERY == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:>6}  {loss.item():>10.6f}  {cos:>10.4f}  {current_lr:>12.2e}")

        if stale_count >= patience:
            print(f"\n  Early stop at epoch {epoch}  (no improvement for {patience} epochs)")
            break

    print(f"\n  Final  loss={history['loss'][-1]:.6f}  "
          f"cosine={history['cosine'][-1]:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(
    model: BeliefTransitionGRU,
    S: torch.Tensor,       # (1, K, 256)
    boundaries: list[tuple[int, int]],
    sr: int,
) -> dict:
    """
    1. Per-slot prediction cosine similarity
    2. Mismatch signal r_k = 1 - cosine(pred_k, target_k)
    3. Belief evolution: cosine(B_k, B_{k+1})

    Returns a dict with numpy arrays for all signals.
    """
    model.eval()
    with torch.no_grad():
        B, pred, L = model(S)

    B_np    = B.squeeze(0).cpu().numpy()       # (K, 256)
    pred_np = pred.squeeze(0).cpu().numpy()    # (K-1, 256)
    tgt_np  = S[:, 1:, :].squeeze(0).cpu().numpy()  # (K-1, 256)

    # 1. Per-slot cosine similarity
    pred_t = torch.from_numpy(pred_np)
    tgt_t  = torch.from_numpy(tgt_np)
    cos_per_slot = F.cosine_similarity(pred_t, tgt_t, dim=-1).numpy()  # (K-1,)

    # 2. Mismatch signal
    mismatch = 1.0 - cos_per_slot                     # (K-1,)

    # 3. Belief evolution
    B_t   = torch.from_numpy(B_np)
    bel_cos = F.cosine_similarity(
        B_t[:-1], B_t[1:], dim=-1
    ).numpy()                                          # (K-1,)

    # Slot durations for timeline
    durations_ms = np.array(
        [(e - s + 1) * 320 / sr * 1000 for s, e in boundaries]
    )

    # Print summary
    print(f"\n  1. Reconstruction quality (cos per predicted slot):")
    print(f"     mean={cos_per_slot.mean():.4f}  "
          f"min={cos_per_slot.min():.4f}  max={cos_per_slot.max():.4f}")

    print(f"\n  2. Top-5 highest-mismatch slots (most surprised):")
    top5_idx = np.argsort(mismatch)[::-1][:5]
    for rank, idx in enumerate(top5_idx):
        print(f"     #{rank+1}  slot k={idx:2d}  mismatch={mismatch[idx]:.4f}  "
              f"(predicting slot {idx+1})")

    print(f"\n  3. Belief evolution cosine(B_k, B_{{k+1}}):")
    print(f"     mean={bel_cos.mean():.4f}  "
          f"min={bel_cos.min():.4f}  max={bel_cos.max():.4f}")
    print(f"     Healthy range: 0.6–0.95  "
          f"({'✓ within range' if 0.5 < bel_cos.mean() < 0.98 else '⚠ check'})")

    return {
        "B":              B_np,
        "pred":           pred_np,
        "target":         tgt_np,
        "cos_per_slot":   cos_per_slot,
        "mismatch":       mismatch,
        "belief_cos":     bel_cos,
        "durations_ms":   durations_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _znorm(X: np.ndarray) -> np.ndarray:
    return (X - X.mean()) / (X.std() + 1e-8)


# VIZ 1 — Training curves ─────────────────────────────────────────────────────

def viz6_1_training(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.45})
    fig.suptitle("Viz 6-1 — Training Curve (one utterance overfit)",
                 color="white", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["loss"], color="#4fc3f7", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlim(1, len(epochs))
    _style_ax(axes[0],
              title="MSE loss (log scale) — should fall monotonically",
              xlabel="Epoch", ylabel="Loss")

    axes[1].plot(epochs, history["cosine"], color="#81c784", lw=1.2)
    axes[1].axhline(1.0, color=SPINE_C, lw=0.6, linestyle="--")
    axes[1].axhline(np.max(history["cosine"]), color="#ffd740",
                    lw=0.8, linestyle=":",
                    label=f"peak = {np.max(history['cosine']):.4f}")
    axes[1].set_xlim(1, len(epochs))
    axes[1].set_ylim(min(history["cosine"]) - 0.05, 1.05)
    _style_ax(axes[1],
              title="Mean cosine(pred, target) — should climb toward 1.0",
              xlabel="Epoch", ylabel="Mean cosine")
    axes[1].legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
                   labelcolor="white", edgecolor=SPINE_C)

    out = VIZ_DIR / "viz6_1_training.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# VIZ 2 — Predicted vs actual next slots ─────────────────────────────────────

def viz6_2_pred_vs_actual(diag: dict) -> None:
    pred    = diag["pred"]      # (K-1, 256)
    target  = diag["target"]   # (K-1, 256)
    cos_ps  = diag["cos_per_slot"]   # (K-1,)
    K1      = pred.shape[0]

    # Shared colourscale across both heatmaps
    joint   = np.concatenate([target, pred], axis=0)
    vmin, vmax = np.percentile(joint, 2), np.percentile(joint, 98)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.5, "height_ratios": [3, 3, 2]})
    fig.suptitle(
        f"Viz 6-2 — Predicted vs Actual Next Slots  "
        f"(mean cos = {cos_ps.mean():.4f})",
        color="white", fontsize=13, fontweight="bold",
    )

    im = axes[0].imshow(target.T, aspect="auto", origin="lower",
                        extent=(0.0, float(K1), 0.0, 256.0),
                        cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
    _style_ax(axes[0],
              title=f"Actual target slots  S[1:]  ({K1} × 256)",
              ylabel="Dim")

    axes[1].imshow(pred.T, aspect="auto", origin="lower",
                   extent=(0.0, float(K1), 0.0, 256.0),
                   cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    _style_ax(axes[1],
              title=f"Predicted next slots  Ŝ  ({K1} × 256)",
              ylabel="Dim")

    k_axis = np.arange(K1)
    axes[2].bar(k_axis, cos_ps,
                color=["#66bb6a" if c > 0.9 else "#ffd740" if c > 0.7
                       else "#ff7043" for c in cos_ps],
                edgecolor=PANEL_BG, linewidth=0.3)
    axes[2].axhline(cos_ps.mean(), color="white", lw=1.0, linestyle=":",
                    label=f"mean={cos_ps.mean():.4f}")
    axes[2].axhline(0.9, color="#66bb6a", lw=0.8, linestyle="--", alpha=0.5)
    axes[2].set_xlim(-0.5, K1 - 0.5)
    axes[2].set_ylim(max(-0.1, cos_ps.min() - 0.1), 1.05)
    _style_ax(axes[2],
              title="Per-slot cosine(pred, target)  [green>0.9 | yellow 0.7–0.9 | red<0.7]",
              xlabel="Slot index k",
              ylabel="Cosine")
    axes[2].legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
                   labelcolor="white", edgecolor=SPINE_C)

    fig.colorbar(im, ax=axes[:2].tolist(), pad=0.01,   # type: ignore[arg-type]
                 fraction=0.008).ax.tick_params(colors=TICK_C)

    out = VIZ_DIR / "viz6_2_pred_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# VIZ 3 — Belief trajectory ───────────────────────────────────────────────────

def viz6_3_belief_trajectory(diag: dict) -> None:
    B = diag["B"]    # (K, 256)
    K = B.shape[0]
    Z = pca_2d(B)    # (K, 2)

    cmap   = plt.cm.coolwarm  # type: ignore[attr-defined]
    colors = [cmap(i / (K - 1)) for i in range(K)]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
    fig.suptitle("Viz 6-3 — Belief Trajectory in PCA Space  (B_k post-training)",
                 color="white", fontsize=13, fontweight="bold")

    for i in range(K - 1):
        ax.annotate("",
                    xy=(Z[i + 1, 0], Z[i + 1, 1]),
                    xytext=(Z[i, 0], Z[i, 1]),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=(*colors[i][:3], 0.55),
                                    lw=1.1, mutation_scale=9))

    sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(K), cmap="coolwarm",
                    s=65, zorder=5, edgecolors="#333333", linewidths=0.5)
    plt.colorbar(sc, ax=ax, pad=0.01,
                 label="Belief slot k (blue=early, red=late)").ax.tick_params(
                     colors=TICK_C)

    for i in range(0, K, 5):
        ax.text(Z[i, 0] + 0.015, Z[i, 1] + 0.015,
                f"k={i}", color="white", fontsize=8, fontweight="bold")
    ax.text(Z[0, 0], Z[0, 1] - 0.04, "start", color="#4fc3f7", fontsize=8)
    ax.text(Z[-1, 0], Z[-1, 1] - 0.04, "end",   color="#ff7043", fontsize=8)

    _style_ax(ax, xlabel="PC 1", ylabel="PC 2")
    ax.text(0.02, 0.04,
            "Each dot = B_k after training.\n"
            "A coherent path shows the GRU is accumulating\n"
            "meaningful sequential context — not random walk.",
            transform=ax.transAxes, color="#888888", fontsize=8, va="bottom")

    out = VIZ_DIR / "viz6_3_belief_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# VIZ 4 — Mismatch / surprise signal ─────────────────────────────────────────

def viz6_4_mismatch(diag: dict, boundaries: list[tuple[int, int]]) -> None:
    mismatch     = diag["mismatch"]        # (K-1,)
    durations_ms = diag["durations_ms"]    # (K,)
    K1 = len(mismatch)

    top5_idx = np.argsort(mismatch)[::-1][:5]
    bar_colors = ["#ff7043" if i in top5_idx else "#4fc3f7"
                  for i in range(K1)]

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.55, "height_ratios": [3, 2]})
    fig.suptitle("Viz 6-4 — Mismatch Signal  r_k = 1 − cos(Ŝ_{k+1}, S_{k+1})",
                 color="white", fontsize=13, fontweight="bold")

    k_axis = np.arange(K1)
    axes[0].bar(k_axis, mismatch, color=bar_colors,
                edgecolor=PANEL_BG, linewidth=0.3)
    axes[0].axhline(mismatch.mean(), color="white", lw=1.0, linestyle=":",
                    label=f"mean = {mismatch.mean():.4f}")

    # Annotate top-5 (text only — no arrowprops to avoid bbox explosion)
    y_cap = max(float(mismatch.max()), 1e-4)
    for rank, idx in enumerate(top5_idx):
        axes[0].text(
            idx, y_cap * 1.05, f"k={idx}",
            color="#ff7043", fontsize=7, ha="center", va="bottom",
        )

    axes[0].set_xlim(-0.5, K1 - 0.5)
    axes[0].set_ylim(-y_cap * 0.05, y_cap * 1.35)
    _style_ax(axes[0],
              title="Mismatch (surprise) per slot  — red = top-5 most surprised",
              xlabel="Slot index k",
              ylabel="Mismatch (1 − cosine)")
    axes[0].legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
                   labelcolor="white", edgecolor=SPINE_C)

    # Duration timeline below
    dur_plot = durations_ms[:K1]  # align to K-1
    safe_vmax = max(float(mismatch.max()), 1e-6)
    sc = axes[1].scatter(k_axis, dur_plot, c=mismatch, cmap="YlOrRd",
                         vmin=0.0, vmax=safe_vmax,
                         s=50, zorder=3)
    axes[1].plot(k_axis, dur_plot, color="#444444", lw=0.8, zorder=2)
    plt.colorbar(sc, ax=axes[1], pad=0.01,
                 label="Mismatch").ax.tick_params(colors=TICK_C)
    _style_ax(axes[1],
              title="Syllable duration (ms) — colour = mismatch intensity",
              xlabel="Slot index k",
              ylabel="Duration (ms)")
    axes[1].set_xlim(-0.5, K1 - 0.5)

    out = VIZ_DIR / "viz6_4_mismatch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# VIZ 5 — Belief smoothness ───────────────────────────────────────────────────

def viz6_5_belief_smoothness(diag: dict) -> None:
    bel_cos = diag["belief_cos"]    # (K-1,)
    K1 = len(bel_cos)
    mean_val = float(bel_cos.mean())

    # Drops: 2σ below mean
    drop_thresh = mean_val - 2.0 * bel_cos.std()
    drop_mask   = bel_cos < drop_thresh

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Viz 6-5 — Belief Smoothness: cos(B_k, B_{k+1})",
                 color="white", fontsize=13, fontweight="bold")

    k_axis = np.arange(K1)
    ax.plot(k_axis, bel_cos, color="#4fc3f7", lw=1.3, zorder=3)
    ax.fill_between(k_axis, bel_cos, mean_val,
                    where=bel_cos < mean_val,
                    color="#ff7043", alpha=0.25, label="below mean")
    ax.fill_between(k_axis, bel_cos, mean_val,
                    where=bel_cos >= mean_val,
                    color="#66bb6a", alpha=0.20, label="above mean")

    # Mark drops
    for i in np.where(drop_mask)[0]:
        ax.axvline(i, color="#ffd740", lw=1.0, linestyle="--", alpha=0.7)
        ax.text(i + 0.1, bel_cos[i] - 0.015,
                f"k={i}", color="#ffd740", fontsize=7)

    ax.axhline(mean_val, color="white", lw=1.0, linestyle=":",
               label=f"mean = {mean_val:.4f}")
    ax.axhline(0.95, color="#cccccc", lw=0.6, linestyle="--", alpha=0.4,
               label="0.95 (near-static)")
    ax.axhline(0.60, color="#ff7043", lw=0.6, linestyle="--", alpha=0.4,
               label="0.60 (rapid update)")
    ax.set_xlim(0, K1 - 1)
    ax.set_ylim(min(bel_cos) - 0.05, 1.05)
    _style_ax(ax,
              title="Belief smoothness  —  dips = strong belief update at that tick",
              xlabel="Slot index k",
              ylabel="cos(B_k, B_{k+1})")
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=SPINE_C, loc="lower right")

    out = VIZ_DIR / "viz6_5_belief_smoothness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n" + "█" * 60)
    print("  PHASE 6 — BELIEF TRANSITION GRU")
    print("  Speech World Model: HuPER Implementation")
    print("█" * 60)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    print("\n--- Step 2: Prepare training data ---")
    S, target, _, K, duration_s = prepare_data(
        DATA_DIR / "slots_mean.pkl", device,
    )

    # Also load boundaries for diagnostics / mismatch timeline
    with open(DATA_DIR / "syllable_boundaries.pkl", "rb") as fh:
        bdata = pickle.load(fh)
    boundaries: list[tuple[int, int]] = bdata["boundaries"]
    sr = int(bdata["sr"])

    # ── Step 1: Build model ───────────────────────────────────────────────────
    print("\n--- Step 1: BeliefTransitionGRU ---")
    model = BeliefTransitionGRU(d=D, n_layers=GRU_LAYERS).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters  : {total_params:,}")
    print(f"  GRU               : input={D}, hidden={D}, layers={GRU_LAYERS}")
    print(f"  NextSlotPredictor : {D}→{D}→GELU→{D}")
    print(f"  LanguagePriorHead : {D}→{D}→GELU→{D}")

    # Dry-run forward pass to verify shapes
    with torch.no_grad():
        B_dry, pred_dry, L_dry = model(S)
    log_shape("B  (beliefs, dry run)", B_dry)
    log_shape("pred  (next-slot pred, dry run)", pred_dry)
    log_shape("L  (lang prior, dry run)", L_dry)
    log_shape("target  S[1:]", target)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n--- Step 3: Training ---")
    history = train(model, S, target, device)

    # ── Step 4: Diagnostics ───────────────────────────────────────────────────
    print("\n--- Step 4: Diagnostics ---")
    diag = compute_diagnostics(model, S, boundaries, sr)

    # ── Step 5: Visualizations ────────────────────────────────────────────────
    print("\n--- Step 5: Visualizations ---")
    print("  Viz 6-1: training curves …")
    viz6_1_training(history)

    print("  Viz 6-2: predicted vs actual slots …")
    viz6_2_pred_vs_actual(diag)

    print("  Viz 6-3: belief trajectory …")
    viz6_3_belief_trajectory(diag)

    print("  Viz 6-4: mismatch signal …")
    viz6_4_mismatch(diag, boundaries)

    print("  Viz 6-5: belief smoothness …")
    viz6_5_belief_smoothness(diag)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\n--- Step 6: Save outputs ---")
    torch.save(model.state_dict(), DATA_DIR / "belief_model.pt")
    print(f"  belief_model.pt saved")

    with open(DATA_DIR / "beliefs.pkl", "wb") as fh:
        pickle.dump({"B": diag["B"], "K": K, "duration_s": duration_s,
                     "boundaries": boundaries}, fh)
    with open(DATA_DIR / "pred_slots.pkl", "wb") as fh:
        pickle.dump({"pred": diag["pred"], "target": diag["target"],
                     "cos_per_slot": diag["cos_per_slot"]}, fh)
    with open(DATA_DIR / "mismatch.pkl", "wb") as fh:
        pickle.dump({"mismatch": diag["mismatch"],
                     "belief_cos": diag["belief_cos"],
                     "boundaries": boundaries}, fh)
    print(f"  beliefs.pkl, pred_slots.pkl, mismatch.pkl saved")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    worst_k = int(diag["mismatch"].argmax())
    print(f"\n{'─' * 46}")
    print(f"  Belief Model Summary")
    print(f"  ├── Input slots:           ({K}, {D})")
    print(f"  ├── Belief states B_k:     ({K}, {D})")
    print(f"  ├── Predicted next slots:  ({K-1}, {D})")
    print(f"  ├── Final train loss:      {history['loss'][-1]:.6f}")
    print(f"  ├── Mean pred cosine:      {diag['cos_per_slot'].mean():.4f}")
    print(f"  ├── Mean belief cosine:    {diag['belief_cos'].mean():.4f}")
    print(f"  └── Max mismatch at k=:    {worst_k}  "
          f"(r={diag['mismatch'][worst_k]:.4f})")
    print(f"{'─' * 46}\n")

    print("  Visualizations saved to:")
    for f in sorted(VIZ_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    run_demo()
