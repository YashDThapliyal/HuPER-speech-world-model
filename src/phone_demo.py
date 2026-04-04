"""
Module: phone_demo
Phase: 2
Goal: Understand what phones are and how CTC loss enables alignment-free recognition.

What this teaches:
- Phones are the atomic units of spoken sound (~40 in English)
- Phone recognizers output a probability distribution over phones AT EVERY FRAME (~50/s)
- CTC (Connectionist Temporal Classification) collapses that noisy per-frame stream
  into a clean phone sequence — without needing manually aligned labels
- The [PAD] blank token is the key: it separates repeated phones and marks silence
"""

# --- Imports ---
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import librosa

matplotlib.use("Agg")

# --- Constants ---
MODEL_ID: str    = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
TARGET_SR: int   = 16_000
BLANK_TOKEN: str = "[PAD]"   # CTC blank in this model's vocab
WORD_BOUNDARY: str = "|"     # word separator token


# ─────────────────────────────────────────────────────────────────────────────
# ARPABET REFERENCE TABLE
# (Standard 39-phone set used in TIMIT / referenced throughout HuPER paper)
# ─────────────────────────────────────────────────────────────────────────────
# Maps ARPAbet label → (IPA symbol, description, example word, example position)
ARPABET_TABLE: list[tuple[str, str, str, str, str]] = [
    # ── Vowels ────────────────────────────────────────────────────────────────
    ("AA", "ɑ",  "Monophthong",    "father",  "fAther"),
    ("AE", "æ",  "Monophthong",    "cat",     "cAt"),
    ("AH", "ə/ʌ","Monophthong",    "cut/about","cUt"),
    ("AO", "ɔ",  "Monophthong",    "thought", "thOUght"),
    ("AW", "aʊ", "Diphthong",      "cow",     "cOW"),
    ("AY", "aɪ", "Diphthong",      "hide",    "hIde"),
    ("EH", "ɛ",  "Monophthong",    "bed",     "bEd"),
    ("ER", "ɝ",  "R-colored vowel","bird",    "bIRd"),
    ("EY", "eɪ", "Diphthong",      "bait",    "bAIt"),
    ("IH", "ɪ",  "Monophthong",    "bit",     "bIt"),
    ("IY", "i",  "Monophthong",    "beet",    "bEEt"),
    ("OW", "oʊ", "Diphthong",      "boat",    "bOAt"),
    ("OY", "ɔɪ", "Diphthong",      "boy",     "bOY"),
    ("UH", "ʊ",  "Monophthong",    "book",    "bOOk"),
    ("UW", "u",  "Monophthong",    "boot",    "bOOt"),
    # ── Consonants — stops ───────────────────────────────────────────────────
    ("B",  "b",  "Voiced stop",    "bat",     "Bat"),
    ("D",  "d",  "Voiced stop",    "dig",     "Dig"),
    ("G",  "g",  "Voiced stop",    "gas",     "Gas"),
    ("K",  "k",  "Unvoiced stop",  "cat",     "Cat"),
    ("P",  "p",  "Unvoiced stop",  "pat",     "Pat"),
    ("T",  "t",  "Unvoiced stop",  "tap",     "Tap"),
    # ── Consonants — fricatives ──────────────────────────────────────────────
    ("DH", "ð",  "Voiced fric.",   "the",     "THe"),
    ("F",  "f",  "Unvoiced fric.", "fat",     "Fat"),
    ("HH", "h",  "Unvoiced fric.", "hat",     "Hat"),
    ("S",  "s",  "Unvoiced fric.", "sat",     "Sat"),
    ("SH", "ʃ",  "Unvoiced fric.", "ship",    "SHip"),
    ("TH", "θ",  "Unvoiced fric.", "thin",    "THin"),
    ("V",  "v",  "Voiced fric.",   "vat",     "Vat"),
    ("Z",  "z",  "Voiced fric.",   "zip",     "Zip"),
    ("ZH", "ʒ",  "Voiced fric.",   "measure", "meaZure"),
    # ── Consonants — affricates ──────────────────────────────────────────────
    ("CH", "ʧ",  "Unvoiced affr.", "chin",    "CHin"),
    ("JH", "ʤ",  "Voiced affr.",   "gin",     "Gin"),
    # ── Consonants — nasals ──────────────────────────────────────────────────
    ("M",  "m",  "Nasal",          "mat",     "Mat"),
    ("N",  "n",  "Nasal",          "nap",     "Nap"),
    ("NG", "ŋ",  "Nasal",          "sing",    "siNG"),
    # ── Consonants — approximants ────────────────────────────────────────────
    ("L",  "l",  "Lateral approx.","let",     "Let"),
    ("R",  "r",  "Approximant",    "red",     "Red"),
    ("W",  "w",  "Approximant",    "wet",     "Wet"),
    ("Y",  "j",  "Approximant",    "yet",     "Yet"),
]


def print_arpabet_table() -> None:
    """Print the ARPAbet phoneme set in a readable table."""
    print("\n" + "=" * 70)
    print("ARPABET — 39 English Phonemes")
    print("(The alphabet HuPER uses for phone recognition)")
    print("=" * 70)
    print(f"  {'ARPAbet':<8} {'IPA':<6} {'Type':<22} {'Word':<10} {'Highlight'}")
    print(f"  {'-'*7:<8} {'-'*5:<6} {'-'*21:<22} {'-'*8:<10} {'-'*12}")

    current_type = ""
    for arp, ipa, ptype, word, highlight in ARPABET_TABLE:
        category = ptype.split()[0]  # "Vowel" / "Consonant" etc — first word
        if category != current_type:
            current_type = category
        print(f"  {arp:<8} {ipa:<6} {ptype:<22} {word:<10} {highlight}")

    print(f"\n  Total: {len(ARPABET_TABLE)} phones")
    print("  [KEY INSIGHT] 'water' ≠ W-A-T-E-R")
    print("  In ARPAbet: W AO T ER — the T is a flap (ɾ) in American English,")
    print("  realized between vowels. Spelling and phonetics diverge constantly.")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_phone_recognizer(
    model_id: str = MODEL_ID,
) -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC, torch.device]:
    """
    Load wav2vec2 phone recognizer from HuggingFace.

    Returns
    -------
    processor : handles feature extraction + tokenizer
    model     : Wav2Vec2ForCTC outputting per-frame phone logits
    device    : mps or cpu
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Loading {model_id}")
    print(f"  Device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model     = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)  # type: ignore[arg-type]
    model.eval()

    vocab     = processor.tokenizer.get_vocab()  # type: ignore[attr-defined]
    n_phones  = len(vocab)
    print(f"  Phone vocab size: {n_phones}  (incl. blank, unk, boundaries)")
    return processor, model, device


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_phone_recognition(
    waveform: np.ndarray,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> tuple[np.ndarray, list[str], str]:
    """
    Run phone recognizer on waveform.

    Returns
    -------
    posteriors    : np.ndarray  shape (T_frames, n_phones)  — softmax probs
    phone_labels  : list[str]   ordered phone label strings (vocab order)
    decoded_str   : str         CTC-decoded phone sequence
    """
    # Feature extraction: waveform → input values
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)  # shape: (1, num_samples)

    with torch.no_grad():
        logits = model(input_values).logits  # shape: (1, T_frames, n_phones)

    # shape: (T_frames, n_phones) — convert to probabilities
    posteriors = F.softmax(logits[0], dim=-1).cpu().numpy()  # type: ignore[attr-defined]

    # Build ordered label list (index → phone string)
    vocab       = processor.tokenizer.get_vocab()  # type: ignore[attr-defined]
    id_to_phone = {v: k for k, v in vocab.items()}
    n_phones    = logits.shape[-1]
    phone_labels = [id_to_phone[i] for i in range(n_phones)]

    # CTC greedy decode: argmax each frame, then collapse
    pred_ids    = np.argmax(posteriors, axis=-1)   # shape: (T_frames,)
    decoded_str = processor.decode(pred_ids)

    return posteriors, phone_labels, decoded_str


# ─────────────────────────────────────────────────────────────────────────────
# CTC EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

def explain_ctc(decoded_str: str, phone_labels: list[str], posteriors: np.ndarray) -> None:
    """
    Print a text-visual explanation of CTC using actual model output.

    CTC collapses the raw per-frame prediction stream → clean phone sequence via:
      Step 1: take argmax at each frame  → raw frame-level prediction
      Step 2: remove adjacent duplicates → merge repeated predictions
      Step 3: remove blank tokens         → final phone sequence
    """
    print("\n" + "=" * 70)
    print("CTC — CONNECTIONIST TEMPORAL CLASSIFICATION")
    print("=" * 70)
    print(textwrap.dedent("""\
    WHY CTC EXISTS
    ─────────────
    Problem: we have T ≈ 330 acoustic frames but only ~20 phones in the
    utterance. We don't know which frames correspond to which phones.
    Manual alignment is expensive. CTC solves this automatically.

    HOW IT WORKS
    ─────────────
    At every frame, the model outputs P(phone | frame) for all phones.
    CTC introduces a special [PAD] blank token, then decodes via:

      Raw stream   →  merge repeats  →  remove blanks  →  phone sequence

    The blank serves TWO purposes:
      1. Represents silence / transition between sounds
      2. Allows the same phone to appear twice: 'HH [PAD] HH' → 'HH HH'
         (without blank, 'HH HH' would collapse to just 'HH')
    """))

    # Build a short visual from actual model output (first 40 frames)
    n_show    = min(40, posteriors.shape[0])
    pred_ids  = np.argmax(posteriors[:n_show], axis=-1)
    raw_seq   = [phone_labels[i] for i in pred_ids]

    # Step 1: show raw (truncated for readability, show every 2nd frame)
    sampled = raw_seq[::2][:20]
    raw_str = " ".join(f"{p:>5}" for p in sampled)
    print(f"  RAW frames (every 2nd, first {n_show} frames of actual output):")
    print(f"  {raw_str}")

    # Step 2: merge adjacent duplicates
    merged: list[str] = []
    for p in raw_seq:
        if not merged or merged[-1] != p:
            merged.append(p)
    merge_str = " ".join(f"{p:>5}" for p in merged[:20])
    print(f"\n  AFTER merging adjacent duplicates:")
    print(f"  {merge_str}")

    # Step 3: remove blanks
    no_blank = [p for p in merged if p not in (BLANK_TOKEN, WORD_BOUNDARY, " ")]
    no_blank_str = " ".join(no_blank[:20])
    print(f"\n  AFTER removing blank+boundary tokens:")
    print(f"  {no_blank_str}")

    print(f"\n  FULL decoded sequence (entire clip):")
    # clean up the decoded string for display
    clean = decoded_str.replace("|", " | ").replace("  ", " ").strip()
    wrapped = textwrap.fill(clean, width=66, initial_indent="  ", subsequent_indent="  ")
    print(wrapped)

    print("\n  [KEY INSIGHT] The model never needed to know WHEN each phone")
    print("  occurred — CTC inferred the alignment from the data alone.")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_phone_posteriors(
    posteriors: np.ndarray,
    phone_labels: list[str],
    waveform: np.ndarray,
    sr: int,
    decoded_str: str,
    save_path: Path,
) -> None:
    """
    Three-panel figure:
      1. Waveform (time reference)
      2. Phone posterior heatmap (T_frames × n_phones)
         — blank/boundary tokens dimmed, top-activated phones highlighted
      3. Argmax phone track over time (which phone is dominant each frame)
    """
    duration_s  = len(waveform) / sr
    n_frames, n_phones = posteriors.shape
    time_frames = np.linspace(0, duration_s, n_frames)
    time_samples = np.linspace(0, duration_s, len(waveform))

    # ── Filter: exclude blank + word-boundary for the heatmap y-axis ─────────
    exclude = {BLANK_TOKEN, WORD_BOUNDARY, "[UNK]", "<s>", "</s>", " "}
    phone_indices  = [i for i, p in enumerate(phone_labels) if p not in exclude]
    phone_names    = [phone_labels[i] for i in phone_indices]
    post_filtered  = posteriors[:, phone_indices]   # shape: (T, n_real_phones)

    # Sort phones by mean activation for readability
    mean_act    = post_filtered.mean(axis=0)
    sort_order  = np.argsort(mean_act)[::-1]
    post_sorted = post_filtered[:, sort_order]
    names_sorted = [phone_names[i] for i in sort_order]

    fig = plt.figure(figsize=(16, 12), facecolor="#0e1117")
    fig.suptitle("Phase 2 — Phone Posteriors", color="white", fontsize=14,
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45, height_ratios=[1, 3, 1])

    # ── Panel 1: Waveform ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_samples, waveform, color="#4fc3f7", linewidth=0.4, alpha=0.8)
    ax1.set_facecolor("#1a1d23")
    ax1.set_title(f"Waveform  ({duration_s:.2f} s)", color="white", fontsize=10)
    ax1.set_xlim(0, duration_s)
    ax1.tick_params(colors="#666666")
    ax1.spines[:].set_color("#333333")
    ax1.set_ylabel("Amp", color="#aaaaaa", fontsize=8)

    # ── Panel 2: Posterior heatmap ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    img = ax2.imshow(
        post_sorted.T,             # shape: (n_real_phones, T_frames)
        aspect="auto",
        origin="lower",
        extent=(0.0, float(duration_s), -0.5, float(len(names_sorted)) - 0.5),
        cmap="inferno",
        vmin=0.0,
        vmax=0.6,
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax2, pad=0.01, label="Probability")
    ax2.set_facecolor("#1a1d23")
    ax2.set_yticks(range(len(names_sorted)))
    ax2.set_yticklabels(names_sorted, fontsize=7, color="#cccccc")
    ax2.set_title(
        f"Phone Posteriors  —  {n_frames} frames × {len(phone_indices)} phones  "
        f"(sorted by mean activation)",
        color="white", fontsize=10,
    )
    ax2.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
    ax2.tick_params(colors="#666666")
    ax2.spines[:].set_color("#333333")
    ax2.set_xlim(0, duration_s)

    # ── Panel 3: Argmax phone track ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    pred_ids = np.argmax(posteriors, axis=-1)         # shape: (T_frames,)
    # Remap to real-phone indices (blanks → -1)
    blank_idx = phone_labels.index(BLANK_TOKEN)
    is_blank  = pred_ids == blank_idx

    # For display: assign y-position based on phone rank in sorted list
    phone_to_rank: dict[str, int] = {name: i for i, name in enumerate(names_sorted)}
    default_rank = len(names_sorted) // 2
    y_track = np.array([
        phone_to_rank.get(phone_labels[pid], default_rank)
        for pid in pred_ids
    ], dtype=float)
    y_track[is_blank] = np.nan   # hide blanks (they become gaps in the line)

    ax3.scatter(time_frames[~is_blank], y_track[~is_blank],
                c="#81c784", s=2, alpha=0.6, linewidths=0)
    ax3.scatter(time_frames[is_blank],
                np.zeros(is_blank.sum()),
                c="#ff7043", s=1, alpha=0.3, linewidths=0, label="blank frame")
    ax3.set_facecolor("#1a1d23")
    ax3.set_title("Argmax phone per frame  (green=phone, red=blank)",
                  color="white", fontsize=10)
    ax3.set_xlim(0, duration_s)
    ax3.set_yticks(range(0, len(names_sorted), max(1, len(names_sorted) // 10)))
    ax3.set_yticklabels(
        [names_sorted[i] for i in range(0, len(names_sorted),
                                         max(1, len(names_sorted) // 10))],
        fontsize=7, color="#cccccc",
    )
    ax3.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
    ax3.tick_params(colors="#666666")
    ax3.spines[:].set_color("#333333")
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1d23")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log_shape(name: str, arr: np.ndarray) -> None:
    print(f"  {name:20s}: shape={arr.shape}  dtype={arr.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n" + "█" * 60)
    print("  PHASE 2 — PHONES & CTC")
    print("  Speech World Model: HuPER Implementation")
    print("█" * 60)

    project_root = Path(__file__).parent.parent
    audio_path   = project_root / "data" / "samples" / "librispeech_sample.wav"
    plot_path    = project_root / "data" / "phase2_phone_posteriors.png"

    # ── 1. ARPAbet reference ─────────────────────────────────────────────────
    print_arpabet_table()

    # ── 2. Load audio ────────────────────────────────────────────────────────
    print("--- Loading audio ---")
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio sample not found at {audio_path}. "
            "Run phase1/audio_explorer.py first to download it."
        )
    waveform, _sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    sr: int = int(_sr)
    log_shape("waveform", waveform)
    duration_s = len(waveform) / sr
    print(f"  Duration: {duration_s:.2f} s  |  SR: {sr:,} Hz\n")

    # ── 3. Load model ────────────────────────────────────────────────────────
    print("--- Loading phone recognizer ---")
    processor, model, device = load_phone_recognizer()
    print()

    # ── 4. Run inference ─────────────────────────────────────────────────────
    print("--- Running phone recognition ---")
    posteriors, phone_labels, decoded_str = run_phone_recognition(
        waveform, sr, processor, model, device,
    )
    log_shape("posteriors", posteriors)
    n_phones = posteriors.shape[1]
    print(f"  Frames (T)    : {posteriors.shape[0]}")
    print(f"  Phone vocab   : {n_phones}  (incl. blank/boundaries)")
    print(f"  Frames/second : {posteriors.shape[0] / duration_s:.1f} Hz")

    # ── 5. CTC explanation ───────────────────────────────────────────────────
    explain_ctc(decoded_str, phone_labels, posteriors)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    print("--- Plotting phone posteriors ---")
    plot_phone_posteriors(
        posteriors, phone_labels, waveform, sr, decoded_str, plot_path,
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    n_real = sum(1 for p in phone_labels if p not in
                 {BLANK_TOKEN, WORD_BOUNDARY, "[UNK]", "<s>", "</s>", " "})
    blank_frames = int((np.argmax(posteriors, axis=-1) ==
                        phone_labels.index(BLANK_TOKEN)).sum())
    pct_blank = 100 * blank_frames / posteriors.shape[0]

    print(f"\n{'=' * 60}")
    print("  Phase 2 complete.")
    print("  What we learned:")
    print(f"  1. English speech maps to ~{n_real} distinct phone types (IPA/ARPAbet)")
    print(f"  2. At 50Hz the model emits {n_phones}-dim prob vector every 20ms")
    print(f"  3. {blank_frames}/{posteriors.shape[0]} frames ({pct_blank:.0f}%) are blank")
    print(f"     — silence, transitions, uncertainty between phones")
    print( "  4. CTC collapses (T=330) noisy frames → short phone sequence")
    print( "     without ever needing frame-level alignment labels")
    print( "  5. This per-frame stream IS the 'evidence' E in (T, 1024)")
    print( "     HuPER's WavLM hidden states carry richer info than just argmax")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_demo()
