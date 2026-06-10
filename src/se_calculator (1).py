"""
Symbolic Category Entropy (SE) Calculator
==========================================
Computes SE heatmaps and the L2-norm Cohen's d scalar for the shuffle test.

Paper: Kurian & Monroy, "Symbolic Category Entropy: A New Entropy Metric for
       Order-Sensitive Narrative Semantics", IEEE CyberHumanities 2026.

GitHub: https://github.com/Mkurian99/Symbolic-Semantic-Entropy-Calculator-Basic

USAGE
-----
1. Set file paths and TARGET_WINDOWS in CONFIG below.
2. Replace MOTIF_DICT with your own categories.
3. Run:  python se_calculator.py

OUTPUT
------
  {prefix}_results.csv          per-window H, Sigma, L2-norm for each condition
  {prefix}_cohens_d.csv         scalar Cohen's d (L2-norm) per shuffle condition
  {prefix}_heatmap_orig.png     data-only heatmap, original
  {prefix}_heatmap_word.png     data-only heatmap, word-shuffled
  {prefix}_heatmap_sent.png     data-only heatmap, sentence-shuffled  (if provided)

PAPER PARAMETERS
----------------
  LOTR (~182 000 tokens):   TARGET_WINDOWS = 120  ->  window ~3 009 tok
  Genesis (~1 900 tokens):  TARGET_WINDOWS = 40   ->  window ~92 tok
"""

# ============================================================================
# CONFIG — edit before running
# ============================================================================

ORIGINAL_FILE      = "original.txt"          # .txt or .docx
WORD_SHUFFLE_FILE  = "word_shuffled.txt"      # .txt or .docx
SENT_SHUFFLE_FILE  = "sent_shuffled.txt"      # .txt or .docx  (set to "" to skip)

OUTPUT_PREFIX      = "se_output"              # prefix for all output files
TARGET_WINDOWS     = 120                      # 120 for long texts, 40 for short

# ─── Motif dictionary ────────────────────────────────────────────────────────
# Format A — simple word list:
#   "Category Name": ["word1", "word2", ...]
#
# Format B — phrases + words (phrases merged before tokenising):
#   "Category Name": {"phrases": ["multi word phrase"], "words": ["word1"]}
#
# Replace with your own categories. Default: LOTR Fellowship of the Ring (15 cat).

MOTIF_DICT = {
    "The One Ring": [
        "ring", "precious", "gold", "band", "circle", "ruling", "gollum",
        "invisibility", "chain", "burden", "magic", "finger", "pocket",
        "vanish", "secret", "found", "gave", "took", "kept", "possess",
        "will", "power"],
    "The Fellowship": [
        "fellowship", "companions", "company", "quest", "group", "nine",
        "walkers", "unity", "alliance", "brotherhood", "journey", "friends",
        "party", "together", "travel", "set", "band", "walking", "adventure",
        "frodo", "sam", "samwise", "merry", "meriadoc", "pippin", "peregrin",
        "aragorn", "strider", "legolas", "gimli", "boromir", "gandalf"],
    "The Shire": [
        "shire", "hobbiton", "bag", "end", "bywater", "hobbit", "green",
        "hill", "westfarthing", "eastfarthing", "southfarthing", "northfarthing",
        "home", "garden", "field", "post", "road", "row", "gaffer", "hole",
        "peaceful", "comfort", "folk"],
    "The Road/Journey": [
        "road", "journey", "path", "travel", "wander", "quest", "adventure",
        "trail", "route", "passage", "walking", "miles", "crossing", "errand",
        "march", "start", "way", "ahead", "behind", "go", "leave", "walk",
        "step"],
    "Light and Darkness": [
        "light", "darkness", "shadow", "dark", "bright", "gloom", "shining",
        "night", "dawn", "dusk", "sunlight", "lantern", "lamp", "moon", "star",
        "glow", "fire", "sun", "morning", "evening", "shine", "shadowy",
        "black", "white", "pale", "stars"],
    "The Shadow": [
        "shadow", "sauron", "enemy", "black", "rider", "nazgul", "ringwraith",
        "wraith", "darkness", "evil", "threat", "eye", "mordor", "pursuit",
        "fear", "dread", "cloak", "hood", "sniff", "follow", "hunt", "search",
        "danger", "servant", "master", "power"],
    "Nature and the Old Forest": [
        "forest", "tree", "old", "woods", "river", "willow", "grass", "leaf",
        "root", "hedge", "meadow", "glade", "bark", "moss", "stream",
        "thicket", "field", "hill", "water", "wood", "branch", "earth",
        "green", "wild"],
    "Songs and Poetry": [
        "song", "singing", "poem", "verse", "tune", "chant", "music",
        "melody", "rhyme", "ballad", "lay", "recite", "chorus", "elven",
        "hobbit", "sing", "voice", "words", "dance", "laugh", "cheer",
        "merry"],
    "Hospitality and Feasting": [
        "feast", "supper", "meal", "party", "table", "food", "drink", "ale",
        "breakfast", "lunch", "dinner", "banquet", "kitchen", "cook",
        "provision", "toast", "wine", "eat", "mug", "jug", "bread", "beer",
        "plate", "dish", "bottle", "present", "host", "guest", "welcome",
        "invitation"],
    "Gifts and Tokens": [
        "gift", "token", "present", "cloak", "sword", "treasure", "mathom",
        "keepsake", "heirloom", "trinket", "ring", "blade", "shield",
        "necklace", "relic", "label", "envelope", "parcel", "letter", "gold",
        "silver", "spoon", "pen", "ink", "key", "chest", "bag", "package"],
    "Hidden Identity and Disguise": [
        "disguise", "hidden", "invisibility", "secret", "cloak", "underhill",
        "mask", "concealed", "alias", "shadowed", "pseudonym", "eavesdrop",
        "spy", "vanish", "slip", "unseen", "invisible", "pocket", "name",
        "stranger", "unknown", "hiding"],
    "Friendship and Loyalty": [
        "friend", "friendship", "loyal", "companion", "trust", "bond",
        "fellowship", "support", "devotion", "kinship", "ally", "camaraderie",
        "faith", "together", "help", "faithful", "dear", "close"],
    "Ancient Lore and History": [
        "lore", "history", "legend", "tale", "story", "ancient", "elendil",
        "myth", "record", "annals", "memory", "past", "old", "days",
        "genealogy", "saga", "told", "remember"],
    "Guardians and Guides": [
        "guide", "guardian", "gandalf", "elrond", "tom", "bombadil",
        "protector", "mentor", "leader", "watcher", "helper", "advisor",
        "steward", "teacher", "wisdom", "help", "advice", "counsel", "lead",
        "protect", "watch", "wise"],
    "Temptation and Choice": [
        "temptation", "choice", "choose", "decision", "will", "mercy",
        "struggle", "test", "resolve", "dilemma", "trial", "fate", "free",
        "crossroads", "resist", "desire", "wish", "decide", "must", "cannot",
        "want", "should", "need", "willpower"],
}

# ============================================================================
# DEPENDENCIES
# ============================================================================
import re
import os
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — heatmaps will not be saved (pip install matplotlib)")

# ============================================================================
# FILE READING
# ============================================================================

def read_file(path):
    """Read .txt or .docx. Returns plain text string."""
    if not path or not os.path.exists(path):
        return None
    if path.lower().endswith(".docx"):
        try:
            from docx import Document
            return " ".join(p.text for p in Document(path).paragraphs)
        except ImportError:
            sys.exit("python-docx required for .docx files: pip install python-docx")
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    sys.exit(f"Cannot decode {path}")

# ============================================================================
# MOTIF DICTIONARY NORMALISATION
# ============================================================================

def normalise_motif_dict(raw):
    """
    Accept both simple lists and {'phrases': [...], 'words': [...]} format.
    Returns:
      categories  — ordered list of category names
      motif_sets  — {category: set of word-tokens (including merged phrases)}
      phrase_map  — [(phrase_text, merged_token), ...] sorted longest-first
    """
    categories = list(raw.keys())
    motif_sets = {}
    phrase_map = []

    for cat, value in raw.items():
        if isinstance(value, dict):
            words   = list(value.get("words", []))
            phrases = list(value.get("phrases", []))
        else:
            words   = list(value)
            phrases = []

        for ph in phrases:
            merged = ph.lower().replace(" ", "_")
            phrase_map.append((ph.lower(), merged))
            words.append(merged)

        motif_sets[cat] = set(w.lower() for w in words)

    phrase_map.sort(key=lambda x: -len(x[0]))   # longest phrase first
    return categories, motif_sets, phrase_map

# ============================================================================
# TOKENISATION
# ============================================================================

def tokenise(text, phrase_map):
    """
    1. Lowercase.
    2. Merge multi-word phrases (longest-first) into single underscore tokens.
    3. Split on word boundaries.
    """
    text = text.lower()
    for phrase, merged in phrase_map:
        text = text.replace(phrase, merged)
    text = re.sub(r"[^a-z\s_'\-]", " ", text)
    return re.findall(r"\b[a-z_][a-z_'\-]*\b", text)

# ============================================================================
# WINDOWING
# ============================================================================

def make_windows(tokens, target_windows):
    """
    Adaptive sliding windows with 50% overlap.
    window_size = total_tokens / (1 + (target_windows - 1) / 2)
    Returns list of token slices.
    """
    N           = len(tokens)
    window_size = max(20, int(N / (1 + (target_windows - 1) / 2)))
    step_size   = window_size // 2
    windows     = []
    start       = 0
    while start + window_size <= N:
        windows.append(tokens[start : start + window_size])
        start += step_size
    return windows, window_size, step_size

# ============================================================================
# GLOBAL BASELINE  (computed from original only)
# ============================================================================

def global_baseline(tokens, categories, motif_sets):
    """pi_k = proportion of motif-k tokens in the full original text."""
    N = len(tokens)
    return {cat: sum(1 for t in tokens if t in motif_sets[cat]) / N
            for cat in categories}

# ============================================================================
# KL MATRIX  (n_windows × n_categories)
# ============================================================================

def kl_matrix(windows, categories, motif_sets, baseline):
    """
    Each cell = p_k(w) * log2(p_k(w) / pi_k) if both > 0, else 0.
    p_k(w) = count of motif-k tokens in window / window size.
    pi_k   = global baseline proportion.
    """
    rows = []
    for window in windows:
        ws  = len(window)
        row = []
        for cat in categories:
            count = sum(1 for t in window if t in motif_sets[cat])
            p_k   = count / ws
            pi_k  = baseline[cat]
            row.append(p_k * np.log2(p_k / pi_k)
                       if p_k > 0 and pi_k > 0 else 0.0)
        rows.append(row)
    return np.array(rows)   # shape: (n_windows, n_categories)

# ============================================================================
# SHANNON ENTROPY  (per window)
# ============================================================================

def shannon_entropy(window):
    from collections import Counter
    counts = Counter(window)
    n      = len(window)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())

# ============================================================================
# COHEN'S d  (pooled SD)
# ============================================================================

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled  = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

# ============================================================================
# DATA-ONLY HEATMAP  (no axes, no chrome — shared colour scale)
# ============================================================================

def save_heatmap(kl_mat, path, vmin, vmax):
    """
    Saves a pure plasma heatmap (no axes, no title, no colorbar).
    All conditions use vmin/vmax from the original so pixels are comparable.
    """
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(kl_mat.T, aspect="auto", cmap="plasma",
              interpolation="nearest", origin="lower",
              vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def run(orig_text, word_text, sent_text,
        motif_dict, target_windows, prefix):

    categories, motif_sets, phrase_map = normalise_motif_dict(motif_dict)
    n_cats = len(categories)

    # ── Tokenise ─────────────────────────────────────────────────────────────
    orig_tok = tokenise(orig_text, phrase_map)
    word_tok = tokenise(word_text, phrase_map)
    sent_tok = tokenise(sent_text, phrase_map) if sent_text else None

    print(f"\n  Tokens  —  orig: {len(orig_tok):,}  "
          f"word: {len(word_tok):,}"
          + (f"  sent: {len(sent_tok):,}" if sent_tok else ""))

    # ── Windows ───────────────────────────────────────────────────────────────
    orig_wins, ws, ss = make_windows(orig_tok, target_windows)
    word_wins, _,  _  = make_windows(word_tok, target_windows)
    sent_wins         = make_windows(sent_tok, target_windows)[0] if sent_tok else None

    print(f"  Window  —  size: {ws:,} tok  step: {ss:,} tok  "
          f"~{len(orig_wins)} windows (orig)")

    # ── Baseline from original ────────────────────────────────────────────────
    baseline = global_baseline(orig_tok, categories, motif_sets)
    nz = sum(1 for v in baseline.values() if v > 0)
    print(f"  Baseline — {nz}/{n_cats} categories non-zero")

    # ── KL matrices ───────────────────────────────────────────────────────────
    print("  Building KL matrices...")
    orig_mat = kl_matrix(orig_wins, categories, motif_sets, baseline)
    word_mat = kl_matrix(word_wins, categories, motif_sets, baseline)
    sent_mat = kl_matrix(sent_wins, categories, motif_sets, baseline) \
               if sent_wins else None

    # ── Shannon H per window ──────────────────────────────────────────────────
    orig_H = np.array([shannon_entropy(w) for w in orig_wins])
    word_H = np.array([shannon_entropy(w) for w in word_wins])
    sent_H = np.array([shannon_entropy(w) for w in sent_wins]) \
             if sent_wins else None

    # ── L2-norm per window  (paper scalar) ───────────────────────────────────
    # Each window's KL contribution vector has shape (n_categories,).
    # L2-norm = sqrt(sum of squares) = total motif concentration magnitude.
    orig_l2 = np.linalg.norm(orig_mat, axis=1)
    word_l2 = np.linalg.norm(word_mat, axis=1)
    sent_l2 = np.linalg.norm(sent_mat, axis=1) if sent_mat is not None else None

    # ── Cohen's d  (paper Table II scalar) ───────────────────────────────────
    d_word = cohens_d(orig_l2, word_l2)
    d_sent = cohens_d(orig_l2, sent_l2) if sent_l2 is not None else None

    # ── Sigma (aggregate KL per window, for reference) ───────────────────────
    orig_sigma = orig_mat.sum(axis=1)
    word_sigma = word_mat.sum(axis=1)

    # ── Console report ────────────────────────────────────────────────────────
    def rating(d):
        d = abs(d)
        if d >= 1.5: return "Strong"
        if d >= 0.8: return "Moderate"
        if d >= 0.5: return "Weak"
        return "Fail"

    print(f"\n{'='*55}")
    print("  SHUFFLE TEST RESULTS")
    print(f"  Scale: Strong≥1.5 | Moderate≥0.8 | Weak≥0.5 | Fail<0.5")
    print(f"  (Cohen 1988; Sawilowsky 2009)")
    print(f"{'='*55}")
    print(f"  {'Condition':<20} {'L2-norm d':>10}  {'Rating'}")
    print(f"  {'-'*45}")
    print(f"  {'Word-shuffle':<20} {d_word:>10.4f}  {rating(d_word)}")
    if d_sent is not None:
        print(f"  {'Sentence-shuffle':<20} {d_sent:>10.4f}  {rating(d_sent)}")
    print(f"{'='*55}")

    # ── Save CSV  (per-window results) ────────────────────────────────────────
    rows = []
    for i, (h, sigma, l2) in enumerate(zip(orig_H, orig_sigma, orig_l2)):
        rows.append({"condition": "original", "window": i,
                     "H": h, "Sigma": sigma, "L2": l2})
    for i, (h, l2) in enumerate(zip(word_H, word_l2)):
        rows.append({"condition": "word_shuffle", "window": i,
                     "H": h, "Sigma": word_mat[i].sum(), "L2": l2})
    if sent_l2 is not None:
        for i, (h, l2) in enumerate(zip(sent_H, sent_l2)):
            rows.append({"condition": "sent_shuffle", "window": i,
                         "H": h, "Sigma": sent_mat[i].sum(), "L2": l2})

    results_csv = f"{prefix}_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    # ── Save Cohen's d summary ────────────────────────────────────────────────
    d_rows = [{"condition": "word_shuffle",
               "cohens_d_L2": round(d_word, 4),
               "rating": rating(d_word),
               "n_windows_orig": len(orig_l2),
               "n_windows_shuf": len(word_l2)}]
    if d_sent is not None:
        d_rows.append({"condition": "sent_shuffle",
                       "cohens_d_L2": round(d_sent, 4),
                       "rating": rating(d_sent),
                       "n_windows_orig": len(orig_l2),
                       "n_windows_shuf": len(sent_l2)})
    d_csv = f"{prefix}_cohens_d.csv"
    pd.DataFrame(d_rows).to_csv(d_csv, index=False)

    print(f"\n  Saved: {results_csv}")
    print(f"  Saved: {d_csv}")

    # ── Save heatmaps  (data-only, shared vmin/vmax from original) ────────────
    vmin, vmax = 0, orig_mat.max()
    save_heatmap(orig_mat, f"{prefix}_heatmap_orig.png", vmin, vmax)
    save_heatmap(word_mat, f"{prefix}_heatmap_word.png", vmin, vmax)
    if sent_mat is not None:
        save_heatmap(sent_mat, f"{prefix}_heatmap_sent.png", vmin, vmax)
    if HAS_MPL:
        print(f"  Saved: {prefix}_heatmap_orig/word/sent.png")

    return d_word, d_sent


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("SE Calculator — reading files...")
    orig = read_file(ORIGINAL_FILE)
    word = read_file(WORD_SHUFFLE_FILE)
    sent = read_file(SENT_SHUFFLE_FILE) if SENT_SHUFFLE_FILE else None

    if orig is None: sys.exit(f"Original file not found: {ORIGINAL_FILE}")
    if word is None: sys.exit(f"Word-shuffle file not found: {WORD_SHUFFLE_FILE}")

    run(orig, word, sent, MOTIF_DICT, TARGET_WINDOWS, OUTPUT_PREFIX)
