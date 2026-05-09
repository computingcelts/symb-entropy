"""
================================================================================
SYMBOLIC CATEGORY ENTROPY (SE) MASTER CALCULATOR - Three-Way Comparison Edition
================================================================================

FEATURES:
✓ Core SE Analysis: (SΣ, H) with proper KL divergence
✓ Multi-word Phrase Support (semantic token definition)
✓ Multi-format Support (.txt and .docx files)
✓ THREE-WAY Shuffle Validation: Original vs Word-Shuffle vs Sentence-Shuffle

THEORETICAL FOUNDATION:
    SE = (H, Σ)
    
    Where:
        H = Shannon entropy (bits/-token) - lexical unpredictability
        Σ = KL divergence (bits/-token) - motif clustering beyond baseline

KEY INNOVATION - Semantic Token Definition:
    - "One Ring" = 1 semantic token (merged pre-processing)
    - Phrases in motif dictionary are pre-merged: "one ring" → "one_ring"
    - Results in bits per SEMANTIC TOKEN, as well as word-tokens
    - Captures linguistic compression naturally

THREE-WAY SHUFFLE VALIDATION:
    - Accepts three input files: original, word-shuffled, sentence-shuffled
    - All heatmaps use ORIGINAL z-axis scale for valid comparison
    - Reveals hierarchical sensitivity to structural destruction:
        * Word shuffle: Destroys ALL structure → Σ collapses maximally
        * Sentence shuffle: Preserves intra-sentence structure → Σ partially preserved
    - Expected collapse ratios:
        * Word shuffle: 10-20x collapse
        * Sentence shuffle: 3-8x collapse (intermediate)

MOTIF DICTIONARY FORMAT:
    motif_dict = {
        'Category Name': {
            'phrases': ['multi word phrase', 'another phrase'],  # Optional
            'words': ['single', 'word', 'tokens']                # Optional
        }
    }
    
    - Phrase-only motifs: Only include 'phrases' key
    - Word-only motifs: Only include 'words' key  
    - Mixed motifs: Include both keys

IMPLEMENTATION STANDARDS:
    - Adaptive window sizing (auto-scales to text length)
    - Default: ~120 windows with 50% overlap
    - To adjust granularity: Change TARGET_WINDOWS (line ~120)
      - Higher value = more windows = finer resolution
      - Lower value = fewer windows = coarser resolution
    - Global baseline (π_k) from full text distribution
    - Whole-word/phrase matching with word boundaries
    - Falsifiable via shuffle testing (Σ → 0 when randomized)

OUTPUTS (Original):
    - <textname>_se_heatmap.png          (KL heatmap with H real-scale axis)
    - <textname>_se_results.csv          (full numerical results)

OUTPUTS (Word Shuffle):
    - <textname>_word_shuffle_se_results.csv

OUTPUTS (Sentence Shuffle):
    - <textname>_sent_shuffle_se_results.csv

OUTPUTS (Comparison):
    - <textname>_3way_comparison.png     (clean trifold with H real-scale axis)
    - <textname>_validation_summary.csv  (statistical comparison)

DEPENDENCIES:
    - numpy, pandas, matplotlib, scipy
    - python-docx (for .docx support: pip install python-docx)

USAGE:
    python SE_Master_Calculator_3Way.py <original> <word_shuffle> <sentence_shuffle>
    
    Or edit TEXT_FILE, WORD_SHUFFLE_FILE, SENT_SHUFFLE_FILE variables
    
    To adjust window granularity:
    - Edit TARGET_WINDOWS (line ~120)
    - Default: 120 windows
    - Higher = finer resolution (more windows, smaller size)
    - Lower = coarser resolution (fewer windows, larger size)

VERSION: 3.0.0 - Three-way comparison with hierarchical validation
AUTHOR: Kurian, M. (2025)
================================================================================
"""

import re
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import docx for .docx file support
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ============================================================================
# Documents and Shuffles
# ============================================================================

# Default text files (can be overridden via command line)
# Set to None or empty string to skip that condition
TEXT_FILE = "C:/Users/Michael Kurian/Desktop/Advanced Researches/TXT Files for SE/LOTR FELLOWSHIP SE.txt"
WORD_SHUFFLE_FILE = "C:/Users/Michael Kurian/Desktop/Advanced Researches/TXT Files for SE/LOTR Word Randomized SE.txt"
SENT_SHUFFLE_FILE = "C:/Users/Michael Kurian/Desktop/Advanced Researches/TXT Files for SE/LOTR Sentence Shuffled SE.txt"

# Random seed for reproducibility
RANDOM_SEED = 42

# Window parameters - ADAPTIVE sizing
# The window size auto-adjusts to text length to produce ~120 windows
TARGET_WINDOWS = 120  # Adjust this to change granularity (higher = more windows)
# Note: Actual window size will be calculated as: total_tokens / (1 + (TARGET_WINDOWS - 1) / 2)

# ============================================================================
# MOTIF DICTIONARY - Example: FELLOWSHIP OF THE RING
# ============================================================================

motif_dict = {
    'The One Ring': [
        'ring', 'precious', 'gold', 'band', 'circle', 'master-ring', 'ruling', 
        'gollum', 'invisibility', 'chain', 'burden', 'magic', 'finger', 'pocket', 
        'vanish', 'secret', 'found', 'gave', 'took', 'kept', 'possess', 'will', 'power'
    ],
    
    'The Fellowship': [
        'fellowship', 'companions', 'company', 'quest', 'group', 'nine', 'walkers', 
        'unity', 'alliance', 'brotherhood', 'journey', 'friends', 'party', 'together', 
        'travel', 'set', 'band', 'walking', 'adventure',
        'frodo', 'sam', 'samwise', 'merry', 'meriadoc', 'pippin', 'peregrin',
        'aragorn', 'strider', 'legolas', 'gimli', 'boromir', 'gandalf'
    ],
    
    'The Shire': [
        'shire', 'hobbiton', 'bag', 'end', 'bywater', 'hobbit', 'green', 'hill', 
        'westfarthing', 'eastfarthing', 'southfarthing', 'northfarthing', 'home', 
        'garden', 'field', 'post', 'road', 'row', 'gaffer', 'hole', 'peaceful', 
        'comfort', 'folk'
    ],
    
    'The Road/Journey': [
        'road', 'journey', 'path', 'travel', 'wander', 'quest', 'adventure', 'trail', 
        'route', 'passage', 'walking', 'miles', 'crossing', 'errand', 'march', 'start', 
        'way', 'ahead', 'behind', 'go', 'leave', 'walk', 'step'
    ],
    
    'Light and Darkness': [
        'light', 'darkness', 'shadow', 'dark', 'bright', 'gloom', 'shining', 'night', 
        'dawn', 'dusk', 'sunlight', 'lantern', 'lamp', 'moon', 'star', 'glow', 'fire', 
        'sun', 'morning', 'evening', 'shine', 'shadowy', 'black', 'white', 'pale', 'stars'
    ],
    
    'The Shadow': [
        'shadow', 'sauron', 'enemy', 'black', 'rider', 'nazgul', 'ringwraith', 'wraith', 
        'darkness', 'evil', 'threat', 'eye', 'mordor', 'pursuit', 'fear', 'dread', 
        'cloak', 'hood', 'sniff', 'follow', 'hunt', 'search', 'danger', 'servant', 
        'master', 'power'
    ],
    
    'Nature and the Old Forest': [
        'forest', 'tree', 'old', 'woods', 'river', 'willow', 'grass', 'leaf', 'root', 
        'hedge', 'meadow', 'glade', 'bark', 'moss', 'stream', 'thicket', 'field', 
        'hill', 'water', 'wood', 'branch', 'earth', 'green', 'wild'
    ],
    
    'Songs and Poetry': [
        'song', 'singing', 'poem', 'verse', 'tune', 'chant', 'music', 'melody', 'rhyme', 
        'ballad', 'lay', 'recite', 'chorus', 'elven', 'hobbit', 'sing', 'voice', 'words', 
        'dance', 'laugh', 'cheer', 'merry'
    ],
    
    'Hospitality and Feasting': [
        'feast', 'supper', 'meal', 'party', 'table', 'food', 'drink', 'ale', 'breakfast', 
        'lunch', 'dinner', 'banquet', 'kitchen', 'cook', 'provision', 'toast', 'wine', 
        'eat', 'mug', 'jug', 'bread', 'beer', 'plate', 'dish', 'bottle', 'present', 
        'host', 'guest', 'welcome', 'invitation'
    ],
    
    'Gifts and Tokens': [
        'gift', 'token', 'present', 'cloak', 'sword', 'treasure', 'mathom', 'keepsake', 
        'heirloom', 'trinket', 'ring', 'blade', 'shield', 'necklace', 'relic', 'label', 
        'envelope', 'parcel', 'letter', 'gold', 'silver', 'spoon', 'pen', 'ink', 'key', 
        'chest', 'bag', 'package'
    ],
    
    'Hidden Identity and Disguise': [
        'disguise', 'hidden', 'invisibility', 'secret', 'cloak', 'underhill', 'mask', 
        'concealed', 'alias', 'shadowed', 'pseudonym', 'eavesdrop', 'spy', 'vanish', 
        'slip', 'unseen', 'invisible', 'pocket', 'name', 'stranger', 'unknown', 'hiding'
    ],
    
    'Friendship and Loyalty': [
        'friend', 'friendship', 'loyal', 'companion', 'trust', 'bond', 'fellowship', 
        'support', 'devotion', 'kinship', 'ally', 'camaraderie', 'faith', 'together', 
        'help', 'faithful', 'dear', 'close'
    ],
    
    'Ancient Lore and History': [
        'lore', 'history', 'legend', 'tale', 'story', 'ancient', 'elendil', 'myth', 
        'record', 'annals', 'memory', 'past', 'old', 'days', 'genealogy', 'saga', 
        'told', 'remember'
    ],
    
    'Guardians and Guides': [
        'guide', 'guardian', 'gandalf', 'elrond', 'tom', 'bombadil', 'protector', 
        'mentor', 'leader', 'watcher', 'helper', 'advisor', 'steward', 'teacher', 
        'wisdom', 'help', 'advice', 'counsel', 'lead', 'protect', 'watch', 'wise'
    ],
    
    'Temptation and Choice': [
        'temptation', 'choice', 'choose', 'decision', 'will', 'mercy', 'struggle', 
        'test', 'resolve', 'dilemma', 'trial', 'fate', 'free', 'crossroads', 'resist', 
        'desire', 'wish', 'decide', 'must', 'cannot', 'want', 'should', 'need', 'willpower'
    ]
}


# ============================================================================
# TEXT PREPROCESSING WITH MULTI-WORD PHRASE SUPPORT
# ============================================================================

def extract_text_from_file(file_path):
    """
    Extract text from .txt or .docx file with robust encoding handling.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.docx':
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    
    elif file_ext == '.txt':
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file with any of: {encodings}")
    
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Use .txt or .docx")


def merge_phrases_in_text(text, motif_dict):
    """
    Merge multi-word phrases in text into single tokens.
    Example: "tree of life" -> "tree_of_life"
    
    Returns: (modified_text, phrase_count)
    """
    phrase_count = 0
    
    # Extract all phrases from motif dictionary
    all_phrases = []
    for category_data in motif_dict.values():
        if 'phrases' in category_data:
            all_phrases.extend(category_data['phrases'])
    
    # Sort by length (longest first) to handle overlapping phrases
    all_phrases = sorted(set(all_phrases), key=len, reverse=True)
    
    # Replace each phrase with underscored version
    modified_text = text.lower()
    for phrase in all_phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        replacement = phrase.replace(' ', '_')
        matches = len(re.findall(pattern, modified_text))
        if matches > 0:
            modified_text = re.sub(pattern, replacement, modified_text)
            phrase_count += matches
    
    return modified_text, phrase_count


def tokenize_text(text):
    """
    Tokenize text into words (after phrase merging).
    Uses word boundary matching.
    """
    # Extract words (including underscored phrases)
    tokens = re.findall(r'\b[\w]+\b', text.lower())
    return tokens


# ============================================================================
# ADAPTIVE WINDOW SIZING
# ============================================================================

def calculate_adaptive_window_size(total_tokens, target_windows=120):
    """
    Calculate window size to produce approximately target_windows.
    With 50% overlap: n_windows = 1 + (total_tokens - window_size) / step_size
                                = 1 + (total_tokens - window_size) / (window_size/2)
    Solving: window_size ≈ total_tokens / (1 + (target_windows - 1)/2)
    """
    window_size = int(total_tokens / (1 + (target_windows - 1) / 2))
    window_size = max(window_size, 50)  # Floor to prevent degenerate windows
    
    return window_size


# ============================================================================
# SHANNON ENTROPY CALCULATION
# ============================================================================

def calculate_shannon_entropy(tokens):
    """
    Calculate Shannon entropy H for a list of tokens.
    H = -Σ p(x) log₂ p(x)
    Returns bits per token.
    """
    if not tokens:
        return 0.0
    
    freq = Counter(tokens)
    total = len(tokens)
    
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


# ============================================================================
# BASELINE CALCULATION (GLOBAL TEXT DISTRIBUTION)
# ============================================================================

def calculate_global_baseline(tokens, motif_dict):
    """
    Calculate baseline (π_k) as proportion of each motif in full text.
    This represents the text's overall distribution.
    """
    N = len(tokens)
    baseline = {}
    
    for category, category_data in motif_dict.items():
        total_count = 0
        
        # Count words
        if 'words' in category_data:
            for word in category_data['words']:
                total_count += tokens.count(word)
        
        # Count phrases (now merged with underscores)
        if 'phrases' in category_data:
            for phrase in category_data['phrases']:
                merged_phrase = phrase.replace(' ', '_')
                total_count += tokens.count(merged_phrase)
        
        baseline[category] = total_count / N if N > 0 else 0.0
    
    return baseline


# ============================================================================
# SIGMA (Σ) CALCULATION - KL DIVERGENCE
# ============================================================================

def calculate_sigma_kl(observed, baseline, window_size):
    """
    Calculate Σ using proper KL divergence.
    Formula: Σ_KL = Σ_k p_k × log₂(p_k / π_k)
    Returns bits per token.
    """
    sigma_kl = 0.0
    
    for category in observed.keys():
        obs_count = observed[category]
        p_k = obs_count / window_size
        pi_k = baseline[category]
        
        # Only calculate when both probabilities are non-zero
        if p_k > 0 and pi_k > 0:
            sigma_kl += p_k * np.log2(p_k / pi_k)
    
    return sigma_kl


# ============================================================================
# MOTIF COUNTING FOR WINDOWS
# ============================================================================

def count_motifs_in_window(window_tokens, motif_dict):
    """
    Count occurrences of each motif category in window.
    Returns dict of {category: count}.
    """
    counts = {}
    
    for category, category_data in motif_dict.items():
        count = 0
        
        # Count words
        if 'words' in category_data:
            for word in category_data['words']:
                count += window_tokens.count(word)
        
        # Count merged phrases
        if 'phrases' in category_data:
            for phrase in category_data['phrases']:
                merged_phrase = phrase.replace(' ', '_')
                count += window_tokens.count(merged_phrase)
        
        counts[category] = count
    
    return counts


# ============================================================================
# MOTIF DICTIONARY NORMALIZATION
# ============================================================================

def normalize_motif_dict(motif_dict):
    """
    Normalize motif dictionary to canonical format.
    
    Accepts two formats:
    
    1. Words-only (flat list):
        {'Category': ['word1', 'word2', 'word3']}
    
    2. Structured (words and/or phrases):
        {'Category': {'words': ['word1'], 'phrases': ['multi word phrase']}}
    
    Returns: dict in structured format (format 2).
    """
    normalized = {}
    
    for category, category_data in motif_dict.items():
        if isinstance(category_data, list):
            # Flat list format → wrap in {'words': [...]}
            normalized[category] = {'words': category_data}
        elif isinstance(category_data, dict):
            # Already structured format
            normalized[category] = category_data
        else:
            raise ValueError(
                f"Motif category '{category}' has unsupported type: {type(category_data)}. "
                f"Expected list of words or dict with 'words'/'phrases' keys."
            )
    
    return normalized


# ============================================================================
# MAIN SE ANALYSIS FUNCTION
# ============================================================================

def run_se_analysis(text_path, motif_dict):
    """
    Run complete Symbolic Category Entropy analysis on text.
    Returns: (results_df, raw_densities, kl_contributions, motif_dict, 
              baseline, window_size, total_tokens, tokens)
    """
    # Normalize motif dictionary to handle both flat-list and structured formats
    motif_dict = normalize_motif_dict(motif_dict)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {os.path.basename(text_path)}")
    print(f"{'='*70}\n")
    
    # Extract text
    print("Step 1/6: Extracting text...")
    raw_text = extract_text_from_file(text_path)
    
    # Merge multi-word phrases
    print("Step 2/6: Merging multi-word phrases...")
    text, n_phrases = merge_phrases_in_text(raw_text, motif_dict)
    print(f"  → Merged {n_phrases} phrase occurrences")
    
    # Tokenize
    print("Step 3/6: Tokenizing...")
    tokens = tokenize_text(text)
    total_tokens = len(tokens)
    print(f"  → {total_tokens:,} semantic tokens")
    
    # Calculate adaptive window size
    window_size = calculate_adaptive_window_size(total_tokens, TARGET_WINDOWS)
    step_size = window_size // 2  # 50% overlap
    print(f"\nStep 4/6: Calculating adaptive window parameters...")
    print(f"  → Window size: {window_size} tokens")
    print(f"  → Step size: {step_size} tokens (50% overlap)")
    
    # Calculate global baseline
    print("Step 5/6: Computing global baseline...")
    baseline = calculate_global_baseline(tokens, motif_dict)
    
    # Sliding window analysis
    print("Step 6/6: Running sliding window analysis...")
    results = []
    raw_densities = []
    kl_contributions = []
    
    n_windows = 1 + (total_tokens - window_size) // step_size
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        if end > total_tokens:
            break
        
        window_tokens = tokens[start:end]
        
        # Shannon entropy
        H = calculate_shannon_entropy(window_tokens)
        
        # Motif counts
        observed = count_motifs_in_window(window_tokens, motif_dict)
        
        # Sigma (KL divergence)
        sigma = calculate_sigma_kl(observed, baseline, window_size)
        
        # Store results
        results.append({
            'window_index': i,
            'start_token': start,
            'end_token': end,
            'H': H,
            'Sigma': sigma
        })
        
        # Store raw densities (for heatmap)
        raw_densities.append([observed[cat] / window_size for cat in motif_dict.keys()])
        
        # Store KL contributions (for heatmap)
        kl_contribs = []
        for cat in motif_dict.keys():
            p_k = observed[cat] / window_size
            pi_k = baseline[cat]
            if p_k > 0 and pi_k > 0:
                contrib = p_k * np.log2(p_k / pi_k)
            else:
                contrib = 0.0
            kl_contribs.append(contrib)
        kl_contributions.append(kl_contribs)
    
    results_df = pd.DataFrame(results)
    
    print(f"  → Generated {len(results_df)} windows")
    print(f"✓ Analysis complete!\n")
    
    return (results_df, np.array(raw_densities), np.array(kl_contributions), 
            motif_dict, baseline, window_size, total_tokens, tokens)


# ============================================================================
# ROBUST LINE SCALING FOR HEATMAP OVERLAYS
# ============================================================================

def scale_line_for_heatmap(values, n_categories):
    """
    Scale H or Σ values to [0, n_categories-1] for heatmap overlay.
    Straight min-max: each point shows what proportion of the way
    from minimum to maximum it sits.
    """
    lo, hi = values.min(), values.max()
    return (n_categories - 1) * (values - lo) / (hi - lo + 1e-10)


# Create a plasma colormap copy with black for exact-zero (masked) cells
_plasma_zero_black = plt.cm.plasma.copy()
_plasma_zero_black.set_bad('#12103a')


def _make_kl_hover(ax, unmasked_data, categories, results_df=None):
    """
    Attach a format_coord to ax so hovering reports KL values,
    plus H and Σ if results_df is provided.
    """
    def format_coord(x, y):
        col = int(round(x))
        row = int(round(y))
        parts = []
        if 0 <= row < unmasked_data.shape[0] and 0 <= col < unmasked_data.shape[1]:
            val = unmasked_data[row, col]
            cat = categories[row] if row < len(categories) else ''
            parts.append(f'Window {col}, {cat}: {val:.6f} bits/token')
        if results_df is not None and 0 <= col < len(results_df):
            row_data = results_df.iloc[col]
            parts.append(f'H={row_data["H"]:.4f}, SΣ={row_data["Sigma"]:.6f}')
        return '  |  '.join(parts) if parts else ''
    ax.format_coord = format_coord


def _make_kl_hover_twin(ax_twin, unmasked_data, categories, n_categories, results_df=None):
    """
    Attach format_coord to a twinx axes, mapping its Y coordinates
    back to heatmap row indices for KL value lookup.
    """
    def format_coord(x, y_h):
        col = int(round(x))
        # Map from twin Y (H bits/token) to heatmap row index
        h_min, h_max = ax_twin.get_ylim()
        if h_max - h_min > 0:
            frac = (y_h - h_min) / (h_max - h_min)
            row = int(round(-0.5 + frac * n_categories))
        else:
            row = -1
        parts = []
        if 0 <= row < unmasked_data.shape[0] and 0 <= col < unmasked_data.shape[1]:
            val = unmasked_data[row, col]
            cat = categories[row] if row < len(categories) else ''
            parts.append(f'Window {col}, {cat}: {val:.6f} bits/token')
        if results_df is not None and 0 <= col < len(results_df):
            row_data = results_df.iloc[col]
            parts.append(f'H={row_data["H"]:.4f}, SΣ={row_data["Sigma"]:.6f}')
        return '  |  '.join(parts) if parts else ''
    ax_twin.format_coord = format_coord


# ============================================================================
# VISUALIZATION: DUAL HEATMAP
# ============================================================================

def plot_dual_heatmap(results_df, raw_densities, kl_contributions, 
                     motif_dict, output_prefix, z_limits=None, condition_label=None):
    """
    Plot dual heatmap: Raw motif density + KL contributions.
    Includes overlaid line plots of H (cyan) and Σ (white) on the KL heatmap.
    If z_limits provided, use those for color scaling (for shuffle comparison).
    z_limits format: {'density': (vmin, vmax), 'kl': (vmin, vmax)}
    condition_label: 'original', 'word_shuffle', 'sent_shuffle' for filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    window_indices = results_df['window_index'].values
    
    # Normalize H and Σ for overlay (scale to heatmap Y-axis range)
    H_values = results_df['H'].values
    sigma_values = results_df['Sigma'].values
    
    # Robust percentile-based scaling to [0, n_categories-1] range
    H_scaled = scale_line_for_heatmap(H_values, n_categories)
    sigma_scaled = scale_line_for_heatmap(sigma_values, n_categories)
    
    # LEFT: Raw density heatmap
    if z_limits and 'density' in z_limits:
        vmin_d, vmax_d = z_limits['density']
    else:
        vmin_d, vmax_d = raw_densities.min(), raw_densities.max()
    
    raw_data = raw_densities.T
    raw_display = np.ma.masked_where(raw_data == 0, raw_data)
    im1 = ax1.imshow(raw_display, aspect='auto', cmap=_plasma_zero_black,
                     interpolation='nearest', origin='lower', 
                     vmin=vmin_d, vmax=vmax_d)
    _make_kl_hover(ax1, raw_data, categories)
    ax1.set_ylim(-0.5, n_categories - 0.5)
    ax1.set_xlabel('Window Index', fontsize=12)
    ax1.set_ylabel('Motif Category', fontsize=12)
    ax1.set_yticks(range(n_categories))
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_title('Method 1: RAW DENSITY\n(Simple frequency counting)', 
                  fontsize=13, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Proportion of Window', fontsize=10)
    
    # RIGHT: KL Divergence with line overlays
    if z_limits and 'kl' in z_limits:
        vmin_k, vmax_k = z_limits['kl']
    else:
        vmin_k, vmax_k = kl_contributions.min(), kl_contributions.max()
    
    kl_data = kl_contributions.T
    kl_display = np.ma.masked_where(kl_data == 0, kl_data)
    im2 = ax2.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                     interpolation='nearest', origin='lower',
                     vmin=vmin_k, vmax=vmax_k)
    _make_kl_hover(ax2, kl_data, categories)
    
    # Overlay H and Σ lines (thin and transparent to not block heatmap)
    ax2.plot(window_indices, sigma_scaled, color='white', linewidth=1.0, 
             alpha=0.4, label='Σ (white)')
    ax2.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, 
             alpha=0.4, label='H (cyan)')
    
    ax2.set_ylim(-0.5, n_categories - 0.5)
    ax2.set_xlabel('Window Index', fontsize=12)
    ax2.set_ylabel('Motif Category', fontsize=12)
    ax2.set_yticks(range(n_categories))
    ax2.set_yticklabels(categories, fontsize=9)
    ax2.set_title('Method 2: KL DIVERGENCE (Σ_KL)\n(Structural Surprise - Where motifs cluster)', 
                  fontsize=13, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('KL Contribution (bits/token)', fontsize=10)
    
    # Add legend for line overlays
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.8)
    
    # Determine title based on condition
    title_suffix = ""
    if condition_label == 'word_shuffle':
        title_suffix = " - WORD SHUFFLED"
    elif condition_label == 'sent_shuffle':
        title_suffix = " - SENTENCE SHUFFLED"
    
    fig.suptitle(f'Symbolic Category Entropy Analysis - Dual Method Visualization{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Determine filename based on condition
    if condition_label == 'word_shuffle':
        filename = f'{output_prefix}_word_shuffle_heatmap.png'
    elif condition_label == 'sent_shuffle':
        filename = f'{output_prefix}_sent_shuffle_heatmap.png'
    else:
        filename = f'{output_prefix}_se_heatmap.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()
    
    # Return z_limits for potential reuse
    if not z_limits:
        return {
            'density': (vmin_d, vmax_d),
            'kl': (vmin_k, vmax_k)
        }
    return z_limits


# ============================================================================
# VISUALIZATION: TIME SERIES
# ============================================================================

def plot_timeseries(results_df, output_prefix):
    """
    Plot H and Σ as time series.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Shannon Entropy (H)
    ax1.plot(results_df['window_index'], results_df['H'], 
             color='steelblue', linewidth=1.5, label='Shannon Entropy (H)')
    ax1.fill_between(results_df['window_index'], results_df['H'], 
                     alpha=0.3, color='steelblue')
    ax1.set_ylabel('H (bits/token)', fontsize=12)
    ax1.set_title('Shannon Entropy Over Text', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Sigma (Σ)
    ax2.plot(results_df['window_index'], results_df['Sigma'], 
             color='crimson', linewidth=1.5, label='Sigma (Σ)')
    ax2.fill_between(results_df['window_index'], results_df['Sigma'], 
                     alpha=0.3, color='crimson')
    ax2.set_xlabel('Window Index', fontsize=12)
    ax2.set_ylabel('Σ (bits/token)', fontsize=12)
    ax2.set_title('Sigma (Motif Concentration) Over Text', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    filename = f'{output_prefix}_se_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


# ============================================================================
# PEAK/VALLEY DETECTION WITH TEXT EXTRACTION
# ============================================================================

def extract_window_text(window_idx, tokens, window_size, step_size, context_words=50):
    """
    Extract text for a given window with context.
    Returns dict with window text, context, and positions.
    """
    start = window_idx * step_size
    end = start + window_size
    
    # Extract window tokens
    window_tokens = tokens[start:end]
    
    # Extract context (before and after)
    context_start = max(0, start - context_words)
    context_end = min(len(tokens), end + context_words)
    full_context_tokens = tokens[context_start:context_end]
    
    # Join tokens back to text (replace underscores with spaces for readability)
    window_text = ' '.join(window_tokens).replace('_', ' ')
    full_context = ' '.join(full_context_tokens).replace('_', ' ')
    
    return {
        'window_text': window_text,
        'full_context': full_context,
        'start_position': start,
        'end_position': end
    }


def analyze_window_motifs(window_tokens, motif_dict):
    """
    Analyze which motifs are present in a window.
    Returns dict of {category: count}, sorted by count descending.
    """
    counts = count_motifs_in_window(window_tokens, motif_dict)
    
    # Sort by count, descending
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    # Filter to only non-zero
    filtered = {k: v for k, v in sorted_counts.items() if v > 0}
    
    return filtered


def plot_peaks_and_valleys(results_df, tokens, window_size, step_size,
                           motif_dict, output_prefix, n_peaks=3):
    """
    Identify and visualize top peaks and valleys in Sigma with text excerpts.
    """
    sigma_values = results_df['Sigma'].values
    
    # Find peaks (local maxima)
    peak_indices, _ = find_peaks(sigma_values, height=np.percentile(sigma_values, 75))
    peak_values = sigma_values[peak_indices]
    
    # Sort peaks by height
    sorted_peak_idx = np.argsort(peak_values)[::-1]
    top_peaks = [(peak_indices[i], peak_values[i]) for i in sorted_peak_idx[:n_peaks]]
    
    # Find valleys (local minima) - invert signal
    valley_indices, _ = find_peaks(-sigma_values, height=-np.percentile(sigma_values, 25))
    valley_values = sigma_values[valley_indices]
    
    # Sort valleys by depth
    sorted_valley_idx = np.argsort(valley_values)
    top_valleys = [(valley_indices[i], valley_values[i]) for i in sorted_valley_idx[:n_peaks]]
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Main time series plot (top row, spans all columns)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(results_df['window_index'], sigma_values, 
                color='crimson', linewidth=2, label='Sigma (Σ)')
    ax_main.fill_between(results_df['window_index'], sigma_values, 
                        alpha=0.2, color='crimson')
    
    # Mark peaks
    for idx, val in top_peaks:
        ax_main.scatter(idx, val, color='darkred', s=100, zorder=5, 
                       marker='^', edgecolors='black', linewidths=1.5)
    
    # Mark valleys
    for idx, val in top_valleys:
        ax_main.scatter(idx, val, color='darkblue', s=100, zorder=5, 
                       marker='v', edgecolors='black', linewidths=1.5)
    
    ax_main.set_xlabel('Window Index', fontsize=12)
    ax_main.set_ylabel('Σ (bits/token)', fontsize=12)
    ax_main.set_title('Sigma Peaks (▲) and Valleys (▼) with Text Excerpts', 
                     fontsize=14, fontweight='bold')
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc='upper right')
    
    # Peak excerpts (second row)
    for i, (idx, val) in enumerate(top_peaks):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        
        # Extract text
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        excerpt = text_data['window_text'][:300] + '...'
        
        # Wrap text
        import textwrap
        wrapped_excerpt = '\n'.join(textwrap.wrap(excerpt, width=40))
        
        # Get dominant motifs
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        # Format motif info
        motif_text = 'Dominant motifs:\n'
        for cat, count in list(motifs.items())[:3]:
            motif_text += f'• {cat}: {count}\n'
        
        # Title
        title_text = f'Peak #{i+1}\nWindow {idx} | Σ = {val:.4f}'
        
        # Display text
        ax.text(0.5, 0.95, title_text, 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.6))
        
        ax.text(0.5, 0.75, motif_text, 
               transform=ax.transAxes, fontsize=8,
               ha='center', va='top', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.text(0.5, 0.50, wrapped_excerpt, 
               transform=ax.transAxes, fontsize=7,
               ha='center', va='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Valley excerpts (third row)
    for i, (idx, val) in enumerate(top_valleys):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        
        # Extract text
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        excerpt = text_data['window_text'][:300] + '...'
        
        # Wrap text
        import textwrap
        wrapped_excerpt = '\n'.join(textwrap.wrap(excerpt, width=40))
        
        # Get dominant motifs
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        # Format motif info
        motif_text = 'Dominant motifs:\n'
        for cat, count in list(motifs.items())[:3]:
            motif_text += f'• {cat}: {count}\n'
        if not motifs:
            motif_text += '(minimal motif presence)'
        
        # Title
        title_text = f'Valley #{i+1}\nWindow {idx} | Σ = {val:.4f}'
        bgcolor = 'lightblue'
        
        # Display text
        ax.text(0.5, 0.95, title_text, 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bgcolor, alpha=0.6))
        
        ax.text(0.5, 0.75, motif_text, 
               transform=ax.transAxes, fontsize=8,
               ha='center', va='top', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.text(0.5, 0.50, wrapped_excerpt, 
               transform=ax.transAxes, fontsize=7,
               ha='center', va='top', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                        alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    filename = f'{output_prefix}_peaks_valleys.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()
    
    # Export detailed CSV
    export_peaks_valleys_csv(top_peaks, top_valleys, tokens, window_size, step_size, 
                             motif_dict, output_prefix)


def export_peaks_valleys_csv(peaks, valleys, tokens, window_size, step_size,
                             motif_dict, output_prefix):
    """
    Export peak and valley text excerpts to CSV for detailed analysis.
    """
    data = []
    
    # Add peaks
    for rank, (idx, val) in enumerate(peaks, 1):
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        data.append({
            'type': 'Peak',
            'rank': rank,
            'window_idx': idx,
            'sigma_value': val,
            'excerpt': text_data['window_text'][:500],
            'top_motif_1': list(motifs.keys())[0] if len(motifs) > 0 else '',
            'top_motif_1_count': list(motifs.values())[0] if len(motifs) > 0 else 0,
            'top_motif_2': list(motifs.keys())[1] if len(motifs) > 1 else '',
            'top_motif_2_count': list(motifs.values())[1] if len(motifs) > 1 else 0,
            'top_motif_3': list(motifs.keys())[2] if len(motifs) > 2 else '',
            'top_motif_3_count': list(motifs.values())[2] if len(motifs) > 2 else 0,
        })
    
    # Add valleys
    for rank, (idx, val) in enumerate(valleys, 1):
        text_data = extract_window_text(idx, tokens, window_size, step_size, 
                                       context_words=100)
        start = text_data['start_position']
        end = text_data['end_position']
        window_tokens = tokens[start:end]
        motifs = analyze_window_motifs(window_tokens, motif_dict)
        
        data.append({
            'type': 'Valley',
            'rank': rank,
            'window_idx': idx,
            'sigma_value': val,
            'excerpt': text_data['window_text'][:500],
            'top_motif_1': list(motifs.keys())[0] if len(motifs) > 0 else '',
            'top_motif_1_count': list(motifs.values())[0] if len(motifs) > 0 else 0,
            'top_motif_2': list(motifs.keys())[1] if len(motifs) > 1 else '',
            'top_motif_2_count': list(motifs.values())[1] if len(motifs) > 1 else 0,
            'top_motif_3': list(motifs.keys())[2] if len(motifs) > 2 else '',
            'top_motif_3_count': list(motifs.values())[2] if len(motifs) > 2 else 0,
        })
    
    df = pd.DataFrame(data)
    filename = f'{output_prefix}_peaks_valleys_text.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")


# ============================================================================
# PUBLICATION STATISTICS
# ============================================================================

def print_publication_statistics(results_df, text_path, total_tokens, 
                                 window_size, n_windows, n_phrases, is_shuffle=False):
    """
    Print comprehensive statistics for publication.
    """
    label = "SHUFFLE" if is_shuffle else "ORIGINAL"
    print(f"\n{'='*70}")
    print(f"PUBLICATION-READY STATISTICS ({label})")
    print(f"{'='*70}")
    print(f"Text: {os.path.basename(text_path)}")
    print(f"Total semantic tokens: {total_tokens:,}")
    print(f"  (includes {n_phrases} merged multi-word phrases)")
    print(f"Window size: {window_size} tokens")
    print(f"Number of windows: {n_windows}")
    print(f"")
    print(f"Shannon Entropy (H):")
    print(f"  Mean: {results_df['H'].mean():.4f} ± {results_df['H'].std():.4f} bits/token")
    print(f"  Range: [{results_df['H'].min():.4f}, {results_df['H'].max():.4f}]")
    print(f"")
    print(f"Sigma (Σ):")
    print(f"  Mean: {results_df['Sigma'].mean():.6f} ± {results_df['Sigma'].std():.6f} bits/token")
    print(f"  Range: [{results_df['Sigma'].min():.6f}, {results_df['Sigma'].max():.6f}]")
    print(f"")
    print(f"Note: 'bits/token' refers to bits per SEMANTIC TOKEN")
    print(f"      Multi-word phrases are merged into single tokens")
    print(f"{'='*70}\n")


# ============================================================================
# THREE-WAY COMPARISON VISUALIZATION
# ============================================================================

def plot_3way_comparison(orig_kl, word_kl, sent_kl, motif_dict, output_prefix, z_limits,
                         orig_df=None, word_df=None, sent_df=None):
    """
    Create side-by-side KL heatmaps for all three conditions.
    All use the same z-axis from the original for valid comparison.
    Includes overlaid H (cyan) and Σ (white) line plots on each heatmap.
    """
    # Create figure with extra space at bottom for colorbar
    fig = plt.figure(figsize=(24, 10))
    
    # Create GridSpec: 3 columns for heatmaps, separate row for colorbar
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.25)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[1, :])
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    
    vmin_k, vmax_k = z_limits['kl']
    
    titles = ['ORIGINAL', 'SENTENCE SHUFFLED', 'WORD SHUFFLED']
    data_sets = [orig_kl, sent_kl, word_kl]
    results_dfs = [orig_df, sent_df, word_df]
    
    im = None  # Store last valid imshow for colorbar
    
    for ax, title, kl_data, rdf in zip(axes, titles, data_sets, results_dfs):
        if kl_data is not None:
            kl_unmasked = kl_data.T
            kl_display = np.ma.masked_where(kl_unmasked == 0, kl_unmasked)
            im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                          interpolation='nearest', origin='lower',
                          vmin=vmin_k, vmax=vmax_k)
            _make_kl_hover(ax, kl_unmasked, categories)
            ax.set_ylim(-0.5, n_categories - 0.5)
            ax.set_xlabel('Window Index', fontsize=11)
            ax.set_ylabel('Motif Category', fontsize=11)
            ax.set_yticks(range(n_categories))
            ax.set_yticklabels(categories, fontsize=7)  # Reduced from 9 to 7
            
            # Overlay H and Σ lines if results DataFrame is available
            if rdf is not None:
                window_indices = rdf['window_index'].values
                H_values = rdf['H'].values
                sigma_values = rdf['Sigma'].values
                
                # Robust percentile-based scaling to [0, n_categories-1] range
                H_scaled = scale_line_for_heatmap(H_values, n_categories)
                sigma_scaled = scale_line_for_heatmap(sigma_values, n_categories)
                
                ax.plot(window_indices, sigma_scaled, color='white', linewidth=1.0, 
                        alpha=0.4, label='Σ (white)')
                ax.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, 
                        alpha=0.4, label='H (cyan)')
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        else:
            ax.text(0.5, 0.5, 'Not Provided', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add horizontal colorbar in dedicated axes at bottom
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('KL Contribution (bits/token)', fontsize=12)
    
    fig.suptitle('Three-Way Comparison: Hierarchical Structure Destruction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    filename = f'{output_prefix}_3way_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def plot_3way_clean(orig_kl, word_kl, sent_kl, motif_dict, output_prefix, z_limits,
                    orig_df=None, word_df=None, sent_df=None):
    """
    Clean three-way comparison: shared Y-axis labels on left panel only,
    no legends, horizontal colorbar at bottom.
    """
    fig = plt.figure(figsize=(24, 10))
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.25,
                          wspace=0.05)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[1, :])
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    
    vmin_k, vmax_k = z_limits['kl']
    
    titles = ['ORIGINAL', 'SENTENCE SHUFFLED', 'WORD SHUFFLED']
    data_sets = [orig_kl, sent_kl, word_kl]
    results_dfs = [orig_df, sent_df, word_df]
    
    im = None
    
    for i, (ax, title, kl_data, rdf) in enumerate(zip(axes, titles, data_sets, results_dfs)):
        if kl_data is not None:
            kl_unmasked = kl_data.T
            kl_display = np.ma.masked_where(kl_unmasked == 0, kl_unmasked)
            im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                          interpolation='nearest', origin='lower',
                          vmin=vmin_k, vmax=vmax_k)
            _make_kl_hover(ax, kl_unmasked, categories)
            ax.set_ylim(-0.5, n_categories - 0.5)
            ax.set_xlabel('Window Index', fontsize=11)
            ax.set_yticks(range(n_categories))
            
            # Y-axis labels only on leftmost panel
            if i == 0:
                ax.set_ylabel('Motif Category', fontsize=11)
                ax.set_yticklabels(categories, fontsize=7)
            else:
                ax.set_yticklabels([])
            
            # Overlay H and Σ lines (no labels, no legend)
            if rdf is not None:
                window_indices = rdf['window_index'].values
                H_values = rdf['H'].values
                sigma_values = rdf['Sigma'].values
                
                H_scaled = scale_line_for_heatmap(H_values, n_categories)
                sigma_scaled = scale_line_for_heatmap(sigma_values, n_categories)
                
                ax.plot(window_indices, sigma_scaled, color='white', linewidth=1.0, alpha=0.4)
                ax.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, alpha=0.4)
        else:
            ax.text(0.5, 0.5, 'Not Provided', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Horizontal colorbar across bottom
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('KL Contribution (bits/token)', fontsize=12)
    
    fig.suptitle('Three-Way Comparison: Hierarchical Structure Destruction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    filename = f'{output_prefix}_3way_clean.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def plot_heatmap_H_only(results_df, kl_contributions, motif_dict, output_prefix, 
                        z_limits=None):
    """
    Single KL divergence heatmap for original text with only the cyan H line overlay.
    No Σ line, no legend.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    window_indices = results_df['window_index'].values
    
    # KL heatmap
    if z_limits and 'kl' in z_limits:
        vmin_k, vmax_k = z_limits['kl']
    else:
        vmin_k, vmax_k = kl_contributions.min(), kl_contributions.max()
    
    kl_data = kl_contributions.T
    kl_display = np.ma.masked_where(kl_data == 0, kl_data)
    im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                   interpolation='nearest', origin='lower',
                   vmin=vmin_k, vmax=vmax_k)
    
    ax.set_ylim(-0.5, n_categories - 0.5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Motif Category', fontsize=12)
    ax.set_yticks(range(n_categories))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_title('KL Divergence (Σ_KL) with Shannon Entropy (H) Overlay', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('KL Contribution (bits/token)', fontsize=10)
    
    # H line on secondary Y-axis with real bits/token scale
    ax_twin = ax.twinx()
    H_values = results_df['H'].values
    ax_twin.plot(window_indices, H_values, color='cyan', linewidth=1.0, alpha=0.4)
    ax_twin.set_ylabel('H (bits/token)', fontsize=12, color='gray')
    ax_twin.tick_params(axis='y', colors='gray')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_ylim(H_values.min(), H_values.max())
    ax_twin.set_yticks(np.linspace(H_values.min(), H_values.max(), 6))
    ax_twin.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Attach hover to twin (topmost axes intercepts mouse)
    _make_kl_hover_twin(ax_twin, kl_data, categories, n_categories, results_df=results_df)
    
    plt.tight_layout()
    filename = f'{output_prefix}_se_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def plot_3way_H_only(orig_kl, word_kl, sent_kl, motif_dict, output_prefix, z_limits,
                     orig_df=None, word_df=None, sent_df=None):
    """
    Clean three-way comparison with cyan H line on twinx() real-scale axis.
    Shared motif labels on left panel only, H axis label on right panel only.
    """
    fig = plt.figure(figsize=(24, 10))
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.25,
                          wspace=0.12)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[1, :])
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    
    vmin_k, vmax_k = z_limits['kl']
    
    titles = ['ORIGINAL', 'SENTENCE SHUFFLED', 'WORD SHUFFLED']
    data_sets = [orig_kl, sent_kl, word_kl]
    results_dfs = [orig_df, sent_df, word_df]
    
    # Compute global H range across all conditions for consistent scale
    all_H = []
    for rdf in results_dfs:
        if rdf is not None:
            all_H.extend(rdf['H'].values)
    H_global_min, H_global_max = min(all_H), max(all_H)
    
    im = None
    
    for i, (ax, title, kl_data, rdf) in enumerate(zip(axes, titles, data_sets, results_dfs)):
        if kl_data is not None:
            kl_unmasked = kl_data.T
            kl_display = np.ma.masked_where(kl_unmasked == 0, kl_unmasked)
            im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                          interpolation='nearest', origin='lower',
                          vmin=vmin_k, vmax=vmax_k)
            ax.set_ylim(-0.5, n_categories - 0.5)
            ax.set_xlabel('Window Index', fontsize=11)
            ax.set_yticks(range(n_categories))
            
            # Motif labels only on leftmost panel
            if i == 0:
                ax.set_ylabel('Motif Category', fontsize=11)
                ax.set_yticklabels(categories, fontsize=7)
            else:
                ax.set_yticklabels([])
            
            # H line on twinx with shared global scale
            if rdf is not None:
                window_indices = rdf['window_index'].values
                H_values = rdf['H'].values
                ax_twin = ax.twinx()
                ax_twin.plot(window_indices, H_values, color='cyan', linewidth=1.0, alpha=0.4)
                ax_twin.set_xlim(ax.get_xlim())
                ax_twin.set_ylim(H_global_min, H_global_max)
                ax_twin.set_yticks(np.linspace(H_global_min, H_global_max, 6))
                ax_twin.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                
                # H axis label only on rightmost panel, tick values on all
                ax_twin.tick_params(axis='y', colors='gray')
                if i == 2:
                    ax_twin.set_ylabel('H (bits/token)', fontsize=11, color='gray')
                
                # Attach hover to twin (topmost axes intercepts mouse)
                _make_kl_hover_twin(ax_twin, kl_unmasked, categories, n_categories, results_df=rdf)
        else:
            ax.text(0.5, 0.5, 'Not Provided', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Horizontal colorbar across bottom
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('KL Contribution (bits/token)', fontsize=12)
    
    fig.suptitle('Three-Way Comparison: Hierarchical Structure Destruction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    filename = f'{output_prefix}_3way_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def plot_heatmap_full(results_df, kl_contributions, motif_dict, output_prefix, 
                      z_limits=None):
    """
    Single KL divergence heatmap for original text with:
    - Cyan H line overlay (scaled to heatmap Y-axis)
    - White Σ line on twinx() right Y-axis with real bits/token scale
    No legends.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    window_indices = results_df['window_index'].values
    
    # KL heatmap
    if z_limits and 'kl' in z_limits:
        vmin_k, vmax_k = z_limits['kl']
    else:
        vmin_k, vmax_k = kl_contributions.min(), kl_contributions.max()
    
    kl_data = kl_contributions.T
    kl_display = np.ma.masked_where(kl_data == 0, kl_data)
    im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                   interpolation='nearest', origin='lower',
                   vmin=vmin_k, vmax=vmax_k)
    _make_kl_hover(ax, kl_data, categories)
    
    # H line (scaled to heatmap Y-axis)
    H_values = results_df['H'].values
    H_scaled = scale_line_for_heatmap(H_values, n_categories)
    ax.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, alpha=0.4)
    
    ax.set_ylim(-0.5, n_categories - 0.5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Motif Category', fontsize=12)
    ax.set_yticks(range(n_categories))
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_title('KL Divergence with H (cyan) and Σ (white) Overlays', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('KL Contribution (bits/token)', fontsize=10)
    
    # Σ line on secondary Y-axis with real scale
    ax_twin = ax.twinx()
    sigma_values = results_df['Sigma'].values
    ax_twin.plot(window_indices, sigma_values, color='white', linewidth=1.0, alpha=0.4)
    ax_twin.set_ylabel('Σ (bits/token)', fontsize=12, color='gray')
    ax_twin.tick_params(axis='y', colors='gray')
    ax_twin.set_xlim(ax.get_xlim())
    
    plt.tight_layout()
    filename = f'{output_prefix}_se_heatmap_full.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def plot_3way_full(orig_kl, word_kl, sent_kl, motif_dict, output_prefix, z_limits,
                   orig_df=None, word_df=None, sent_df=None):
    """
    Clean three-way comparison with:
    - Cyan H line overlay (scaled to heatmap Y-axis)
    - White Σ line on twinx() right Y-axis with real bits/token scale
    Shared motif labels on left panel only, Σ axis label on right panel only.
    No legends.
    """
    fig = plt.figure(figsize=(24, 10))
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.25,
                          wspace=0.12)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[1, :])
    
    categories = list(motif_dict.keys())
    n_categories = len(categories)
    
    vmin_k, vmax_k = z_limits['kl']
    
    titles = ['ORIGINAL', 'SENTENCE SHUFFLED', 'WORD SHUFFLED']
    data_sets = [orig_kl, sent_kl, word_kl]
    results_dfs = [orig_df, sent_df, word_df]
    
    im = None
    
    for i, (ax, title, kl_data, rdf) in enumerate(zip(axes, titles, data_sets, results_dfs)):
        if kl_data is not None:
            kl_unmasked = kl_data.T
            kl_display = np.ma.masked_where(kl_unmasked == 0, kl_unmasked)
            im = ax.imshow(kl_display, aspect='auto', cmap=_plasma_zero_black,
                          interpolation='nearest', origin='lower',
                          vmin=vmin_k, vmax=vmax_k)
            _make_kl_hover(ax, kl_unmasked, categories)
            ax.set_ylim(-0.5, n_categories - 0.5)
            ax.set_xlabel('Window Index', fontsize=11)
            ax.set_yticks(range(n_categories))
            
            # Motif labels only on leftmost panel
            if i == 0:
                ax.set_ylabel('Motif Category', fontsize=11)
                ax.set_yticklabels(categories, fontsize=7)
            else:
                ax.set_yticklabels([])
            
            if rdf is not None:
                window_indices = rdf['window_index'].values
                
                # H line (scaled to heatmap Y-axis)
                H_values = rdf['H'].values
                H_scaled = scale_line_for_heatmap(H_values, n_categories)
                ax.plot(window_indices, H_scaled, color='cyan', linewidth=1.0, alpha=0.4)
                
                # Σ line on secondary Y-axis with real scale
                ax_twin = ax.twinx()
                sigma_values = rdf['Sigma'].values
                ax_twin.plot(window_indices, sigma_values, color='white', linewidth=1.0, alpha=0.4)
                ax_twin.set_xlim(ax.get_xlim())
                
                # Σ axis label only on rightmost panel
                if i == 2:
                    ax_twin.set_ylabel('Σ (bits/token)', fontsize=11, color='gray')
                    ax_twin.tick_params(axis='y', colors='gray')
                else:
                    ax_twin.set_yticklabels([])
                    ax_twin.tick_params(axis='y', length=0)
        else:
            ax.text(0.5, 0.5, 'Not Provided', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Horizontal colorbar across bottom
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('KL Contribution (bits/token)', fontsize=12)
    
    fig.suptitle('Three-Way Comparison: Hierarchical Structure Destruction', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    filename = f'{output_prefix}_3way_full.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.show()
    plt.close()


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def export_validation_summary(orig_df, word_df, sent_df, output_prefix):
    """
    Export comprehensive validation statistics to CSV.
    Includes Cohen's d effect sizes.
    """
    rows = []
    
    # Original stats
    rows.append({
        'Condition': 'Original',
        'H_mean': orig_df['H'].mean(),
        'H_std': orig_df['H'].std(),
        'Sigma_mean': orig_df['Sigma'].mean(),
        'Sigma_std': orig_df['Sigma'].std(),
        'Collapse_Ratio': 1.0,
        'Cohens_d_vs_Original': 0.0
    })
    
    # Word shuffle stats
    if word_df is not None:
        collapse_word = orig_df['Sigma'].mean() / word_df['Sigma'].mean() if word_df['Sigma'].mean() > 0 else float('inf')
        cohens_d_word = calculate_cohens_d(orig_df['Sigma'].values, word_df['Sigma'].values)
        rows.append({
            'Condition': 'Word_Shuffle',
            'H_mean': word_df['H'].mean(),
            'H_std': word_df['H'].std(),
            'Sigma_mean': word_df['Sigma'].mean(),
            'Sigma_std': word_df['Sigma'].std(),
            'Collapse_Ratio': collapse_word,
            'Cohens_d_vs_Original': cohens_d_word
        })
    
    # Sentence shuffle stats
    if sent_df is not None:
        collapse_sent = orig_df['Sigma'].mean() / sent_df['Sigma'].mean() if sent_df['Sigma'].mean() > 0 else float('inf')
        cohens_d_sent = calculate_cohens_d(orig_df['Sigma'].values, sent_df['Sigma'].values)
        rows.append({
            'Condition': 'Sentence_Shuffle',
            'H_mean': sent_df['H'].mean(),
            'H_std': sent_df['H'].std(),
            'Sigma_mean': sent_df['Sigma'].mean(),
            'Sigma_std': sent_df['Sigma'].std(),
            'Collapse_Ratio': collapse_sent,
            'Cohens_d_vs_Original': cohens_d_sent
        })
    
    df = pd.DataFrame(rows)
    filename = f'{output_prefix}_validation_summary.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Handle command-line arguments
    if len(sys.argv) > 1:
        TEXT_FILE = sys.argv[1]
        print(f"Using command-line specified original file: {TEXT_FILE}")
    if len(sys.argv) > 2:
        WORD_SHUFFLE_FILE = sys.argv[2]
        print(f"Using command-line specified word shuffle file: {WORD_SHUFFLE_FILE}")
    if len(sys.argv) > 3:
        SENT_SHUFFLE_FILE = sys.argv[3]
        print(f"Using command-line specified sentence shuffle file: {SENT_SHUFFLE_FILE}")
    
    # Check original file exists
    if not os.path.exists(TEXT_FILE):
        print(f"ERROR: Original file not found: {TEXT_FILE}")
        print(f"Usage: python {sys.argv[0]} <original> [word_shuffle] [sentence_shuffle]")
        sys.exit(1)
    
    # Generate output prefix from original filename
    output_prefix = os.path.splitext(os.path.basename(TEXT_FILE))[0]
    
    # Storage for comparison
    word_results_df = None
    sent_results_df = None
    word_kl_contributions = None
    sent_kl_contributions = None
    
    # ========================================================================
    # PHASE 1: ANALYZE ORIGINAL TEXT
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: ANALYZING ORIGINAL TEXT")
    print("="*70)
    
    try:
        (results_df, raw_densities, kl_contributions, motif_dict_used, 
         baseline, window_size, total_tokens, tokens) = run_se_analysis(TEXT_FILE, motif_dict)
    except Exception as e:
        print(f"ERROR during original analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    n_windows = len(results_df)
    n_phrases = sum(1 for token in tokens if '_' in token)
    
    # Print statistics for original
    print_publication_statistics(results_df, TEXT_FILE, total_tokens, 
                                 window_size, n_windows, n_phrases, is_shuffle=False)
    
    # Generate visualizations for original
    print("Generating original text visualizations...")
    z_limits = {
        'density': (raw_densities.min(), raw_densities.max()),
        'kl': (kl_contributions.min(), kl_contributions.max())
    }
    
    # Generate SE heatmap with H overlay
    print("Generating SE heatmap...")
    plot_heatmap_H_only(results_df, kl_contributions, motif_dict_used, 
                        output_prefix, z_limits=z_limits)
    
    # Save original results
    csv_filename = f'{output_prefix}_se_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Original results saved: {csv_filename}")
    
    # Store original KL for comparison
    orig_kl_contributions = kl_contributions
    
    # ========================================================================
    # PHASE 2: ANALYZE WORD SHUFFLE (if provided)
    # ========================================================================
    if WORD_SHUFFLE_FILE and os.path.exists(WORD_SHUFFLE_FILE):
        print("\n" + "="*70)
        print("PHASE 2: ANALYZING WORD-SHUFFLED TEXT")
        print("="*70)
        
        try:
            (word_results_df, word_raw_densities, word_kl_contributions, 
             _, _, word_window_size, word_total_tokens, word_tokens) = run_se_analysis(WORD_SHUFFLE_FILE, motif_dict)
        except Exception as e:
            print(f"ERROR during word shuffle analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        else:
            word_n_windows = len(word_results_df)
            word_n_phrases = sum(1 for token in word_tokens if '_' in token)
            
            # Print statistics
            print_publication_statistics(word_results_df, WORD_SHUFFLE_FILE, word_total_tokens, 
                                         word_window_size, word_n_windows, word_n_phrases, 
                                         is_shuffle=True)
            
            # Save results
            word_csv = f'{output_prefix}_word_shuffle_se_results.csv'
            word_results_df.to_csv(word_csv, index=False)
            print(f"✓ Word shuffle results saved: {word_csv}")
    
    elif WORD_SHUFFLE_FILE:
        print(f"\n⚠ Warning: Word shuffle file specified but not found: {WORD_SHUFFLE_FILE}")
    
    # ========================================================================
    # PHASE 3: ANALYZE SENTENCE SHUFFLE (if provided)
    # ========================================================================
    if SENT_SHUFFLE_FILE and os.path.exists(SENT_SHUFFLE_FILE):
        print("\n" + "="*70)
        print("PHASE 3: ANALYZING SENTENCE-SHUFFLED TEXT")
        print("="*70)
        
        try:
            (sent_results_df, sent_raw_densities, sent_kl_contributions, 
             _, _, sent_window_size, sent_total_tokens, sent_tokens) = run_se_analysis(SENT_SHUFFLE_FILE, motif_dict)
        except Exception as e:
            print(f"ERROR during sentence shuffle analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        else:
            sent_n_windows = len(sent_results_df)
            sent_n_phrases = sum(1 for token in sent_tokens if '_' in token)
            
            # Print statistics
            print_publication_statistics(sent_results_df, SENT_SHUFFLE_FILE, sent_total_tokens, 
                                         sent_window_size, sent_n_windows, sent_n_phrases, 
                                         is_shuffle=True)
            
            # Save results
            sent_csv = f'{output_prefix}_sent_shuffle_se_results.csv'
            sent_results_df.to_csv(sent_csv, index=False)
            print(f"✓ Sentence shuffle results saved: {sent_csv}")
    
    elif SENT_SHUFFLE_FILE:
        print(f"\n⚠ Warning: Sentence shuffle file specified but not found: {SENT_SHUFFLE_FILE}")
    
    # ========================================================================
    # PHASE 4: THREE-WAY COMPARISON
    # ========================================================================
    if word_kl_contributions is not None or sent_kl_contributions is not None:
        print("\n" + "="*70)
        print("PHASE 4: THREE-WAY COMPARISON")
        print("="*70)
        
        # Three-way comparison with H overlay
        print("Generating three-way comparison visualization...")
        plot_3way_H_only(orig_kl_contributions, word_kl_contributions, 
                         sent_kl_contributions, motif_dict_used, output_prefix, z_limits,
                         orig_df=results_df, word_df=word_results_df, 
                         sent_df=sent_results_df)
        
        # Export validation summary with Cohen's d
        print("Exporting validation summary...")
        summary_df = export_validation_summary(results_df, word_results_df, 
                                               sent_results_df, output_prefix)
        
        # Print validation results
        print("\n" + "="*70)
        print("THREE-WAY VALIDATION RESULTS")
        print("="*70)
        
        original_sigma_mean = results_df['Sigma'].mean()
        print(f"Original Σ (mean): {original_sigma_mean:.6f} bits/token")
        print()
        
     
        # Hierarchical validation check
        if word_results_df is not None and sent_results_df is not None:
            word_sigma = word_results_df['Sigma'].mean()
            sent_sigma = sent_results_df['Sigma'].mean()
            if word_sigma < sent_sigma < original_sigma_mean:
                print("✓ HIERARCHICAL VALIDATION PASSED:")
                print(f"  Original ({original_sigma_mean:.6f}) > Sentence ({sent_sigma:.6f}) > Word ({word_sigma:.6f})")
                print("  This confirms SE detects multiple levels of semantic structure.")
            else:
                print("⚠ Hierarchical ordering not as expected.")
                print(f"  Original: {original_sigma_mean:.6f}, Sentence: {sent_sigma:.6f}, Word: {word_sigma:.6f}")
        
        print("="*70)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Original text outputs:")
    print(f"  - {output_prefix}_se_heatmap.png")
    print(f"  - {output_prefix}_se_results.csv")
    
    if word_results_df is not None:
        print(f"\nWord shuffle outputs:")
        print(f"  - {output_prefix}_word_shuffle_se_results.csv")
    
    if sent_results_df is not None:
        print(f"\nSentence shuffle outputs:")
        print(f"  - {output_prefix}_sent_shuffle_se_results.csv")
    
    if word_kl_contributions is not None or sent_kl_contributions is not None:
        print(f"\nComparison outputs:")
        print(f"  - {output_prefix}_3way_comparison.png")
        print(f"  - {output_prefix}_validation_summary.csv")
    
    print(f"\nFor publication, cite:")
    print(f"  Kurian, M. (2025). Symbolic Category Entropy: A Mathematical Framework")
    print(f"  for Quantifying Meaning Density in Text.")
    print(f"{'='*70}\n")
