"""Figures for the Intro to Scaled Qualitative Analysis notebook.

Two plots: a word cloud over the text column, and a co-occurrence heatmap over
the codes the notebook itself produces.

PROVENANCE. Four functions in this module are recycled from the CMAP
Visualization Toolkit, which is BSD 3-Clause licensed:

    https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit
    Copyright (c) 2025, Computational Ethnography Lab. All rights reserved.
    Source: visualization_toolkit_final.ipynb, code cell 6, release tag v0.9.6

They are copied verbatim apart from four deliberate adaptations, each noted
inline at the line it touches:

    A1  The module-level OUTPUT_DIR global becomes an explicit ``out_dir``
        argument defaulting to ``Path("output")``, so nothing depends on a
        notebook global.
    A3  ``os.makedirs(out_dir, exist_ok=True)`` runs before the first write.
        Git does not track an empty ignored directory, so a clean checkout has
        no ``output/`` for the first save to land in.
    A4  The pydantic ``HeatmapInput`` model becomes plain keyword arguments,
        removing a dependency whose v1/v2 split is a live hazard. The model was
        never constructed inside the function; the body only unpacked it. But
        it did carry two checks of its own, an existence check on ``filepath``
        and a ``num_codes > 0`` validator, so both are restated explicitly at
        the top of the function. Without that, dropping the model would have
        quietly dropped the validation with it.
    A5  ``stopwords.words("english")`` and ``word_tokenize`` become the
        ``_stopwords()`` and ``_tokenize()`` helpers below, so the notebook
        runs whether or not NLTK's corpora were downloaded.

Everything else in the recycled spans is byte-identical to the source. The
helpers and stop-word lists below the imports are written for this repository.
"""

import ast
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from scipy.spatial.distance import pdist
from wordcloud import WordCloud

try:  # NLTK is optional -- see A5 and _stopwords()/_tokenize() below.
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
except ImportError:  # pragma: no cover - exercised only where nltk is absent
    _nltk_stopwords = None
    _nltk_word_tokenize = None


# --- Stop words and tokenizing (authored here) -------------------------------
#
# The recycled word cloud looks up stop words BEFORE it tokenizes, so a
# tokenizer-only fallback would still die on a missing corpus. Both halves are
# covered: NLTK is used when its data is present, and the bundled list plus a
# regex tokenizer are used when it is not. Nothing here downloads anything.

# A standard English function-word list, sufficient for a demo word cloud.
_BUNDLED_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Corpus-specific stop words, seeded from the CMAP Visualization Toolkit's
# tracked input/additional_stops.txt (BSD 3-Clause). These are oral-history
# filler words that swamp a word cloud if left in.
_CORPUS_STOPWORDS = frozenset(
    "also could get hah like lot may one well would".split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _stopwords():
    """Return the English stop-word set, from NLTK if available else bundled."""
    base = None
    if _nltk_stopwords is not None:
        try:
            base = set(_nltk_stopwords.words("english"))
        except LookupError:
            base = None  # corpus not downloaded -- fall through to the bundle
    if base is None:
        base = set(_BUNDLED_STOPWORDS)
    return base | set(_CORPUS_STOPWORDS)


def _tokenize(text):
    """Tokenize with NLTK if its punkt data is available, else with a regex."""
    if _nltk_word_tokenize is not None:
        try:
            return _nltk_word_tokenize(text)
        except LookupError:
            pass  # punkt not downloaded -- fall through to the regex
    return _WORD_RE.findall(text)

# -----------------------------------------------------------------------------
# RECYCLED VERBATIM -- CMAP Visualization Toolkit, BSD 3-Clause
#   Copyright (c) 2025, Computational Ethnography Lab. All rights reserved.
#   Source: visualization_toolkit_final.ipynb, code cell 6, lines 20-23
#   Release tag: v0.9.6
#   Adaptations: none.
# -----------------------------------------------------------------------------

def make_circular_mask(diam: int = 1600, border: int = 5) -> np.ndarray:
    img = Image.new("L", (diam, diam), 0)
    ImageDraw.Draw(img).ellipse([(border, border), (diam - border, diam - border)], fill=255)
    return 255 - np.array(img)  # WordCloud expects black = non-fillable

# -----------------------------------------------------------------------------
# RECYCLED -- CMAP Visualization Toolkit, BSD 3-Clause
#   Copyright (c) 2025, Computational Ethnography Lab. All rights reserved.
#   Source: visualization_toolkit_final.ipynb, code cell 6, lines 26-108
#   Release tag: v0.9.6
#   Adaptations: A1 (out_dir argument), A5 (optional NLTK). Three lines changed;
#   the other 80 are byte-identical to the source.
# -----------------------------------------------------------------------------

def generate_wordcloud(
    text_series,
    stopwords_path=None,
    title="Wordcloud",
    out_dir=Path("output"),
    categories=None
):
    print("\n✔ [OK] Building word-cloud…")

    # Stopwords
    stop_words = set(_stopwords())
    if stopwords_path and os.path.exists(stopwords_path):
        with open(stopwords_path, 'r') as f:
            stop_words.update(f.read().splitlines())

    # Tokenize and filter
    combined_text = ' '.join(text_series.dropna().astype(str))
    tokens = _tokenize(combined_text.lower())
    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 2]
    paragraph_count = len(text_series)

    # Mask
    mask = make_circular_mask()
    print(f"✔ [OK] Mask ready {mask.shape}")

    # Frequencies
    word_freq = Counter(filtered_tokens)
    print(f"✔ [OK] {len(word_freq):,} unique tokens")

    # Categories
    if categories is None:
        categories = {}
    
    word2cat = {w: cat for cat, info in categories.items() for w in info["words"]}

    # Color function
    def colour_for_word(word, **_):
        cat = word2cat.get(word.lower())
        if cat:
            r, g, b, _ = categories[cat]["color"]
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        return "#bcbcbc"

    # Generate WordCloud
    wc = (WordCloud(
        width=1600, height=1600, mask=mask, background_color="white",
        max_words=600, min_font_size=5, max_font_size=160, font_step=1,
        margin=1, prefer_horizontal=0.3, random_state=42,
        collocations=False, repeat=True, mode="RGBA"
    )
        .generate_from_frequencies(word_freq)
        .recolor(color_func=colour_for_word, random_state=42))

    print("✔ [OK] WordCloud generated")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    fig.patch.set_facecolor("white")
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")

    fig.suptitle(title, fontsize=36, fontweight="bold", y=1.05, color="black")
    fig.text(0.5, 0.975, f"Analysis of {paragraph_count:,} Paragraphs of Text",
             ha="center", va="top", fontsize=18, style="italic", color="#333333")

    handles = [Patch(color=info["color"], label=cat) for cat, info in categories.items()]
    legend = ax.legend(handles=handles,
                       loc="lower center", bbox_to_anchor=(0.5, -0.085),
                       ncol=3, frameon=False, fontsize=14)
    for txt in legend.get_texts():
        txt.set_color("#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    os.makedirs(out_dir, exist_ok=True)
    base = "wordcloud_latest"

    fig.savefig(os.path.join(out_dir, f"{base}.png"),
                dpi=300, bbox_inches="tight", format="png")
    print(f"✔ [OK] Saved {base}.png")

    plt.show()

# -----------------------------------------------------------------------------
# RECYCLED VERBATIM -- CMAP Visualization Toolkit, BSD 3-Clause
#   Copyright (c) 2025, Computational Ethnography Lab. All rights reserved.
#   Source: visualization_toolkit_final.ipynb, code cell 6, lines 1541-1580
#           (span starts at the @lru_cache decorator, not the def)
#   Release tag: v0.9.6
#   Adaptations: none.
# -----------------------------------------------------------------------------

@lru_cache(maxsize=10000)
def parse_string_list(value: Union[str, list, None]) -> List[str]:
    """
    Performance-optimized string-formatted list parser for code lists.
    
    Args:
        value: String, list or None containing codes
        
    Returns:
        List[str]: Cleaned and parsed list of codes
    """
    if pd.isna(value) or value == "" or value is None:
        return []
    
    if isinstance(value, list):
        return [str(item).lower().strip() for item in value if item]
        
    if isinstance(value, str):
        value = value.strip()
        
        if value in ["[]", "['']", '[""]', "nan", "NaN"]:
            return []
            
        try:
            if value.startswith("[") and value.endswith("]"):
                parsed = ast.literal_eval(value) 
                if isinstance(parsed, list):
                    return [str(item).lower().strip() for item in parsed if item and str(item).strip()]
        except (ValueError, SyntaxError):
            pass
            
        try:
            cleaned = value.strip("[]").replace("'", "").replace('"', "")
            if cleaned:
                items = [item.strip().lower() for item in cleaned.split(",")]
                return [item for item in items if item]
        except Exception:
            pass
    
    return []

# -----------------------------------------------------------------------------
# RECYCLED -- CMAP Visualization Toolkit, BSD 3-Clause
#   Copyright (c) 2025, Computational Ethnography Lab. All rights reserved.
#   Source: visualization_toolkit_final.ipynb, code cell 6, lines 1583-1695
#   Release tag: v0.9.6
#   Adaptations: A4 (pydantic HeatmapInput -> keyword arguments; the signature
#   and the five unpack lines it replaced), A3 (makedirs before the first
#   write), A1 (out_dir argument, two save paths). 113 source lines become 108;
#   the remaining body is byte-identical to the source.
# -----------------------------------------------------------------------------

def create_code_cooccurrence_heatmap(
    filepath,
    *,
    num_codes=10,
    seed_codes=None,
    projects=None,
    data_groups=None,
    clustered=True,
    out_dir=Path("output"),
):
    # A4, continued. The pydantic model this replaced did two things besides
    # carry types: `filepath: FilePath` refused a path that does not exist, and
    # a field validator refused num_codes <= 0. Dropping the model would have
    # dropped both, turning an invalid num_codes into a confusing "No codes
    # found" message instead of an error. Restated here so the replacement is
    # faithful rather than merely equivalent in the happy path.
    if int(num_codes) <= 0:
        raise ValueError("num_codes must be greater than 0")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(filepath)

    # Filter by projects 
    if projects:
        df = df[df['project'].isin(projects)]

    # Filter by data groups
    if data_groups:
        df = df[df['data_group'].apply(lambda x: any(g in parse_string_list(x) for g in data_groups))]

    # Vectorized code parsing
    all_codes = []
    for codes in df['codes'].dropna():
        all_codes.extend(parse_string_list(codes))

    # Get top N most frequent codes
    if seed_codes:
        seed_set = set(code.lower().strip() for code in seed_codes)
        cooccurrence_counter = Counter()

        # Count co-occurring codes with seeds
        for codes in df['codes'].dropna():
            code_list = set(parse_string_list(codes))
            if seed_set & code_list:  # If any seed is in the list
                overlapping = code_list - seed_set
                cooccurrence_counter.update(overlapping)

        # Add top-N co-occurring codes
        top_overlap = [code for code, _ in cooccurrence_counter.most_common(num_codes)]
        selected_codes = list(seed_set) + top_overlap
        print(f"Selected codes: {selected_codes}")
    else:
        # Use top-N most frequent codes in corpus
        selected_codes = [code for code, _ in Counter(all_codes).most_common(num_codes)]
        print(f"Top {num_codes} most frequent codes: {selected_codes}")

    # Check if we have any codes to analyze
    if not selected_codes:
        print("No codes found to analyze. Please check your input data.")
        return

    # Co-occurrence matrix using vectorized operations
    cooc_matrix = np.zeros((len(selected_codes), len(selected_codes)))
    
    for codes in df['codes'].dropna():
        codes_set = set(parse_string_list(codes)).intersection(selected_codes)
        for i, code1 in enumerate(selected_codes):
            for j in range(i + 1, len(selected_codes)):
                code2 = selected_codes[j]
                if code1 in codes_set and code2 in codes_set:
                    cooc_matrix[i][j] += 1
                    cooc_matrix[j][i] += 1

    # Check if matrix is empty
    if np.all(cooc_matrix == 0):
        print("No co-occurrences found. Please check your input data.")
        return

    heatmap_df = pd.DataFrame(cooc_matrix, index=selected_codes, columns=selected_codes)
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 10))

    
    if clustered:
        # Convert co-occurrence matrix to proper distance format for linkage
        row_linkage = hierarchy.linkage(pdist(heatmap_df), method='ward')
        col_linkage = hierarchy.linkage(pdist(heatmap_df.T), method='ward')
        
        g = sns.clustermap(heatmap_df,
                        annot=True,
                        fmt='g',
                        cmap='inferno',
                        row_linkage=row_linkage,
                        col_linkage=col_linkage,
                        figsize=(12, 10),
                        dendrogram_ratio=0.2,
                        colors_ratio=0.03)
        g.fig.patch.set_facecolor('black')
        g.ax_heatmap.set_facecolor('black')

        for item in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
            item.set_facecolor('black')
            for c in item.collections:
                c.set_color('white')

        filename = f"code_heatmap_clustered_{num_codes}.png"
        out_path = os.path.join(out_dir, filename)
        g.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✔ [OK] Saved clustered heatmap: {out_path}")
    else:
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='g',
            cmap='inferno'
        )
        ax.set_title('Code Co-occurrence Matrix')

        filename = f"code_heatmap_plain_{num_codes}.png"
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✔ [OK] Saved plain heatmap: {out_path}")

    plt.tight_layout()
    plt.show()
