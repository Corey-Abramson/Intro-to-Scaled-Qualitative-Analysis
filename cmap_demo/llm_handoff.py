"""Classification-lane helpers: the shared code vocabulary and the LLM handoff.

This module holds one definition of the code vocabulary so every lane agrees on
it, plus the pieces of the large-language-model lane that are not the model:

* ``CONCEPT_DICTIONARIES`` / ``ALLOWED_CODES`` -- the vocabulary itself;
* ``code_by_dictionary()`` -- the dictionary/regex lane, with real matches;
* ``build_llm_prompt()`` -- a ready-to-paste prompt naming the public data and
  embedding the exact rows to code;
* ``check_llm_response()`` -- the grounding check run on whatever comes back;
* ``code_by_stub()`` -- a deterministic stand-in so the notebook always
  produces a column, clearly labelled as not being a model result.

Nothing here calls a model, opens a network connection, or reads a key. The
model step happens wherever you already work, and you paste the result back.
"""

import ast
import io
import re

import pandas as pd

# The public copy of this corpus, for a model that can browse.
PUBLIC_CSV_URL = (
    "https://github.com/Computational-Ethnography-Lab/"
    "cmap_visualization_toolkit/blob/main/data/1_cleaned_data.csv"
)

# Keep prompt excerpts short. The corpus carries a quotation restriction --
# see SAMPLE_DATA.md.
EXCERPT_CHARS = 300

# One vocabulary, four concepts. Patterns are deliberately broad and are
# validated against real corpus text rather than assumed to hit.
CONCEPT_DICTIONARIES = {
    "education": re.compile(
        r"\b(school|college|univers\w*|degree|professor|student|studi\w*|"
        r"class|classes|classroom|teach\w*|educat\w*|graduat\w*|thesis|"
        r"undergraduate|doctorate)\b", re.IGNORECASE),
    "career": re.compile(
        r"\b(job|jobs|career|work|worked|working|employ\w*|hired|position|"
        r"company|laborator\w*|promot\w*|salary|colleague|retire\w*|"
        r"profession\w*)\b", re.IGNORECASE),
    "military": re.compile(
        r"\b(army|navy|air force|marines|military|war|wartime|enlist\w*|"
        r"soldier|veteran|combat|corps|battalion|draft|drafted)\b",
        re.IGNORECASE),
    "family": re.compile(
        r"\b(father|mother|dad|mom|parent|parents|brother|sister|wife|"
        r"husband|son|daughter|child|children|famil\w*|grandfather|"
        r"grandmother|marri\w*|uncle|aunt|cousin)\b", re.IGNORECASE),
}

ALLOWED_CODES = sorted(CONCEPT_DICTIONARIES)


def code_by_dictionary(text):
    """Return the codes whose dictionary matches ``text``."""
    if not isinstance(text, str):
        return []
    return [code for code, pattern in CONCEPT_DICTIONARIES.items()
            if pattern.search(text)]


def dictionary_matches(text, limit=3):
    """Return ``(code, matched_word)`` pairs, so hits can be shown not claimed."""
    hits = []
    if not isinstance(text, str):
        return hits
    for code, pattern in CONCEPT_DICTIONARIES.items():
        found = pattern.findall(text)
        for word in found[:limit]:
            hits.append((code, word if isinstance(word, str) else word[0]))
    return hits


def build_llm_prompt(df, n_rows=5, csv_url=PUBLIC_CSV_URL):
    """Build a ready-to-paste prompt for coding ``n_rows`` rows.

    The prompt names the public dataset, so a model that browses can go and
    read it, and embeds the exact rows with their ``doc_id``s, so a model that
    cannot browse still has the text in front of it. The embedded ``doc_id``s
    are what ``check_llm_response()`` later checks the answer against.
    """
    sample = df.head(n_rows)
    allowed = ", ".join(ALLOWED_CODES)

    lines = [
        "You are helping code qualitative interview data.",
        "",
        f"The full dataset is public: {csv_url}",
        "Below are the exact rows to code, excerpted from that dataset.",
        "",
        "TASK",
        f"Assign codes to each of the {len(sample)} rows below.",
        "",
        "RULES",
        f"- Allowed codes, and no others: {allowed}",
        "- A row may take several codes, or none.",
        f"- Return exactly {len(sample)} rows, one per input row.",
        "- Return CSV only. No prose, no explanation, no code fences.",
        "- Use exactly these two columns, with this header:",
        "      doc_id,codes",
        "- Put multiple codes in one quoted, semicolon-separated field,",
        '      for example: 1234,"career;family"',
        "- Reuse each doc_id exactly as given below. Do not renumber them.",
        "",
        "ROWS",
    ]

    for _, row in sample.iterrows():
        text = str(row["text"]).replace("\n", " ").strip()
        if len(text) > EXCERPT_CHARS:
            text = text[:EXCERPT_CHARS].rstrip() + " [...]"
        lines.append(f'doc_id={row["doc_id"]}: {text}')
        lines.append("")

    lines.append("Return the CSV now.")
    return "\n".join(lines)


def check_llm_response(response_csv, source_df):
    """Check a returned CSV against the rows that were actually sent.

    A model with no access to the data can return five plausible, perfectly
    formed rows. Parsing proves syntax, not grounding. This compares the
    returned ``doc_id``s and row count against the local source and refuses
    anything that disagrees.

    Returns a frame with ``doc_id`` and a parsed ``codes`` list. Raises
    ``ValueError`` on any mismatch.
    """
    try:
        returned = pd.read_csv(io.StringIO(response_csv.strip()))
    except Exception as exc:
        raise ValueError(f"Response is not parseable CSV: {exc}") from exc

    missing = {"doc_id", "codes"} - set(returned.columns)
    if missing:
        raise ValueError(
            f"Response is missing column(s) {sorted(missing)}; "
            f"got {list(returned.columns)}"
        )

    expected_ids = [int(x) for x in source_df["doc_id"]]
    got_ids = [int(x) for x in returned["doc_id"]]

    if len(got_ids) != len(expected_ids):
        raise ValueError(
            f"Response has {len(got_ids)} rows; {len(expected_ids)} were sent."
        )
    if sorted(got_ids) != sorted(expected_ids):
        raise ValueError(
            "Returned doc_ids do not match the rows that were sent.\n"
            f"  sent    : {sorted(expected_ids)}\n"
            f"  returned: {sorted(got_ids)}\n"
            "A self-consistent answer that disagrees with the source means "
            "the answer is wrong, not the source."
        )

    parsed = []
    unknown = set()
    for value in returned["codes"]:
        if pd.isna(value):
            codes = []
        else:
            codes = [c.strip().lower() for c in str(value).split(";") if c.strip()]
        unknown.update(set(codes) - set(ALLOWED_CODES))
        parsed.append(codes)

    if unknown:
        raise ValueError(
            f"Response used codes outside the allowed list: {sorted(unknown)}"
        )

    out = returned[["doc_id"]].copy()
    out["codes"] = parsed
    print(
        f"[OK] Grounding check passed: {len(out)} rows, doc_ids match source, "
        f"all codes within the allowed list."
    )
    return out


def code_by_stub(df, verbose=True):
    """Deterministic stand-in for the model, so the notebook always finishes.

    This is NOT a model result. It applies the same dictionaries as lane 1, so
    it is reproducible and inspectable, and it is labelled wherever it prints.
    """
    if verbose:
        print("SIMULATED -- deterministic stand-in, not a model result.")
        print("The real lane is the prompt above, run in your own assistant.")
    return [code_by_dictionary(text) for text in df["text"]]


def merge_code_columns(*code_lists):
    """Merge several per-row code lists into one sorted, de-duplicated list."""
    merged = []
    for row_codes in zip(*code_lists):
        combined = set()
        for codes in row_codes:
            combined.update(codes)
        merged.append(sorted(combined))
    return merged


def codes_to_source_shape(code_lists):
    """Render code lists in the corpus's stringified-list shape.

    The recycled toolkit visualizations parse this shape, so writing it is what
    lets them run over the notebook's own codes unmodified.
    """
    return [str(list(codes)) for codes in code_lists]


def parse_codes_column(series):
    """Read the stringified-list ``codes`` column back into lists."""
    return [ast.literal_eval(str(value)) if str(value).strip() else []
            for value in series]


__all__ = [
    "PUBLIC_CSV_URL", "CONCEPT_DICTIONARIES", "ALLOWED_CODES",
    "code_by_dictionary", "dictionary_matches", "build_llm_prompt",
    "check_llm_response", "code_by_stub", "merge_code_columns",
    "codes_to_source_shape", "parse_codes_column",
]
