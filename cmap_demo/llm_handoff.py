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

    The prompt names the public dataset, so a model that browses can read the
    full text, and embeds the selected rows with their ``doc_id``s so a model
    that cannot browse still has something to work from. The embedded
    ``doc_id``s are what ``check_llm_response()`` checks the answer against.

    Row text is **truncated to ``EXCERPT_CHARS`` characters**, and the prompt
    says so. The corpus carries a restriction on quotation, so long excerpts
    are not shipped. The practical cost is real: if the only evidence for a
    code sits past the cutoff, a non-browsing model cannot see it. Raise
    ``EXCERPT_CHARS`` for your own unrestricted data.

    Returns the prompt. Pass the SAME frame you passed here to
    ``check_llm_response()``, i.e. ``df.head(n_rows)``, not the full frame.
    """
    sample = df.head(n_rows)
    allowed = ", ".join(ALLOWED_CODES)

    lines = [
        "You are helping code qualitative interview data.",
        "",
        f"The full dataset is public: {csv_url}",
        f"Below are the rows to code. Each one is truncated to about",
        f"{EXCERPT_CHARS} characters; a trailing [...] marks where text was cut.",
        "If you can open the URL, read the full text for these doc_ids there.",
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


def _exact_int(value, where):
    """Convert to int only if the value is exactly integral."""
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {value!r} is not a number.") from exc
    if as_float != int(as_float):
        raise ValueError(
            f"{where}: {value!r} is not a whole number. doc_ids are integers; "
            "a fractional value means the answer was altered in transit."
        )
    return int(as_float)


def check_llm_response(response_csv, source_df):
    """Check a returned CSV against the rows that were actually sent.

    This is an identity and schema check, not a check that the codes are
    correct. It proves the answer is about the rows you sent: the row count
    matches, every ``doc_id`` is one you supplied and is exactly integral, and
    every code is in the allowed list. That is what catches a model inventing
    plausible rows for text it never read.

    It cannot tell you the codes are *right*. An answer that returns your
    ``doc_id``s and labels every row ``family`` passes this check. Judging
    whether a code fits the text is still your job.

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

    expected_ids = [_exact_int(x, "source doc_id") for x in source_df["doc_id"]]
    got_ids = [_exact_int(x, "returned doc_id") for x in returned["doc_id"]]
    returned["doc_id"] = got_ids

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
        f"[OK] Identity check passed: {len(out)} rows, doc_ids match the rows "
        f"that were sent, all codes within the allowed list."
    )
    print("     This proves the answer is about your rows. It does not prove "
          "the codes are correct.")
    return out


def code_by_stub(df, verbose=True):
    """Deterministic stand-in for the model, so the notebook always finishes.

    This is NOT a model result, and it is not an independent third opinion
    either: it reuses the same dictionaries as lane 1, so it agrees with lane 1
    by construction. Its only job is to guarantee the notebook produces a
    coded column when nobody has pasted a real answer back.
    """
    if verbose:
        print("SIMULATED -- deterministic stand-in, not a model result.")
        print("It reuses the lane 1 dictionaries, so it agrees with lane 1 by")
        print("construction. The real lane is the prompt above, run in your")
        print("own assistant and pasted back.")
    return [code_by_dictionary(text) for text in df["text"]]


def apply_llm_codes(base_codes, df, checked):
    """Overlay checked model codes onto the base codes, matched by ``doc_id``.

    ``base_codes`` is one list of codes per row of ``df``. Rows whose
    ``doc_id`` appears in ``checked`` take the model's codes; every other row
    keeps what it had. Returns the merged list and the number of rows replaced.
    """
    replacement = {int(row.doc_id): list(row.codes)
                   for row in checked.itertuples()}
    merged, replaced = [], 0
    for codes, doc_id in zip(base_codes, df["doc_id"]):
        key = int(doc_id)
        if key in replacement:
            merged.append(replacement[key]); replaced += 1
        else:
            merged.append(codes)
    return merged, replaced


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
    "PUBLIC_CSV_URL", "EXCERPT_CHARS", "CONCEPT_DICTIONARIES", "ALLOWED_CODES",
    "code_by_dictionary", "dictionary_matches", "build_llm_prompt",
    "check_llm_response", "code_by_stub", "apply_llm_codes",
    "merge_code_columns", "codes_to_source_shape", "parse_codes_column",
]
