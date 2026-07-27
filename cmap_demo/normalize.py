"""Turn a raw transcript into machine-readable CMAP rows, and validate them.

This module is written for this repository. It is a clean implementation of a
small, generic version of the normalization stage: filename to metadata, strip
timestamps, split on speaker turns, one row per turn, then validate the result.

The transformation is reconstructible. Every row records the character offsets
of its text within the normalized document, so the normalized document can be
rebuilt from the rows alone -- see ``reconstruct_text()``, which the notebook
calls to prove it rather than assert it.

The CMAP schema used here is the same twelve columns as the corpus in
``data/1_cleaned_data.csv``, so rows produced by this module and rows read from
that file are interchangeable.
"""

import ast
import re
from datetime import datetime

import pandas as pd

# The twelve CMAP columns, in corpus order.
CMAP_COLUMNS = [
    "project", "number", "reference", "text", "document",
    "start_position", "end_position", "data_group", "text_length",
    "word_count", "doc_id", "codes",
]

# Filename convention: datasource_subject_date.txt, date as YYYYMMDD.
_FILENAME_RE = re.compile(
    r"^(?P<data_source>[A-Za-z]+)_(?P<subject>[A-Za-z0-9]+)_(?P<date>\d{8})$"
)

# Timestamps in square or round brackets, [HH:MM:SS] / (MM:SS) and so on.
_TIMESTAMP_RE = re.compile(r"[\[(]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\])]")

# A speaker label at the start of a line: "Speaker 1:" or "NAME:". Deliberately
# generic -- no project-specific identifier shapes.
_SPEAKER_RE = re.compile(
    r"^\s*(?P<speaker>Speaker\s+\d+|[A-Z][A-Za-z.'\- ]{0,40}?)\s*:\s*(?P<rest>.*)$"
)


def parse_filename(filename):
    """Map a transcript filename to metadata. Fails loud on a bad name.

    ``interview_demo01_20240115.txt`` becomes
    ``{"data_source": "interview", "subject": "demo01", "date": date(2024, 1, 15)}``.

    The date is parsed strictly, so an impossible date such as ``20240132``
    raises rather than passing through as a string.
    """
    stem = str(filename).rsplit("/", 1)[-1]
    if stem.lower().endswith(".txt"):
        stem = stem[: -len(".txt")]

    match = _FILENAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Filename {stem!r} does not follow the convention "
            "datasource_subject_date.txt with an 8-digit YYYYMMDD date "
            "(for example: interview_demo01_20240115.txt)."
        )

    raw_date = match.group("date")
    try:
        parsed = datetime.strptime(raw_date, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Filename {stem!r} carries {raw_date!r}, which is not a real date."
        ) from exc

    return {
        "data_source": match.group("data_source").lower(),
        "subject": match.group("subject").lower(),
        "date": parsed,
    }


def remove_timestamps(text):
    """Strip bracketed timestamps such as ``[00:04:12]`` or ``(4:12)``."""
    return _TIMESTAMP_RE.sub("", text)


def detect_speaker(line):
    """Split a leading speaker label off a line.

    Returns ``(speaker, remainder)``, or ``(None, line)`` when the line does
    not open with a label.
    """
    match = _SPEAKER_RE.match(line)
    if not match:
        return None, line
    return match.group("speaker").strip(), match.group("rest").strip()


def collapse_whitespace(text):
    """Collapse runs of spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text):
    """Remove timestamps and collapse whitespace, leaving speaker labels."""
    return collapse_whitespace(remove_timestamps(text))


def segment_turns(raw_text):
    """Split a raw transcript into one record per speaker turn.

    A turn starts at a line carrying a speaker label and runs until the next
    such line. Text before the first label is dropped as front matter.
    """
    turns = []
    current = None

    for line in remove_timestamps(raw_text).splitlines():
        speaker, rest = detect_speaker(line)
        if speaker is not None:
            if current is not None:
                turns.append(current)
            current = {"speaker": speaker, "parts": [rest] if rest else []}
        elif current is not None:
            current["parts"].append(line)

    if current is not None:
        turns.append(current)

    records = []
    for turn in turns:
        text = collapse_whitespace(" ".join(turn["parts"]))
        if text:
            records.append({"speaker": turn["speaker"], "text": text})
    return records


def build_cmap_frame(turns, metadata, project=None):
    """Build a CMAP-schema frame from segmented turns.

    ``start_position`` and ``end_position`` are character offsets into the
    normalized document -- the single space -joined concatenation of every
    turn's text, in order. That is what makes the rows reconstructible.
    """
    data_source = metadata["data_source"]
    document = f"{data_source}_{metadata['subject']}"
    project = project or f"{data_source}_demo"

    rows = []
    cursor = 0
    for index, turn in enumerate(turns):
        text = turn["text"]
        start = cursor
        end = start + len(text)
        cursor = end + 1  # the single space that joins this turn to the next
        rows.append({
            "project": project,
            "number": turn["speaker"],   # segment label; here, the speaker turn
            "reference": index,
            "text": text,
            "document": document,
            "start_position": start,
            "end_position": end,
            "data_group": str([data_source]),
            "text_length": len(text),
            "word_count": len(text.split()),
            "doc_id": index,
            "codes": "[]",
        })

    return pd.DataFrame(rows).reindex(columns=CMAP_COLUMNS)


def reconstruct_text(df):
    """Rebuild the normalized document from the rows, using the offsets.

    Returns the reconstructed string. Raises ``ValueError`` if the offsets are
    not contiguous and ordered, which would mean the rows have lost the
    information needed to recover the original.
    """
    ordered = df.sort_values("start_position")
    buffer = ""
    for _, row in ordered.iterrows():
        start, end = int(row["start_position"]), int(row["end_position"])
        text = str(row["text"])
        if end - start != len(text):
            raise ValueError(
                f"Row doc_id={row['doc_id']} offsets span {end - start} "
                f"characters but its text is {len(text)} characters long."
            )
        if start < len(buffer):
            raise ValueError(
                f"Row doc_id={row['doc_id']} starts at {start}, inside text "
                f"already reconstructed to {len(buffer)}."
            )
        buffer += " " * (start - len(buffer)) + text
    return buffer


def _as_str_list(value):
    """Parse a stringified list into ``list[str]``, or raise."""
    if isinstance(value, list):
        parsed = value
    else:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"{value!r} does not parse to a list.")
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{value!r} contains non-string items.")
    return parsed


def validate_cmap_frame(df, csv_path=None, verbose=True):
    """Validate a frame against the CMAP schema. Raises on the first failure.

    Checks, in order:

    1. every CMAP column is present;
    2. ``project``, ``text``, and ``document`` are present and non-empty;
    3. ``codes`` and ``data_group`` round-trip through ``ast.literal_eval``
       to ``list[str]``;
    4. ``doc_id`` is unique;
    5. ``word_count`` equals the recomputed whitespace token count;
    6. if ``csv_path`` is given, the file re-reads to an identical frame.

    ``text_length`` is deliberately not recomputed. In the shipped corpus it
    was measured before a later cleaning pass, so it disagrees with
    ``len(text)`` on most rows -- a check that rejects the canonical corpus
    would be a defect in the check, not in the data.

    Dates are validated where they live, at ``parse_filename()``, which parses
    strictly and refuses an impossible date. The CMAP schema carries no date
    column, so none is invented here.

    Returns a dict of counts on success.
    """
    missing = [column for column in CMAP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CMAP columns: {missing}")

    if len(df) == 0:
        raise ValueError("Frame is empty.")

    for column in ("project", "text", "document"):
        blank = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if blank.any():
            raise ValueError(
                f"Column {column!r} is empty in {int(blank.sum())} row(s): "
                f"doc_id {list(df.loc[blank, 'doc_id'])[:5]}"
            )

    for column in ("codes", "data_group"):
        for doc_id, value in zip(df["doc_id"], df[column]):
            try:
                _as_str_list(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Column {column!r} does not round-trip to list[str] at "
                    f"doc_id={doc_id}: {value!r} ({exc})"
                ) from exc

    if not df["doc_id"].is_unique:
        duplicates = df.loc[df["doc_id"].duplicated(), "doc_id"].tolist()
        raise ValueError(f"doc_id is not unique; duplicates: {duplicates[:5]}")

    recounted = df["text"].astype(str).str.split().str.len()
    bad_counts = recounted != df["word_count"]
    if bad_counts.any():
        first = df.loc[bad_counts].iloc[0]
        raise ValueError(
            f"word_count disagrees with the text in {int(bad_counts.sum())} "
            f"row(s); first at doc_id={first['doc_id']}: stored "
            f"{first['word_count']}, recounted {recounted[bad_counts].iloc[0]}"
        )

    if csv_path is not None:
        reread = pd.read_csv(csv_path)
        expected = df[CMAP_COLUMNS].reset_index(drop=True)
        actual = reread[CMAP_COLUMNS].reset_index(drop=True)
        if not expected.astype(str).equals(actual.astype(str)):
            raise ValueError(
                f"{csv_path} does not re-read to an identical frame."
            )

    report = {
        "rows": len(df),
        "documents": int(df["document"].nunique()),
        "columns": len(CMAP_COLUMNS),
        "reread_verified": csv_path is not None,
    }
    if verbose:
        reread_note = f", re-read from {csv_path}" if csv_path else ""
        print(
            f"[OK] CMAP schema valid: {report['rows']:,} rows, "
            f"{report['documents']} document(s), "
            f"{report['columns']} columns{reread_note}"
        )
    return report


__all__ = [
    "CMAP_COLUMNS", "parse_filename", "remove_timestamps", "detect_speaker",
    "collapse_whitespace", "normalize_text", "segment_turns",
    "build_cmap_frame", "reconstruct_text", "validate_cmap_frame",
]
