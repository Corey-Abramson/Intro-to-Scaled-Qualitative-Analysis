# Running the language-model step for real

The notebook never calls a model. It prints a prompt, you run it wherever you
already work, and you bring the answer back. That keeps the demo free of API
keys, endpoints, and configuration, and it means nothing leaves your machine
unless you send it.

## The loop

1. Run the prompt cell. It prints a complete prompt between two rules, and
   below them the shape a good answer comes back in.
2. Copy everything between the rules into your assistant of choice.
3. Copy the CSV it returns.
4. Check it before you use it. The notebook does not do this for you: pass the
   CSV and the same frame you prompted with to `check_llm_response()`, as in
   the example under "Doing this on your own data" below.

## What the prompt does, and why

The prompt names the public dataset **and** embeds the rows to code. Those two
things serve different readers. A model that can browse can go and read the
corpus; a model that cannot still has text in front of it. Either way the
answer is checkable, because the rows carry their `doc_id`s.

**The embedded rows are excerpts, not full text.** Each is truncated to about
300 characters, and the prompt says so. This corpus restricts quotation, so
long excerpts are not shipped in a prompt. The cost is real and worth stating:
if the only evidence for a code sits past the cutoff, a model that cannot open
the URL will not see it and may return an incomplete code set that still passes
every check below. Raise `EXCERPT_CHARS` in `cmap_demo/llm_handoff.py` for data
of your own that carries no such restriction.

The rules in the prompt are not stylistic:

| Rule | What it prevents |
|---|---|
| CSV only, no prose | A model that may explain itself will, and you will be parsing English |
| An explicit list of allowed codes | Synonyms, invented categories, inconsistent capitalisation |
| Exactly one row out per row in | Silent misalignment between your codes and your data |
| Reuse the given `doc_id`s | An answer that cannot be checked against the source |
| A verbatim quote per coded row | Codes asserted about text the model never read |
| A hard character cap on the quote | Returning the passage when the phrase would do, against a corpus that restricts quotation |
| A minimum length on the quote | Passing the grounding check with a common word that appears in every row |
| Post-process outside the model | Splitting and lowercasing are deterministic; do not ask a model to do them |

## What the check proves, and what it does not

A model with no access to the data can return five perfectly formed, entirely
invented rows. Parsing proves syntax, not grounding.

So `check_llm_response()` runs two checks that answer different questions. The
**identity check** compares the returned `doc_id`s and row count against the
local source and rejects codes outside the allowed list. The **grounding check**
takes each non-empty `quote` and requires it to appear verbatim in the row it is
attached to, at a length between `QUOTE_MIN_CHARS` and `QUOTE_CHARS`. If a
returned answer is internally consistent but disagrees with the source, the
answer is wrong, not the source.

The grounding check is the stronger of the two, and it is why the third column
exists. An identifier can be copied straight back out of the prompt by a model
that read nothing, and so can a single common word. "the" appears verbatim in
almost every row in this corpus, so a check that only asked for a verbatim
match would accept `0,"family","the"` five times over and report it grounded.
The minimum length is what closes that: a span of `QUOTE_MIN_CHARS` or more has
to belong to the row it is attached to. It makes fabrication more work than
reading the text rather than impossible, which is the honest description of
what any check like this buys. Whitespace is normalized on both sides first,
because a model reflowing a line is a transport artefact rather than a
paraphrase; everything else must match exactly.

It rejects, with a specific message: a wrong row count, fabricated or renumbered
`doc_id`s, an identifier that is not exactly a whole number, a missing column,
prose instead of CSV, any code outside the allowed four, a quote that cannot be
found in the row it claims to come from, a quote over `QUOTE_CHARS`, and a quote
under `QUOTE_MIN_CHARS`.

**Be clear about the limit.** Together these prove the answer is about the rows
you sent and drawn from their words. Neither tells you the codes are right: an
answer that quotes every row honestly and labels them all `family` passes both
cleanly. Nothing here substitutes for reading the text, and on real work you
would validate a sample against human coding.

## The reference answer

`llm_reference_output.csv` is a real answer, not a mock-up.

It was produced by giving an assistant the prompt text and nothing else. The
model was a GPT one, where this repository was otherwise assembled with Claude,
so the reference answer and the prompt it answers do not come from the same
family. It ran in a read-only sandbox. The result was then put through
`check_llm_response()`, which passed both checks: five rows, `doc_id`s
`0, 1, 3, 5, 7` matching the source, every code within the allowed four, and
four quotes each located verbatim in its own row. The fifth row took no codes
and so returned no quote, which the prompt allows and the check accepts.

Three caveats, in descending order of importance. **This provenance is reported
by the author, not provable from the files here**: the repository holds the
output, not a transcript of its generation, and the sandbox the model ran in
could read this directory, even though nothing in the answer suggests it did.
Take it as a worked example rather than as evidence. It also comes from one
assistant on one occasion, which is not a benchmark. And these models are not
deterministic, so your own run need not match it row for row. That last point
is the argument for checking every answer rather than trusting any single one.

## Doing this on your own data

The prompt builder takes any frame with `doc_id` and `text` columns:

```python
from cmap_demo.llm_handoff import build_llm_prompt, check_llm_response

# Select the rows ONCE and pass that same frame to both calls. Passing the
# full frame to the check while prompting with only its first five rows
# rejects a perfectly good answer for the wrong row count.
sample = my_rows.head(5)

prompt = build_llm_prompt(sample)
print(prompt)
# ... paste into your assistant, copy the CSV back ...
codes = check_llm_response(returned_csv, sample)
```

Change the code vocabulary by editing `CONCEPT_DICTIONARIES` in
`cmap_demo/llm_handoff.py`. The allowed-values list in the prompt and the
validation in the check both read from it, so they cannot drift apart.

Start with five rows while you are tuning the prompt. Scale up once the answers
come back clean, and keep checking them, because the failure mode is a
confident, well-formed answer about text the model never read.
