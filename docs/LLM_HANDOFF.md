# Running the language-model step for real

The notebook never calls a model. It prints a prompt, you run it wherever you
already work, and you paste the answer back. That keeps the demo free of API
keys, endpoints, and configuration, and it means nothing leaves your machine
unless you send it.

## The loop

1. Run the prompt cell. It prints a complete prompt between two rules.
2. Copy everything between the rules into your assistant of choice.
3. Copy the CSV it returns.
4. Paste that CSV into the `llm_response` string in the next cell and re-run
   it. The grounding check runs automatically.

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
| Post-process outside the model | Splitting and lowercasing are deterministic; do not ask a model to do them |

## What the check proves, and what it does not

A model with no access to the data can return five perfectly formed, entirely
invented rows. Parsing proves syntax, not grounding.

So `check_llm_response()` compares the returned `doc_id`s and row count against
the local source before accepting anything, and rejects codes outside the
allowed list. If a returned answer is internally consistent but disagrees with
the source, the answer is wrong, not the source.

It rejects, with a specific message: a wrong row count, fabricated or renumbered
`doc_id`s, an identifier that is not exactly a whole number, a missing column,
prose instead of CSV, and any code outside the allowed four.

**Be clear about the limit.** This is an identity and schema check. It proves
the answer is about the rows you sent. It cannot tell you the codes are right:
an answer that returns your `doc_id`s and labels every row `family` passes it
cleanly. Nothing here substitutes for reading the text, and on real work you
would validate a sample against human coding.

## The reference answer

`llm_reference_output.csv` is a real answer, not a mock-up.

It was produced by giving an AI assistant the prompt text and nothing else: no
access to this repository, no access to the dataset, and no tools of any kind.
The result was then run through `check_llm_response()`, which passed: five
rows, `doc_id`s `0, 1, 3, 5, 7` matching the source, all codes within the
allowed list.

Three caveats, in descending order of importance. **This provenance is reported
by the author, not provable from the files here**: the repository contains the
output, not a transcript of its generation, so take it as a worked example
rather than evidence. It also comes from one assistant on one occasion, which
is not a benchmark. And these models are not deterministic, so your own run
need not match it row for row. That last point is the argument for checking
every answer rather than trusting any single one.

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
