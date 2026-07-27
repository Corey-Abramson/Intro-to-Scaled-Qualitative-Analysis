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

The prompt names the public dataset **and** embeds the exact rows to code. Those
two things serve different readers. A model that can browse can go and read the
corpus; a model that cannot still has the text in front of it. Either way the
answer is checkable, because the rows carry their `doc_id`s.

The rules in the prompt are not stylistic:

| Rule | What it prevents |
|---|---|
| CSV only, no prose | A model that may explain itself will, and you will be parsing English |
| An explicit list of allowed codes | Synonyms, invented categories, inconsistent capitalisation |
| Exactly one row out per row in | Silent misalignment between your codes and your data |
| Reuse the given `doc_id`s | An answer that cannot be checked against the source |
| Post-process outside the model | Splitting and lowercasing are deterministic; do not ask a model to do them |

## Why the check matters

A model with no access to the data can return five perfectly formed, entirely
invented rows. Parsing proves syntax, not grounding.

So `check_llm_response()` compares the returned `doc_id`s and row count against
the local source before accepting anything, and rejects codes outside the
allowed list. If a returned answer is internally consistent but disagrees with
the source, the answer is wrong — not the source.

It rejects, with a specific message: a wrong row count, fabricated or renumbered
`doc_id`s, a missing column, prose instead of CSV, and any code outside the
allowed four.

## The reference answer

`llm_reference_output.csv` is a real answer, not a mock-up.

It was produced by giving an AI assistant the prompt text and nothing else — no
access to this repository, no access to the dataset, and no tools of any kind —
and then running the result through `check_llm_response()`, which passed: five
rows, `doc_id`s `0, 1, 3, 5, 7` matching the source, all codes within the
allowed list.

Two honest caveats. It comes from one assistant on one occasion, so treat it as
a worked example rather than a benchmark. And because these models are not
deterministic, your own run will not necessarily match it row for row — which is
the point of checking every answer rather than trusting one.

## Doing this on your own data

The prompt builder takes any frame with `doc_id` and `text` columns:

```python
from cmap_demo.llm_handoff import build_llm_prompt, check_llm_response

prompt = build_llm_prompt(my_rows, n_rows=5)
print(prompt)
# ... paste into your assistant, copy the CSV back ...
codes = check_llm_response(returned_csv, my_rows)
```

Change the code vocabulary by editing `CONCEPT_DICTIONARIES` in
`cmap_demo/llm_handoff.py`. The allowed-values list in the prompt and the
validation in the check both read from it, so they cannot drift apart.

Start with five rows while you are tuning the prompt. Scale up once the answers
come back clean — and keep checking them, because the failure mode is a
confident, well-formed answer about text the model never read.
