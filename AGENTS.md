# AGENTS.md

Orientation for an AI agent or a newcomer working in this repository.

## What this is

A teaching repository: one notebook that runs a qualitative analysis workflow end to end
on public interview data. It is built to be **plug and play**: someone opens it in Colab
during a workshop and it runs. Optimize every change for that.

## Layout

```
notebook/intro_scaled_qualitative_analysis.ipynb   the deliverable
cmap_demo/          all the real code; notebook cells stay thin
  viz.py            RECYCLED from the CMAP Visualization Toolkit (BSD-3)
  normalize.py      raw text -> CMAP rows; original work
  header.py         plain-ASCII run header
  llm_handoff.py    prompt builder + deterministic stub
data/1_cleaned_data.csv          vendored corpus; NOT BSD (see SAMPLE_DATA.md)
data/raw/interview_demo01_*.txt  synthetic, authored here
docs/               LLM handoff notes, reference outputs, figures
output/             gitignored; the notebook writes here
```

## Rules that are not negotiable

**The dataset is not BSD.** `data/1_cleaned_data.csv` carries an IEEE History Center
restriction: no part may be quoted for publication without written permission. The
repository's BSD 3-Clause LICENSE covers **code only**. Never let a change imply otherwise.
See `SAMPLE_DATA.md`.

**Recycled code is attributed, not paraphrased.** Functions in `cmap_demo/viz.py` copied
from the CMAP Visualization Toolkit carry a BSD-3 header naming the source file and
release tag `v0.9.6`. If you modify one, say so in that header. Do not silently reword
recycled code into look-alike code; that loses the attribution chain.

**`lizhuofan95/ASA2022_Workshop` has no license.** Link it and cite it. Never copy its
code, redistribute its data, or reuse images from its slide deck.

**No secrets, no environment.** No API keys, no `.env`, no endpoint configuration, no
absolute local paths, no internal hostnames. The LLM lane prints a prompt for a human to
paste; the notebook never calls a model.

**Nothing simulated is presented as real.** If a step produces illustrative rather than
computed output, it says so in its own output. Prefer real output over a convincing
placeholder.

## Working on the notebook

- Every cell must run top to bottom from a fresh kernel, with no reruns and no
  interactive `input()`.
- Markdown cell states what and why; the code cell does it. Keep prose short.
- It must run **locally and in Colab**. Colab resolves the repository root by looking for
  a sentinel; if absent it clones this repo at the pinned release tag. Do not make that
  clone unconditional; it breaks local runs.
- Pin seeds (`random_state=42`). Two runs should produce identical CSV output.
- NLTK is optional. The word cloud falls back to a bundled stopword list and a regex
  tokenizer when NLTK data is unavailable. Test with downloads disabled.
- Figures and CSVs go to `output/`, which is gitignored. Create it before writing.

## Verifying a change

```bash
pip install -r requirements.txt
python -c "import cmap_demo.viz"          # must import in a clean subprocess

# nbconvert is a development tool, not a runtime dependency, so it is not in
# requirements.txt. Install it only if you want to execute the notebook headless.
pip install nbconvert ipykernel
jupyter nbconvert --to notebook --execute \
  notebook/intro_scaled_qualitative_analysis.ipynb --stdout > /dev/null
```

Importing is not enough on its own: call each recycled function once as well, or a
`NameError` that only fires mid-plot will pass a clean import.

The real test is a **clean clone** with `output/`, `__pycache__`, and NLTK caches
removed, not the working tree that built it.

## Citations

Every work cited is in `REFERENCES.md` with a resolved DOI. Citations are facts: verify
before adding one, and never approximate a locator. If you cannot verify it, leave it out.
