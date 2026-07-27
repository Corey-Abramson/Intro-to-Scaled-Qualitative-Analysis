# Intro to Scaled Qualitative Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Corey-Abramson/Intro-to-Scaled-Qualitative-Analysis/blob/main/notebook/intro_scaled_qualitative_analysis.ipynb)

One notebook that runs a qualitative analysis workflow end to end on public interview
data: **normalize → read into a table → classify → visualize**. Runs locally or in
Google Colab. No API keys, no environment setup, no model downloads.

Built for the Stanford / CASBS **AIMS** session on qualitative and textual data.

---

## Run it

**Colab (easiest):** click the badge. The notebook fetches everything it needs.

**Locally:**

```bash
git clone https://github.com/Corey-Abramson/Intro-to-Scaled-Qualitative-Analysis.git
cd Intro-to-Scaled-Qualitative-Analysis
pip install -r requirements.txt

# requirements.txt holds what the analysis needs, not a notebook front end.
# Skip this line if you already have Jupyter, or open the file in VS Code.
pip install notebook

jupyter notebook notebook/intro_scaled_qualitative_analysis.ipynb
```

Run the cells top to bottom. Figures and tables land in `output/`.

---

## What the notebook does

| Stage | What you see |
|---|---|
| **1. Normalize** | A messy raw transcript becomes machine-readable rows: one row per unit of talk, with stable identifiers, so the original can be reconstructed |
| **2. Read into a table** | A coded corpus in the same format: 29,090 rows across 384 documents |
| **3. Classify** | Three ways to code text at scale: a dictionary/regex lane, the machine-learning lane, and a large-language-model lane |
| **4. Visualize** | A word cloud and a code co-occurrence heatmap, generated from the codes the notebook itself produced |

De-identification is explained where it belongs in the pipeline but not run here, because the
demo corpus is already public. It is **required** for human-subjects data.

---

## The three coding lanes

**Dictionary / regex.** Concept dictionaries matched against the text. Transparent, fast,
reproducible. Always validate hits against source text and tune terms to your corpus.

**Machine learning.** Embed segments with a transformer and classify. The notebook links
the live version rather than shipping a model download. See Zhuofan Li's
[ASA 2022 workshop](https://github.com/lizhuofan95/ASA2022_Workshop)
([Colab](https://colab.research.google.com/drive/1qMwvjaY6DKQ-jxFTyXt3S3qNQdpV_S9n)) and
the [CMAP Visualization Toolkit](https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit).
The hybrid human–machine result behind this lane is Li, Dohan and Abramson (2021).

**Large language model.** The notebook prints a ready-to-paste prompt pointing at the
public data, which you run in whatever assistant you use, and returns CSV. Nothing is
sent anywhere by the notebook itself: no keys, no endpoints, no configuration. Good
prompt practice here means CSV-only output, an explicit list of allowed values, exactly
one row per input row, and deterministic post-processing outside the model.

---

## Data

`data/1_cleaned_data.csv` is a corpus of engineering oral histories, webscraped from the
IEEE History Center's Engineering and Technology History Wiki.

> **The data carries its own restriction and is NOT covered by this repository's license.**
> No part of the data may be quoted for publication without written permission from the
> Director of the IEEE History Center.

Full notice, provenance chain, and file hash: [`SAMPLE_DATA.md`](SAMPLE_DATA.md).

`data/raw/interview_demo01_20240115.txt` is **synthetic**, written for this repository so
the normalization step has something messy to clean. It is not real interview data.

---

## Learn more

- **[AI wiki](https://github.com/Computational-Ethnography-Lab/ai-wiki)**: concepts,
  glossary, and a curated reading list on AI and social science
- **[Teaching materials + bibliography](https://github.com/Computational-Ethnography-Lab/teaching#v-bibliography)**:
  the curated topical bibliography with DOIs
- **[CMAP Visualization Toolkit](https://github.com/Computational-Ethnography-Lab/cmap_visualization_toolkit)**:
  the full interactive visuals (t-SNE, embeddings, clustering, semantic networks)
- **[CMAP QDPX Converter](https://github.com/Computational-Ethnography-Lab/cmap_qdpx_converter)**:
  get data out of ATLAS.ti / NVivo / MAXQDA and into this format
- **[Computational Ethnography Lab](https://computationalethnography.org/)**

Works cited: [`REFERENCES.md`](REFERENCES.md).

---

## License

**BSD 3-Clause, code only.** See [`LICENSE`](LICENSE).

The license covers the notebook, the `cmap_demo/` package, and the synthetic transcript.
It does **not** cover `data/1_cleaned_data.csv`, which keeps the IEEE restriction above.

Parts of `cmap_demo/viz.py` are recycled from the CMAP Visualization Toolkit (BSD
3-Clause) and carry per-function attribution headers naming their source and release tag.

To cite this repository, see [`CITATION.cff`](CITATION.cff).

## Disclosures

- An AI assistant was used to help write and check code and documentation in this
  repository. All analytic choices, citations, and text are the author's.
- This free software carries no warranty or guarantee.
- Questions or collaboration: corey.abramson@rice.edu
