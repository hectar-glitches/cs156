# Has my writing become more predictable?

An n-gram language model study of my own essays: train a model on one period of my writing, score held-out essays from the other, and compare. Inspired by StoryScope (Russell et al., 2026, arXiv:2604.03136).

**Short answer:** by the decision rule fixed in advance, no. Recent essays are *less* predictable on all words (more specialised vocabulary) but *more* predictable on function words alone. Course level and genre are confounded with period, so the data cannot say why.

## Layout

```
code/                  cleaning scripts and study scripts
data/cleaned_essays/   20 cleaned essay texts (my own writing; names carry the period)
results/               JSON output of every analysis
figures/               fig1_models, fig2_essays (PNG for previews, PDF for LaTeX)
run_cleaning.sh        raw text -> cleaned essays            (needs the original PDFs, not included)
run_study.sh           cleaned essays -> results + figures   (works from this repo alone)
```

## Reproduce the study

```bash
pip install -r requirements.txt
bash run_study.sh        # a few minutes; fixed seed (156)
```

## Cleaning steps

```bash
bash code/01_extract_text.sh my_pdfs raw_text     # pdftotext -layout on each PDF
bash run_cleaning.sh raw_text data/cleaned_essays
```

Run order matters: `clean_tech.py` decides whether to rejoin a line-end hyphen by checking which words appear in the
essays already in the output folder, so `clean_essays.py` must run first.

## Data

| Group | Essays | Use |
|---|---|---|
| Early solo (Fall 2023 to Spring 2024) | 9 | main analysis |
| Recent solo (Fall 2025 to Spring 2026) | 6 | main analysis |
| `_GROUP` essays | 4 | robustness run only (written with classmates) |
| `food_insecurity_part1_fall2023` | 1 | excluded, near-duplicate of Part 2 |

Which essays belong to which period is set at the top of `code/predictability.py`. Cleaning removes cover pages, page headers and footers, footnote tags, equations, tables and figure text, repairs extraction artifacts (Cyrillic look-alike letters, zero-width characters, line-end hyphens), and for the SS152 template keeps only my own paragraphs.

## What each script does

| Script | Output |
|---|---|
| `predictability.py` | add-alpha unigram and bigram, 2x2 train/test perplexity table, exact permutation tests, function-word and with-group runs (`results.json`) |
| `kn_model.py` | the same study with an interpolated Kneser-Ney bigram (`kn_results.json`) |
| `overlap.py` | unseen-word share and word overlap between periods |
| `classifier.py` | leave-one-essay-out logistic regression (function words, TF-IDF) and compression checks |
| `course_test.py` | same-course vs different-course vocabulary overlap |
| `control.py` | negative control: the classifier on arbitrary labels within a period |
| `make_figures.py` | the two report figures |

## Method in brief

- Metric: bits per word (cross-entropy); perplexity is 2 to that power.
- Whole essays are held out. Within a period each essay is left out in turn. Training size is equal across periods
  (7,400 predicted words), drawn at random and averaged over 20 draws.
- Smoothing (alpha, or the Kneser-Ney discount) is chosen by leave-one-essay-out inside the training essays only.
- Period differences use an exact permutation test on per-essay bits (9 vs 6 essays, 5,005 splits).
- Decision rule fixed before running: "more predictable" needs lower recent perplexity, p < 0.05, and agreement from the
  function-word rerun.

## Known limitations

- 15 essays by one writer; course level, genre and assignment rules vary together with period.
- The vocabulary (words seen at least twice) is counted over all 15 solo essays, including held-out ones.
- The Kneser-Ney model was adopted after the add-alpha bigram lost to the unigram; both are reported.
- No transformer baseline was run.
