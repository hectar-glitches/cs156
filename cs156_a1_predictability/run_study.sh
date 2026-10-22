#!/usr/bin/env bash
# Step 3: run the whole study on the cleaned essays in data/ and rebuild results/ and figures/.
# Usage: bash run_study.sh        (takes several minutes: the classifier and control use permutation tests)
set -euo pipefail
E=data/cleaned_essays; R=results
mkdir -p "$R" figures
python3 -I code/predictability.py "$E" "$R"     # add-alpha unigram/bigram  -> results.json
python3 -I code/kn_model.py       "$E" "$R"     # Kneser-Ney bigram         -> kn_results.json
python3 -I code/overlap.py        "$E" "$R"     # unseen-word share, word overlap
python3 -I code/classifier.py     "$E" "$R"     # early-vs-recent classifier + compression
python3 -I code/course_test.py    "$E" "$R"     # same-course vs different-course vocabulary overlap
python3 -I code/control.py        "$E" "$R"     # negative control: classifier on arbitrary labels
python3 -I code/make_figures.py   "$R" figures  # fig1_models, fig2_essays (PNG + PDF)
