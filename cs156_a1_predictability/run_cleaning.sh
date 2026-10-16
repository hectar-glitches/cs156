#!/usr/bin/env bash
# Step 2 (optional): raw text -> cleaned essays (removes headers, footers, footnote tags, equations, tables, etc.)
# Usage: bash run_cleaning.sh [RAW_TEXT_DIR] [OUT_DIR]      Needs the raw text from step 1, which is not in this repo.
# Order matters: clean_technical_reports.py learns which line-end hyphens to rejoin from the files already in OUT_DIR.
set -euo pipefail
RAW=${1:-raw_text}; OUT=${2:-data/cleaned_essays}
mkdir -p "$OUT"
python3 -I code/clean_essays.py "$RAW" "$OUT"          # Word-style PDFs (page-based cleaner)
python3 -I code/clean_technical_reports.py   "$RAW" "$OUT"          # LaTeX / technical reports (prose-only extractor)
python3 -I code/clean_ss152.py  "$RAW/SS152_Assignment_2_LBA_Part_I_1_1.txt" \
                                "$OUT/essay_pica_playgrounds_lba_spring2026.txt"   # keeps only my own paragraphs of the template
