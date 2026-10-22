#!/usr/bin/env bash
# Step 1 (optional): turn the original PDFs into raw text. The PDFs themselves are not in this repo.
# Usage: bash code/01_extract_text.sh PDF_DIR RAW_TEXT_DIR      (needs poppler's pdftotext)
set -euo pipefail
mkdir -p "$2"
for f in "$1"/*.pdf; do
  pdftotext -layout "$f" "$2/$(basename "${f%.pdf}").txt"
done
