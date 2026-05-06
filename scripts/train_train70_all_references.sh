#!/usr/bin/env bash
set -euo pipefail

# Train one reference-specific HMM per recording, using only the first
# TRAIN_PERCENT percent of sorted recordings as training queries.
#
# The remaining recordings are not used as training queries, so they can be
# reserved for held-out testing.
#
# Usage from the repo root on the server:
#   bash scripts/train_train70_all_references.sh
#
# Optional environment overrides:
#   TRAIN_PERCENT=70
#   DATA_ROOT=data
#   OUTPUT_DIR=artifacts/train70_models
#   PYTHON_BIN=/home/dweiss/ttmp/miniconda3/bin/python

TRAIN_PERCENT="${TRAIN_PERCENT:-70}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/train70_models}"
PYTHON_BIN="${PYTHON_BIN:-/home/dweiss/ttmp/miniconda3/bin/python}"

pieces=(
  "Chopin_Op017No4"
  "Chopin_Op024No2"
  "Chopin_Op030No2"
  "Chopin_Op063No3"
  "Chopin_Op068No3"
)

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$OUTPUT_DIR"

for piece in "${pieces[@]}"; do
  piece_dir="$DATA_ROOT/wav_22050_mono/$piece"
  if [[ ! -d "$piece_dir" ]]; then
    echo "Missing piece directory: $piece_dir" >&2
    exit 1
  fi

  mapfile -t recordings < <(find "$piece_dir" -maxdepth 1 -type f -name "*.wav" | sort)
  total="${#recordings[@]}"
  train_count=$(( total * TRAIN_PERCENT / 100 ))

  if (( train_count < 2 )); then
    echo "Skipping $piece: need at least 2 training recordings, got $train_count / $total"
    continue
  fi

  echo "=== $piece: $total reference models; $train_count training queries (${TRAIN_PERCENT}%) ==="

  for (( ref_idx = 0; ref_idx < total; ref_idx++ )); do
    output="$OUTPUT_DIR/${piece}_reference${ref_idx}_train${TRAIN_PERCENT}_hmm.npz"

    if [[ -f "$output" ]]; then
      echo "Skipping existing model: $output"
      continue
    fi

    echo "--- $piece reference index $ref_idx ---"
    "$PYTHON_BIN" -m scripts.train_hmm \
      --piece "$piece" \
      --data-root "$DATA_ROOT" \
      --reference-index "$ref_idx" \
      --train-query-count "$train_count" \
      --output "$output"
  done
done

echo "Done. Models are in $OUTPUT_DIR"
