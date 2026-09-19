#!/usr/bin/env bash
# ==============================================================================
# Script: data/download.sh
# Purpose: Pulls the CoIR apps dataset from HuggingFace
# Note: Person A uses train and validation splits exclusively.
# The test split is evaluated only at final submission in Phase 6.
# ==============================================================================

set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/raw" && pwd)"
echo "Downloading CoIR apps dataset to ${DEST_DIR}..."

python -c "
import os
from datasets import load_dataset

dest = '${DEST_DIR}'
os.makedirs(dest, exist_ok=True)
print('Fetching CoIR-Retrieval/apps (train and validation splits)...')
try:
    ds = load_dataset('CoIR-Retrieval/apps', split=['train', 'validation'])
    ds[0].save_to_disk(os.path.join(dest, 'train'))
    ds[1].save_to_disk(os.path.join(dest, 'validation'))
    print('Download complete: train and validation splits stored in', dest)
except Exception as e:
    print('Download notice:', e)
"
