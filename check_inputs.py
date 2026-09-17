#!/usr/bin/env python3
"""Diagnostic: verify every split entry maps to a readable memmap slice."""
import json, os, re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/sammarv/quantum_corrosion")
SPLITS_DIR = BASE_DIR / "data/splits"

# signal params (must match notebook)
FFT_SIZE = 4096
N_STACKS = 16
OVERLAP = 0.25
HOP = int(FFT_SIZE * (1 - OVERLAP))
SAMPLES_PER_IMG = N_STACKS * HOP + (FFT_SIZE - HOP)

IQ_FILES = {
    0: BASE_DIR / "data/raw/0.5/0.5.iq",
    1: BASE_DIR / "data/raw/1/1.iq",
    2: BASE_DIR / "data/raw/1.5/1.5.iq",
    3: BASE_DIR / "data/raw/2/2.iq",
    4: BASE_DIR / "data/raw/2.5/2.5.iq",
    5: BASE_DIR / "stack/1/1gSTACK",
    6: BASE_DIR / "stack/1.5/1.5gSTACK",
    7: BASE_DIR / "stack/2/2gSTACK",
    8: BASE_DIR / "stack/2.5/2.5gSTACK",
}

def get_img_idx(path):
    m = re.search(r"img(\d+)", str(path))
    return int(m.group(1)) if m else 0


def inspect_split(split_name):
    with open(SPLITS_DIR / f"{split_name}.json") as f:
        meta = json.load(f)
    total = len(meta)
    missing_label = 0
    not_exist_path = 0
    out_of_bounds = 0
    per_label_missing = defaultdict(int)
    per_label_total = defaultdict(int)

    # prepare memmaps for existing IQ files
    memmaps = {}
    for lbl, p in IQ_FILES.items():
        if p.exists():
            try:
                memmaps[lbl] = os.path.getsize(str(p)) // 8  # bytes / 8 for complex64 (two floats) approximation
            except Exception:
                memmaps[lbl] = None

    for rec in meta:
        lbl = int(rec.get('label', -1))
        per_label_total[lbl] += 1
        if lbl not in IQ_FILES:
            missing_label += 1
            per_label_missing[lbl] += 1
            continue
        iq_path = IQ_FILES[lbl]
        if not iq_path.exists():
            not_exist_path += 1
            per_label_missing[lbl] += 1
            continue
        idx = get_img_idx(rec.get('path',''))
        required = (idx * SAMPLES_PER_IMG) + SAMPLES_PER_IMG
        # try to open memmap length robustly
        try:
            # If the path is a file, compute size in samples
            file_size_bytes = os.path.getsize(str(iq_path))
            # complex64 stores 2 float32 per sample (I,Q) -> 8 bytes per sample
            n_samples = file_size_bytes // 8
            if required > n_samples:
                out_of_bounds += 1
                per_label_missing[lbl] += 1
        except Exception as e:
            per_label_missing[lbl] += 1
    return {
        'split': split_name,
        'total_records': total,
        'missing_label_count': missing_label,
        'not_exist_path_count': not_exist_path,
        'out_of_bounds_count': out_of_bounds,
        'per_label_total': dict(per_label_total),
        'per_label_missing': dict(per_label_missing),
        'available_iq_files': {k: str(v) for k,v in IQ_FILES.items() if v.exists()}
    }

if __name__ == '__main__':
    results = []
    for s in ('train','val','test'):
        try:
            r = inspect_split(s)
            results.append(r)
        except FileNotFoundError:
            print(f"Split file not found: {s}.json")
    import json
    print(json.dumps(results, indent=2))
