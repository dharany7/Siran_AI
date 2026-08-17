"""
setup_real_data.py — Populate audio_ml/samples/ from the local UrbanSound8K cache,
retrain the siren classifier on real audio only, and report honest held-out accuracy.

Dataset location (confirmed):
  C:\\Users\\dhara\\.cache\\kagglehub\\datasets\\chrisfilo\\urbansound8k\\versions\\1

Class mapping:
  siren   → classID 8                (all real siren files)
  traffic → classID 1,5,9            (car_horn, engine_idling, street_music — sample ~150)
  other   → classID 0,2,3,4,6,7     (remaining — sample ~150)

Steps:
  1. Parse UrbanSound8K.csv
  2. WIPE existing synthetic/proxy files from audio_ml/samples/
  3. Copy real audio files into each bucket
  4. Train on real data only
  5. Report honest held-out accuracy (flag if >=95%)
"""
from __future__ import annotations
import csv, random, shutil, sys
from pathlib import Path
from collections import Counter, defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\dhara\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1")
CSV_PATH     = DATASET_ROOT / "UrbanSound8K.csv"
SAMPLES_BASE = Path(__file__).parent / "audio_ml" / "samples"

SIREN_IDS   = {8}
TRAFFIC_IDS = {1, 5, 9}    # car_horn, engine_idling, street_music
OTHER_IDS   = {0, 2, 3, 4, 6, 7}

N_TRAFFIC = 150
N_OTHER   = 150

RNG_SEED = 42
rng = random.Random(RNG_SEED)

SEP = "=" * 62

# ─────────────────────────────────────────────────────────────────────────────
# 1. Validate dataset path
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  Step 1 — Validating dataset path")
print(SEP)

if not DATASET_ROOT.exists():
    print(f"  ERROR: Dataset not found at {DATASET_ROOT}")
    sys.exit(1)
if not CSV_PATH.exists():
    print(f"  ERROR: UrbanSound8K.csv not found at {CSV_PATH}")
    sys.exit(1)

print(f"  Dataset root : {DATASET_ROOT}")
print(f"  CSV          : {CSV_PATH}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Parse CSV and resolve file paths
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  Step 2 — Parsing UrbanSound8K.csv")
print(SEP)

rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"  Total rows: {len(rows)}")
dist = Counter(f"classID={r['classID']:>2}  {r['class']}" for r in rows)
for k, v in sorted(dist.items()):
    print(f"    {k}: {v} files")
print()

# Resolve each row to an actual file path
def resolve(row: dict) -> Path | None:
    p = DATASET_ROOT / f"fold{row['fold']}" / row["slice_file_name"]
    return p if p.exists() else None

buckets: dict[str, list[Path]] = {"siren": [], "traffic": [], "other": []}
missing = 0
class_breakdown: dict[str, Counter] = defaultdict(Counter)

for row in rows:
    cid = int(row["classID"])
    cls_name = row["class"]
    p = resolve(row)
    if p is None:
        missing += 1
        continue
    if cid in SIREN_IDS:
        buckets["siren"].append(p)
        class_breakdown["siren"][cls_name] += 1
    elif cid in TRAFFIC_IDS:
        buckets["traffic"].append(p)
        class_breakdown["traffic"][cls_name] += 1
    elif cid in OTHER_IDS:
        buckets["other"].append(p)
        class_breakdown["other"][cls_name] += 1

if missing:
    print(f"  WARNING: {missing} rows could not be resolved to a file.")

print("  Resolved files per bucket:")
for bucket, files in buckets.items():
    print(f"    {bucket:<10}: {len(files)} files")
    for cls_name, cnt in sorted(class_breakdown[bucket].items()):
        print(f"               {cls_name}: {cnt}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Wipe synthetic/proxy files, copy real files
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  Step 3 — Wiping synthetic proxies, copying real audio")
print(SEP)

def wipe_and_copy(src_files: list[Path], dest_dir: Path,
                  n: int | None, label: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Wipe ALL existing files (synthetic proxy + old chirp samples)
    old_files = list(dest_dir.glob("*.wav")) + list(dest_dir.glob("*.ogg"))
    for f in old_files:
        f.unlink()
    print(f"  {label:<10}: removed {len(old_files)} old file(s)")

    # Sample if needed
    subset = src_files if n is None else rng.sample(src_files, min(n, len(src_files)))

    # Copy with collision-safe naming (different folds may share filenames)
    copied = 0
    for src in subset:
        fold = src.parent.name          # e.g. "fold3"
        dst  = dest_dir / f"{fold}_{src.name}"
        shutil.copy2(src, dst)
        copied += 1

    print(f"  {label:<10}: copied {copied} real file(s) -> {dest_dir}")
    return copied

n_siren   = wipe_and_copy(buckets["siren"],   SAMPLES_BASE / "siren",   None,      "siren")
n_traffic = wipe_and_copy(buckets["traffic"], SAMPLES_BASE / "traffic", N_TRAFFIC, "traffic")
n_other   = wipe_and_copy(buckets["other"],   SAMPLES_BASE / "other",   N_OTHER,   "other")

total = n_siren + n_traffic + n_other
print()
print(f"  Total real files staged: {total}  ({n_siren} siren, {n_traffic} traffic, {n_other} other)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  Step 4 — Training RandomForest on real UrbanSound8K audio")
print(SEP)
print()

# Force fresh import (clear any cached state from earlier runs)
for mod in list(sys.modules.keys()):
    if mod.startswith("audio_ml"):
        del sys.modules[mod]

sys.path.insert(0, str(Path(__file__).parent))
from audio_ml.train import train
train()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Integrity check on the saved model
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  Step 5 — Sanity check & leakage flag")
print(SEP)

import pickle
with open(Path(__file__).parent / "audio_ml" / "model.pkl", "rb") as f:
    meta = pickle.load(f)

test_acc = meta.get("test_acc", None)
cv_mean  = meta.get("cv_mean",  None)
cv_std   = meta.get("cv_std",   None)
synthetic = meta.get("synthetic", False)

print(f"  synthetic flag : {synthetic}  (should be False)")
print(f"  test_acc       : {test_acc:.3f}" if test_acc is not None else "  test_acc: N/A")
print(f"  cv_mean        : {cv_mean:.3f}" if cv_mean  is not None else "  cv_mean:  N/A")
print(f"  cv_std         : {cv_std:.3f}"  if cv_std   is not None else "  cv_std:   N/A")
print()

if test_acc is not None and test_acc >= 0.95:
    print("  *** DATA LEAKAGE WARNING ***")
    print(f"  Test accuracy = {test_acc:.1%} is suspiciously high for real UrbanSound8K data.")
    print("  Possible causes:")
    print("    - Files from the same original recording split across train/test")
    print("    - Synthetic files still mixed into samples/ (check manually)")
    print("    - Class imbalance (929 sirens vs 150 each) dominating the split")
    print("  Recommend: use fold-aware cross-validation (leave-one-fold-out)")
    print("  instead of random train/test split to prevent cross-recording leakage.")
elif test_acc is not None:
    print(f"  Accuracy {test_acc:.1%} is in a plausible range for real UrbanSound8K.")

print()
print(SEP)
print("  Done.")
print(SEP)
