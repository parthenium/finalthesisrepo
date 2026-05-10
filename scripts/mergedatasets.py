import json
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATASET1_ROOT = r'C:\Users\agniv\OneDrive\Desktop\lawyer\mergedataset\echrclassification'
DATASET2_PATH = r'C:\Users\agniv\OneDrive\Desktop\lawyer\mergedataset\merged\.json'
OUTPUT_DIR    = r'C:\Users\agniv\OneDrive\Desktop\lawyer\mergedataset'

# ---------------------------------------------------------------------------
# Load dataset1
# ---------------------------------------------------------------------------

print("Loading dataset 1 (individual JSON files)...", flush=True)
dataset1 = []
json_files = list(Path(DATASET1_ROOT).rglob('*.json'))
print(f"  Found {len(json_files)} JSON files, reading...", flush=True)

for i, json_file in enumerate(json_files):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            rec = json.load(f)
            if 'ITEMID' not in rec:
                rec['ITEMID'] = json_file.stem
            dataset1.append(rec)
    except Exception as e:
        print(f"  WARNING: could not read {json_file.name}: {e}", flush=True)
    if (i + 1) % 1000 == 0:
        print(f"  ... read {i + 1} files so far", flush=True)

print(f"  Done. Loaded {len(dataset1)} records from dataset 1", flush=True)

# ---------------------------------------------------------------------------
# Load dataset2
# ---------------------------------------------------------------------------

print("Loading dataset 2 (output.json)...", flush=True)
with open(DATASET2_PATH, 'r', encoding='utf-8') as f:
    dataset2 = json.load(f)
print(f"  Done. Loaded {len(dataset2)} records from dataset 2", flush=True)

# ---------------------------------------------------------------------------
# Build lookup from dataset2
# ---------------------------------------------------------------------------

print("Building lookup...", flush=True)
d2_lookup = {r['itemid'].lower().strip(): r for r in dataset2}

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

print("Merging...", flush=True)
merged  = []
d1_only = []

for rec in dataset1:
    key = rec['ITEMID'].lower().strip()
    if key in d2_lookup:
        d2_rec = d2_lookup.pop(key)
        merged_rec = {
            'itemid':                   rec['ITEMID'],
            'docname':                  rec.get('DOCNAME'),
            'respondent':               rec.get('RESPONDENT'),
            'date':                     rec.get('DATE'),
            'branch':                   rec.get('BRANCH'),
            'importance':               rec.get('IMPORTANCE'),
            'conclusion':               rec.get('CONCLUSION'),
            'violated_articles':        rec.get('VIOLATED_ARTICLES', []),
            'violated_paragraphs':      rec.get('VIOLATED_PARAGRAPHS', []),
            'violated_bulletpoints':    rec.get('VIOLATED_BULLETPOINTS', []),
            'non_violated_articles':    rec.get('NON_VIOLATED_ARTICLES', []),
            'text':                     rec.get('TEXT', []),
            'num_applicants':           d2_rec.get('num_applicants'),
            'total_damage_claimed':     d2_rec.get('total_damage_claimed'),
            'total_damage_awarded':     d2_rec.get('total_damage_awarded'),
        }
        merged.append(merged_rec)
    else:
        d1_only.append(rec)

d2_only = list(d2_lookup.values())

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

out = Path(OUTPUT_DIR)

print("Saving merged.json...", flush=True)
with open(out / 'merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Saving d1_only.json...", flush=True)
with open(out / 'd1_only.json', 'w', encoding='utf-8') as f:
    json.dump(d1_only, f, indent=2, ensure_ascii=False)

print("Saving d2_only.json...", flush=True)
with open(out / 'd2_only.json', 'w', encoding='utf-8') as f:
    json.dump(d2_only, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"Dataset 1 total      : {len(dataset1)}")
print(f"Dataset 2 total      : {len(dataset2)}")
print(f"Overlap (merged)     : {len(merged)}")
print(f"Dataset 1 only       : {len(d1_only)}  (violation labels, no monetary data)")
print(f"Dataset 2 only       : {len(d2_only)}  (monetary data, no violation labels)")
print(f"\nSaved to: {OUTPUT_DIR}")
print(f"  merged.json  -> {len(merged)} records")
print(f"  d1_only.json -> {len(d1_only)} records (classifier head only)")
print(f"  d2_only.json -> {len(d2_only)} records (regressor head only)")