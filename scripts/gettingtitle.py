"""
merge_docname.py
----------------
Merges `docname` from the ECHR metadata file into the damages dataset (output.json),
joining on `itemid`.

Usage:
    python merge_docname.py --damages output.json --metadata metadata.json --out merged.json

Arguments:
    --damages    Path to your damages JSON file (output.json)
    --metadata   Path to your large metadata JSON file (array of case objects)
    --out        Path for the merged output file (default: merged.json)
"""

import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge docname into damages dataset")
    parser.add_argument("--damages",  required=True, help="Path to damages JSON (output.json)")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON (large file)")
    parser.add_argument("--out",      default="merged.json", help="Output file path")
    args = parser.parse_args()

    print(f"Loading damages data from: {args.damages}")
    with open(args.damages, encoding="utf-8") as f:
        damages = json.load(f)
    print(f"  → {len(damages)} records loaded")

    print(f"Loading metadata from: {args.metadata}")
    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  → {len(metadata)} records loaded")

    # Build a lookup: itemid → docname
    docname_lookup = {}
    for record in metadata:
        itemid  = record.get("itemid")
        docname = record.get("docname")
        if itemid and docname:
            docname_lookup[itemid] = docname

    print(f"  → {len(docname_lookup)} itemid→docname mappings built")

    # Merge
    matched   = 0
    unmatched = []

    for record in damages:
        itemid = record.get("itemid")
        if itemid in docname_lookup:
            record["docname"] = docname_lookup[itemid]
            matched += 1
        else:
            record["docname"] = None   # flag as unmatched
            unmatched.append(itemid)

    print(f"\nResults:")
    print(f"  Matched:   {matched}")
    print(f"  Unmatched: {len(unmatched)}")

    if unmatched:
        unmatched_path = args.out.replace(".json", "_unmatched.json")
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, indent=2)
        print(f"  Unmatched itemids saved to: {unmatched_path}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(damages, f, indent=2, ensure_ascii=False)
    print(f"\nMerged file saved to: {args.out}")

if __name__ == "__main__":
    main()