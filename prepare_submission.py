#!/usr/bin/env python3
"""
Prepare final submission file.
Removes reasoning field and ensures correct format.
"""

import json
import sys
import os


def prepare_submission(input_file: str, output_file: str):
    """Convert prediction file to submission format."""
    records = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                # Only keep id and prediction
                clean_rec = {
                    "id": str(rec["id"]),
                    "prediction": float(rec["prediction"])
                }
                records.append(clean_rec)
            except Exception as e:
                print(f"Error parsing line: {e}")
    
    # Sort by ID
    records.sort(key=lambda x: int(x["id"]))
    
    # Write output
    with open(output_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"Prepared {len(records)} predictions")
    print(f"Output: {output_file}")
    
    # Verify IDs
    ids = [int(r["id"]) for r in records]
    expected = list(range(len(records)))
    missing = set(expected) - set(ids)
    if missing:
        print(f"WARNING: Missing IDs: {sorted(missing)[:10]}...")
    else:
        print(f"All IDs 0-{len(records)-1} present ✓")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_submission.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    prepare_submission(sys.argv[1], sys.argv[2])
