import json
import glob

for fpath in glob.glob("data/*.json"):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            print(f"{fpath}: {len(data)} items. Keys: {list(data.keys())[:5]} ... {list(data.keys())[-1]}")
    except Exception as e:
        print(f"{fpath}: Error {e}")
