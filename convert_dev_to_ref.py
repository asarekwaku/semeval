import json
import sys

def convert(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    with open(output_path, 'w') as f_out:
        for key, value in data.items():
            # scoring.py expects "id" and "label" (which correlates to "choices" in dev.json)
            record = {
                "id": key,
                "label": value["choices"]
            }
            f_out.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert.py input.json output.jsonl")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
