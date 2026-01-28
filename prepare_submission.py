import json
import argparse
import sys

def clean_file(input_path, output_path):
    print(f"Cleaning {input_path} -> {output_path}...")
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                # Strict format: only id and prediction
                record = {
                    "id": str(data["id"]),
                    "prediction": float(data["prediction"]) # Ensure number
                }
                f_out.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"Skipping bad line: {line[:50]}... Error: {e}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input predictions.jsonl with reasoning")
    parser.add_argument("output", help="Output Clean submission.jsonl")
    args = parser.parse_args()
    
    clean_file(args.input, args.output)
