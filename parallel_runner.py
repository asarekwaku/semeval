#!/usr/bin/env python3
"""
Parallel Runner for Dual-GPU Inference

Distributes samples across two Ollama instances running on different GPUs
for 2x throughput.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_inference_batch(samples: dict, gpu_id: int, api_port: int, model: str,
                        ensemble: int, output_file: str, train_data: str) -> dict:
    """Run inference on a batch of samples using a specific GPU/port."""
    
    # Create temporary input file for this batch
    temp_input = f"/tmp/batch_gpu{gpu_id}.json"
    with open(temp_input, 'w') as f:
        json.dump(samples, f)
    
    # Run the v2 script for this batch
    cmd = [
        "python3", "generative_reasoning_v2.py",
        "--input", temp_input,
        "--output", output_file,
        "--model", model,
        "--api-url", f"http://localhost:{api_port}",
        "--ensemble", str(ensemble),
        "--train-data", train_data,
        "--self-correction"
    ]
    
    logger.info(f"GPU {gpu_id}: Starting inference on {len(samples)} samples")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        if result.returncode != 0:
            logger.error(f"GPU {gpu_id} error: {result.stderr}")
        else:
            logger.info(f"GPU {gpu_id}: Completed batch")
    except subprocess.TimeoutExpired:
        logger.error(f"GPU {gpu_id}: Timeout on batch")
    except Exception as e:
        logger.error(f"GPU {gpu_id}: Exception {e}")
    
    # Cleanup temp file
    try:
        os.remove(temp_input)
    except:
        pass
    
    return {"gpu_id": gpu_id, "samples_processed": len(samples)}


def split_data(data: dict, num_splits: int) -> list:
    """Split data dictionary into roughly equal parts."""
    items = list(data.items())
    split_size = len(items) // num_splits
    remainder = len(items) % num_splits
    
    splits = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (1 if i < remainder else 0)
        splits.append(dict(items[start:end]))
        start = end
    
    return splits


def merge_outputs(output_files: list, final_output: str):
    """Merge multiple output files into one, removing duplicates."""
    seen_ids = set()
    all_records = []
    
    for fpath in output_files:
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec['id'] not in seen_ids:
                            seen_ids.add(rec['id'])
                            all_records.append(rec)
                    except:
                        pass
    
    # Sort by ID
    all_records.sort(key=lambda x: int(x['id']))
    
    with open(final_output, 'w') as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    
    logger.info(f"Merged {len(all_records)} records to {final_output}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Dual-GPU Runner")
    parser.add_argument("--input", default="data/test.json", help="Input data file")
    parser.add_argument("--output", default="predictions/parallel_submission.jsonl", help="Output file")
    parser.add_argument("--model", default="llama3.1:70b", help="Model to use")
    parser.add_argument("--ensemble", type=int, default=9, help="Ensemble size")
    parser.add_argument("--train-data", default="data/train.json", help="Training data for RAG")
    parser.add_argument("--gpu0-port", type=int, default=11434, help="Ollama port for GPU 0")
    parser.add_argument("--gpu1-port", type=int, default=11435, help="Ollama port for GPU 1")
    parser.add_argument("--single-gpu", action="store_true", help="Use single GPU only")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.single_gpu:
        # Single GPU mode - just run directly
        logger.info("Running in single-GPU mode")
        cmd = [
            "python3", "generative_reasoning_v2.py",
            "--input", args.input,
            "--output", args.output,
            "--model", args.model,
            "--api-url", f"http://localhost:{args.gpu0_port}",
            "--ensemble", str(args.ensemble),
            "--train-data", args.train_data,
            "--self-correction"
        ]
        subprocess.run(cmd)
    else:
        # Dual GPU mode
        logger.info("Running in dual-GPU mode")
        
        # Split data
        splits = split_data(data, 2)
        
        output_files = [
            f"/tmp/gpu0_output.jsonl",
            f"/tmp/gpu1_output.jsonl"
        ]
        
        # Run in parallel using threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            futures.append(executor.submit(
                run_inference_batch, splits[0], 0, args.gpu0_port, 
                args.model, args.ensemble, output_files[0], args.train_data
            ))
            futures.append(executor.submit(
                run_inference_batch, splits[1], 1, args.gpu1_port,
                args.model, args.ensemble, output_files[1], args.train_data
            ))
            
            for future in as_completed(futures):
                result = future.result()
                logger.info(f"Completed: {result}")
        
        # Merge outputs
        merge_outputs(output_files, args.output)
        
        # Cleanup
        for f in output_files:
            try:
                os.remove(f)
            except:
                pass
    
    logger.info(f"Done! Output saved to {args.output}")


if __name__ == "__main__":
    main()
