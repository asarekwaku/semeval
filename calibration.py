#!/usr/bin/env python3
"""
Score Calibration Script

Uses the dev set to learn a calibration mapping from raw model scores
to better-calibrated predictions.
"""

import argparse
import json
import logging
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_predictions(filepath: str) -> dict:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                predictions[str(rec['id'])] = rec['prediction']
            except:
                pass
    return predictions


def load_gold_data(filepath: str) -> dict:
    """Load gold data from JSONL file."""
    gold = {}
    with open(filepath, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                gold[str(rec['id'])] = {
                    'label': rec['label'],
                    'average': sum(rec['label']) / len(rec['label'])
                }
            except:
                pass
    return gold


def linear_calibration(pred: float, a: float, b: float) -> float:
    """Apply linear calibration: y = a*x + b, clamped to [1, 5]."""
    return max(1.0, min(5.0, a * pred + b))


def isotonic_calibration(preds: np.ndarray, targets: np.ndarray) -> callable:
    """Fit isotonic regression for calibration."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=1.0, y_max=5.0)
    ir.fit(preds, targets)
    return lambda x: ir.predict([x])[0]


def optimize_linear_params(predictions: dict, gold: dict) -> tuple:
    """Find optimal linear calibration parameters."""
    pred_list = []
    target_list = []
    
    for sample_id, pred in predictions.items():
        if sample_id in gold:
            pred_list.append(pred)
            target_list.append(gold[sample_id]['average'])
    
    preds = np.array(pred_list)
    targets = np.array(target_list)
    
    def loss(params):
        a, b = params
        calibrated = np.clip(a * preds + b, 1.0, 5.0)
        return np.mean((calibrated - targets) ** 2)
    
    result = minimize(loss, [1.0, 0.0], method='Nelder-Mead')
    return tuple(result.x)


def apply_calibration(input_file: str, output_file: str, a: float, b: float):
    """Apply calibration to predictions file."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                rec = json.loads(line)
                raw_score = rec['prediction']
                calibrated_score = linear_calibration(raw_score, a, b)
                rec['prediction'] = round(calibrated_score, 1)
                rec['raw_prediction'] = raw_score
                f_out.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.warning(f"Error processing line: {e}")
                f_out.write(line)
    
    logger.info(f"Calibrated predictions saved to {output_file}")


def evaluate(predictions: dict, gold: dict) -> dict:
    """Evaluate predictions against gold data."""
    pred_list = []
    target_list = []
    correct = 0
    total = 0
    
    for sample_id, pred in predictions.items():
        if sample_id in gold:
            avg = gold[sample_id]['average']
            labels = gold[sample_id]['label']
            stdev = np.std(labels) if len(labels) > 1 else 0
            
            pred_list.append(pred)
            target_list.append(avg)
            
            # Accuracy: within 1 OR within stdev
            if abs(pred - avg) < 1 or (stdev > 0 and (avg - stdev) < pred < (avg + stdev)):
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    spearman = spearmanr(pred_list, target_list)[0] if len(pred_list) > 1 else 0
    
    return {
        'accuracy': accuracy,
        'spearman': spearman,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Score Calibration")
    parser.add_argument("--train-preds", help="Predictions on dev set for training calibration")
    parser.add_argument("--gold", default="data/dev_ref.jsonl", help="Gold data for dev set")
    parser.add_argument("--input", help="Predictions to calibrate")
    parser.add_argument("--output", help="Output calibrated predictions")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate, don't calibrate")
    
    args = parser.parse_args()
    
    if args.evaluate_only and args.input:
        # Just evaluate
        preds = load_predictions(args.input)
        gold = load_gold_data(args.gold)
        results = evaluate(preds, gold)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Spearman: {results['spearman']:.4f}")
        return
    
    if args.train_preds:
        # Learn calibration from dev set
        preds = load_predictions(args.train_preds)
        gold = load_gold_data(args.gold)
        
        # Evaluate before calibration
        before = evaluate(preds, gold)
        logger.info(f"Before calibration: Acc={before['accuracy']:.4f}, Spearman={before['spearman']:.4f}")
        
        # Find optimal parameters
        a, b = optimize_linear_params(preds, gold)
        logger.info(f"Optimal calibration: y = {a:.4f}*x + {b:.4f}")
        
        # Evaluate after calibration
        calibrated_preds = {k: linear_calibration(v, a, b) for k, v in preds.items()}
        after = evaluate(calibrated_preds, gold)
        logger.info(f"After calibration: Acc={after['accuracy']:.4f}, Spearman={after['spearman']:.4f}")
        
        # Save calibration parameters
        params_file = "calibration_params.json"
        with open(params_file, 'w') as f:
            json.dump({'a': a, 'b': b}, f)
        logger.info(f"Saved calibration parameters to {params_file}")
        
        # Apply to input if provided
        if args.input and args.output:
            apply_calibration(args.input, args.output, a, b)
    
    elif args.input and args.output:
        # Apply existing calibration
        if os.path.exists("calibration_params.json"):
            with open("calibration_params.json", 'r') as f:
                params = json.load(f)
            apply_calibration(args.input, args.output, params['a'], params['b'])
        else:
            logger.error("No calibration parameters found. Run with --train-preds first.")


if __name__ == "__main__":
    import os
    main()
