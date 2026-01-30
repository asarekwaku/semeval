#!/usr/bin/env python3
"""
Enhanced Generative Reasoning for AmbiStory (SemEval 2026 Task 5)
Version 2.0 - High Accuracy Configuration

Improvements over v1:
1. Uses example_sentence from dataset
2. RAG retrieval from training data for similar homonyms
3. Multi-meaning disambiguation prompting
4. Higher ensemble voting with majority aggregation
5. Better score calibration
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from utils import load_data, format_story_for_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingDataRetriever:
    """RAG-style retrieval from training data for similar examples."""
    
    def __init__(self, train_path: str = "data/train.json"):
        self.examples_by_homonym: Dict[str, List[dict]] = defaultdict(list)
        self.loaded = False
        
        if os.path.exists(train_path):
            try:
                with open(train_path, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                
                for sample_id, sample in train_data.items():
                    homonym = sample.get('homonym', '').lower().strip()
                    if homonym:
                        self.examples_by_homonym[homonym].append({
                            'id': sample_id,
                            'homonym': sample.get('homonym', ''),
                            'judged_meaning': sample.get('judged_meaning', ''),
                            'precontext': sample.get('precontext', ''),
                            'sentence': sample.get('sentence', ''),
                            'ending': sample.get('ending', ''),
                            'average': sample.get('average', 3.0),
                            'example_sentence': sample.get('example_sentence', '')
                        })
                self.loaded = True
                logger.info(f"Loaded {len(train_data)} training examples for {len(self.examples_by_homonym)} unique homonyms")
            except Exception as e:
                logger.warning(f"Could not load training data: {e}")
    
    def get_similar_examples(self, homonym: str, limit: int = 3) -> List[dict]:
        """Retrieve examples with the same homonym from training data."""
        homonym_key = homonym.lower().strip()
        examples = self.examples_by_homonym.get(homonym_key, [])
        
        # Sort by score extremity (most informative examples)
        examples_sorted = sorted(examples, key=lambda x: abs(x['average'] - 3.0), reverse=True)
        return examples_sorted[:limit]


def parse_response(output: str) -> Tuple[Optional[float], str]:
    """Extract the score and reasoning from the LLM output."""
    # Look for "Score: X.X" pattern
    score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    score = None
    if score_match:
        try:
            score = float(score_match.group(1))
            # Clamp to valid range
            score = max(1.0, min(5.0, score))
        except ValueError:
            pass
    
    reasoning = output.strip()
    return score, reasoning


def query_ollama(model: str, prompt: str, api_url: str, temperature: float = 0.0) -> Optional[str]:
    """Send a request to the Ollama API."""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    encoded_data = json.dumps(data).encode('utf-8')
    try:
        req = urllib.request.Request(
            f"{api_url}/api/generate",
            data=encoded_data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '')
    except Exception as e:
        logger.error(f"Error querying Ollama ({model}): {e}")
        logger.error(f"  -> Is Ollama running? Check with 'curl {api_url}' or 'systemctl status ollama'")
        return None


def query_ollama_with_retry(model: str, prompt: str, api_url: str, 
                            temperature: float = 0.0, max_retries: int = 3) -> Optional[str]:
    """Query Ollama with retry logic."""
    for i in range(max_retries):
        response = query_ollama(model, prompt, api_url, temperature)
        if response:
            return response
        logger.warning(f"Retry {i+1}/{max_retries} waiting...")
        time.sleep(2 ** i)  # Exponential backoff
    return None


def generate_enhanced_prompt(story_text: str, homonym: str, judged_meaning: str,
                             example_sentence: str = "", 
                             similar_examples: List[dict] = None) -> str:
    """Generate an enhanced prompt with example sentence and RAG examples."""
    
    # Build RAG examples section
    rag_section = ""
    if similar_examples:
        rag_section = "\n## Reference Examples (same homonym from training data):\n"
        for i, ex in enumerate(similar_examples[:2], 1):
            rag_section += f"""
Example {i}:
- Meaning: "{ex['judged_meaning']}"
- Context: "{ex['sentence']}"
- Human Average Score: {ex['average']:.1f}
"""
    
    # Include example sentence if available
    example_sentence_section = ""
    if example_sentence:
        example_sentence_section = f'\nExample usage of this meaning: "{example_sentence}"'
    
    prompt = f"""You are an expert linguist analyzing word sense disambiguation in context.

## Task
Determine how plausible the proposed meaning is for the target word in this specific story context.

## Scoring Scale
- 1.0 = Impossible/Makes no sense in this context
- 2.0 = Very unlikely
- 3.0 = Ambiguous/Could go either way
- 4.0 = Likely
- 5.0 = Very plausible/Perfect fit

## Story Context
{story_text}

## Analysis Target
Target Word: "{homonym}"
Proposed Meaning: "{judged_meaning}"{example_sentence_section}
{rag_section}

## Instructions
1. First, identify what other meanings the word "{homonym}" could have
2. Analyze the precontext - what situation is being set up?
3. Analyze the sentence containing "{homonym}" - how is it used?
4. Analyze the ending - does it confirm or contradict the proposed meaning?
5. Compare the proposed meaning against the context clues
6. Assign a score based on how well the proposed meaning fits

## Response Format
Reasoning:
[Your step-by-step analysis]

Score: [X.X]
"""
    return prompt


def generate_critique_prompt(original_prompt: str, initial_response: str, 
                             initial_score: float) -> str:
    """Generate a self-critique prompt."""
    return f"""{original_prompt}

---
## Your Initial Analysis:
{initial_response}

## Your Initial Score: {initial_score}

---
## Self-Critique Required:

Review your initial analysis:
1. Did you correctly identify the story's setting and context?
2. Did you check if the ending confirms or contradicts your interpretation?
3. Could you have missed contextual clues?
4. Is your score appropriate given the evidence?

If you find errors in your reasoning, provide a corrected analysis.
Otherwise, confirm your original score.

Reasoning:
[Your critique and any corrections]

Score: [Your final score X.X]
"""


def aggregate_scores(scores: List[float], method: str = "median") -> float:
    """Aggregate multiple scores into a final prediction."""
    if not scores:
        return 3.0  # Default to middle
    
    if method == "median":
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            return (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            return sorted_scores[n//2]
    elif method == "mean":
        return sum(scores) / len(scores)
    elif method == "trimmed_mean":
        # Remove highest and lowest, then average
        if len(scores) <= 2:
            return sum(scores) / len(scores)
        sorted_scores = sorted(scores)[1:-1]
        return sum(sorted_scores) / len(sorted_scores)
    else:
        return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Generative Reasoning for AmbiStory v2")
    parser.add_argument("--input", default="data/dev.json", help="Path to input JSON file")
    parser.add_argument("--output", default="predictions/predictions_v2.jsonl", help="Path to output JSONL file")
    parser.add_argument("--model", default="llama3.1:70b-instruct-q4_0", help="Ollama model to use")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--ensemble", type=int, default=9, help="Number of ensemble votes (default: 9)")
    parser.add_argument("--self-correction", action="store_true", default=True, help="Enable self-correction")
    parser.add_argument("--no-self-correction", action="store_false", dest="self_correction")
    parser.add_argument("--train-data", default="data/train.json", help="Path to training data for RAG")
    parser.add_argument("--aggregation", choices=["mean", "median", "trimmed_mean"], 
                        default="median", help="Score aggregation method")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Initialize RAG retriever
    retriever = TrainingDataRetriever(args.train_data)
    
    logger.info(f"Loading data from {args.input}...")
    try:
        data = load_data(args.input)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} samples.")
    logger.info(f"Configuration: model={args.model}, ensemble={args.ensemble}, "
                f"self_correction={args.self_correction}, aggregation={args.aggregation}")
    
    # Resume logic
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try:
                    rec = json.loads(line)
                    processed_ids.add(str(rec['id']))
                except:
                    pass
    logger.info(f"Found {len(processed_ids)} already processed samples.")
    
    # Process samples
    with open(args.output, 'a', encoding='utf-8') as f_out:
        count = 0
        total = len(data)
        
        for sample_id, sample in data.items():
            if str(sample_id) in processed_ids:
                continue
            
            if args.limit and count >= args.limit:
                break
            
            story_text, homonym, judged_meaning = format_story_for_prompt(sample)
            example_sentence = sample.get('example_sentence', '')
            
            # Get similar examples from training data
            similar_examples = retriever.get_similar_examples(homonym) if retriever.loaded else []
            
            # Generate enhanced prompt
            prompt = generate_enhanced_prompt(
                story_text, homonym, judged_meaning,
                example_sentence, similar_examples
            )
            
            logger.info(f"[{count+1}/{total}] Processing ID {sample_id} ({homonym})...")
            
            scores = []
            reasonings = []
            
            # Ensemble loop
            for vote_idx in range(args.ensemble):
                # Use temperature variation for ensemble diversity
                temp = 0.0 if vote_idx == 0 else 0.7
                
                response_text = query_ollama_with_retry(args.model, prompt, args.api_url, temperature=temp)
                
                if response_text:
                    score, reasoning = parse_response(response_text)
                    
                    # Self-correction on first vote only (to save time)
                    if args.self_correction and score is not None and vote_idx == 0:
                        logger.debug(f"  Self-correcting (Initial: {score})...")
                        critique_prompt = generate_critique_prompt(prompt, reasoning, score)
                        critique_response = query_ollama_with_retry(
                            args.model, critique_prompt, args.api_url, temperature=0.0
                        )
                        if critique_response:
                            new_score, new_reasoning = parse_response(critique_response)
                            if new_score is not None:
                                logger.debug(f"  Corrected Score: {new_score}")
                                score = new_score
                                reasoning = f"[Critique]: {new_reasoning}\n[Original]: {reasoning}"
                    
                    if score is not None:
                        scores.append(score)
                        reasonings.append(reasoning)
                    else:
                        logger.warning(f"  Vote {vote_idx+1}: Could not parse score")
                else:
                    logger.warning(f"  Vote {vote_idx+1}: No response")
            
            if scores:
                # Aggregate scores
                final_score = aggregate_scores(scores, args.aggregation)
                
                # Round to 1 decimal place
                final_score = round(final_score, 1)
                
                logger.info(f"  Scores: {scores} -> Final: {final_score}")
                
                # Combine reasonings (just keep first for efficiency)
                final_reasoning = reasonings[0] if reasonings else ""
                
                # Write result
                record = {
                    "id": sample_id,
                    "prediction": final_score,
                    "reasoning": final_reasoning,
                    "raw_scores": scores
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
            else:
                logger.error(f"No valid response for {sample_id} after {args.ensemble} tries")
            
            count += 1
    
    logger.info(f"Finished. Results saved to {args.output}")


if __name__ == "__main__":
    main()
