import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error

from utils import load_data, format_story_for_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_response(output):
    """
    Extract the score and reasoning from the LLM output.
    Returns (score, reasoning_text).
    """
    # Look for "Score: 4.5"
    score_match = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    score = None
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            pass
            
    # Reasoning is everything before the Score? 
    # Or strict parsing? Let's take the whole response as reasoning if not parsed strictly,
    # or remove the "Score:" line.
    # Simple approach: save the full raw response as reasoning/evidence.
    reasoning = output.strip()
    
    return score, reasoning

    return score, reasoning

def query_ollama(model, prompt, api_url, temperature=0.0):
    """
    Send a request to the Ollama API.
    """
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
        return None

def query_ollama_with_retry(model, prompt, api_url, temperature=0.0, max_retries=3):
    for i in range(max_retries):
        response = query_ollama(model, prompt, api_url, temperature)
        if response:
            return response
        logger.warning(f"Retry {i+1}/{max_retries} waiting...")
        time.sleep(2)
    return None

FEW_SHOT_EXAMPLES = """
Example 1:
Story:
John glanced at his watch and sighed. It had been a busy afternoon filled with back-to-back meetings. The evening sky was beginning to darken as he packed up his bag.
He was ready to drive after a long day.
He started the car, and the engine roared to life. He sped off down the country lanes into the sunset; he'd been waiting for this moment all day.

Target Word: "drive"
Proposed Meaning: "a journey in a vehicle (usually an automobile)"

Reasoning:
1. The target word "drive" is in the sentence "He was ready to drive after a long day."
2. The precontext mentions a busy day and packing up.
3. The ending explicitly mentions "started the car", "engine roared to life", "sped off".
4. This perfectly matches the meaning of driving a vehicle.
Score: 5.0

Example 2:
Story:
John glanced at his watch and sighed. It had been a busy afternoon filled with back-to-back meetings. The evening sky was beginning to darken as he packed up his bag.
He was ready to drive after a long day.
As he settled into his car, he couldn't help but think about the satisfying feeling of sending the ball soaring down the fairway during his weekend games.

Target Word: "drive"
Proposed Meaning: "hitting a golf ball off of a tee with a driver"

Reasoning:
1. The sentence says "He was ready to drive".
2. The ending mentions "settled into his car", which supports driving a vehicle.
3. However, it also mentions "sending the ball soaring down the fairway".
4. While the character is *thinking* about golf, the action he is doing right now (settling into car) is related to driving a car.
5. The sentence "He was ready to drive" in this specific context (after work, getting into car) primarily refers to the commute, though the ending introduces ambiguity about his thoughts.
6. The proposed meaning (golf) is less plausible for the act of "driving" in this exact moment, though the word is used in a golfing context in the character's thoughts.
7. Actually, wait. "He was ready to drive" -> ending says "settled into his car". The physical action is driving a car. The thought is about golf. The judging is about the word in *that* sentence.
8. The sentence sets up the car journey. The golf meaning is low plausibility here.
Score: 1.6
"""

def generate_prompt(story_text, homonym, judged_meaning):
    return f"""Task: Determine the plausibility of a word meaning in the context of a story.
Valuation: Assign a score from 1.0 (Impossible) to 5.0 (Very Plausible).

{FEW_SHOT_EXAMPLES}

Now, your turn:

Story:
{story_text}

Target Word: "{homonym}"
Proposed Meaning: "{judged_meaning}"

Reasoning Instructions:
1. Analyze the context (Precontext, Sentence, Ending).
2. Determine if the Proposed Meaning makes sense for the Target Word *in that specific sentence* given the surrounding details.
3. Be careful with homonyms that fit the sentence grammatically but clash with the story flow.
4. Provide step-by-step reasoning.
5. End with "Score: <number>".

Output Format:
Reasoning: <your reasoning>
Score: <score>
"""

def generate_critique_prompt(original_prompt, initial_response, initial_score):
    return f"""{original_prompt}

Your Initial Response:
{initial_response}

Your Initial Score: {initial_score}

Critique Request:
1. Review the Initial Reasoning. Does it blindly accept the Proposed Meaning, or does it critically check the context (Precontext/Ending)?
2. Is the Initial Score ({initial_score}) justified?
   - If the meaning is "Impossible" in this context, score should be near 1.0.
   - If "Very Plausible", score should be near 5.0.
   - If ambiguous, is it truly ambiguous or just a mismatch?
3. Provide a corrected reasoning and a final score.

Output Format:
Reasoning: <critique and final reasoning>
Score: <final score>
"""

def main():
    parser = argparse.ArgumentParser(description="Generative Reasoning for AmbiStory")
    parser.add_argument("--input", default="data/dev.json", help="Path to input JSON file")
    parser.add_argument("--output", default="predictions/predictions.jsonl", help="Path to output JSONL file")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--api-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of times to run for self-consistency (default: 1)")
    parser.add_argument("--self-correction", action="store_true", help="Enable self-correction/critique loop")
    
    args = parser.parse_args()
    
    # ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    logger.info(f"Loading data from {args.input}...")
    try:
        data = load_data(args.input)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
        
    logger.info(f"Loaded {len(data)} samples.")
    
    # Resume logic: read existing file to see what we have done
    processed_ids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try:
                    rec = json.loads(line)
                    processed_ids.add(str(rec['id'])) # Ensure string comparison
                except:
                    pass
    logger.info(f"Found {len(processed_ids)} already processed samples.")

    # Prepare output file in APPEND mode
    with open(args.output, 'a', encoding='utf-8') as f_out:
        count = 0
        for sample_id, sample in data.items():
            if str(sample_id) in processed_ids:
                continue
                
            if args.limit and count >= args.limit:
                break
            
            story_text, homonym, judged_meaning = format_story_for_prompt(sample)
            prompt = generate_prompt(story_text, homonym, judged_meaning)
            
            logger.info(f"Processing ID {sample_id} ({homonym})...")
            
            scores = []
            reasonings = []
            
            # Ensemble loop
            for _ in range(args.ensemble):
                temp = 0.7 if args.ensemble > 1 else 0.0
                response_text = query_ollama_with_retry(args.model, prompt, args.api_url, temperature=temp)
                
                if response_text:
                    score, reasoning = parse_response(response_text)
                    
                    # Self-Correction Loop
                    if args.self_correction and score is not None:
                        logger.info(f"  Self-correcting (Initial: {score})...")
                        critique_prompt = generate_critique_prompt(prompt, reasoning, score)
                        critique_response = query_ollama_with_retry(args.model, critique_prompt, args.api_url, temperature=0.0) # Critique should be focused
                        if critique_response:
                            new_score, new_reasoning = parse_response(critique_response)
                            if new_score is not None:
                                logger.info(f"  Corrected Score: {new_score}")
                                score = new_score
                                reasoning = f"[Critique]: {new_reasoning}\n[Original]: {reasoning}"
                            else:
                                logger.info("  Critique failed to parse, keeping original.")
                        
                    if score is not None:
                        scores.append(score)
                        reasonings.append(reasoning)
                    else:
                        logger.warning(f"Could not parse score. Partial: {response_text[:50]}...")
            
            if scores:
                final_score = sum(scores) / len(scores)
                # Combine reasonings
                final_reasoning = "\n---\n".join(reasonings) if len(reasonings) > 1 else reasonings[0]
                
                # Write to file
                record = {
                    "id": sample_id, 
                    "prediction": final_score,
                    "reasoning": final_reasoning
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
            else:
                logger.error(f"No valid response for {sample_id} after {args.ensemble} tries")
            
            count += 1
            
    logger.info(f"Finished. Results saved to {args.output}")

if __name__ == "__main__":
    main()
