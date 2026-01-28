import json
import os

def load_data(file_path):
    """
    Load the AmbiStory dataset from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_story_for_prompt(sample):
    """
    Format a sample into a readable story string for the prompt.
    
    Args:
        sample (dict): A single sample from the dataset.
        
    Returns:
        str: Formatted story text.
        str: Homonym.
        str: Judged meaning.
    """
    precontext = sample.get('precontext', '').strip()
    sentence = sample.get('sentence', '').strip()
    ending = sample.get('ending', '').strip()
    homonym = sample.get('homonym', '').strip()
    judged_meaning = sample.get('judged_meaning', '').strip()
    
    # Check if ending exists (some samples might not have it)
    if ending:
        story_text = f"{precontext}\n{sentence}\n{ending}"
    else:
        story_text = f"{precontext}\n{sentence}"
        
    return story_text, homonym, judged_meaning
