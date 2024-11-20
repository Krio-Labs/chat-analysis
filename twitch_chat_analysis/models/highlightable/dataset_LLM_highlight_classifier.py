import pandas as pd
import time
from pathlib import Path
import sys
from tqdm import tqdm
import json
from typing import List
from multiprocessing import Pool, cpu_count
import google.generativeai as genai

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import config (assuming config.py is in the project root)
from config import GEMINI_API_KEY

# Configure the Gemini API with safety settings
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-8b',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Cache file path
CACHE_FILE = Path(__file__).parent / 'emotion_cache.json'

# Load cache if exists
try:
    with open(CACHE_FILE, 'r') as f:
        highlight_cache = json.load(f)
except FileNotFoundError:
    highlight_cache = {}

def get_highlight_score(message: str) -> int:
    """Get the highlight-worthiness score (0-3) for a single message using Google Gemini API."""
    retries = 5
    for attempt in range(retries):
        try:
            prompt = f"""You are a text classification expert tasked with analyzing Twitch chat messages. Your goal is to classify each message into one of four categories based on its relevance to highlight-worthy moments in a Twitch stream. Use the following guidelines to assign a score:

            Classification Scale:

            0 (Not Highlight-Worthy): The message contains no significant emotional reaction, hype, or relevance to noteworthy events. These are generic or filler messages (e.g., "hi," "okay," "yes," or unrelated discussions).
            1 (Mildly Highlight-Worthy): The message shows slight engagement or relevance, such as low-key reactions or basic commentary. These might include casual observations, mild humor, or single keywords/emotes without much context (e.g., "nice," "lol," "close one").
            2 (Moderately Highlight-Worthy): The message indicates stronger engagement, moderate hype, or emotional reaction to events in the stream. This includes brief expressions of excitement, humor, or disappointment related to the stream, often with game-specific or emotional keywords/emotes (e.g., "That was insane!" or "Rip Pog").
            3 (Highly Highlight-Worthy): The message clearly indicates a significant highlight-worthy moment. These messages typically involve intense reactions, collective excitement, or emotional spikes, often supported by context-specific references, multiple emotes, or repeated patterns (e.g., "NO WAY!!! POG POG POG," "What a clutch play!" or "LMAO THAT FAIL WAS AMAZING").
            
            Contextual Notes:

            Twitch chat often reflects real-time events in the stream, with keywords like "clutch," "fail," "win," and emotes like "PogChamp" or "OMEGALUL" signaling viewer reactions.
            Ignore unrelated or low-value messages that do not contribute to identifying highlights.
            Consider emotional intensity, relevance to the stream, and the presence of keywords/emotes to classify appropriately.

            Task: For each Twitch chat message, analyze the text and assign a score (0, 1, 2, or 3) based on the criteria above. Return only the score.
            
            Message: "{message}"
            Output only one number based on the categories above, nothing else."""

            response = model.generate_content(prompt)
            
            # Check if response has content
            if not response.candidates or not response.candidates[0].content:
                print(f"No valid response for message: {message}")
                return 0
                
            score = response.text.strip()
            
            # Validate the score is 0-3
            try:
                score = int(score)
                if score not in [0, 1, 2, 3]:
                    return 0
                return score
            except ValueError:
                return 0

        except Exception as e:
            if "Rate limit" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"An error occurred: {e}")
                if attempt == retries - 1:  # Only on last retry
                    return 0
                time.sleep(1)

    return 0

def process_message(message: str) -> int:
    """Process a single message and return its highlight score."""
    # Check cache first
    if message in highlight_cache:
        return highlight_cache[message]
    
    # Get highlight score
    score = get_highlight_score(message)
    
    # Update cache with new results
    highlight_cache[message] = score
    return score

def process_messages(messages: List[str]) -> List[int]:
    """Process all messages using multiprocessing."""
    with Pool(processes=cpu_count() - 1) as pool:
        all_scores = list(tqdm(pool.imap(process_message, messages), total=len(messages), desc="Processing messages"))
    return all_scores

def main():
    # Set up file paths
    base_path = Path(__file__).parent.parent.parent
    input_file = base_path / 'models' / 'dataset' / 'twitch_chat_preprocessed.csv'
    output_file = base_path / 'models' / 'highlightable' / 'highlight_scores.csv'

    # Read input data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    
    # Process messages
    print("Processing messages...")
    messages = df['message'].tolist()
    scores = process_messages(messages)
    
    # Create results
    results = []
    for idx, (message, score) in enumerate(zip(messages, scores)):
        results.append({
            '': idx,
            'Score': score,
            'Text': message
        })

    # Create output dataframe and save
    output_df = pd.DataFrame(results)
    print(f"Saving results to {output_file}...")
    output_df.to_csv(output_file, index=False)
    
    # Save cache
    print("Saving cache...")
    with open(CACHE_FILE, 'w') as f:
        json.dump(highlight_cache, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
