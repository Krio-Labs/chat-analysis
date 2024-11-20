import pandas as pd
import time
from pathlib import Path
import sys
from tqdm import tqdm
import json
from typing import List
from multiprocessing import Pool, cpu_count
import google.generativeai as genai

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
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
        emotion_cache = json.load(f)
except FileNotFoundError:
    emotion_cache = {}

def get_emotion_label(message: str) -> str:
    """Get the emotion label for a single message using Google Gemini API."""
    retries = 5
    for attempt in range(retries):
        try:
            prompt = f"""Classify the emotion expressed in the following Twitch chat message into exactly one of the following categories: Excitement, Humor, Joy, Anger, Sadness, or Neutral.

            When classifying, consider the following:

            Twitch Context: Account for popular Twitch slang, emotes (e.g., "PogChamp," "Kappa"), and emojis, as they often convey nuanced emotions.
            Sarcasm & Irony: Be attentive to sarcastic or ironic tones, which are common in Twitch chat and can modify the intended emotion.
            Ambiguity: If the message is ambiguous, use your best judgment to select the most fitting category based on the context and sentiment expressed.
            One-Word Response: Provide only the category name as the output (e.g., "Excitement").

            Message: "{message}"
            Output only one word from the categories above, nothing else."""

            response = model.generate_content(prompt)
            
            # Check if response has content
            if not response.candidates or not response.candidates[0].content:
                print(f"No valid response for message: {message}")
                return "Neutral"
                
            emotion = response.text.strip()
            
            # Validate the emotion is one of our categories
            valid_emotions = {"Excitement", "Humor", "Joy", "Anger", "Sadness", "Neutral"}
            if emotion not in valid_emotions:
                return "Neutral"
                
            return emotion

        except Exception as e:
            if "Rate limit" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"An error occurred: {e}")
                if attempt == retries - 1:  # Only on last retry
                    return "Neutral"
                time.sleep(1)  # Wait a bit before retrying

    return "Neutral"

def process_message(message: str) -> str:
    """Process a single message and return its emotion."""
    # Check cache first
    if message in emotion_cache:
        return emotion_cache[message]
    
    # Get emotion label
    emotion = get_emotion_label(message)
    
    # Update cache with new results
    emotion_cache[message] = emotion
    return emotion

def process_messages(messages: List[str]) -> List[str]:
    """Process all messages using multiprocessing."""
    with Pool(processes=cpu_count() - 1) as pool:
        all_emotions = list(tqdm(pool.imap(process_message, messages), total=len(messages), desc="Processing messages"))
    return all_emotions

def main():
    # Set up file paths
    base_path = Path(__file__).parent.parent.parent
    input_file = base_path / 'models' / 'dataset' / 'twitch_chat_preprocessed.csv'
    output_file = base_path / 'models' / 'emotion'/ 'emotion_dataset.csv'

    # Read input data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    
    # Process messages
    print("Processing messages...")
    messages = df['message'].tolist()  # Process all messages
    emotions = process_messages(messages)
    
    # Create results
    results = []
    for idx, (message, emotion) in enumerate(zip(messages, emotions)):
        results.append({
            '': idx,
            'Emotion': emotion,
            'Text': message
        })

    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save to CSV
    print(f"Saving results to {output_file}...")
    output_df.to_csv(output_file, index=False)
    
    # Save cache
    print("Saving cache...")
    with open(CACHE_FILE, 'w') as f:
        json.dump(emotion_cache, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
