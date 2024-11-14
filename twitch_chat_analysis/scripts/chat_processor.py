import os
import re
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Directory and file setup
script_dir = Path(__file__).parent.parent
data_dir = script_dir / 'data'
dict_dir = script_dir / 'dictionary'
outputs_dir = script_dir / 'outputs'

# Make sure outputs directory exists
os.makedirs(outputs_dir, exist_ok=True)

# File paths
raw_data_file = data_dir / 'twitch_chat_raw.csv'
filtered_data_file = outputs_dir / 'twitch_chat_filtered.csv'
preprocessed_data_file = outputs_dir / 'twitch_chat_preprocessed.csv'
emote_dict_file = dict_dir / 'emote_dictionary.csv'
emoji_dict_file = dict_dir / 'emoji_dictionary.csv'
slang_dict_file = dict_dir / 'slang_dictionary.csv'

# Define filtering patterns
NUMERIC_PATTERN = re.compile(r'^\d+$')
COMMAND_PATTERN = re.compile(r'^[/!].*')
URL_PATTERN = re.compile(r'http[s]?://')
GREETING_PATTERN = re.compile(r'^(hi|hey|hello|sup)\s+\w+$')

ADVERTISEMENT_KEYWORDS = ['buy', 'subscribe', 'follow']
GIFTING_KEYWORDS = ['gift', 'sub', 'donate']
GENERIC_EXPRESSIONS = ['nice', 'good game', 'lol', 'gg']

# Constants and keyword maps for filtering
BOT_NAMES = {
    "nightbot": True,
    "streamelements": True,
    "moobot": True
}

SUBSCRIPTION_KEYWORDS = ADVERTISEMENT_KEYWORDS + GIFTING_KEYWORDS + GENERIC_EXPRESSIONS

def filters(message_data):
    """Filter chat messages based on various criteria."""
    username = message_data['username']
    message = message_data['message']
    
    # Early returns for basic checks
    if pd.isna(message) or not isinstance(message, str) or not message.strip():
        return False
        
    if username.lower() in BOT_NAMES:
        return False
        
    if '@' in message:
        return False
    
    message_lower = message.lower()
    
    # Check against keyword maps
    if any(keyword in message_lower for keyword in SUBSCRIPTION_KEYWORDS):
        return False
    
    # Check against patterns    
    if any([
        NUMERIC_PATTERN.match(message),
        COMMAND_PATTERN.match(message),
        GREETING_PATTERN.match(message_lower),
        URL_PATTERN.search(message)
    ]):
        return False

    return True

# Load dictionaries from CSV files
emote_df = pd.read_csv(emote_dict_file)
emoji_df = pd.read_csv(emoji_dict_file)
slang_df = pd.read_csv(slang_dict_file)

# Convert dictionaries to usable format
emote_dict = pd.Series(emote_df['meaning'].values, index=emote_df['emote']).to_dict()
emoji_dict = pd.Series(emoji_df['meaning'].values, index=emoji_df['emoji']).to_dict()
slang_dict = pd.Series(slang_df['meaning'].values, index=slang_df['slang']).to_dict()

# Define a function to preprocess messages to handle Twitch slang, emotes, emojis, and sarcasm
def preprocess_message(message):
    """Preprocess Twitch chat messages to handle slang, emotes, emojis, and expressions.
    
    Args:
        message: The message to preprocess
        
    Returns:
        str: The preprocessed message
    """
    # Handle non-string messages
    if not isinstance(message, str):
        return ""
        
    # Normalize excessive whitespace
    message = re.sub(r'\s+', ' ', message).strip()
    
    try:
        # Handle repeated patterns with intensity markers
        patterns = [
            # Repeated words with spaces
            (r'(\b\w+\b)(\s*\1\s*){2,}', lambda m: f"{m.group(1)} (repeated, very intense)"),
            
            # Win celebrations - only match standalone W's
            (r'\b(W)\b', "Win!"),  # Single W
            (r'\b(W{2})\b', "Win!"),  # WW
            (r'\b(W{3,5})\b', "huge win, very exciting!"),  # WWW to WWWWW
            (r'\b(W{6,})\b', "massive win, extremely exciting!"),  # WWWWWW or more
            
            # Mixed emotions
            (r'(\!\?)+', "excited but also confused"),
            
            # Common expressions
            (r'\b(GG)\b', "good game, showing sportsmanship"),
            (r'\b(BRB)\b', "be right back, stepping away for a moment"),
            
            # URLs and mentions
            (r'http\S+|www\.\S+', '[link]'),
            (r'@\w+', '[user]'),
            
            # User engagement
            (r'\b(follow|like|subscribe)\b', lambda m: f"asking users to {m.group(1)}")
        ]
        
        # Apply all patterns
        for pattern, replacement in patterns:
            try:
                if callable(replacement):
                    message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
                else:
                    message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
            except re.error as e:
                print(f"Regex error with pattern {pattern}: {e}")
                continue
                
        # Replace emotes, emojis and slang using dictionaries
        try:
            for emote, meaning in emote_dict.items():
                message = re.sub(rf'\b{re.escape(emote)}\b', meaning, message, flags=re.IGNORECASE)
        except Exception as e:
            print(f"Error processing emotes: {e}")
            
        try:
            for emoji, meaning in emoji_dict.items():
                message = message.replace(emoji, meaning)
        except Exception as e:
            print(f"Error processing emojis: {e}")
            
        try:
            for slang, meaning in slang_dict.items():
                message = re.sub(rf'\b{re.escape(slang)}\b', meaning, message, flags=re.IGNORECASE)
        except Exception as e:
            print(f"Error processing slang: {e}")
            
        # Handle excessive repetition
        message = re.sub(r'(\S)\1{20,}', 
                        lambda m: f"{m.group(1)} (extreme repetition)", 
                        message)
                        
        return message.strip()
        
    except Exception as e:
        print(f"Error preprocessing message: {e}")
        return message  # Return original message if processing fails

def process_chat_data():
    """Process the chat data with improved error handling and filtering."""
    try:
        # Load data
        df = pd.read_csv(raw_data_file)
        print(f"Initial row count: {len(df)}")
        
        # Drop NaN values
        df = df.dropna(subset=['message', 'username'])
        print(f"Row count after dropping NaN: {len(df)}")
        
        # Apply filters and create a copy
        filtered_df = df[df.apply(filters, axis=1)].copy()
        print(f"Row count after filtering: {len(filtered_df)}")
        
        # Remove duplicate messages from same user within their last 10 messages
        filtered_df['message_lower'] = filtered_df['message'].str.lower()
        
        def remove_recent_duplicates(group):
            """Remove duplicate messages within last 10 messages for each user."""
            group['is_duplicate'] = False
            for i in range(len(group)):
                current_msg = group.iloc[i]['message_lower']
                # Look at previous 10 messages from this user
                start_idx = max(0, i - 10)
                prev_msgs = group.iloc[start_idx:i]['message_lower']
                if current_msg in prev_msgs.values:
                    group.iloc[i, group.columns.get_loc('is_duplicate')] = True
            return group

        # Group by username and apply duplicate removal
        filtered_df = filtered_df.groupby('username', group_keys=False).apply(remove_recent_duplicates)
        filtered_df = filtered_df[~filtered_df['is_duplicate']].drop(['message_lower', 'is_duplicate'], axis=1)
        print(f"Row count after removing recent duplicates: {len(filtered_df)}")
        
        # Preprocess messages
        filtered_df['message'] = filtered_df['message'].apply(preprocess_message)
        print(f"Final row count: {len(filtered_df)}")
        
        # Save processed data
        filtered_df.to_csv(preprocessed_data_file, index=False)
        print(f"Preprocessed chat data saved to {preprocessed_data_file}")
        
    except Exception as e:
        print(f"Error processing chat data: {e}")

if __name__ == "__main__":
    process_chat_data()