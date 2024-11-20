import os
import re
import pandas as pd
from datetime import timedelta
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Use absolute import
from twitch_chat_analysis.archived import chat_dictionary

# Directory and file setup
script_dir = Path(__file__).parent.parent.parent
data_dir = script_dir / 'models' / 'dataset'
dict_dir = script_dir / 'dictionary'
outputs_dir = script_dir / 'models' / 'dataset'

# Make sure outputs directory exists
os.makedirs(outputs_dir, exist_ok=True)

# Run the dictionary creation
try:
    chat_dictionary.create_dictionaries()  # Call the function that creates the dictionaries
except Exception as e:
    print(f"Error running dictionary creation: {e}")

# File paths
raw_data_file = data_dir / 'twitch_chat_sampled_150k.csv'
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

def load_dictionaries():
    """Load and validate all dictionaries."""
    try:
        # Load dictionaries from CSV files
        emote_df = pd.read_csv(emote_dict_file)
        emoji_df = pd.read_csv(emoji_dict_file)
        slang_df = pd.read_csv(slang_dict_file)
        
        # Convert to usable format with error checking
        dictionaries = {
            'emote': pd.Series(emote_df['meaning'].values, index=emote_df['emote']).to_dict(),
            'emoji': pd.Series(emoji_df['meaning'].values, index=emoji_df['emoji']).to_dict(),
            'slang': pd.Series(slang_df['meaning'].values, index=slang_df['slang']).to_dict()
        }
        
        print("Successfully loaded all dictionaries")
        return dictionaries
        
    except FileNotFoundError as e:
        print(f"Dictionary file not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading dictionaries: {e}")
        return None

def apply_dictionaries(message, dictionaries):
    """Apply dictionary replacements to a message."""
    try:
        if not isinstance(message, str):
            return ""
            
        # Replace emotes with word boundaries
        for emote, meaning in dictionaries['emote'].items():
            message = re.sub(rf'\b{re.escape(emote)}\b', meaning, message, flags=re.IGNORECASE)
            
        # Replace emojis (direct replacement)
        for emoji, meaning in dictionaries['emoji'].items():
            message = message.replace(emoji, meaning)
            
        # Replace slang with word boundaries
        for slang, meaning in dictionaries['slang'].items():
            message = re.sub(rf'\b{re.escape(slang)}\b', meaning, message, flags=re.IGNORECASE)
            
        return message.strip()
        
    except Exception as e:
        print(f"Error applying dictionaries: {e}")
        return message

def preprocess_message(message, dictionaries=None):
    """Preprocess Twitch chat messages with optional dictionary replacement."""
    if not isinstance(message, str):
        return ""
        
    # Normalize whitespace
    message = re.sub(r'\s+', ' ', message).strip()
    
    try:
        # Apply pattern replacements (keep existing patterns code)
        patterns = [
            # Win celebrations - only match standalone W's
            (r'\b(W)\b', "Yes!"),  # Single W
            (r'\b(W{2})\b', "YES!"),  # WW
            (r'\b(W{3,5})\b', "YES!!"),  # WWW to WWWWW
            (r'\b(W{6,})\b', "YES!!!!!"),  # WWWWWW or more
            
            # Loss expressions - only match standalone L's
            (r'\b(L)\b', "😔"),  # Single L
            (r'\b(L{2})\b', "😔😔"),  # LL
            (r'\b(L{3,5})\b', "😔😔😔"),  # LLL to LLLLL
            (r'\b(L{6,})\b', "😔😔😔😔😔😔"),  # LLLLLL or more
            
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
        
        # Apply dictionary replacements if dictionaries are provided
        if dictionaries:
            message = apply_dictionaries(message, dictionaries)
            
        # Handle excessive repetition
        message = re.sub(r'(\S)\1{20,}', 
                        lambda m: f"{m.group(1)} (extreme repetition)", 
                        message)
                        
        return message.strip()
        
    except Exception as e:
        print(f"Error preprocessing message: {e}")
        return message

def process_chat_data():
    """Process the chat data with improved error handling and filtering."""
    try:
        # Load dictionaries first
        dictionaries = load_dictionaries()
        
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
        
        # Preprocess messages with dictionaries
        filtered_df['message'] = filtered_df['message'].apply(
            lambda x: preprocess_message(x, dictionaries)
        )
        print(f"Final row count: {len(filtered_df)}")
        
        # Save processed data
        filtered_df.to_csv(preprocessed_data_file, index=False)
        print(f"Preprocessed chat data saved to {preprocessed_data_file}")
        
    except Exception as e:
        print(f"Error processing chat data: {e}")

if __name__ == "__main__":
    process_chat_data()
