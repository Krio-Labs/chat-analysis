import pandas as pd
import re
import os
from pathlib import Path

# Get the script's directory and create paths relative to it
script_dir = Path(__file__).parent.parent  # Go up one level to reach twitch_chat_analysis
dictionary_dir = script_dir / 'dictionary'

# Create the data directory if it doesn't exist
os.makedirs(dictionary_dir, exist_ok=True)

# Update file paths to use the data directory
emote_file_path = dictionary_dir / 'emote_dictionary.csv'
emoji_file_path = dictionary_dir / 'emoji_dictionary.csv'
slang_file_path = dictionary_dir / 'slang_dictionary.csv'

# Create emote dictionary with expanded popular Twitch emotes
emote_data = {
    'emote': [
        'Kappa', 'PogChamp', 'FeelsBadMan', 'FeelsGoodMan', 'BibleThump', 'LUL', 'ResidentSleeper', 'W', 'L', 'Kreygasm', 'PepeHands',
        'OMEGALUL', 'PogU', 'Sadge', 'Pepega', 'monkaS', 'TriHard', '4Head', 'AYAYA', 'BlessRNG', 'CoolStoryBob', 'CmonBruh', 'Clap',
        'KappaPride', 'gachiBASS', 'gachiHYPER', 'PepeLaugh', 'Kekw', 'PepoG', 'CatJAM', 'FeelsBirthdayMan', 'PogO', 'FeelsWeirdMan',
        'OkayChamp', 'HYPERS', 'PepeD', 'HandsUp', 'POGGERS', 'peepoHappy'
    ],
    'meaning': [
        'being sarcastic, not serious', 'super hype, very exciting', 'feeling really sad', 'feeling happy and content', 'crying out of sadness', 'laughing hard', 'extremely boring', 'big win, great success', 'huge loss, disappointing', 'intense excitement', 'feeling sad and helpless',
        'uncontrollable laughter', 'really hyped and excited', 'feeling a bit down', 'acting foolish', 'feeling nervous', 'super pumped', 'simple joke but funny', 'feeling very excited', 'hoping for the best', 'interesting but questionable story', 'skeptical reaction', 'showing applause',
        'pride in identity', 'extremely awesome', 'high energy', 'laughing loudly', 'absolutely hilarious', 'mind blown', 'vibing with it', 'celebrating birthday', 'excited but uneasy', 'feeling awkward',
        'it’s okay', 'feeling hyper and excited', 'dancing with energy', 'celebrating a big moment', 'super pumped up', 'feeling genuinely happy'
    ]
}
emote_df = pd.DataFrame(emote_data)
emote_df.to_csv(emote_file_path, index=False)

# Create emoji dictionary with expanded popular emojis
emoji_data = {
    'emoji': [
        '😂', '😍', '😢', '😡', '🔥', '💀', '👍', '🍔', '🙏', '👑', '💩', '😎', '😭', '🤔', '😱', '🥳', '🤯', '🤡', '🤬', '🥺', '👀', '🤗', '😇', '🤤', '😴', '😈', '🤮', '😵', '🙌', '👏', '🤝', '💪', '🤑'
    ],
    'meaning': [
        'laughing uncontrollably', 'feeling love and admiration', 'deep sadness', 'extremely angry', 'something awesome or exciting', 'laughing so hard it hurts', 'thumbs up, approval', 'delicious food', 'hoping for good outcome', 'feeling like royalty', 'this is bad or awful', 'feeling cool and confident', 'crying hard', 'deep thinking', 'shocked and surprised', 'celebration time!', 'mind completely blown', 'feeling silly or foolish', 'very angry', 'pleading or begging', 'watching closely', 'sending a hug', 'feeling blessed', 'craving this', 'very tired or sleepy', 'feeling mischievous', 'disgusted', 'feeling dizzy or overwhelmed', 'celebration success', 'applause for good work', 'making an agreement', 'feeling strong', 'thinking about money'
    ]
}
emoji_df = pd.DataFrame(emoji_data)
emoji_df.to_csv(emoji_file_path, index=False)

# Create slang dictionary with expanded popular Twitch slangs
slang_data = {
    'slang': [
        'GG', 'EZ', 'bruh', 'hype', 'salty', 'cringe', 'rekt', 'noob', 'kek', 'pepelaugh', 'pog', 'monkas', 'smh', 'based', 'yeet', 'boomer', 'zoomie', 'copium', 'cracked', 'sus', 'pepega', 'poggers', 'f', 'bop', 'malding', 'dank', 'weeb', 'chad', 'glhf', 'omega'
    ],
    'meaning': [
        'good game, well played', 'too easy, no challenge', 'are you serious?', 'let’s get excited!', 'feeling upset', 'extremely awkward', 'completely defeated', 'new or inexperienced player', 'laughing out loud', 'sarcastic laughter', 'amazing!', 'feeling anxious or nervous', 'shaking my head in disbelief', 'authentic and true', 'throwing it away confidently', 'old-fashioned person', 'young and energetic', 'trying to cope with difficulty', 'playing extremely well', 'that’s suspicious', 'acting very foolish', 'incredibly exciting!', 'showing respect', 'great move!', 'angry and balding at the same time', 'really cool', 'anime lover', 'confident and assertive', 'good luck, have fun', 'that’s massive, really impressive'
    ]
}
slang_df = pd.DataFrame(slang_data)
slang_df.to_csv(slang_file_path, index=False)

# Function to preprocess messages to handle repetition of emotes, emojis, and slangs
def preprocess_repeated_chars(message):
    # Replace consecutive repeated emojis/emotes/slangs with a single instance
    message = re.sub(r'(\b\w+\b)(\s*\1\s*)+', r'\1', message)  # Handle repeated words like "W W W" or "LOL LOL"
    message = re.sub(r'(.)\1{2,}', r'\1', message)  # Handle repeated characters like "!!!!!" or "?????"
    return message

print(f"Dictionaries created and saved to: \n{emote_file_path}\n{emoji_file_path}\n{slang_file_path}")
