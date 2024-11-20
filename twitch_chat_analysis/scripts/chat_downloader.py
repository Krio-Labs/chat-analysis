import csv
import logging
import sys
import re
import argparse
from pathlib import Path
from chat_downloader import ChatDownloader
from tqdm import tqdm
import multiprocessing as mp
import psutil

def extract_vod_id(vod_link):
    # Example URL: https://www.twitch.tv/videos/2280260735
    return vod_link.rstrip('/').split('/')[-1]

def fetch_chat_data(vod_link):
    """
    Fetches chat data from a Twitch VOD using optimized parameters.
    
    Parameters based on: https://chat-downloader.readthedocs.io/en/latest/cli.html#command-line-usage
    """
    try:
        downloader = ChatDownloader()
        
        # Use only documented parameters for Twitch
        chat = downloader.get_chat(
            url=vod_link,
            max_attempts=15,  # Default value from docs
            retry_timeout=2,
            message_groups=['messages'],
            message_receive_timeout=0.1,  # Twitch-specific parameter
            buffer_size=32768  # Increased from default 4096 for better performance
        )
        
        logging.info(f"Successfully initialized chat download for VOD: {vod_link}")
        return chat
        
    except Exception as e:
        logging.error(f"Failed to fetch chat data: {str(e)}")
        raise

def process_message_batch(messages):
    """
    Process a batch of messages with memory optimization
    """
    processed = []
    for message in messages:
        if message is None:
            continue
            
        try:
            # Extract only needed fields to reduce memory usage
            processed.append([
                message.get('time_in_seconds', 0),
                message.get('author', {}).get('name', 'Unknown'),
                message.get('message', '')
            ])
                
        except (AttributeError, TypeError) as e:
            logging.warning(f"Skipping invalid message format: {e}")
            continue
            
    # Clear message batch from memory
    messages.clear()
    return processed

def write_chat_to_csv(chat, output_file='/Users/aman/Documents/KrioGit/Chat Analyzer/twitch_chat_analysis/data/twitch_chat_raw.csv'):
    """
    Writes chat data to CSV using optimized batch processing and multiprocessing
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optimize CPU usage
    available_cpus = psutil.cpu_count(logical=True)
    num_cpus = max(1, min(available_cpus - 1, 8))  # Use up to 8 cores, leave 1 for system
    
    # Increase batch size for better throughput
    BATCH_SIZE = 5000  # Increased from 5000
    message_batch = []
    total_messages = 0
    
    try:
        with output_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'username', 'message'])
            
            with mp.Pool(processes=num_cpus) as pool:
                pbar = tqdm(desc="Processing messages", unit="msgs")
                
                # Process messages in chunks
                chunks = []
                for message in chat:
                    message_batch.append(message)
                    
                    if len(message_batch) >= BATCH_SIZE:
                        chunks.append(message_batch)
                        message_batch = []
                        
                        if len(chunks) >= num_cpus:
                            # Process multiple batches in parallel
                            results = pool.map(process_message_batch, chunks)
                            for batch_results in results:
                                for msg in batch_results:
                                    writer.writerow(msg)
                                    
                            total_messages += sum(len(batch) for batch in chunks)
                            pbar.update(total_messages - pbar.n)
                            chunks = []
                
                # Process remaining messages
                if message_batch:
                    chunks.append(message_batch)
                if chunks:
                    results = pool.map(process_message_batch, chunks)
                    for batch_results in results:
                        for msg in batch_results:
                            writer.writerow(msg)
                            
                    total_messages += sum(len(batch) for batch in chunks)
                    pbar.update(total_messages - pbar.n)
                
                pbar.close()
                
        logging.info(f"Successfully processed {total_messages} messages")
        
    except Exception as e:
        logging.error(f"Error processing messages: {str(e)}")
        raise

def is_valid_vod_link(vod_link):
    """
    Validates the Twitch VOD link format.
    
    Args:
        vod_link (str): The VOD link to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    pattern = r'^https:\/\/www\.twitch\.tv\/videos\/\d+$'
    return re.match(pattern, vod_link) is not None

def main(vod_url=None):
    """Download chat data from a Twitch VOD.
    
    Args:
        vod_url (str, optional): URL of the Twitch VOD to analyze.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('twitch_chat_downloader.log')
        ]
    )
    logging.info('Process started.')

    # If no VOD link provided, prompt user for input
    if vod_url is None:
        vod_url = input("Please enter a Twitch VOD URL: ").strip()
        if not vod_url:
            raise ValueError("VOD URL is required")

    # Validate the VOD link
    if not is_valid_vod_link(vod_url):
        error_msg = "Invalid Twitch VOD link format. Please use format: 'https://www.twitch.tv/videos/1234567890'"
        logging.error(error_msg)
        raise ValueError(error_msg)

    try:
        logging.info(f"Fetching chat data from VOD: {vod_url}")
        chat = fetch_chat_data(vod_url)
        logging.info("Chat data fetching initiated.")
        
        logging.info("Writing chat data to CSV...")
        write_chat_to_csv(chat)
        
        logging.info('CSV file created successfully.')
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    logging.info('Process completed successfully.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Twitch chat data from a VOD')
    parser.add_argument('--vod', type=str, help='Twitch VOD URL')
    args = parser.parse_args()
    
    try:
        main(args.vod)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)
