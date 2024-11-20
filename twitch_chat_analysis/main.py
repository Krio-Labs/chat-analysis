# twitch_chat_analysis/main.py

import argparse
import logging
from pathlib import Path
import sys
import time
import os

from scripts.chat_downloader import main as download_chat
from scripts.chat_processor import process_chat_data
from scripts.chat_sentiment import main as analyze_sentiment
from scripts.chat_analysis import main as generate_final_analysis

def setup_logging():
    """Configure logging for the main process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('twitch_analysis.log')
        ]
    )

def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'twitch_chat_analysis/data',
        'twitch_chat_analysis/outputs',
        'twitch_chat_analysis/models'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_vod_url(args):
    """Get VOD URL from args or prompt user"""
    if args.vod:
        return args.vod
    return input("Please enter a Twitch VOD URL: ").strip()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze Twitch chat from a VOD')
    parser.add_argument('--vod', type=str, help='Twitch VOD URL')
    return parser.parse_args()

def cleanup_resources():
    """Cleanup any temporary resources"""
    try:
        # Add cleanup logic here if needed
        pass
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def run_pipeline_step(step_name, step_function, *args):
    """Execute a pipeline step with timing and error handling"""
    logging.info(f"Step {step_name}: Starting...")
    step_start = time.time()
    try:
        result = step_function(*args)
        step_time = time.time() - step_start
        logging.info(f"Step {step_name}: Completed in {step_time:.2f} seconds")
        return result
    except Exception as e:
        logging.error(f"Step {step_name}: Failed - {str(e)}")
        raise

def run_analysis_pipeline(vod_url):
    """Execute the main analysis pipeline steps"""
    steps = [
        ("Download chat", download_chat, vod_url),
        ("Process chat", process_chat_data),
        ("Sentiment analysis", analyze_sentiment),
        ("Final analysis", generate_final_analysis)
    ]
    
    for step_num, (step_name, step_func, *args) in enumerate(steps, 1):
        run_pipeline_step(f"{step_num}: {step_name}", step_func, *args)

def main():
    """Main entry point for the analysis pipeline"""
    total_start_time = time.time()
    
    try:
        # Initialize
        setup_logging()
        logging.info("Starting Twitch chat analysis pipeline")
        args = parse_arguments()
        vod_url = get_vod_url(args)
        ensure_directories()
        
        # Run pipeline
        logging.info(f"Analyzing VOD: {vod_url}")
        run_analysis_pipeline(vod_url)
        
        # Log completion time
        total_time = time.time() - total_start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        logging.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        sys.exit(1)
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()
