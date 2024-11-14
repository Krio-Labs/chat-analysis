# twitch_chat_analysis/main.py

import argparse
import logging
from pathlib import Path
import sys
import time
import os

# Add the project root directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# Import the script functions
from twitch_chat_analysis.scripts.twitch_chat_downloader import main as download_chat
from twitch_chat_analysis.scripts.chat_processor import process_chat_data
from twitch_chat_analysis.scripts.chat_sentiment import main as analyze_sentiment
from twitch_chat_analysis.scripts.chat_analysis import analyze_chat_intervals

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
    """Ensure all required directories exist"""
    base_dir = Path(__file__).parent
    required_dirs = ['data', 'outputs', 'dictionary']
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Twitch Chat Analysis Pipeline')
    parser.add_argument('--vod', type=str, help='Twitch VOD URL to analyze')
    return parser.parse_args()

def cleanup_resources():
    """Clean up multiprocessing resources properly"""
    try:
        # Clean up any remaining multiprocessing resources
        import multiprocessing as mp
        if mp.current_process().name == 'MainProcess':
            # Ensure all pools are closed
            mp.get_context('spawn').Pool(1).close()
            
            # Clean up any remaining resource trackers
            import atexit
            atexit._run_exitfuncs()
            
    except Exception as e:
        logging.warning(f"Resource cleanup warning: {str(e)}")

def main():
    """Run the Twitch chat analysis pipeline"""
    total_start_time = time.time()
    setup_logging()
    logging.info("Starting Twitch chat analysis pipeline")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Ensure directories exist
        ensure_directories()
        
        # Step 1: Download chat data
        logging.info("Step 1: Downloading chat data...")
        step_start = time.time()
        try:
            download_chat(args.vod)
            step_time = time.time() - step_start
            logging.info(f"Chat download completed successfully in {step_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Chat download failed: {str(e)}")
            raise
            
        # Step 2: Process chat data
        logging.info("Step 2: Processing chat data...")
        step_start = time.time()
        try:
            process_chat_data()
            step_time = time.time() - step_start
            logging.info(f"Chat processing completed successfully in {step_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Chat processing failed: {str(e)}")
            raise
            
        # Step 3: Sentiment analysis
        logging.info("Step 3: Performing sentiment analysis...")
        step_start = time.time()
        try:
            analyze_sentiment()
            step_time = time.time() - step_start
            logging.info(f"Sentiment analysis completed successfully in {step_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            raise
            
        # Step 4: Final analysis and visualization
        logging.info("Step 4: Generating final analysis and visualizations...")
        step_start = time.time()
        try:
            input_file = "twitch_chat_analysis/outputs/twitch_chat_sentiment_emotion_analysis.csv"
            analyze_chat_intervals(input_file)
            step_time = time.time() - step_start
            logging.info(f"Final analysis completed successfully in {step_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Final analysis failed: {str(e)}")
            raise
            
        # Calculate and log total execution time
        total_time = time.time() - total_start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        logging.info("Analysis pipeline completed successfully!")
        logging.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
    except Exception as e:
        total_time = time.time() - total_start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        logging.error(f"Pipeline failed after {int(hours)}h {int(minutes)}m {seconds:.2f}s: {str(e)}")
        sys.exit(1)
    finally:
        cleanup_resources()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure cleanup happens even if the script is interrupted
        cleanup_resources()
