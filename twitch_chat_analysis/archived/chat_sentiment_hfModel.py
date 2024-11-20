import pandas as pd
from transformers import pipeline
import torch
import platform
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import psutil
from functools import partial
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_models():
    """Initialize models with optimized settings."""
    if torch.cuda.is_available():
        device = 0
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = "mps"
    else:
        device = -1
    
    # Load model once and share across processes
    emotion_classifier = pipeline(
        'text-classification', 
        model="SamLowe/roberta-base-go_emotions",
        device=device,
        truncation=True,
        max_length=128,
        top_k=None,
        batch_size=32
    )
    
    return emotion_classifier

def process_chunk(messages, device_id=-1):
    """Process a chunk of messages with optimized batch size"""
    # Convert messages to list of strings if not already
    messages = [str(msg) if msg is not None else "" for msg in messages]
    
    # Calculate optimal batch size based on available memory
    mem = psutil.virtual_memory()
    available_mem = mem.available / (1024 * 1024 * 1024)  # Convert to GB
    optimal_batch_size = min(256 if available_mem > 16 else 128, len(messages))
    
    emotion_classifier = pipeline(
        'text-classification', 
        model="SamLowe/roberta-base-go_emotions",
        device=device_id,
        truncation=True,
        max_length=128,
        batch_size=optimal_batch_size,  # Use dynamic batch size
        top_k=None
    )
    
    # Define emotion clusters
    emotion_clusters = {
        'excitement': ['admiration', 'excitement', 'curiosity', 'surprise', 'desire', 'pride', 'optimism'],
        'funny': ['amusement', 'confusion', 'embarrassment', 'relief', 'surprise'],
        'happiness': ['joy', 'gratitude', 'approval', 'love', 'caring', 'relief', 'optimism'],
        'anger': ['anger', 'annoyance', 'disapproval', 'disgust', 'embarrassment', 'remorse'],
        'sadness': ['sadness', 'disappointment', 'grief', 'remorse', 'fear', 'nervousness'],
        'neutral': ['neutral', 'realization']
    }
    
    try:
        # Process entire batch at once instead of one message at a time
        emotion_outputs = emotion_classifier(messages, batch_size=32)
        
        results = []
        for message, emotion_output in zip(messages, emotion_outputs):
            # Create a dictionary with message and initial scores for all emotions
            result = {
                'message': message,
                'admiration': 0, 'amusement': 0, 'anger': 0, 'annoyance': 0,
                'approval': 0, 'caring': 0, 'confusion': 0, 'curiosity': 0,
                'desire': 0, 'disappointment': 0, 'disapproval': 0, 'disgust': 0,
                'embarrassment': 0, 'excitement': 0, 'fear': 0, 'gratitude': 0,
                'grief': 0, 'joy': 0, 'love': 0, 'nervousness': 0,
                'optimism': 0, 'pride': 0, 'realization': 0, 'relief': 0,
                'remorse': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0
            }
            
            # Update emotion scores from model output
            for item in emotion_output:
                result[item['label']] = float(item['score'])
            
            # Calculate cluster scores
            cluster_scores = {}
            for cluster_name, emotions in emotion_clusters.items():
                # Handle emotions that appear in multiple clusters by taking their maximum contribution
                cluster_score = sum(result[emotion] for emotion in emotions)
                # Normalize by number of emotions in cluster to get average
                cluster_scores[cluster_name] = round(cluster_score / len(emotions), 3)
            
            # Calculate sentiment score (using original positive/negative emotions)
            pos_emotions = ['admiration', 'amusement', 'approval', 'gratitude', 
                          'joy', 'love', 'optimism', 'pride', 'relief']
            neg_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 
                          'disgust', 'embarrassment', 'fear', 'grief', 
                          'nervousness', 'remorse', 'sadness']
            
            pos_score = sum(result[emotion] for emotion in pos_emotions)
            neg_score = sum(result[emotion] for emotion in neg_emotions)
            
            sentiment_score = pos_score - neg_score
            if sentiment_score > 1.0:
                sentiment_score = 1.0
            if sentiment_score < -1.0:
                sentiment_score = -1.0
            
            # Add sentiment score and cluster scores to result
            result = {
                'message': message,
                'sentiment_score': round(sentiment_score, 3),
                **cluster_scores  # Add all cluster scores
            }
            
            results.append(result)
            
        return results
        
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        return [{'message': m, 'sentiment_score': None, 'excitement': None, 
                'funny': None, 'happiness': None, 'anger': None, 
                'sadness': None, 'neutral': None} for m in messages]

def clean_up_workers():
    """Kill any remaining worker processes and cleanup resources."""
    try:
        # Kill child processes
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Clean up multiprocessing resources - Python 3.12 compatible
        import multiprocessing.resource_tracker
        try:
            # Get the resource tracker singleton
            tracker = multiprocessing.resource_tracker._resource_tracker
            if tracker is not None:
                tracker.clear()
        except Exception as e:
            logging.debug(f"Resource tracker cleanup skipped: {str(e)}")

    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def main():
    try:
        # Load data in chunks to reduce memory usage
        chunk_size = 10000
        for chunk in pd.read_csv('twitch_chat_analysis/outputs/twitch_chat_preprocessed.csv', 
                               chunksize=chunk_size):
            
            # Convert messages to list of strings and handle NaN/None values
            messages = [str(msg) if msg is not None else "" for msg in chunk['message'].values]
            
            # Optimize batch size based on available memory
            mem = psutil.virtual_memory()
            available_mem = mem.available / (1024 * 1024 * 1024)  # Convert to GB
            batch_size = min(256 if available_mem > 16 else 128, len(messages))
            
            # Create batches as list of lists
            message_batches = [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]
            
            # Process batches
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                process_fn = partial(process_chunk, device_id=0 if torch.cuda.is_available() else -1)
                batch_results = list(executor.map(process_fn, message_batches))
                # Flatten results
                for batch_result in batch_results:
                    results.extend(batch_result)
            
            # Create DataFrame from results
            output_df = pd.DataFrame(results)
            
            if not output_df.empty:
                # Ensure we have the same number of rows as input
                if len(output_df) != len(chunk):
                    logging.error(f"Mismatch in results length: got {len(output_df)}, expected {len(chunk)}")
                    continue
                    
                # Add index for merging
                output_df.index = range(len(output_df))
                chunk.index = range(len(chunk))
                
                # Merge the dataframes
                final_df = pd.concat([
                    chunk[['time', 'username', 'message']],
                    output_df.drop('message', axis=1)
                ], axis=1)
                
                # Round scores for better readability
                score_columns = [col for col in final_df.columns 
                               if col not in ['time', 'username', 'message']]
                final_df[score_columns] = final_df[score_columns].round(3)
                
                # Save results
                output_file = 'twitch_chat_analysis/outputs/twitch_chat_sentiment_emotion_analysis_hfModel.csv'
                final_df.to_csv(output_file, index=False)
                logging.info(f"Analysis completed. Results saved to {output_file}")
            else:
                logging.error("No results were generated from the analysis")
                
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
    finally:
        clean_up_workers()

if __name__ == '__main__':
    try:
        main()
    finally:
        clean_up_workers()  # Ensure cleanup happens even on script exit
