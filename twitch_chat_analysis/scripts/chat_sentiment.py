import pandas as pd
from transformers import pipeline
import torch
import platform
import logging
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import psutil
from functools import partial
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_models():
    """Initialize models with appropriate device settings."""
    device = "mps" if torch.backends.mps.is_available() and platform.system() == "Darwin" else (0 if torch.cuda.is_available() else -1)
    
    emotion_classifier = pipeline('text-classification', 
                                model="SamLowe/roberta-base-go_emotions", 
                                device=device,
                                truncation=True,
                                max_length=128,
                                top_k=None)  # Ensure we get all emotion probabilities
    
    return emotion_classifier

def process_chunk(messages, device_id=-1):
    """Process a chunk of messages for emotion analysis."""
    emotion_classifier = pipeline(
        'text-classification', 
        model="SamLowe/roberta-base-go_emotions",
        device=device_id,
        truncation=True,
        max_length=128,
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
    
    results = []
    for message in messages:
        try:
            # Get emotion classification with all probabilities
            emotion_output = emotion_classifier(message)
            
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
            for item in emotion_output[0]:
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
            
        except Exception as e:
            logging.error(f"Error processing message: {message}")
            logging.error(str(e))
            results.append({
                'message': message,
                'sentiment_score': None,
                'excitement': None,
                'funny': None,
                'happiness': None,
                'anger': None,
                'sadness': None,
                'neutral': None
            })
    
    return results

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
        # Set up logging with more detail
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load preprocessed messages
        preprocessed_file_path = 'twitch_chat_analysis/outputs/twitch_chat_preprocessed.csv'
        df_preprocessed = pd.read_csv(preprocessed_file_path)
        
        # Process messages in parallel
        messages = df_preprocessed['message'].values
        available_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)
        num_cpus = max(1, min(physical_cpus, int(available_cpus * 0.8)))
        batch_size = 32
        
        message_batches = np.array_split(messages, len(messages) // batch_size + 1)
        all_results = []
        
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            try:
                process_fn = partial(process_chunk, device_id=-1)
                for i, batch_results in enumerate(executor.map(process_fn, message_batches)):
                    all_results.extend(batch_results)
                    logging.info(f"Processed batch {i + 1}/{len(message_batches)}")
            except Exception as e:
                logging.error(f"Error processing batches: {str(e)}")
                raise e
        
        # Create DataFrame from results and ensure it has the same length as input
        output_df = pd.DataFrame(all_results)
        
        if not output_df.empty:
            # Ensure we have the same number of rows as input
            if len(output_df) != len(df_preprocessed):
                logging.error(f"Mismatch in results length: got {len(output_df)}, expected {len(df_preprocessed)}")
                return
                
            # Add index for merging
            output_df.index = range(len(output_df))
            df_preprocessed.index = range(len(df_preprocessed))
            
            # Merge the dataframes on index
            final_df = pd.concat([
                df_preprocessed[['time', 'username', 'message']],
                output_df.drop('message', axis=1)
            ], axis=1)
            
            # Round scores for better readability
            score_columns = [col for col in final_df.columns 
                           if col not in ['time', 'username', 'message']]
            final_df[score_columns] = final_df[score_columns].round(3)
            
            # Save results
            output_file = 'twitch_chat_analysis/outputs/twitch_chat_sentiment_emotion_analysis.csv'
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
