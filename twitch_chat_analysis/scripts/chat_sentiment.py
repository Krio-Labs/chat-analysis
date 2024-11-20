import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
from functools import partial
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define emotion clusters mapping
emotion_clusters = {
    'excitement': ['Excitement'],
    'funny': ['Humor'],
    'happiness': ['Joy'],
    'anger': ['Anger'],
    'sadness': ['Sadness'],
    'neutral': ['Neutral']
}

def load_models():
    """Load both emotion and highlight classifier models."""
    try:
        with open('twitch_chat_analysis/models/emotion/emotion_classifier_pipe_lr.pkl', 'rb') as f:
            emotion_model = pickle.load(f)
        with open('twitch_chat_analysis/models/highlightable/highlight_classifier_pipe_lr.pkl', 'rb') as f:
            highlight_model = pickle.load(f)
        return emotion_model, highlight_model
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def process_chunk(messages, models=None):
    """Process a chunk of messages using both classifiers"""
    if models is None:
        models = load_models()
    emotion_model, highlight_model = models
    
    try:
        # Get predictions from both models
        emotion_predictions = emotion_model.predict(messages)
        emotion_probabilities = emotion_model.predict_proba(messages)
        highlight_scores = highlight_model.predict(messages)
        
        results = []
        for message, prediction, probs, highlight_score in zip(
            messages, emotion_predictions, emotion_probabilities, highlight_scores
        ):
            # Create base result dictionary with highlight score
            result = {
                'message': message,
                'sentiment_score': 0.0,
                'highlight_score': float(highlight_score),  # Add highlight score
                'excitement': 0.0,
                'funny': 0.0,
                'happiness': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'neutral': 0.0
            }
            
            # Get the probability for each emotion class
            emotion_probs = dict(zip(emotion_model.classes_, probs))
            
            # Calculate cluster scores
            for cluster_name, emotions in emotion_clusters.items():
                cluster_score = sum(emotion_probs.get(emotion, 0.0) for emotion in emotions)
                result[cluster_name] = round(cluster_score, 3)
            
            # Calculate sentiment score
            pos_emotions = ['Excitement', 'Joy', 'Humor']
            neg_emotions = ['Anger', 'Sadness']
            
            pos_score = sum(emotion_probs.get(emotion, 0.0) for emotion in pos_emotions)
            neg_score = sum(emotion_probs.get(emotion, 0.0) for emotion in neg_emotions)
            
            sentiment_score = pos_score - neg_score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
            result['sentiment_score'] = round(sentiment_score, 3)
            
            results.append(result)
            
        return results
        
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        return [{'message': m, 'sentiment_score': None, 'highlight_score': None,
                'excitement': None, 'funny': None, 'happiness': None, 
                'anger': None, 'sadness': None, 'neutral': None} for m in messages]

def main():
    try:
        # Load both models once
        models = load_models()
        
        # Initialize an empty DataFrame to store all results
        all_final_df = pd.DataFrame()
        
        # Load data in chunks to reduce memory usage
        chunk_size = 10000
        for chunk_num, chunk in enumerate(pd.read_csv('twitch_chat_analysis/data/twitch_chat_preprocessed.csv', 
                                                    chunksize=chunk_size)):
            
            logging.info(f"Processing chunk {chunk_num + 1}")
            messages = chunk['message'].values
            
            # Calculate optimal batch size based on available memory
            mem = psutil.virtual_memory()
            available_mem = mem.available / (1024 * 1024 * 1024)  # Convert to GB
            batch_size = min(256 if available_mem > 16 else 128, len(messages))
            
            # Create batches for processing
            message_batches = np.array_split(messages, max(1, len(messages) // batch_size))
            
            # Process batches using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                process_fn = partial(process_chunk, models=models)
                results = list(executor.map(process_fn, message_batches))
            
            # Flatten results
            all_results = [item for batch_result in results for item in batch_result]
            
            # Create DataFrame from results
            output_df = pd.DataFrame(all_results)
            
            if not output_df.empty:
                # Add index for merging
                output_df.index = range(len(output_df))
                chunk.index = range(len(chunk))
                
                # Merge the dataframes
                chunk_final_df = pd.concat([
                    chunk[['time', 'username', 'message']],
                    output_df.drop('message', axis=1)
                ], axis=1)
                
                # Round scores
                score_columns = [col for col in chunk_final_df.columns 
                               if col not in ['time', 'username', 'message']]
                chunk_final_df[score_columns] = chunk_final_df[score_columns].round(3)
                
                # Append to main DataFrame
                all_final_df = pd.concat([all_final_df, chunk_final_df], ignore_index=True)
                
                logging.info(f"Processed {len(all_final_df)} messages so far")
            else:
                logging.error(f"No results were generated for chunk {chunk_num + 1}")
        
        if not all_final_df.empty:
            # Save complete results
            output_file = 'twitch_chat_analysis/data/twitch_chat_sentiment_emotion_analysis.csv'
            all_final_df.to_csv(output_file, index=False)
            logging.info(f"Analysis completed. Total messages processed: {len(all_final_df)}")
            logging.info(f"Results saved to {output_file}")
        else:
            logging.error("No results were generated from the analysis")
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main()
