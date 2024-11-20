import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os
import requests
import io

def train_emotion_classifier():
    """Train the emotion classification model"""
    try:
        # Load local data file
        print("Loading training data...")
        data_path = 'twitch_chat_analysis/models/highlightable/highlight_scores.csv'
        
        if os.path.exists(data_path):
            # Load data into pandas DataFrame
            data = pd.read_csv(data_path)
            print(f"Loaded {len(data)} training examples")
            
            # Print the column names to verify structure
            print("Dataset columns:", data.columns.tolist())
            
            # Print first few rows to verify content
            print("\nFirst few rows of the dataset:")
            print(data.head())
            
            # Get the correct column names from the data
            text_column = data.columns[2]  # First column should be the text
            emotion_column = data.columns[1]  # Second column should be the emotion
            
            # Extract texts and labels
            texts = data[text_column].values
            labels = data[emotion_column].values
            
            print(f"\nUnique emotions in dataset: {np.unique(labels)}")
            
            # Create the pipeline
            emotion_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression(max_iter=1000))
            ])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            print(f"\nTraining on {len(X_train)} examples, testing on {len(X_test)} examples")
            
            # Train the model
            print("Training model...")
            emotion_classifier.fit(X_train, y_train)
            
            # Evaluate
            accuracy = emotion_classifier.score(X_test, y_test)
            print(f"Model accuracy: {accuracy:.2f}")
            
            # Save the model
            os.makedirs('twitch_chat_analysis/models', exist_ok=True)
            model_path = 'twitch_chat_analysis/models/highlightable/highlight_classifier_pipe_lr.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(emotion_classifier, f)
            
            print(f"Model saved to {model_path}")
            return True
            
        else:
            print(f"Error: Could not find the dataset at {data_path}")
            print("Please ensure the emotion_dataset.csv file is in the correct location")
            return False
            
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_emotion_classifier() 