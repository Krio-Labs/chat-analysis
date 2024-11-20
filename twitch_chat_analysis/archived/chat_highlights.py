#!/usr/bin/env python3
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def analyze_chat_highlights(input_file, output_dir):
    """
    Analyze chat data to identify highlight moments and emotional peaks.

    Args:
        input_file (str): Path to the input CSV file containing chat analysis data.
        output_dir (str): Directory where output files should be saved.
    """
    # Check if file exists
    if not os.path.exists(input_file):
        logging.error(f"Required file not found: {input_file}")
        logging.info("Please run the full pipeline with a VOD URL first to generate the required data files.")
        return None

    # Load data
    data = pd.read_csv(input_file)

    # Remove any leading/trailing whitespaces in column names
    data.columns = data.columns.str.strip()

    # Display available columns
    print("Available columns in the original data:", list(data.columns))

    # Check for missing necessary columns
    required_columns = [
        'time', 'sentiment_score', 'highlight_score',
        'excitement', 'funny', 'happiness', 'anger', 'sadness', 'neutral', 'message'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns in data: {missing_columns}")
        return None

    # Convert 'time' to datetime if it's not already
    if not np.issubdtype(data['time'].dtype, np.datetime64):
        data['time'] = pd.to_datetime(data['time'], unit='s')  # Adjust unit as necessary

    # Group by 'time' and aggregate the necessary columns
    grouped_data = data.groupby('time').agg({
        'sentiment_score': 'mean',
        'highlight_score': 'mean',
        'excitement': 'mean',
        'funny': 'mean',
        'happiness': 'mean',
        'anger': 'mean',
        'sadness': 'mean',
        'neutral': 'mean',
        'message': 'count'  # Count the number of messages per time window
    }).rename(columns={'message': 'message_count'})

    # Compute weighted_highlight_score based on highlight_score and message_count
    # Adjust the weights according to your requirements
    grouped_data['weighted_highlight_score'] = (
        grouped_data['highlight_score'] * 0.7 +  # Weight for highlight_score
        grouped_data['message_count'] * 0.3     # Weight for message_count (chat density)
    )

    # Detect peaks in weighted_highlight_score
    peaks, properties = find_peaks(
        grouped_data['weighted_highlight_score'],
        prominence=0.7,    # Adjusted prominence
        height=0.1,        # Adjusted minimum height
        width=1            # Minimum width of 1 window
    )

    # Create a DataFrame with all peak moments
    peak_windows = grouped_data.iloc[peaks].copy()
    peak_windows['prominence'] = properties['prominences']

    # Sort peaks by weighted_highlight_score first, then prominence
    peak_windows = peak_windows.sort_values(
        by=['weighted_highlight_score', 'prominence'],
        ascending=[False, False]
    )

    # Function to filter peaks with minimum time distance (10 minutes)
    def filter_peaks_with_distance(peaks_df, min_distance_secs=600):
        selected_peaks = []
        used_times = []

        for idx, row in peaks_df.iterrows():
            current_time = row.name.timestamp()  # Assuming 'time' is datetime
            if not any(abs(current_time - used_time) < min_distance_secs for used_time in used_times):
                selected_peaks.append(idx)
                used_times.append(current_time)

        return peaks_df.loc[selected_peaks]

    # Get top peaks while maintaining minimum 10 minutes distance
    top_windows = filter_peaks_with_distance(peak_windows, min_distance_secs=600).head(10)

    # Sort final results by time for clearer output
    top_windows = top_windows.sort_values('time')

    # Save top highlights to CSV
    top_highlights_file = os.path.join(output_dir, "top_highlights.csv")
    top_windows.to_csv(top_highlights_file)
    print(f"Top highlights saved to {top_highlights_file}")

    # Identify and save top 5 prominent time regions for each emotion
    emotions_of_interest = ['funny', 'anger', 'excitement', 'happiness', 'sadness']
    for emotion in emotions_of_interest:
        # Detect peaks in the specific emotion
        emotion_peaks, emotion_properties = find_peaks(
            grouped_data[emotion],
            prominence=0.5,    # Adjusted prominence for emotion
            height=0.1,        # Adjusted minimum height for emotion
            width=1            # Minimum width of 1 window
        )

        # Create a DataFrame with emotion peak moments
        emotion_peak_windows = grouped_data.iloc[emotion_peaks].copy()
        emotion_peak_windows['prominence'] = emotion_properties['prominences']

        # Sort peaks by emotion score first, then prominence
        emotion_peak_windows = emotion_peak_windows.sort_values(
            by=[emotion, 'prominence'],
            ascending=[False, False]
        )

        # Filter peaks with minimum time distance
        top_emotion_peaks = filter_peaks_with_distance(emotion_peak_windows, min_distance_secs=600).head(5)

        # Sort by time
        top_emotion_peaks = top_emotion_peaks.sort_values('time')

        # Save to CSV
        emotion_file = os.path.join(output_dir, f"top_5_{emotion}_regions.csv")
        top_emotion_peaks.to_csv(emotion_file)
        print(f"Top 5 {emotion.capitalize()} regions saved to {emotion_file}")

    # Detect peaks in sentiment_score
    sentiment_peaks, sentiment_properties = find_peaks(
        grouped_data['sentiment_score'],
        prominence=0.5,    # Adjusted prominence for sentiment
        height=0.1,        # Adjusted minimum height for sentiment
        width=1            # Minimum width of 1 window
    )

    # Create a DataFrame with sentiment peak moments
    sentiment_peak_windows = grouped_data.iloc[sentiment_peaks].copy()
    sentiment_peak_windows['prominence'] = sentiment_properties['prominences']

    # Sort peaks by sentiment score first, then prominence
    sentiment_peak_windows = sentiment_peak_windows.sort_values(
        by=['sentiment_score', 'prominence'],
        ascending=[False, False]
    )

    # Filter peaks with minimum time distance
    top_sentiment_peaks = filter_peaks_with_distance(sentiment_peak_windows, min_distance_secs=600).head(5)

    # Sort by time
    top_sentiment_peaks = top_sentiment_peaks.sort_values('time')

    # Save to CSV
    sentiment_file = os.path.join(output_dir, "top_5_sentiment_peaks.csv")
    top_sentiment_peaks.to_csv(sentiment_file)
    print(f"Top 5 Sentiment peaks saved to {sentiment_file}")

    # Plotting the Smoothened Highlight Score
    plt.figure(figsize=(15, 7))
    x = grouped_data.index
    y = grouped_data['weighted_highlight_score']

    # Prepare data for smoothing
    x_numeric = mdates.date2num(x)
    spline = make_interp_spline(x_numeric, y, k=2)
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 1000)
    y_smooth = spline(x_smooth)

    # Plot the smooth line
    plt.plot_date(mdates.num2date(x_smooth), y_smooth, '-', label='Weighted Highlight Score (Smoothed)', color='blue')

    # Mark peaks with X
    for peak_time in top_windows.index:
        peak_datetime = pd.to_datetime(peak_time)
        plt.axvspan(peak_datetime - pd.Timedelta(seconds=300), peak_datetime + pd.Timedelta(seconds=300), color='yellow', alpha=0.3)
        plt.plot(peak_datetime, grouped_data.loc[peak_time, 'weighted_highlight_score'], 'rx', markersize=10)

    # Improve date formatting on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.HourLocator(interval=1)))
    plt.xticks(rotation=45)

    plt.title("Smoothened Weighted Highlight Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Weighted Highlight Score")
    plt.legend()
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "highlight_score_smooth_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Smoothened highlight score plot saved to {plot_file}")
    

if __name__ == "__main__":
    input_file = "twitch_chat_analysis/data/twitch_chat_sentiment_emotion_analysis.csv"
    output_dir = "twitch_chat_analysis/outputs"
    analyze_chat_highlights(input_file, output_dir)