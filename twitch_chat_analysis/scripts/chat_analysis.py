import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline
import matplotlib.dates as mdates

def plot_metrics(stats, output_dir, window=5):
    """
    Create visualizations with smoothed lines for the interval analysis.
    
    Args:
        stats (pd.DataFrame): DataFrame containing the statistics
        output_dir (str): Directory to save the plots
        window (int): Window size for rolling average smoothing
    """
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_theme()
    
    try:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 35))
        gs = fig.add_gridspec(7, 1, hspace=0.4)
        
        # Calculate rolling averages for smoothing
        smooth_messages = stats['message_count'].rolling(window=window, center=True).mean()
        smooth_sentiment = stats['avg_sentiment'].rolling(window=window, center=True).mean()
        
        # Calculate weighted metrics
        stats['weighted_sentiment'] = stats['avg_sentiment'] * stats['message_count']
        smooth_weighted_sentiment = stats['weighted_sentiment'].rolling(window=window, center=True).mean()
        
        emotions = ['avg_excitement', 'avg_funny', 'avg_happiness', 'avg_anger', 'avg_sadness']
        for emotion in emotions:
            stats[f'weighted_{emotion}'] = stats[emotion] * stats['message_count']
        
        # Add smoothing for highlight scores (corrected)
        smooth_highlight = stats['avg_highlight'].rolling(window=window, center=True).mean()
        stats['weighted_highlight'] = stats['avg_highlight'] * stats['message_count']  # Changed from highlight_score
        smooth_weighted_highlight = stats['weighted_highlight'].rolling(window=window, center=True).mean()
        
        # Plot 1: Message Count over time
        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(stats['time_mid'], stats['message_count'], 
                    alpha=0.2, color='blue', s=20)
        ax1.plot(stats['time_mid'], smooth_messages, 
                 color='darkblue', linewidth=2.5, label='Smoothed trend')
        
        ax1.set_title('Message Frequency Over Time', fontsize=14, pad=15)
        ax1.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax1.set_ylabel('Messages per 30s', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot 2: Raw Sentiment over time
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(stats['time_mid'], stats['avg_sentiment'], 
                    alpha=0.2, color='green', s=20)
        ax2.plot(stats['time_mid'], smooth_sentiment, 
                 color='darkgreen', linewidth=2.5, label='Smoothed trend')
        
        ax2.set_title('Average Sentiment Over Time (Raw)', fontsize=14, pad=15)
        ax2.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax2.set_ylabel('Sentiment Score', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot 3: Weighted Sentiment over time
        ax3 = fig.add_subplot(gs[2])
        ax3.scatter(stats['time_mid'], stats['weighted_sentiment'], 
                    alpha=0.2, color='purple', s=20)
        ax3.plot(stats['time_mid'], smooth_weighted_sentiment, 
                 color='darkmagenta', linewidth=2.5, label='Smoothed trend')
        
        ax3.set_title('Weighted Sentiment Over Time (Sentiment × Message Count)', 
                     fontsize=14, pad=15)
        ax3.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax3.set_ylabel('Weighted Sentiment', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Plot 4: Raw Emotions over time
        ax4 = fig.add_subplot(gs[3])
        colors = sns.color_palette("husl", len(emotions))
        
        for emotion, color in zip(emotions, colors):
            smooth_emotion = stats[emotion].rolling(window=window, center=True).mean()
            ax4.plot(stats['time_mid'], smooth_emotion, 
                     linewidth=2, label=emotion.replace('avg_', ''),
                     color=color)
            ax4.scatter(stats['time_mid'], stats[emotion], 
                       alpha=0.1, s=10, color=color)
        
        ax4.set_title('Emotions Over Time (Raw)', fontsize=14, pad=15)
        ax4.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax4.set_ylabel('Emotion Score', fontsize=12)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 5: Weighted Emotions over time
        ax5 = fig.add_subplot(gs[4])
        
        for emotion, color in zip(emotions, colors):
            weighted_emotion = f'weighted_{emotion}'
            smooth_weighted = stats[weighted_emotion].rolling(window=window, center=True).mean()
            ax5.plot(stats['time_mid'], smooth_weighted, 
                     linewidth=2, label=emotion.replace('avg_', ''),
                     color=color)
            ax5.scatter(stats['time_mid'], stats[weighted_emotion], 
                       alpha=0.1, s=10, color=color)
        
        ax5.set_title('Weighted Emotions Over Time (Emotion × Message Count)', 
                     fontsize=14, pad=15)
        ax5.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax5.set_ylabel('Weighted Emotion Score', fontsize=12)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 6: Highlight Scores over time
        ax6 = fig.add_subplot(gs[5])
        ax6.scatter(stats['time_mid'], stats['avg_highlight'],  # Changed from highlight_score
                   alpha=0.2, color='purple', s=20)
        ax6.plot(stats['time_mid'], smooth_highlight, 
                color='darkviolet', linewidth=2.5, label='Smoothed trend')
        
        ax6.set_title('Average Highlight Score Over Time', fontsize=14, pad=15)
        ax6.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax6.set_ylabel('Highlight Score (0-3)', fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.legend()
        
        # Plot 7: Weighted Highlight Score over time
        ax7 = fig.add_subplot(gs[6])
        ax7.scatter(stats['time_mid'], stats['weighted_highlight'],
                   alpha=0.2, color='orange', s=20)
        ax7.plot(stats['time_mid'], smooth_weighted_highlight,
                 color='darkorange', linewidth=2.5, label='Smoothed trend')
        
        ax7.set_title('Weighted Highlight Score Over Time (Highlight × Message Count)',
                     fontsize=14, pad=15)
        ax7.set_xlabel('Stream Time (minutes)', fontsize=12)
        ax7.set_ylabel('Weighted Highlight Score', fontsize=12)
        ax7.grid(True, linestyle='--', alpha=0.7)
        ax7.legend()
        
        # Adjust layout and save
        plt.savefig(os.path.join(output_dir, 'chat_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        plt.close()

def analyze_chat_intervals(input_file, output_dir, interval_seconds=30):
    """
    Analyze chat messages in intervals and calculate statistics.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save output files
        interval_seconds (int): Size of time intervals in seconds
    """
    try:
        # Read the data
        df = pd.read_csv(input_file)
        
        # Convert time to numeric for binning
        df['time'] = pd.to_numeric(df['time'])
        
        # Create interval bins
        max_time = df['time'].max()
        bins = range(0, int(max_time) + interval_seconds, interval_seconds)
        labels = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins)-1)]
        df['interval'] = pd.cut(df['time'], bins=bins, labels=labels)
        
        # Calculate statistics for each interval
        interval_stats = df.groupby('interval', observed=True).agg({
            'message': 'count',
            'sentiment_score': ['mean', 'std'],
            'highlight_score': 'mean',
            'excitement': 'mean',
            'funny': 'mean',
            'happiness': 'mean',
            'anger': 'mean',
            'sadness': 'mean',
            'neutral': 'mean'
        }).round(3)
        
        # Flatten column names
        interval_stats.columns = [
            'message_count',
            'avg_sentiment',
            'std_sentiment',
            'avg_highlight',
            'avg_excitement',
            'avg_funny',
            'avg_happiness',
            'avg_anger',
            'avg_sadness',
            'avg_neutral'
        ]
        
        # Add time midpoint for plotting
        interval_stats = interval_stats.reset_index()
        interval_stats['time_mid'] = interval_stats['interval'].apply(
            lambda x: (float(x.strip('()[]').split(',')[0]) + 
                      float(x.strip('()[]').split(',')[1])) / 2 / 60
        )
        
        # Save to CSV
        interval_stats.to_csv(os.path.join(output_dir, 'interval_analysis.csv'), index=False)
        
        return interval_stats
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

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

    # Convert time to minutes for better readability in plots
    grouped_data['time_minutes'] = grouped_data.index / 60

    # Compute weighted_highlight_score based on highlight_score and message_count
    grouped_data['weighted_highlight_score'] = (
        grouped_data['highlight_score'] * 0.7 +  # Weight for highlight_score
        grouped_data['message_count'] * 0.3      # Weight for message_count (chat density)
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
            current_time = idx  # Using raw time value
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

    # Plotting the Smoothened Highlight Score
    plt.figure(figsize=(15, 7))
    
    # Get time in minutes for x-axis
    x = grouped_data.index / 60  # Convert seconds to minutes
    y = grouped_data['weighted_highlight_score']

    # Prepare data for smoothing
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    spline = make_interp_spline(x, y, k=1)  # Linear interpolation
    y_smooth = spline(x_smooth)

    # Plot the smooth line
    plt.plot(x_smooth, y_smooth, '-', 
            label='Weighted Highlight Score (Smoothed)', color='blue', alpha=0.7)

    # Read top highlights from CSV and mark them
    top_highlights = pd.read_csv('twitch_chat_analysis/outputs/top_highlights.csv')
    
    # Mark peaks with X and highlight windows
    for _, highlight in top_highlights.iterrows():
        peak_minutes = highlight['time'] / 60  # Convert seconds to minutes
        # Add yellow highlight window
        plt.axvspan(peak_minutes - 2.5, peak_minutes + 2.5,  # +/- 2.5 minutes
                   color='yellow', alpha=0.2)
        # Add X marker at peak
        plt.plot(peak_minutes, 
                highlight['weighted_highlight_score'],
                'rx', markersize=12, markeredgewidth=2,
                label='_nolegend_')  # Add large red X marker

    # Configure axes
    plt.title("Smoothened Weighted Highlight Score Over Time")
    plt.xlabel("Stream Time (minutes)")
    plt.ylabel("Weighted Highlight Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "highlight_score_smooth_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Smoothened highlight score plot saved to {plot_file}")

def main():
    # Define input/output paths
    input_file = 'twitch_chat_analysis/data/twitch_chat_sentiment_emotion_analysis.csv'
    output_dir = 'twitch_chat_analysis/outputs'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze chat intervals
    interval_stats = analyze_chat_intervals(input_file, output_dir)
        
    # Plot metrics
    plot_metrics(interval_stats, output_dir)
    
    # Print summary statistics
    total_messages = interval_stats['message_count'].sum()
    stream_duration = interval_stats['time_mid'].max()
    messages_per_minute = total_messages / stream_duration
    avg_messages_per_interval = interval_stats['message_count'].mean()
    overall_sentiment = interval_stats['avg_sentiment'].mean()
    
    print("\nAnalysis Summary:")
    print("-" * 50)
    print(f"Total messages: {total_messages:.0f}")
    print(f"Stream duration: {stream_duration:.1f} minutes")
    print(f"Messages per minute: {messages_per_minute:.1f}")
    print(f"Average messages per 30s interval: {avg_messages_per_interval:.1f}")
    print(f"Overall sentiment: {overall_sentiment:.3f}")
        
    # Analyze highlights
    analyze_chat_highlights(input_file, output_dir)

if __name__ == "__main__":
    main()
