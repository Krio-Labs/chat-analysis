import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        fig = plt.figure(figsize=(15, 25))
        gs = fig.add_gridspec(5, 1, hspace=0.4)
        
        # Calculate rolling averages for smoothing
        smooth_messages = stats['message_count'].rolling(window=window, center=True).mean()
        smooth_sentiment = stats['avg_sentiment'].rolling(window=window, center=True).mean()
        
        # Calculate weighted metrics
        stats['weighted_sentiment'] = stats['avg_sentiment'] * stats['message_count']
        smooth_weighted_sentiment = stats['weighted_sentiment'].rolling(window=window, center=True).mean()
        
        emotions = ['avg_excitement', 'avg_funny', 'avg_happiness', 'avg_anger', 'avg_sadness']
        for emotion in emotions:
            stats[f'weighted_{emotion}'] = stats[emotion] * stats['message_count']
        
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
        
        # Adjust layout and save
        plt.savefig(os.path.join(output_dir, 'chat_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        plt.close()

def analyze_chat_intervals(input_file, interval_seconds=30):
    """Analyze chat data in time intervals."""
    try:
        # Read the data
        df = pd.read_csv(input_file)
        
        if df.empty:
            print("Error: Input file is empty!")
            return
            
        # Convert time column to numeric (it's already in seconds)
        df['runtime'] = pd.to_numeric(df['time'])
        stream_start = df['runtime'].min()
        df['runtime'] = df['runtime'] - stream_start  # Normalize to start at 0
        
        # Create intervals (30-second bins) without labels first
        bins = np.arange(0, df['runtime'].max() + interval_seconds, interval_seconds)
        df['interval'] = pd.cut(df['runtime'], bins=bins)
        
        # Calculate statistics for each interval
        interval_stats = df.groupby('interval', observed=True).agg({
            'message': 'count',  # Message count
            'sentiment_score': ['mean', 'std'],  # Sentiment statistics
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
            'avg_excitement',
            'avg_funny',
            'avg_happiness',
            'avg_anger',
            'avg_sadness',
            'avg_neutral'
        ]
        
        # Reset index and calculate interval midpoints for plotting
        interval_stats = interval_stats.reset_index()
        
        # Calculate midpoints numerically instead of using interval objects
        interval_stats['time_mid'] = interval_stats['interval'].apply(
            lambda x: (x.left + x.right) / 2 / 60  # Convert to minutes
        )
        
        # Handle any NaN values in numeric columns only
        numeric_cols = interval_stats.select_dtypes(include=[np.number]).columns
        interval_stats[numeric_cols] = interval_stats[numeric_cols].fillna(0)
        
        # Save results
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, 'interval_analysis.csv')
        interval_stats.to_csv(output_file, index=False)
        
        # Create visualizations
        plot_metrics(interval_stats, output_dir)
        
        # Print summary statistics
        print("\nAnalysis Summary:")
        print("-" * 50)
        print(f"Total messages: {df.shape[0]}")
        print(f"Stream duration: {df['runtime'].max()/60:.1f} minutes")
        print(f"Messages per minute: {df.shape[0] / (df['runtime'].max()/60):.1f}")
        print(f"Average messages per 30s interval: {interval_stats['message_count'].mean():.1f}")
        print(f"Overall sentiment: {interval_stats['avg_sentiment'].mean():.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return

if __name__ == "__main__":
    input_file = "twitch_chat_analysis/outputs/twitch_chat_sentiment_emotion_analysis.csv"
    analyze_chat_intervals(input_file)
