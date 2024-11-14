# Twitch Chat Analysis

A comprehensive tool for downloading, processing, and analyzing Twitch chat data to understand viewer engagement and sentiment during streams.

## Input

Provide a Twitch URL to download the chat data.

## Output

The output is a series of CSV files containing the chat data, processed for sentiment analysis, and statistical analysis & visualizations.

## Overview

This project provides a pipeline for analyzing Twitch chat data through four main stages:
1. Chat Download
2. Message Processing
3. Sentiment Analysis
4. Statistical Analysis & Visualization

## Features

- Download chat data from any Twitch VOD
- Filter out noise (bot messages, spam, commands)
- Analyze message sentiment and emotions
- Generate time-series visualizations
- Process emotes, slang, and emoji meanings
- Parallel processing for improved performance

## Requirements

```bash
pip install -r requirements.txt
