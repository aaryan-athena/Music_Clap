#!/usr/bin/env python3
"""
Data analysis script for the music tone dataset
"""

import pandas as pd
import numpy as np
import os
import json

def analyze_dataset():
    """Analyze the dataset structure and content"""
    
    # Load the CSV data
    csv_path = "Data/indexes/index_train.csv"
    df = pd.read_csv(csv_path)
    
    print("=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"\nUnique tone scenarios: {df['scenario'].unique()}")
    print(f"\nTone scenario counts:")
    print(df['scenario'].value_counts())
    
    print(f"\nDataset columns:")
    print(df.columns.tolist())
    
    print(f"\nSample data structure:")
    print(df.head())
    
    # Check normalized parameters
    knob_params = [col for col in df.columns if col.startswith('n_')]
    print(f"\nKnob parameters (normalized): {len(knob_params)}")
    print(knob_params)
    
    # Check if audio files exist
    audio_dir = "Data/generated_audio/train"
    json_dir = "Data/labels_normalised/train"
    
    print(f"\nChecking file existence...")
    missing_audio = 0
    missing_json = 0
    
    for _, row in df.head(10).iterrows():  # Check first 10 for speed
        audio_path = os.path.join(audio_dir, row['audio_path'])
        json_path = os.path.join(json_dir, row['label_path'])
        
        if not os.path.exists(audio_path):
            missing_audio += 1
        if not os.path.exists(json_path):
            missing_json += 1
    
    print(f"Missing audio files (from first 10): {missing_audio}")
    print(f"Missing JSON files (from first 10): {missing_json}")
    
    # Load a sample JSON file
    if len(df) > 0:
        sample_json_path = os.path.join(json_dir, df.iloc[0]['label_path'])
        if os.path.exists(sample_json_path):
            with open(sample_json_path, 'r') as f:
                sample_json = json.load(f)
            print(f"\nSample JSON structure:")
            print(f"Keys: {list(sample_json.keys())}")
            if 'params' in sample_json:
                print(f"Parameter keys: {list(sample_json['params'].keys())}")

if __name__ == "__main__":
    analyze_dataset()
