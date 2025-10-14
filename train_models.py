#!/usr/bin/env python3
"""
Training script for the Music Tone Classifier
Run this first to train the models before using the chatbot
"""

from music_tone_classifier import MusicToneClassifier
import os

def main():
    print("🎸 Music Tone Classifier - Training Script")
    print("=" * 50)
    
    # Initialize classifier
    classifier = MusicToneClassifier()
    
    # Load data
    print("📊 Loading dataset...")
    df = classifier.load_data()
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Tone types: {df['scenario'].unique()}")
    print(f"Distribution:\n{df['scenario'].value_counts()}")
    
    # Train models
    print("\n🤖 Training models...")
    print("This may take a few minutes depending on your hardware...")
    
    # Use more samples for better model performance
    # Options: 
    # - Use None for ALL samples (slow but best accuracy)
    # - Use a reasonable number like 200-500 per class for good balance
    # - Use 50-100 for quick training
    
    # For your 11,937 sample dataset, let's use a good subset
    max_samples = 200  # Use 200 samples per tone type (800 total)
    print(f"Using {max_samples} samples per tone type for training...")
    print(f"Total training samples: {max_samples * len(df['scenario'].unique())} out of {len(df)} available")
    
    classifier.train_models(max_samples_per_class=max_samples)
    
    # Save models
    print("\n💾 Saving trained models...")
    classifier.save_models()
    
    print("\n✅ Training completed successfully!")
    print("🚀 You can now run the chatbot with: python tone_chatbot.py")
    
    # Quick test
    print("\n🧪 Running quick test...")
    
    # Test text matching
    test_descriptions = ["heavy metal", "clean jazz", "ambient reverb"]
    for desc in test_descriptions:
        try:
            result = classifier.match_text_to_tone(desc)
            print(f"'{desc}' → {result['tone_type']} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"Error testing '{desc}': {e}")
    
    print("\n🎉 Setup complete! Ready to use the chatbot.")

if __name__ == "__main__":
    main()
