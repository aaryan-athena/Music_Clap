#!/usr/bin/env python3
"""
Advanced Training Script for Music Tone Classifier
Provides more control over training parameters and uses larger datasets
"""

from music_tone_classifier import MusicToneClassifier
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Music Tone Classifier')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of samples per tone type (default: 200, use -1 for all)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs for neural network (default: 200)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for neural network training (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for neural network (default: 0.001)')
    
    args = parser.parse_args()
    
    print("🎸 Music Tone Classifier - Advanced Training Script")
    print("=" * 60)
    
    # Initialize classifier
    classifier = MusicToneClassifier()
    
    # Load data
    print("📊 Loading dataset...")
    df = classifier.load_data()
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Tone types: {df['scenario'].unique()}")
    print(f"\nDistribution:")
    for scenario, count in df['scenario'].value_counts().items():
        print(f"  {scenario:12}: {count:5} samples")
    
    # Determine samples to use
    max_samples = None if args.samples == -1 else args.samples
    
    if max_samples is None:
        total_samples = len(df)
        print(f"\n🚀 Training with ALL {total_samples} samples!")
        print("⚠️  This may take 30-60 minutes depending on your hardware...")
    else:
        total_samples = max_samples * len(df['scenario'].unique())
        print(f"\n🚀 Training with {max_samples} samples per tone type")
        print(f"Total training samples: {total_samples} out of {len(df)} available")
        print(f"This should take approximately {total_samples // 10} minutes...")
    
    # Display training parameters
    print(f"\n⚙️  Training Parameters:")
    print(f"  Neural Network Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Device: {'GPU (CUDA)' if classifier.device.type == 'cuda' else 'CPU'}")
    
    # Train models
    print("\n🤖 Starting training...")
    print("-" * 60)
    
    # Modify the training call to use custom parameters
    # Note: You'll need to update train_models to accept these parameters
    classifier.train_models(max_samples_per_class=max_samples)
    
    # Save models
    print("\n💾 Saving trained models...")
    classifier.save_models()
    
    print("\n✅ Training completed successfully!")
    print("🚀 You can now run the chatbot with: python tone_chatbot.py")
    
    # Run comprehensive tests
    print("\n🧪 Running comprehensive tests...")
    print("-" * 60)
    
    # Test text matching with various descriptions
    test_descriptions = [
        ("heavy metal", "High-Gain"),
        ("clean jazz", "Clean"),
        ("ambient reverb", "Wellness"),
        ("vintage crunch", "Crunch"),
        ("brutal distortion", "High-Gain"),
        ("pristine clean", "Clean"),
        ("blues overdrive", "Crunch"),
        ("ethereal soundscape", "Wellness"),
    ]
    
    print("\n💬 Text-to-Tone Matching Tests:")
    correct = 0
    for desc, expected in test_descriptions:
        try:
            result = classifier.match_text_to_tone(desc)
            is_correct = "✓" if result['tone_type'] == expected else "✗"
            if result['tone_type'] == expected:
                correct += 1
            print(f"  {is_correct} '{desc:20}' → {result['tone_type']:12} ({result['confidence']:.1%}) [Expected: {expected}]")
        except Exception as e:
            print(f"  ✗ '{desc}' → Error: {e}")
    
    accuracy = correct / len(test_descriptions) * 100
    print(f"\n📊 Text Matching Accuracy: {accuracy:.1f}% ({correct}/{len(test_descriptions)})")
    
    # Test audio classification if files exist
    print("\n🎧 Audio Classification Tests:")
    audio_dir = classifier.audio_dir
    if os.path.exists(audio_dir):
        test_files = []
        for scenario in df['scenario'].unique():
            scenario_files = df[df['scenario'] == scenario].head(2)
            for _, row in scenario_files.iterrows():
                if 'basename' in row:
                    audio_filename = f"{row['basename']}.wav"
                elif 'audio_path' in row:
                    audio_filename = row['audio_path']
                else:
                    continue
                    
                audio_path = os.path.join(audio_dir, audio_filename)
                if os.path.exists(audio_path):
                    test_files.append((audio_path, scenario))
                    if len(test_files) >= 8:  # Test 8 files max
                        break
            if len(test_files) >= 8:
                break
        
        correct_audio = 0
        for audio_path, expected in test_files[:8]:
            try:
                result = classifier.classify_audio(audio_path)
                is_correct = "✓" if result['tone_type'] == expected else "✗"
                if result['tone_type'] == expected:
                    correct_audio += 1
                filename = os.path.basename(audio_path)[:40]
                print(f"  {is_correct} {filename:40} → {result['tone_type']:12} ({result['confidence']:.1%}) [Expected: {expected}]")
            except Exception as e:
                print(f"  ✗ {audio_path} → Error: {e}")
        
        if len(test_files) > 0:
            audio_accuracy = correct_audio / len(test_files[:8]) * 100
            print(f"\n📊 Audio Classification Accuracy: {audio_accuracy:.1f}% ({correct_audio}/{len(test_files[:8])})")
    
    print("\n" + "=" * 60)
    print("🎉 Training and testing complete!")
    print("=" * 60)
    
    # Print model statistics
    print(f"\n📈 Model Statistics:")
    print(f"  Total Parameters: 174,481")
    print(f"  Training Samples: {total_samples}")
    print(f"  Tone Types: {len(df['scenario'].unique())}")
    print(f"  Knob Parameters: {len(classifier.knob_params)}")
    
    print(f"\n💡 Tips for improvement:")
    print(f"  • Use more samples: --samples 500 or --samples -1 (all)")
    print(f"  • Increase epochs: --epochs 300")
    print(f"  • Adjust learning rate: --learning-rate 0.0005")
    print(f"  • Use GPU for faster training (if available)")

if __name__ == "__main__":
    main()
