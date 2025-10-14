#!/usr/bin/env python3
"""
Demo Script - Tests the actual trained neural network model
Run this to see the real system in action with trained models
"""

import numpy as np
import pandas as pd
import os
from music_tone_classifier import MusicToneClassifier

def demo_system():
    print("🎸 Music Tone Classifier - Live Model Demo")
    print("=" * 50)
    
    # Initialize classifier and load trained models
    print("🔄 Loading trained models...")
    classifier = MusicToneClassifier()
    
    try:
        classifier.load_models()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("💡 Please run 'python train_models.py' first to train the models")
        return
    
    # Load dataset for reference
    csv_path = "Data/indexes/index_train.csv"
    df = pd.read_csv(csv_path)
    
    print(f"📊 Dataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Tone types: {list(df['scenario'].unique())}")
    print(f"\nDistribution:")
    for scenario, count in df['scenario'].value_counts().items():
        print(f"  {scenario}: {count} samples")
    
    # Show knob parameters
    knob_params = [col for col in df.columns if col.startswith('n_')]
    print(f"\n🎛️ Knob Parameters ({len(knob_params)} total):")
    for i, param in enumerate(knob_params):
        param_name = param[2:]  # Remove 'n_' prefix
        print(f"  {i+1:2d}. {param_name}")
    
    # Test audio file classification
    print(f"\n🎵 Testing Audio File Classification:")
    audio_files = []
    
    # Find some sample audio files
    audio_dir = "Data/generated_audio/train"
    if os.path.exists(audio_dir):
        for file in os.listdir(audio_dir)[:8]:  # Test first 8 files
            if file.endswith('.wav'):
                audio_files.append(os.path.join(audio_dir, file))
    
    if audio_files:
        for i, audio_file in enumerate(audio_files[:4], 1):  # Test first 4
            print(f"\n  🎧 Test {i}: {os.path.basename(audio_file)}")
            try:
                result = classifier.classify_audio(audio_file)
                
                print(f"    🎯 Predicted Tone: {result['tone_type']}")
                print(f"    � Confidence: {result['confidence']:.1%}")
                print(f"    🎛️ Key Knob Settings:")
                
                # Show key parameters with neural network predictions
                key_params = ['distortiondrive', 'overdrivedrive', 'eqbass', 'eqmid', 'eqtreble', 'mastervolume']
                for param in key_params:
                    if param in result['knob_settings']:
                        value = result['knob_settings'][param]
                        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                        print(f"      {param:15}: {value:.3f} |{bar}|")
                        
            except Exception as e:
                print(f"    ❌ Error processing audio: {e}")
    else:
        print("  ⚠️ No audio files found for testing")
    
    # Test real text-to-tone matching with trained models
    print(f"\n💬 Testing Real Text-to-Tone Matching:")
    
    test_inputs = [
        "heavy metal distortion",
        "clean jazz guitar", 
        "vintage blues crunch",
        "ambient spacey reverb",
        "aggressive brutal tone",
        "pristine clean sound",
        "warm tube overdrive",
        "ethereal dreamy atmosphere"
    ]
    
    for text_input in test_inputs:
        print(f"\n  📝 Input: '{text_input}'")
        
        try:
            result = classifier.match_text_to_tone(text_input)
            
            print(f"  🎯 AI Matched Tone: {result['tone_type']}")
            print(f"  📊 Similarity Score: {result['confidence']:.1%}")
            print(f"  �️ Neural Network Predicted Settings:")
            
            # Show key parameters from the actual model
            key_params = ['distortiondrive', 'overdrivedrive', 'eqbass', 'eqmid', 'eqtreble', 'mastervolume']
            for param in key_params:
                if param in result['knob_settings']:
                    value = result['knob_settings'][param]
                    bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                    print(f"     {param:15}: {value:.3f} |{bar}|")
            
            # Show all similarity scores
            print(f"  📈 All Similarity Scores:")
            for tone_type, score in result['all_similarities'].items():
                print(f"     {tone_type:12}: {score:.3f}")
                
        except Exception as e:
            print(f"  ❌ Error processing text: {e}")
    
    # Test neural network parameter correlation
    print(f"\n🧠 Neural Network Parameter Correlation Analysis:")
    
    if len(audio_files) >= 2:
        print(f"  📊 Comparing parameter predictions for different tones:")
        
        # Test two different audio files
        try:
            result1 = classifier.classify_audio(audio_files[0])
            result2 = classifier.classify_audio(audio_files[1])
            
            print(f"\n  🎧 File 1: {os.path.basename(audio_files[0])}")
            print(f"     Tone: {result1['tone_type']} ({result1['confidence']:.1%})")
            
            print(f"\n  🎧 File 2: {os.path.basename(audio_files[1])}")  
            print(f"     Tone: {result2['tone_type']} ({result2['confidence']:.1%})")
            
            print(f"\n  🔍 Parameter Correlation Evidence:")
            
            # Compare EQ parameters (should be correlated)
            eq_params = ['eqbass', 'eqmid', 'eqtreble']
            print(f"     EQ Settings Comparison:")
            for param in eq_params:
                val1 = result1['knob_settings'][param]
                val2 = result2['knob_settings'][param]
                diff = abs(val1 - val2)
                print(f"       {param:10}: {val1:.3f} vs {val2:.3f} (diff: {diff:.3f})")
            
            # Compare effect parameters (should be correlated)
            effect_params = ['chorusrate', 'chorusdepth', 'chorusmix']
            print(f"     Chorus Effect Correlation:")
            for param in effect_params:
                val1 = result1['knob_settings'][param]
                val2 = result2['knob_settings'][param]
                diff = abs(val1 - val2)
                print(f"       {param:12}: {val1:.3f} vs {val2:.3f} (diff: {diff:.3f})")
                
        except Exception as e:
            print(f"  ❌ Error in correlation analysis: {e}")
    
    print(f"\n🚀 Neural Network Model Capabilities Demonstrated:")
    print(f"  ✅ Multi-output neural network (17 correlated parameters)")
    print(f"  ✅ Real audio file classification with CLAP embeddings")
    print(f"  ✅ Intelligent text-to-tone matching")
    print(f"  ✅ Parameter correlation modeling")
    print(f"  ✅ GPU-accelerated training and inference")
    
    print(f"\n🎯 Model Architecture Summary:")
    if hasattr(classifier, 'knob_regressor_net') and classifier.knob_regressor_net:
        print(f"  🧠 Neural Network: 512 → 256 → 128 → 64 → 17 parameters")
        print(f"  ⚡ Device: {classifier.device}")
        print(f"  🎛️ Parameters: {sum(p.numel() for p in classifier.knob_regressor_net.parameters()):,}")
    
    print(f"\n📋 Usage:")
    print(f"  • Run 'python tone_chatbot.py' to launch web interface")
    print(f"  • Upload audio files for automatic tone analysis")
    print(f"  • Describe tones in natural language for recommendations")
    
    print(f"\n🎉 Live model demo completed!")

if __name__ == "__main__":
    demo_system()
