#!/usr/bin/env python3
"""
Quick test for text-to-tone matching
"""

from music_tone_classifier import MusicToneClassifier

def test_text_matching():
    print("🧪 Testing Text-to-Tone Matching")
    print("=" * 40)
    
    # Initialize and load models
    classifier = MusicToneClassifier()
    classifier.load_models()
    
    # Test a simple text input
    test_text = "heavy metal distortion"
    print(f"Testing: '{test_text}'")
    
    try:
        result = classifier.match_text_to_tone(test_text)
        print(f"✅ Success!")
        print(f"Matched tone: {result['tone_type']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Show a few key parameters
        key_params = ['distortiondrive', 'overdrivedrive', 'mastervolume']
        for param in key_params:
            if param in result['knob_settings']:
                value = result['knob_settings'][param]
                print(f"{param}: {value:.3f}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_matching()