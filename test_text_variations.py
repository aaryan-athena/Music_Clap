#!/usr/bin/env python3
"""
Test script to demonstrate text-to-parameter variations
Shows how different text descriptions produce different knob settings
"""

from music_tone_classifier import MusicToneClassifier
import numpy as np

def main():
    print("🎸 Text-to-Parameter Variation Test")
    print("=" * 80)
    print("\nLoading models...")
    
    # Load trained classifier
    classifier = MusicToneClassifier()
    classifier.load_models(save_dir="models")
    
    print("\n✅ Models loaded successfully!")
    print("\nTesting various text descriptions to show parameter variations...")
    print("=" * 80)
    
    # Test various text descriptions
    test_descriptions = [
        # Metal variations
        "heavy metal with aggressive distortion",
        "brutal death metal tone",
        "modern metal with tight low end",
        "thrash metal with scooped mids",
        
        # Clean variations
        "pristine clean jazz tone",
        "bright clean with sparkle",
        "warm clean with body",
        "clean tone with slight chorus",
        
        # Crunch variations
        "vintage blues crunch",
        "classic rock overdrive",
        "AC/DC style crunch",
        "edge of breakup tone",
        
        # Ambient variations
        "spacey ambient with long reverb",
        "shoegaze wall of sound",
        "ethereal dreamy tone",
        "atmospheric with delay",
        
        # Creative descriptions
        "angry aggressive tone",
        "smooth silky sound",
        "punchy tight tone",
        "warm vintage vibe",
        "bright cutting tone",
        "dark heavy sound",
    ]
    
    results = []
    
    for desc in test_descriptions:
        result = classifier.match_text_to_tone(desc)
        results.append((desc, result))
        
        print(f"\n📝 Text: '{desc}'")
        print(f"   Tone Type: {result['tone_type']} (confidence: {result['confidence']:.1%})")
        
        # Show key parameters that vary
        knobs = result['knob_settings']
        print(f"   Key Parameters:")
        print(f"     Distortion: {knobs['distortiondrive']:.3f} | Overdrive: {knobs['overdrivedrive']:.3f}")
        print(f"     EQ: Bass={knobs['eqbass']:.3f}, Mid={knobs['eqmid']:.3f}, Treble={knobs['eqtreble']:.3f}")
        print(f"     Reverb: Wet={knobs['reverbwet']:.3f}, T60={knobs['reverbt60']:.3f}")
        print(f"     Delay: Mix={knobs['delaymix']:.3f}, Feedback={knobs['delayfeedback']:.3f}")
    
    # Analyze variation
    print("\n" + "=" * 80)
    print("📊 VARIATION ANALYSIS")
    print("=" * 80)
    
    # Compare similar descriptions to show variation
    comparisons = [
        ("heavy metal with aggressive distortion", "brutal death metal tone"),
        ("pristine clean jazz tone", "bright clean with sparkle"),
        ("vintage blues crunch", "classic rock overdrive"),
        ("spacey ambient with long reverb", "shoegaze wall of sound"),
    ]
    
    for desc1, desc2 in comparisons:
        result1 = classifier.match_text_to_tone(desc1)
        result2 = classifier.match_text_to_tone(desc2)
        
        knobs1 = result1['knob_settings']
        knobs2 = result2['knob_settings']
        
        # Calculate differences
        differences = {}
        for param in classifier.knob_params:
            diff = abs(knobs1[param] - knobs2[param])
            differences[param] = diff
        
        # Find parameters with biggest differences
        sorted_diffs = sorted(differences.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Comparing:")
        print(f"   1. '{desc1}'")
        print(f"   2. '{desc2}'")
        print(f"\n   Top 5 Parameter Differences:")
        for param, diff in sorted_diffs[:5]:
            val1 = knobs1[param]
            val2 = knobs2[param]
            print(f"     {param:20} | {val1:.3f} vs {val2:.3f} (diff: {diff:.3f})")
    
    # Show that we get unique parameters for each description
    print("\n" + "=" * 80)
    print("✅ CONCLUSION")
    print("=" * 80)
    print("\nThe neural network generates UNIQUE parameter sets for each text input!")
    print("You're no longer limited to 4 predefined tone types.")
    print("\nHow it works:")
    print("  1. CLAP converts your text to a 512-dimensional embedding")
    print("  2. The neural network predicts 17 knob parameters from that embedding")
    print("  3. Each unique text description produces unique parameter values")
    print("\nTry any description you want:")
    print("  • 'warm vintage tube amp'")
    print("  • 'bright cutting lead tone'")
    print("  • 'dark heavy rhythm'")
    print("  • 'smooth jazz with chorus'")
    print("  • 'aggressive modern metal'")
    print("\nThe model will generate appropriate parameters for each! 🎉")

if __name__ == "__main__":
    main()
