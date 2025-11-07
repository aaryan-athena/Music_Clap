#!/usr/bin/env python3
"""
Test script for the Flask API server
Tests both text input and audio input endpoints
"""

import requests
import json

API_URL = "http://127.0.0.1:5000/predict"

def test_text_input():
    """Test the API with text input"""
    print("=" * 60)
    print("Testing Text Input")
    print("=" * 60)
    
    test_prompts = [
        "heavy metal distortion",
        "clean jazz guitar",
        "ambient reverb",
        "blues crunch tone"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        payload = {"text_input": prompt}
        
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success! Received {len(result.get('PARAMETERS', []))} parameters")
                
                # Display first 5 parameters
                params = result.get('PARAMETERS', [])
                print("\nSample parameters:")
                for param in params[:5]:
                    print(f"  {param['name']:20} = {param['value']:.4f}")
                if len(params) > 5:
                    print(f"  ... and {len(params) - 5} more")
            else:
                print(f"✗ Error: Status code {response.status_code}")
                print(f"  Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("✗ Error: Could not connect to server")
            print("  Make sure the server is running: python API/ai_flask_server.py")
            return
        except Exception as e:
            print(f"✗ Error: {e}")

def test_audio_input():
    """Test the API with audio file input"""
    print("\n" + "=" * 60)
    print("Testing Audio Input")
    print("=" * 60)
    
    # Try to find a sample audio file
    import os
    audio_dir = "Data/generated_audio/train"
    
    if not os.path.exists(audio_dir):
        print("✗ Audio directory not found, skipping audio test")
        return
    
    # Get first audio file
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    if not audio_files:
        print("✗ No audio files found, skipping audio test")
        return
    
    test_file = os.path.join(audio_dir, audio_files[0])
    print(f"\nAudio file: {test_file}")
    
    payload = {"audio_path": test_file}
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success! Received {len(result.get('PARAMETERS', []))} parameters")
            
            # Display first 5 parameters
            params = result.get('PARAMETERS', [])
            print("\nSample parameters:")
            for param in params[:5]:
                print(f"  {param['name']:20} = {param['value']:.4f}")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")
        else:
            print(f"✗ Error: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server")
        print("  Make sure the server is running: python API/ai_flask_server.py")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("\n🎸 Flask API Test Script")
    print("Make sure the server is running before testing!")
    print("Start server with: python API/ai_flask_server.py\n")
    
    # Test text input
    test_text_input()
    
    # Test audio input
    test_audio_input()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
