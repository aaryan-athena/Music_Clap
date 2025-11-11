"""
NAM Real Audio Testing Script

This script tests the trained music tone classifier on real audio samples from the NAM_testing folder.
It evaluates:
1. Tone classification accuracy (comparing predicted vs actual from filename)
2. Knob parameter prediction accuracy (comparing predicted vs JSON values)
3. Per-sample detailed analysis showing exact differences

This helps validate model performance on real-world audio (not synthetic data).
"""

import os
import json
import numpy as np
import pandas as pd
from music_tone_classifier import MusicToneClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}{Colors.ENDC}\n")

def print_section(text):
    """Print formatted section"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(text)}{Colors.ENDC}")

def extract_tone_from_filename(filename):
    """
    Extract expected tone class from filename
    Files start with: clean_, crunch_, high-gain_, Clean__, Crunch__, High-Gain__, Wellness__
    """
    filename_lower = filename.lower()
    
    if filename_lower.startswith('clean'):
        return 'Clean'
    elif filename_lower.startswith('crunch'):
        return 'Crunch'
    elif filename_lower.startswith('high-gain') or filename_lower.startswith('high_gain'):
        return 'High-Gain'
    elif filename_lower.startswith('wellness'):
        return 'Wellness'
    else:
        return 'Unknown'

def load_json_parameters(json_path):
    """Load ground truth parameters from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['params']

def calculate_parameter_differences(predicted, actual):
    """Calculate differences between predicted and actual parameters"""
    differences = {}
    for param in actual.keys():
        if param in predicted:
            diff = abs(predicted[param] - actual[param])
            differences[param] = {
                'predicted': predicted[param],
                'actual': actual[param],
                'difference': diff,
                'percent_error': (diff / (actual[param] + 1e-10)) * 100  # Avoid division by zero
            }
    return differences

def test_nam_samples(classifier, nam_dir):
    """
    Test classifier on NAM real audio samples
    
    Args:
        classifier: Trained MusicToneClassifier
        nam_dir: Path to NAM_testing directory
    
    Returns:
        Dictionary with detailed results
    """
    audio_dir = os.path.join(nam_dir, "generated_audio")
    labels_dir = os.path.join(nam_dir, "labels_normalised")
    
    # Get all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    print(f"Found {len(audio_files)} real audio samples to test")
    
    results = {
        'samples': [],
        'classification_correct': 0,
        'classification_total': 0,
        'all_predicted_params': [],
        'all_actual_params': [],
        'param_names': []
    }
    
    # Process each sample
    for audio_file in sorted(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        json_file = audio_file.replace('.wav', '.json')
        json_path = os.path.join(labels_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"{Colors.WARNING}⚠{Colors.ENDC} JSON file not found for {audio_file}, skipping...")
            continue
        
        # Extract expected tone from filename
        expected_tone = extract_tone_from_filename(audio_file)
        
        # Load ground truth parameters
        actual_params = load_json_parameters(json_path)
        
        # Get model predictions
        try:
            prediction = classifier.classify_audio(audio_path)
            predicted_tone = prediction['tone_type']
            predicted_params = prediction['knob_settings']
            confidence = prediction['confidence']
            
            # Check classification correctness
            is_correct = (predicted_tone == expected_tone)
            if is_correct:
                results['classification_correct'] += 1
            results['classification_total'] += 1
            
            # Calculate parameter differences
            param_diffs = calculate_parameter_differences(predicted_params, actual_params)
            
            # Store results for this sample
            sample_result = {
                'filename': audio_file,
                'expected_tone': expected_tone,
                'predicted_tone': predicted_tone,
                'confidence': confidence,
                'classification_correct': is_correct,
                'parameter_differences': param_diffs,
                'avg_absolute_error': np.mean([d['difference'] for d in param_diffs.values()]),
                'max_error': max([d['difference'] for d in param_diffs.values()]),
                'max_error_param': max(param_diffs.items(), key=lambda x: x[1]['difference'])[0]
            }
            
            results['samples'].append(sample_result)
            
            # Collect for overall statistics
            results['all_predicted_params'].append([predicted_params[p] for p in sorted(actual_params.keys())])
            results['all_actual_params'].append([actual_params[p] for p in sorted(actual_params.keys())])
            if not results['param_names']:
                results['param_names'] = sorted(actual_params.keys())
            
        except Exception as e:
            print(f"{Colors.FAIL}✗{Colors.ENDC} Error processing {audio_file}: {e}")
            continue
    
    return results

def print_detailed_results(results):
    """Print detailed per-sample results"""
    print_section("Per-Sample Detailed Analysis")
    
    for i, sample in enumerate(results['samples'], 1):
        print(f"\n{Colors.BOLD}Sample {i}: {sample['filename']}{Colors.ENDC}")
        print(f"{'─' * 80}")
        
        # Classification result
        if sample['classification_correct']:
            status = f"{Colors.GREEN}✓ CORRECT{Colors.ENDC}"
        else:
            status = f"{Colors.FAIL}✗ INCORRECT{Colors.ENDC}"
        
        print(f"Expected Tone:  {sample['expected_tone']}")
        print(f"Predicted Tone: {sample['predicted_tone']} (confidence: {sample['confidence']:.2%}) {status}")
        
        # Parameter accuracy summary
        print(f"\nParameter Prediction Summary:")
        print(f"  Average Absolute Error: {sample['avg_absolute_error']:.4f}")
        print(f"  Maximum Error:          {sample['max_error']:.4f} (parameter: {sample['max_error_param']})")
        
        # Detailed parameter comparison (top 5 best and worst)
        param_diffs = sample['parameter_differences']
        sorted_params = sorted(param_diffs.items(), key=lambda x: x[1]['difference'])
        
        print(f"\n  {Colors.GREEN}Best Predictions (lowest error):{Colors.ENDC}")
        for param, diff in sorted_params[:5]:
            print(f"    {param:20s}: Pred={diff['predicted']:.4f}, Actual={diff['actual']:.4f}, "
                  f"Diff={diff['difference']:.4f}")
        
        print(f"\n  {Colors.WARNING}Worst Predictions (highest error):{Colors.ENDC}")
        for param, diff in sorted_params[-5:]:
            print(f"    {param:20s}: Pred={diff['predicted']:.4f}, Actual={diff['actual']:.4f}, "
                  f"Diff={diff['difference']:.4f}")

def print_overall_statistics(results):
    """Print overall statistics across all samples"""
    print_section("Overall Statistics")
    
    # Classification accuracy
    classification_acc = (results['classification_correct'] / results['classification_total']) * 100
    print(f"\n{Colors.BOLD}Tone Classification Performance:{Colors.ENDC}")
    print(f"  Correct:  {results['classification_correct']}/{results['classification_total']}")
    print(f"  Accuracy: {Colors.GREEN}{classification_acc:.2f}%{Colors.ENDC}")
    
    # Parameter prediction statistics
    pred_array = np.array(results['all_predicted_params'])
    actual_array = np.array(results['all_actual_params'])
    
    overall_mse = mean_squared_error(actual_array, pred_array)
    overall_mae = mean_absolute_error(actual_array, pred_array)
    overall_r2 = r2_score(actual_array, pred_array)
    
    print(f"\n{Colors.BOLD}Parameter Prediction Performance:{Colors.ENDC}")
    print(f"  Mean Squared Error (MSE):  {overall_mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  R² Score:                  {overall_r2:.4f}")
    
    # Per-parameter statistics
    print(f"\n{Colors.BOLD}Per-Parameter Performance:{Colors.ENDC}")
    print(f"\n{'Parameter':<25} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("─" * 60)
    
    for i, param in enumerate(results['param_names']):
        param_mse = mean_squared_error(actual_array[:, i], pred_array[:, i])
        param_mae = mean_absolute_error(actual_array[:, i], pred_array[:, i])
        param_r2 = r2_score(actual_array[:, i], pred_array[:, i])
        
        print(f"{param:<25} {param_mse:>10.6f} {param_mae:>10.6f} {param_r2:>10.4f}")
    
    # Average errors per sample
    print(f"\n{Colors.BOLD}Average Errors Per Sample:{Colors.ENDC}")
    avg_errors = [s['avg_absolute_error'] for s in results['samples']]
    print(f"  Mean:   {np.mean(avg_errors):.4f}")
    print(f"  Median: {np.median(avg_errors):.4f}")
    print(f"  Min:    {np.min(avg_errors):.4f}")
    print(f"  Max:    {np.max(avg_errors):.4f}")

def save_results_to_file(results, output_path):
    """Save results to a detailed text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NAM REAL AUDIO TESTING RESULTS\n")
        f.write("Music Tone Classifier - Real-World Audio Validation\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        classification_acc = (results['classification_correct'] / results['classification_total']) * 100
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples Tested: {results['classification_total']}\n\n")
        
        f.write("Tone Classification:\n")
        f.write(f"  Correct Predictions: {results['classification_correct']}/{results['classification_total']}\n")
        f.write(f"  Accuracy: {classification_acc:.2f}%\n\n")
        
        # Parameter statistics
        pred_array = np.array(results['all_predicted_params'])
        actual_array = np.array(results['all_actual_params'])
        
        overall_mse = mean_squared_error(actual_array, pred_array)
        overall_mae = mean_absolute_error(actual_array, pred_array)
        overall_r2 = r2_score(actual_array, pred_array)
        
        f.write("Parameter Prediction:\n")
        f.write(f"  Mean Squared Error (MSE):  {overall_mse:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {overall_mae:.6f}\n")
        f.write(f"  R² Score:                  {overall_r2:.4f}\n\n")
        
        # Per-parameter performance
        f.write("PER-PARAMETER PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Parameter':<25} {'MSE':>10} {'MAE':>10} {'R²':>10}\n")
        f.write("-" * 60 + "\n")
        
        for i, param in enumerate(results['param_names']):
            param_mse = mean_squared_error(actual_array[:, i], pred_array[:, i])
            param_mae = mean_absolute_error(actual_array[:, i], pred_array[:, i])
            param_r2 = r2_score(actual_array[:, i], pred_array[:, i])
            f.write(f"{param:<25} {param_mse:>10.6f} {param_mae:>10.6f} {param_r2:>10.4f}\n")
        
        # Detailed per-sample results
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETAILED PER-SAMPLE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, sample in enumerate(results['samples'], 1):
            f.write(f"Sample {i}: {sample['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Expected Tone:  {sample['expected_tone']}\n")
            f.write(f"Predicted Tone: {sample['predicted_tone']} (confidence: {sample['confidence']:.2%})\n")
            f.write(f"Classification: {'CORRECT' if sample['classification_correct'] else 'INCORRECT'}\n\n")
            
            f.write(f"Parameter Accuracy:\n")
            f.write(f"  Average Absolute Error: {sample['avg_absolute_error']:.4f}\n")
            f.write(f"  Maximum Error: {sample['max_error']:.4f} ({sample['max_error_param']})\n\n")
            
            f.write(f"{'Parameter':<25} {'Predicted':>12} {'Actual':>12} {'Difference':>12}\n")
            f.write("-" * 65 + "\n")
            
            for param, diff in sorted(sample['parameter_differences'].items()):
                f.write(f"{param:<25} {diff['predicted']:>12.4f} {diff['actual']:>12.4f} "
                       f"{diff['difference']:>12.4f}\n")
            
            f.write("\n")

def save_csv_comparison(results, output_path):
    """Save comparison data to CSV for further analysis"""
    rows = []
    
    for sample in results['samples']:
        for param, diff in sample['parameter_differences'].items():
            rows.append({
                'filename': sample['filename'],
                'expected_tone': sample['expected_tone'],
                'predicted_tone': sample['predicted_tone'],
                'classification_correct': sample['classification_correct'],
                'confidence': sample['confidence'],
                'parameter': param,
                'predicted_value': diff['predicted'],
                'actual_value': diff['actual'],
                'absolute_difference': diff['difference'],
                'percent_error': diff['percent_error']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved CSV comparison: {output_path}")

def test_set2_predictions(classifier, set2_audio_dir):
    """
    Test classifier on set_2 audio samples (prediction only, no ground truth)
    
    Args:
        classifier: Trained MusicToneClassifier
        set2_audio_dir: Path to set_2 generated_audio directory
    
    Returns:
        List of prediction results
    """
    audio_files = [f for f in os.listdir(set2_audio_dir) if f.endswith('.wav')]
    
    print(f"Found {len(audio_files)} set_2 audio samples (prediction only)")
    
    predictions = []
    
    for audio_file in sorted(audio_files):
        audio_path = os.path.join(set2_audio_dir, audio_file)
        
        try:
            prediction = classifier.classify_audio(audio_path)
            
            predictions.append({
                'filename': audio_file,
                'predicted_tone': prediction['tone_type'],
                'confidence': prediction['confidence'],
                'knob_settings': prediction['knob_settings']
            })
            
        except Exception as e:
            print(f"{Colors.FAIL}✗{Colors.ENDC} Error processing {audio_file}: {e}")
            continue
    
    return predictions

def save_combined_results(set1_results, set2_predictions, output_path):
    """Save combined results for both set_1 and set_2 to a single file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NAM REAL AUDIO TESTING RESULTS - COMBINED REPORT\n")
        f.write("Music Tone Classifier - Real-World Audio Validation\n")
        f.write("=" * 80 + "\n\n")
        
        # ========== SET 1 RESULTS (WITH COMPARISON) ==========
        f.write("█" * 80 + "\n")
        f.write("SET 1: COMPARISON WITH GROUND TRUTH VALUES\n")
        f.write("█" * 80 + "\n\n")
        
        # Overall statistics for set_1
        classification_acc = (set1_results['classification_correct'] / set1_results['classification_total']) * 100
        f.write("OVERALL STATISTICS - SET 1\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples Tested: {set1_results['classification_total']}\n\n")
        
        f.write("Tone Classification:\n")
        f.write(f"  Correct Predictions: {set1_results['classification_correct']}/{set1_results['classification_total']}\n")
        f.write(f"  Accuracy: {classification_acc:.2f}%\n\n")
        
        # Parameter statistics for set_1
        pred_array = np.array(set1_results['all_predicted_params'])
        actual_array = np.array(set1_results['all_actual_params'])
        
        overall_mse = mean_squared_error(actual_array, pred_array)
        overall_mae = mean_absolute_error(actual_array, pred_array)
        overall_r2 = r2_score(actual_array, pred_array)
        
        f.write("Parameter Prediction:\n")
        f.write(f"  Mean Squared Error (MSE):  {overall_mse:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {overall_mae:.6f}\n")
        f.write(f"  R² Score:                  {overall_r2:.4f}\n\n")
        
        # Per-parameter performance for set_1
        f.write("PER-PARAMETER PERFORMANCE - SET 1\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Parameter':<25} {'MSE':>10} {'MAE':>10} {'R²':>10}\n")
        f.write("-" * 60 + "\n")
        
        for i, param in enumerate(set1_results['param_names']):
            param_mse = mean_squared_error(actual_array[:, i], pred_array[:, i])
            param_mae = mean_absolute_error(actual_array[:, i], pred_array[:, i])
            param_r2 = r2_score(actual_array[:, i], pred_array[:, i])
            f.write(f"{param:<25} {param_mse:>10.6f} {param_mae:>10.6f} {param_r2:>10.4f}\n")
        
        # Detailed per-sample results for set_1
        f.write("\n\nDETAILED PER-SAMPLE RESULTS - SET 1\n")
        f.write("-" * 80 + "\n\n")
        
        for i, sample in enumerate(set1_results['samples'], 1):
            f.write(f"SET 1 - Sample {i}: {sample['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Expected Tone:  {sample['expected_tone']}\n")
            f.write(f"Predicted Tone: {sample['predicted_tone']} (confidence: {sample['confidence']:.2%})\n")
            f.write(f"Classification: {'✓ CORRECT' if sample['classification_correct'] else '✗ INCORRECT'}\n\n")
            
            f.write(f"Parameter Accuracy:\n")
            f.write(f"  Average Absolute Error: {sample['avg_absolute_error']:.4f}\n")
            f.write(f"  Maximum Error: {sample['max_error']:.4f} ({sample['max_error_param']})\n\n")
            
            f.write(f"{'Parameter':<25} {'Predicted':>12} {'Actual':>12} {'Difference':>12}\n")
            f.write("-" * 65 + "\n")
            
            for param, diff in sorted(sample['parameter_differences'].items()):
                f.write(f"{param:<25} {diff['predicted']:>12.4f} {diff['actual']:>12.4f} "
                       f"{diff['difference']:>12.4f}\n")
            
            f.write("\n")
        
        # ========== SET 2 RESULTS (PREDICTIONS ONLY) ==========
        f.write("\n\n" + "█" * 80 + "\n")
        f.write("SET 2: PREDICTIONS ONLY (NO GROUND TRUTH AVAILABLE)\n")
        f.write("█" * 80 + "\n\n")
        
        f.write("OVERVIEW - SET 2\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples Predicted: {len(set2_predictions)}\n\n")
        
        # Count predictions by tone type
        tone_counts = {}
        for pred in set2_predictions:
            tone = pred['predicted_tone']
            tone_counts[tone] = tone_counts.get(tone, 0) + 1
        
        f.write("Predicted Tone Distribution:\n")
        for tone, count in sorted(tone_counts.items()):
            f.write(f"  {tone:12s}: {count:3d} samples ({count/len(set2_predictions)*100:.1f}%)\n")
        
        # Average confidence
        avg_confidence = np.mean([p['confidence'] for p in set2_predictions])
        f.write(f"\nAverage Prediction Confidence: {avg_confidence:.2%}\n")
        
        # Detailed predictions for set_2
        f.write("\n\nDETAILED PREDICTIONS - SET 2\n")
        f.write("-" * 80 + "\n\n")
        
        for i, pred in enumerate(set2_predictions, 1):
            f.write(f"SET 2 - Sample {i}: {pred['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Predicted Tone: {pred['predicted_tone']} (confidence: {pred['confidence']:.2%})\n\n")
            
            f.write(f"{'Parameter':<25} {'Predicted Value':>18}\n")
            f.write("-" * 45 + "\n")
            
            for param, value in sorted(pred['knob_settings'].items()):
                f.write(f"{param:<25} {value:>18.4f}\n")
            
            f.write("\n")
        
        # ========== SUMMARY ==========
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"SET 1 (With Ground Truth):\n")
        f.write(f"  - Samples: {set1_results['classification_total']}\n")
        f.write(f"  - Classification Accuracy: {classification_acc:.2f}%\n")
        f.write(f"  - Parameter MAE: {overall_mae:.4f}\n\n")
        
        f.write(f"SET 2 (Predictions Only):\n")
        f.write(f"  - Samples: {len(set2_predictions)}\n")
        f.write(f"  - Average Confidence: {avg_confidence:.2%}\n")
        f.write(f"  - Most Common Prediction: {max(tone_counts, key=tone_counts.get)} "
               f"({tone_counts[max(tone_counts, key=tone_counts.get)]} samples)\n\n")
        
        f.write(f"Total Samples Processed: {set1_results['classification_total'] + len(set2_predictions)}\n")

def main():
    """Main testing function"""
    print_header("NAM REAL AUDIO TESTING - SETS 1 & 2")
    print(f"{Colors.BOLD}Testing model on real-world audio samples{Colors.ENDC}\n")
    
    # Paths
    workspace_dir = r"C:\Users\AIT 33\Desktop\Clap"
    nam_testing_2_dir = os.path.join(workspace_dir, "NAM_testing", "NAM_testing_2")
    
    set1_audio_dir = os.path.join(nam_testing_2_dir, "set_1_generated_audio")
    set1_labels_dir = os.path.join(nam_testing_2_dir, "set_1_labels_normalised")
    set2_audio_dir = os.path.join(nam_testing_2_dir, "set_2_generated_audio")
    
    models_dir = os.path.join(workspace_dir, "models")
    results_dir = os.path.join(workspace_dir, "nam_test_results")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(nam_testing_2_dir):
        print(f"{Colors.FAIL}✗{Colors.ENDC} NAM_testing_2 directory not found: {nam_testing_2_dir}")
        return
    
    # Load classifier
    print_section("Loading Trained Models")
    classifier = MusicToneClassifier()
    
    try:
        classifier.load_models(models_dir)
        print(f"{Colors.GREEN}✓{Colors.ENDC} Successfully loaded models from: {models_dir}")
    except Exception as e:
        print(f"{Colors.FAIL}✗{Colors.ENDC} Error loading models: {e}")
        return
    
    # ========== Process Set 1 (with ground truth) ==========
    print_section("SET 1: Testing with Ground Truth Comparison")
    
    # Temporarily create a dict structure for set_1 that mimics the old NAM_testing structure
    set1_temp_structure = {
        'generated_audio': set1_audio_dir,
        'labels_normalised': set1_labels_dir
    }
    
    # Reuse the existing test function but point to set_1 directories
    audio_files_set1 = [f for f in os.listdir(set1_audio_dir) if f.endswith('.wav')]
    
    set1_results = {
        'samples': [],
        'classification_correct': 0,
        'classification_total': 0,
        'all_predicted_params': [],
        'all_actual_params': [],
        'param_names': []
    }
    
    print(f"Found {len(audio_files_set1)} samples in set_1")
    
    for audio_file in sorted(audio_files_set1):
        audio_path = os.path.join(set1_audio_dir, audio_file)
        json_file = audio_file.replace('.wav', '.json')
        json_path = os.path.join(set1_labels_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"{Colors.WARNING}⚠{Colors.ENDC} JSON file not found for {audio_file}, skipping...")
            continue
        
        expected_tone = extract_tone_from_filename(audio_file)
        actual_params = load_json_parameters(json_path)
        
        try:
            prediction = classifier.classify_audio(audio_path)
            predicted_tone = prediction['tone_type']
            predicted_params = prediction['knob_settings']
            confidence = prediction['confidence']
            
            is_correct = (predicted_tone == expected_tone)
            if is_correct:
                set1_results['classification_correct'] += 1
            set1_results['classification_total'] += 1
            
            param_diffs = calculate_parameter_differences(predicted_params, actual_params)
            
            sample_result = {
                'filename': audio_file,
                'expected_tone': expected_tone,
                'predicted_tone': predicted_tone,
                'confidence': confidence,
                'classification_correct': is_correct,
                'parameter_differences': param_diffs,
                'avg_absolute_error': np.mean([d['difference'] for d in param_diffs.values()]),
                'max_error': max([d['difference'] for d in param_diffs.values()]),
                'max_error_param': max(param_diffs.items(), key=lambda x: x[1]['difference'])[0]
            }
            
            set1_results['samples'].append(sample_result)
            set1_results['all_predicted_params'].append([predicted_params[p] for p in sorted(actual_params.keys())])
            set1_results['all_actual_params'].append([actual_params[p] for p in sorted(actual_params.keys())])
            
            if not set1_results['param_names']:
                set1_results['param_names'] = sorted(actual_params.keys())
            
            if (len(set1_results['samples'])) % 10 == 0:
                print(f"  Processed {len(set1_results['samples'])}/{len(audio_files_set1)} set_1 samples...")
                
        except Exception as e:
            print(f"{Colors.FAIL}✗{Colors.ENDC} Error processing {audio_file}: {e}")
            continue
    
    print(f"{Colors.GREEN}✓{Colors.ENDC} Completed set_1: {set1_results['classification_total']} samples")
    
    # ========== Process Set 2 (predictions only) ==========
    print_section("SET 2: Generating Predictions (No Ground Truth)")
    set2_predictions = test_set2_predictions(classifier, set2_audio_dir)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Completed set_2: {len(set2_predictions)} samples")
    
    # Print summary statistics
    print_section("SET 1 - Summary Statistics")
    classification_acc = (set1_results['classification_correct'] / set1_results['classification_total']) * 100
    print(f"Classification Accuracy: {Colors.GREEN}{classification_acc:.2f}%{Colors.ENDC}")
    
    pred_array = np.array(set1_results['all_predicted_params'])
    actual_array = np.array(set1_results['all_actual_params'])
    overall_mae = mean_absolute_error(actual_array, pred_array)
    print(f"Parameter MAE: {overall_mae:.4f}")
    
    print_section("SET 2 - Summary Statistics")
    tone_counts = {}
    for pred in set2_predictions:
        tone = pred['predicted_tone']
        tone_counts[tone] = tone_counts.get(tone, 0) + 1
    
    print("Predicted Tone Distribution:")
    for tone, count in sorted(tone_counts.items()):
        print(f"  {tone:12s}: {count:3d} samples")
    
    avg_confidence = np.mean([p['confidence'] for p in set2_predictions])
    print(f"Average Confidence: {avg_confidence:.2%}")
    
    # Save combined results
    print_section("Saving Combined Results")
    
    combined_file = os.path.join(results_dir, "nam_combined_sets_results.txt")
    save_combined_results(set1_results, set2_predictions, combined_file)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved combined results: {combined_file}")
    
    # Save CSV for set_1 comparison
    csv_file = os.path.join(results_dir, "nam_set1_parameter_comparison.csv")
    save_csv_comparison(set1_results, csv_file)
    
    # Save CSV for set_2 predictions
    set2_csv_file = os.path.join(results_dir, "nam_set2_predictions.csv")
    set2_rows = []
    for pred in set2_predictions:
        for param, value in pred['knob_settings'].items():
            set2_rows.append({
                'filename': pred['filename'],
                'predicted_tone': pred['predicted_tone'],
                'confidence': pred['confidence'],
                'parameter': param,
                'predicted_value': value
            })
    pd.DataFrame(set2_rows).to_csv(set2_csv_file, index=False)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved set_2 predictions CSV: {set2_csv_file}")
    
    # Summary
    print_header("TESTING COMPLETE")
    
    print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"\n{Colors.CYAN}SET 1 (With Ground Truth):{Colors.ENDC}")
    print(f"  Samples Tested:          {set1_results['classification_total']}")
    print(f"  Classification Accuracy: {Colors.GREEN}{classification_acc:.2f}%{Colors.ENDC}")
    print(f"  Parameter MAE:           {overall_mae:.4f}")
    
    print(f"\n{Colors.CYAN}SET 2 (Predictions Only):{Colors.ENDC}")
    print(f"  Samples Predicted:       {len(set2_predictions)}")
    print(f"  Average Confidence:      {avg_confidence:.2%}")
    print(f"  Most Common Prediction:  {max(tone_counts, key=tone_counts.get)} "
          f"({tone_counts[max(tone_counts, key=tone_counts.get)]} samples)")
    
    print(f"\n{Colors.BOLD}Results saved to:{Colors.ENDC} {results_dir}")
    print(f"  - Combined report:     nam_combined_sets_results.txt")
    print(f"  - Set 1 CSV:          nam_set1_parameter_comparison.csv")
    print(f"  - Set 2 CSV:          nam_set2_predictions.csv")
    print()

if __name__ == "__main__":
    main()
