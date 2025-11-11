"""
Test Model Evaluation Script

This script evaluates the trained music tone classifier on a separate test dataset.
It provides comprehensive performance metrics including:
- Classification accuracy (overall and per-class)
- Confusion matrix
- Precision, Recall, F1-score for each tone class
- Knob parameter prediction errors (MSE, MAE, R²)
- Per-parameter error analysis
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, mean_squared_error, 
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from music_tone_classifier import MusicToneClassifier

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

def load_test_data(test_data_dir):
    """
    Load test data from the Test Data directory
    
    Args:
        test_data_dir: Path to Test Data folder
        
    Returns:
        DataFrame with test data including audio paths and labels
    """
    print_section("Loading Test Data")
    
    # Load test index CSV
    index_path = os.path.join(test_data_dir, "indexes", "index_train.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Test index not found: {index_path}")
    
    df = pd.read_csv(index_path)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Loaded {len(df)} test samples from: {index_path}")
    
    # Add full audio path (Test Data has files directly in generated_audio folder)
    audio_dir = os.path.join(test_data_dir, "generated_audio")
    df['audio_path'] = df['basename'].apply(lambda x: os.path.join(audio_dir, f"{x}.wav"))
    
    # Verify files exist
    missing = df[~df['audio_path'].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"{Colors.WARNING}⚠{Colors.ENDC} Warning: {len(missing)} audio files not found")
        df = df[df['audio_path'].apply(os.path.exists)]
        print(f"{Colors.GREEN}✓{Colors.ENDC} Using {len(df)} samples with available audio files")
    else:
        print(f"{Colors.GREEN}✓{Colors.ENDC} All {len(df)} audio files verified")
    
    # Extract tone class from scenario column
    df['tone_class'] = df['scenario']
    
    print(f"\n{Colors.BOLD}Test Data Distribution:{Colors.ENDC}")
    class_counts = df['tone_class'].value_counts().sort_index()
    for tone, count in class_counts.items():
        print(f"  {tone:12s}: {count:4d} samples")
    
    return df

def evaluate_classification(classifier, test_df):
    """
    Evaluate tone classification performance
    
    Args:
        classifier: Trained MusicToneClassifier instance
        test_df: Test data DataFrame
        
    Returns:
        Dictionary with classification metrics
    """
    print_section("Evaluating Tone Classification")
    
    # Get predictions
    y_true = []
    y_pred = []
    
    print(f"Classifying {len(test_df)} test samples...")
    for idx, row in test_df.iterrows():
        audio_path = row['audio_path']
        true_label = row['tone_class']
        
        try:
            result = classifier.classify_audio(audio_path)
            pred_label = result['tone_type']
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} samples...")
        except Exception as e:
            print(f"{Colors.WARNING}⚠{Colors.ENDC} Error classifying {audio_path}: {e}")
            continue
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{Colors.BOLD}Overall Classification Accuracy:{Colors.ENDC} {Colors.GREEN}{accuracy*100:.2f}%{Colors.ENDC}")
    
    # Classification report
    print(f"\n{Colors.BOLD}Per-Class Performance:{Colors.ENDC}")
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Print in formatted table
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    classes = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
    for cls in classes:
        metrics = report[cls]
        print(f"{cls:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10d}")
    
    print("-" * 60)
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:>10.3f} "
          f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")
    print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:>10.3f} "
          f"{report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(f"\n{Colors.BOLD}Confusion Matrix:{Colors.ENDC}")
    header_label = "True \\ Pred"
    print(f"\n{header_label:<15}", end='')
    for cls in classes:
        print(f"{cls:>12}", end='')
    print()
    print("-" * (15 + 12 * len(classes)))
    
    for i, true_cls in enumerate(classes):
        print(f"{true_cls:<15}", end='')
        for j, pred_cls in enumerate(classes):
            count = cm[i, j]
            if i == j:
                print(f"{Colors.GREEN}{count:>12d}{Colors.ENDC}", end='')
            else:
                print(f"{count:>12d}", end='')
        print()
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'report': report,
        'confusion_matrix': cm,
        'classes': classes
    }

def evaluate_knob_parameters(classifier, test_df):
    """
    Evaluate knob parameter prediction performance
    
    Args:
        classifier: Trained MusicToneClassifier instance
        test_df: Test data DataFrame
        
    Returns:
        Dictionary with parameter prediction metrics
    """
    print_section("Evaluating Knob Parameter Prediction")
    
    # Knob parameter columns (with "n_" prefix in CSV)
    knob_params_csv = [
        'n_overdrivedrive', 'n_distortiondrive', 'n_distortiontone',
        'n_eqbass', 'n_eqmid', 'n_eqtreble',
        'n_chorusrate', 'n_chorusdepth', 'n_chorusmix',
        'n_delaytime', 'n_delayfeedback', 'n_delaymix',
        'n_reverbt60', 'n_reverbdamp', 'n_reverbsize', 'n_reverbwet',
        'n_mastervolume'
    ]
    
    # Knob parameter names in classifier (without "n_" prefix)
    knob_params_model = [
        'overdrivedrive', 'distortiondrive', 'distortiontone',
        'eqbass', 'eqmid', 'eqtreble',
        'chorusrate', 'chorusdepth', 'chorusmix',
        'delaytime', 'delayfeedback', 'delaymix',
        'reverbt60', 'reverbdamp', 'reverbsize', 'reverbwet',
        'mastervolume'
    ]
    
    y_true_params = []
    y_pred_params = []
    
    print(f"Predicting parameters for {len(test_df)} test samples...")
    for idx, row in test_df.iterrows():
        audio_path = row['audio_path']
        
        try:
            result = classifier.classify_audio(audio_path)
            
            # Get true and predicted parameters
            true_vals = [row[csv_param] for csv_param in knob_params_csv]
            pred_vals = [result['knob_settings'][model_param] for model_param in knob_params_model]
            
            y_true_params.append(true_vals)
            y_pred_params.append(pred_vals)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} samples...")
        except Exception as e:
            print(f"{Colors.WARNING}⚠{Colors.ENDC} Error predicting parameters for {audio_path}: {e}")
            continue
    
    y_true_params = np.array(y_true_params)
    y_pred_params = np.array(y_pred_params)
    
    # Overall metrics
    overall_mse = mean_squared_error(y_true_params, y_pred_params)
    overall_mae = mean_absolute_error(y_true_params, y_pred_params)
    overall_r2 = r2_score(y_true_params, y_pred_params)
    
    print(f"\n{Colors.BOLD}Overall Parameter Prediction Performance:{Colors.ENDC}")
    print(f"  Mean Squared Error (MSE):  {overall_mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  R² Score:                  {overall_r2:.4f}")
    
    # Per-parameter metrics
    print(f"\n{Colors.BOLD}Per-Parameter Performance:{Colors.ENDC}")
    print(f"\n{'Parameter':<25} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    
    param_metrics = {}
    for i, csv_param in enumerate(knob_params_csv):
        # Remove "n_" prefix for display
        display_name = csv_param[2:]  # Remove "n_" prefix
        param_mse = mean_squared_error(y_true_params[:, i], y_pred_params[:, i])
        param_mae = mean_absolute_error(y_true_params[:, i], y_pred_params[:, i])
        param_r2 = r2_score(y_true_params[:, i], y_pred_params[:, i])
        
        param_metrics[display_name] = {
            'mse': param_mse,
            'mae': param_mae,
            'r2': param_r2
        }
        
        print(f"{display_name:<25} {param_mse:>10.6f} {param_mae:>10.6f} {param_r2:>10.4f}")
    
    return {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'param_metrics': param_metrics,
        'y_true': y_true_params,
        'y_pred': y_pred_params,
        'param_names': [p[2:] for p in knob_params_csv]  # Remove "n_" prefix for display
    }

def save_confusion_matrix_plot(cm, classes, output_path):
    """Save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved confusion matrix plot: {output_path}")

def save_parameter_error_plot(param_metrics, output_path):
    """Save parameter error visualization"""
    params = list(param_metrics.keys())
    mae_values = [param_metrics[p]['mae'] for p in params]
    
    # Shorten parameter names for better display
    short_names = [p.replace('n_', '').replace('drive', 'drv').replace('tone', 'tn') 
                   for p in params]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(params)), mae_values, color='steelblue', alpha=0.7)
    plt.xlabel('Knob Parameter', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.title('Parameter Prediction Errors - Test Set', fontsize=16, fontweight='bold')
    plt.xticks(range(len(params)), short_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved parameter error plot: {output_path}")

def save_results_to_file(classification_results, parameter_results, output_path):
    """Save comprehensive test results to text file"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MUSIC TONE CLASSIFIER - TEST SET EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Classification results
        f.write("TONE CLASSIFICATION PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Accuracy: {classification_results['accuracy']*100:.2f}%\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 60 + "\n")
        
        report = classification_results['report']
        for cls in classification_results['classes']:
            metrics = report[cls]
            f.write(f"{cls:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                   f"{metrics['f1-score']:>10.3f} {int(metrics['support']):>10d}\n")
        
        f.write("\n")
        f.write(f"{'Macro Avg':<15} {report['macro avg']['precision']:>10.3f} "
               f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}\n")
        f.write(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:>10.3f} "
               f"{report['weighted avg']['recall']:>10.3f} {report['weighted avg']['f1-score']:>10.3f}\n")
        
        # Confusion matrix
        f.write("\n\nConfusion Matrix:\n")
        cm = classification_results['confusion_matrix']
        classes = classification_results['classes']
        
        header_label = "True \\ Pred"
        f.write(f"{header_label:<15}")
        for cls in classes:
            f.write(f"{cls:>12}")
        f.write("\n" + "-" * (15 + 12 * len(classes)) + "\n")
        
        for i, true_cls in enumerate(classes):
            f.write(f"{true_cls:<15}")
            for j in range(len(classes)):
                f.write(f"{cm[i, j]:>12d}")
            f.write("\n")
        
        # Parameter prediction results
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("KNOB PARAMETER PREDICTION PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Mean Squared Error (MSE):  {parameter_results['overall_mse']:.6f}\n")
        f.write(f"  Mean Absolute Error (MAE): {parameter_results['overall_mae']:.6f}\n")
        f.write(f"  R² Score:                  {parameter_results['overall_r2']:.4f}\n\n")
        
        f.write("Per-Parameter Metrics:\n")
        f.write(f"{'Parameter':<25} {'MSE':>10} {'MAE':>10} {'R²':>10}\n")
        f.write("-" * 60 + "\n")
        
        for param, metrics in parameter_results['param_metrics'].items():
            f.write(f"{param:<25} {metrics['mse']:>10.6f} {metrics['mae']:>10.6f} "
                   f"{metrics['r2']:>10.4f}\n")
    
    print(f"{Colors.GREEN}✓{Colors.ENDC} Saved detailed results: {output_path}")

def main():
    """Main evaluation function"""
    print_header("MUSIC TONE CLASSIFIER - TEST SET EVALUATION")
    
    # Paths
    workspace_dir = r"C:\Users\AIT 33\Desktop\Clap"
    test_data_dir = os.path.join(workspace_dir, "Test Data")
    models_dir = os.path.join(workspace_dir, "models")
    results_dir = os.path.join(workspace_dir, "test_results")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if models exist
    tone_model_path = os.path.join(models_dir, "tone_classifier.pkl")
    knob_model_path = os.path.join(models_dir, "knob_regressor_net.pth")
    
    if not os.path.exists(tone_model_path):
        print(f"{Colors.FAIL}✗{Colors.ENDC} Error: Tone classifier model not found: {tone_model_path}")
        print(f"  Please train the model first using train_advanced.py")
        sys.exit(1)
    
    if not os.path.exists(knob_model_path):
        print(f"{Colors.FAIL}✗{Colors.ENDC} Error: Knob parameter model not found: {knob_model_path}")
        print(f"  Please train the model first using train_advanced.py")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓{Colors.ENDC} Found trained models:")
    print(f"  - Tone classifier: {tone_model_path}")
    print(f"  - Knob parameter network: {knob_model_path}")
    
    # Load test data
    test_df = load_test_data(test_data_dir)
    
    # Load classifier
    print_section("Loading Trained Classifier")
    classifier = MusicToneClassifier()
    classifier.load_models(models_dir)
    print(f"{Colors.GREEN}✓{Colors.ENDC} Successfully loaded trained models")
    
    # Evaluate classification
    classification_results = evaluate_classification(classifier, test_df)
    
    # Evaluate knob parameters
    parameter_results = evaluate_knob_parameters(classifier, test_df)
    
    # Save visualizations
    print_section("Saving Results")
    
    cm_plot_path = os.path.join(results_dir, "confusion_matrix_test.png")
    save_confusion_matrix_plot(
        classification_results['confusion_matrix'],
        classification_results['classes'],
        cm_plot_path
    )
    
    param_plot_path = os.path.join(results_dir, "parameter_errors_test.png")
    save_parameter_error_plot(
        parameter_results['param_metrics'],
        param_plot_path
    )
    
    results_file_path = os.path.join(results_dir, "test_evaluation_results.txt")
    save_results_to_file(classification_results, parameter_results, results_file_path)
    
    # Summary
    print_header("EVALUATION COMPLETE")
    print(f"{Colors.BOLD}Test Results Summary:{Colors.ENDC}")
    print(f"  Classification Accuracy: {Colors.GREEN}{classification_results['accuracy']*100:.2f}%{Colors.ENDC}")
    print(f"  Parameter MSE:           {parameter_results['overall_mse']:.6f}")
    print(f"  Parameter MAE:           {parameter_results['overall_mae']:.6f}")
    print(f"  Parameter R²:            {parameter_results['overall_r2']:.4f}")
    print(f"\n{Colors.BOLD}Results saved to:{Colors.ENDC} {results_dir}")
    print(f"  - Detailed report:     test_evaluation_results.txt")
    print(f"  - Confusion matrix:    confusion_matrix_test.png")
    print(f"  - Parameter errors:    parameter_errors_test.png")
    print()

if __name__ == "__main__":
    main()
