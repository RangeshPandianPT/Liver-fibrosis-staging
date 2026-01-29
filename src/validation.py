"""
Validation and metrics utilities for liver fibrosis classification.
Includes confusion matrix generation and Cohen's Kappa calculation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    cohen_kappa_score, 
    classification_report,
    accuracy_score,
    f1_score
)
from typing import List, Tuple, Dict
import json

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import CLASS_NAMES, METRICS_DIR


def generate_confusion_matrix(y_true: List[int],
                               y_pred: List[int],
                               class_names: List[str] = CLASS_NAMES,
                               normalize: bool = True,
                               save_path: str = None,
                               figsize: Tuple[int, int] = (10, 8)) -> np.ndarray:
    """
    Generate and visualize a multi-class confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure (if None, uses default)
        figsize: Figure size
        
    Returns:
        Confusion matrix as numpy array
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
        title = 'Normalized Confusion Matrix - Liver Fibrosis Staging'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix - Liver Fibrosis Staging'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        square=True
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = METRICS_DIR / 'confusion_matrix.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    
    return cm


def compute_cohens_kappa(y_true: List[int],
                          y_pred: List[int],
                          weights: str = 'quadratic',
                          save_path: str = None) -> float:
    """
    Compute Cohen's Kappa score for inter-rater reliability.
    
    Quadratic weighting is appropriate for ordinal data like fibrosis stages
    where the distance between classes matters (F0 vs F4 is more severe than F0 vs F1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        weights: 'linear', 'quadratic', or None
        save_path: Path to save the score
        
    Returns:
        Cohen's Kappa score
    """
    kappa = cohen_kappa_score(y_true, y_pred, weights=weights)
    
    # Interpretation
    if kappa < 0:
        interpretation = "Less than chance agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    # Save to file
    if save_path is None:
        save_path = METRICS_DIR / 'kappa_score.txt'
    
    with open(save_path, 'w') as f:
        f.write(f"Cohen's Kappa Score (Quadratic Weighted)\n")
        f.write(f"=" * 45 + "\n\n")
        f.write(f"Kappa Score: {kappa:.4f}\n")
        f.write(f"Interpretation: {interpretation}\n\n")
        f.write(f"Weighting: {weights}\n")
        f.write(f"Number of samples: {len(y_true)}\n")
    
    print(f"Cohen's Kappa: {kappa:.4f} ({interpretation})")
    print(f"Kappa score saved to: {save_path}")
    
    return kappa


def generate_classification_report(y_true: List[int],
                                    y_pred: List[int],
                                    class_names: List[str] = CLASS_NAMES,
                                    save_path: str = None) -> Dict:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        save_path: Path to save the report
        
    Returns:
        Classification report as dictionary
    """
    # Generate report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names
    )
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Save to file
    if save_path is None:
        save_path = METRICS_DIR / 'classification_report.txt'
    
    with open(save_path, 'w') as f:
        f.write("Classification Report - Liver Fibrosis Staging\n")
        f.write("=" * 55 + "\n\n")
        f.write(report_str)
        f.write("\n" + "=" * 55 + "\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"  F1 (macro): {f1_macro:.4f}\n")
    
    print(f"Classification report saved to: {save_path}")
    
    # Also save as JSON
    json_path = METRICS_DIR / 'classification_report.json'
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    return report_dict


def compute_per_class_accuracy(y_true: List[int],
                                y_pred: List[int],
                                class_names: List[str] = CLASS_NAMES) -> Dict[str, float]:
    """
    Compute accuracy for each class separately.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        
    Returns:
        Dictionary mapping class names to accuracies
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    per_class_acc = {}
    for idx, class_name in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            per_class_acc[class_name] = float(acc)
        else:
            per_class_acc[class_name] = 0.0
    
    return per_class_acc


def generate_all_metrics(y_true: List[int],
                          y_pred: List[int],
                          class_names: List[str] = CLASS_NAMES) -> Dict:
    """
    Generate all validation metrics and save to files.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        
    Returns:
        Dictionary containing all computed metrics
    """
    print("\n" + "=" * 60)
    print("GENERATING VALIDATION METRICS")
    print("=" * 60 + "\n")
    
    # Confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred, class_names)
    
    # Cohen's Kappa
    kappa = compute_cohens_kappa(y_true, y_pred)
    
    # Classification report
    report = generate_classification_report(y_true, y_pred, class_names)
    
    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(y_true, y_pred, class_names)
    
    print("\nPer-class Accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name}: {acc:.4f}")
    
    print("\n" + "=" * 60 + "\n")
    
    return {
        'confusion_matrix': cm,
        'cohens_kappa': kappa,
        'classification_report': report,
        'per_class_accuracy': per_class_acc,
        'overall_accuracy': accuracy_score(y_true, y_pred)
    }


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 100)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(100, 20, replace=False)
    y_pred[noise_idx] = np.random.randint(0, 5, 20)
    
    metrics = generate_all_metrics(y_true.tolist(), y_pred.tolist())
    print(f"\nTest completed. Overall accuracy: {metrics['overall_accuracy']:.4f}")
