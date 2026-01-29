"""
K-Fold Cross-Validation utilities for Liver Fibrosis Staging.
Provides stratified k-fold splitting and results aggregation.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import RANDOM_SEED, METRICS_DIR, CLASS_NAMES


class CrossValidationResults:
    """
    Aggregates and analyzes results from k-fold cross-validation.
    Computes mean, std, and confidence intervals for all metrics.
    """
    
    def __init__(self, num_folds: int = 5, class_names: List[str] = CLASS_NAMES):
        self.num_folds = num_folds
        self.class_names = class_names
        self.fold_results = []
        self.fold_predictions = []
        self.fold_labels = []
    
    def add_fold_result(self, 
                        fold: int,
                        accuracy: float,
                        kappa: float,
                        f1_weighted: float,
                        f1_macro: float,
                        per_class_acc: Dict[str, float],
                        predictions: List[int],
                        labels: List[int],
                        val_loss: float = 0.0):
        """Add results from one fold."""
        self.fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'kappa': kappa,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'per_class_accuracy': per_class_acc,
            'val_loss': val_loss
        })
        self.fold_predictions.append(predictions)
        self.fold_labels.append(labels)
    
    def compute_statistics(self) -> Dict:
        """
        Compute mean, std, and 95% confidence intervals for all metrics.
        
        Returns:
            Dictionary with aggregated statistics
        """
        accuracies = [r['accuracy'] for r in self.fold_results]
        kappas = [r['kappa'] for r in self.fold_results]
        f1_weighted = [r['f1_weighted'] for r in self.fold_results]
        f1_macro = [r['f1_macro'] for r in self.fold_results]
        
        def compute_ci(values, confidence=0.95):
            """Compute confidence interval using t-distribution."""
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(n)
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_value * se
            return mean - margin, mean + margin
        
        # Compute per-class accuracy statistics
        per_class_stats = {}
        for class_name in self.class_names:
            class_accs = [r['per_class_accuracy'].get(class_name, 0.0) 
                         for r in self.fold_results]
            ci_low, ci_high = compute_ci(class_accs)
            per_class_stats[class_name] = {
                'mean': np.mean(class_accs),
                'std': np.std(class_accs, ddof=1),
                'ci_95': (ci_low, ci_high)
            }
        
        # Compute overall statistics
        acc_ci = compute_ci(accuracies)
        kappa_ci = compute_ci(kappas)
        
        return {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies, ddof=1),
                'ci_95': acc_ci,
                'all_folds': accuracies
            },
            'cohens_kappa': {
                'mean': np.mean(kappas),
                'std': np.std(kappas, ddof=1),
                'ci_95': kappa_ci,
                'all_folds': kappas
            },
            'f1_weighted': {
                'mean': np.mean(f1_weighted),
                'std': np.std(f1_weighted, ddof=1),
                'all_folds': f1_weighted
            },
            'f1_macro': {
                'mean': np.mean(f1_macro),
                'std': np.std(f1_macro, ddof=1),
                'all_folds': f1_macro
            },
            'per_class_accuracy': per_class_stats,
            'num_folds': self.num_folds
        }
    
    def save_results(self, output_dir: Path = METRICS_DIR):
        """Save all results to files."""
        output_dir = Path(output_dir)
        cv_dir = output_dir / 'cross_validation'
        cv_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics
        stats = self.compute_statistics()
        
        # Save detailed results as JSON
        results_json = {
            'fold_results': self.fold_results,
            'aggregated_statistics': {
                'accuracy': {
                    'mean': stats['accuracy']['mean'],
                    'std': stats['accuracy']['std'],
                    'ci_95_low': stats['accuracy']['ci_95'][0],
                    'ci_95_high': stats['accuracy']['ci_95'][1],
                    'all_folds': stats['accuracy']['all_folds']
                },
                'cohens_kappa': {
                    'mean': stats['cohens_kappa']['mean'],
                    'std': stats['cohens_kappa']['std'],
                    'ci_95_low': stats['cohens_kappa']['ci_95'][0],
                    'ci_95_high': stats['cohens_kappa']['ci_95'][1],
                    'all_folds': stats['cohens_kappa']['all_folds']
                },
                'f1_weighted': {
                    'mean': stats['f1_weighted']['mean'],
                    'std': stats['f1_weighted']['std']
                },
                'f1_macro': {
                    'mean': stats['f1_macro']['mean'],
                    'std': stats['f1_macro']['std']
                }
            }
        }
        
        with open(cv_dir / 'cv_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save summary text file
        self._save_summary_text(cv_dir, stats)
        
        # Generate visualizations
        self._plot_fold_comparison(cv_dir, stats)
        self._plot_per_class_accuracy(cv_dir, stats)
        
        print(f"\nCross-validation results saved to: {cv_dir}")
        
        return stats
    
    def _save_summary_text(self, output_dir: Path, stats: Dict):
        """Save a human-readable summary."""
        with open(output_dir / 'cv_summary.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("K-FOLD CROSS-VALIDATION RESULTS - LIVER FIBROSIS STAGING\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Number of Folds: {stats['num_folds']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n\n")
            
            # Accuracy
            acc = stats['accuracy']
            f.write(f"Accuracy: {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}%\n")
            f.write(f"  95% CI: [{acc['ci_95'][0]*100:.2f}%, {acc['ci_95'][1]*100:.2f}%]\n")
            f.write(f"  Per fold: {[f'{a*100:.2f}%' for a in acc['all_folds']]}\n\n")
            
            # Cohen's Kappa
            kappa = stats['cohens_kappa']
            f.write(f"Cohen's Kappa (Quadratic): {kappa['mean']:.4f} ± {kappa['std']:.4f}\n")
            f.write(f"  95% CI: [{kappa['ci_95'][0]:.4f}, {kappa['ci_95'][1]:.4f}]\n")
            f.write(f"  Per fold: {[f'{k:.4f}' for k in kappa['all_folds']]}\n\n")
            
            # F1 Scores
            f1w = stats['f1_weighted']
            f1m = stats['f1_macro']
            f.write(f"F1 Score (Weighted): {f1w['mean']:.4f} ± {f1w['std']:.4f}\n")
            f.write(f"F1 Score (Macro): {f1m['mean']:.4f} ± {f1m['std']:.4f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("PER-CLASS ACCURACY\n")
            f.write("-" * 70 + "\n\n")
            
            for class_name, class_stats in stats['per_class_accuracy'].items():
                f.write(f"{class_name}: {class_stats['mean']*100:.2f}% ± {class_stats['std']*100:.2f}%\n")
                f.write(f"  95% CI: [{class_stats['ci_95'][0]*100:.2f}%, {class_stats['ci_95'][1]*100:.2f}%]\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("For your research paper, report results as:\n")
            f.write(f"  \"Accuracy: {acc['mean']*100:.1f}% (95% CI: {acc['ci_95'][0]*100:.1f}%-{acc['ci_95'][1]*100:.1f}%)\"\n")
            f.write(f"  \"Cohen's κ: {kappa['mean']:.3f} (95% CI: {kappa['ci_95'][0]:.3f}-{kappa['ci_95'][1]:.3f})\"\n")
            f.write("=" * 70 + "\n")
    
    def _plot_fold_comparison(self, output_dir: Path, stats: Dict):
        """Create bar chart comparing folds."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [f"Fold {i+1}" for i in range(self.num_folds)]
        
        # Accuracy plot
        ax1 = axes[0]
        accuracies = [a * 100 for a in stats['accuracy']['all_folds']]
        bars1 = ax1.bar(folds, accuracies, color='steelblue', alpha=0.8)
        ax1.axhline(y=stats['accuracy']['mean'] * 100, color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {stats['accuracy']['mean']*100:.2f}%")
        ax1.fill_between(range(-1, self.num_folds + 1), 
                        stats['accuracy']['ci_95'][0] * 100,
                        stats['accuracy']['ci_95'][1] * 100,
                        alpha=0.2, color='red', label='95% CI')
        ax1.set_xlim(-0.5, self.num_folds - 0.5)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy per Fold', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Kappa plot
        ax2 = axes[1]
        kappas = stats['cohens_kappa']['all_folds']
        bars2 = ax2.bar(folds, kappas, color='forestgreen', alpha=0.8)
        ax2.axhline(y=stats['cohens_kappa']['mean'], color='red',
                   linestyle='--', linewidth=2, label=f"Mean: {stats['cohens_kappa']['mean']:.4f}")
        ax2.fill_between(range(-1, self.num_folds + 1),
                        stats['cohens_kappa']['ci_95'][0],
                        stats['cohens_kappa']['ci_95'][1],
                        alpha=0.2, color='red', label='95% CI')
        ax2.set_xlim(-0.5, self.num_folds - 0.5)
        ax2.set_ylabel("Cohen's Kappa", fontsize=12)
        ax2.set_title("Cohen's Kappa per Fold", fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars2, kappas):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved fold comparison chart")
    
    def _plot_per_class_accuracy(self, output_dir: Path, stats: Dict):
        """Create per-class accuracy chart with error bars."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(stats['per_class_accuracy'].keys())
        means = [stats['per_class_accuracy'][c]['mean'] * 100 for c in classes]
        stds = [stats['per_class_accuracy'][c]['std'] * 100 for c in classes]
        
        x = np.arange(len(classes))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='coral', alpha=0.8,
                     error_kw={'linewidth': 2, 'capthick': 2})
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Fibrosis Stage', fontsize=12)
        ax.set_title('Per-Class Accuracy (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved per-class accuracy chart")


def create_stratified_kfold_splits(dataset, num_folds: int = 5, 
                                    seed: int = RANDOM_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-fold splits for the dataset.
    
    Args:
        dataset: The full dataset
        num_folds: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    splits = []
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        splits.append((train_idx, val_idx))
    
    return splits


def compute_fold_metrics(predictions: List[int], 
                         labels: List[int],
                         class_names: List[str] = CLASS_NAMES) -> Dict:
    """
    Compute all metrics for a single fold.
    
    Args:
        predictions: Model predictions
        labels: True labels
        class_names: Names of classes
        
    Returns:
        Dictionary with all computed metrics
    """
    accuracy = accuracy_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
    f1_w = f1_score(labels, predictions, average='weighted')
    f1_m = f1_score(labels, predictions, average='macro')
    
    # Per-class accuracy
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)
    
    per_class_acc = {}
    for idx, class_name in enumerate(class_names):
        mask = labels_arr == idx
        if mask.sum() > 0:
            per_class_acc[class_name] = float((preds_arr[mask] == labels_arr[mask]).mean())
        else:
            per_class_acc[class_name] = 0.0
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'per_class_accuracy': per_class_acc
    }


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing CrossValidationResults...")
    
    cv_results = CrossValidationResults(num_folds=5)
    
    # Simulate 5 folds
    for fold in range(5):
        np.random.seed(42 + fold)
        # Simulate slightly varying results
        accuracy = 0.85 + np.random.uniform(-0.05, 0.05)
        kappa = 0.80 + np.random.uniform(-0.05, 0.05)
        f1_w = 0.84 + np.random.uniform(-0.05, 0.05)
        f1_m = 0.82 + np.random.uniform(-0.05, 0.05)
        
        per_class = {
            'F0': 0.90 + np.random.uniform(-0.05, 0.05),
            'F1': 0.75 + np.random.uniform(-0.08, 0.08),
            'F2': 0.72 + np.random.uniform(-0.08, 0.08),
            'F3': 0.78 + np.random.uniform(-0.08, 0.08),
            'F4': 0.88 + np.random.uniform(-0.05, 0.05),
        }
        
        cv_results.add_fold_result(
            fold=fold,
            accuracy=accuracy,
            kappa=kappa,
            f1_weighted=f1_w,
            f1_macro=f1_m,
            per_class_acc=per_class,
            predictions=list(range(100)),
            labels=list(range(100))
        )
    
    stats = cv_results.save_results()
    
    print(f"\nTest completed!")
    print(f"Mean Accuracy: {stats['accuracy']['mean']*100:.2f}% ± {stats['accuracy']['std']*100:.2f}%")
    print(f"Mean Kappa: {stats['cohens_kappa']['mean']:.4f} ± {stats['cohens_kappa']['std']:.4f}")
