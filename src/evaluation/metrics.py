"""
src/evaluation/metrics.py
Comprehensive evaluation metrics for ECD delay prediction models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, cohen_kappa_score, matthews_corrcoef
)

def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC metrics)
    
    Returns:
        Dict with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # AUC metrics (if probabilities provided)
    if y_proba is not None:
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = 0.5
            metrics['avg_precision'] = y_true.mean()
    
    return metrics

def print_classification_summary(y_true, y_pred, y_proba=None, target_names=None):
    """
    Print comprehensive classification summary
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        target_names: Class names (optional)
    """
    if target_names is None:
        target_names = ['On Track', 'Delayed']
    
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    
    metrics = calculate_all_metrics(y_true, y_pred, y_proba)
    
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"   {metric:20}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = np.array(metrics['confusion_matrix'])
    print(f"   TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"   FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Calculate rates
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   True Negative Rate (Specificity): {tn/(tn+fp):.4f}")
    print(f"   True Positive Rate (Sensitivity): {tp/(tp+fn):.4f}")
    print(f"   Positive Predictive Value:        {tp/(tp+fp):.4f}")
    print(f"   Negative Predictive Value:        {tn/(tn+fn):.4f}")

def calculate_ci(metric_value, n_samples, confidence=0.95):
    """
    Calculate confidence interval for a metric
    
    Args:
        metric_value: Metric value (0-1)
        n_samples: Sample size
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Wilson score interval for proportions
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / n_samples
    centre = (metric_value + z**2 / (2 * n_samples)) / denominator
    margin = z * np.sqrt((metric_value * (1 - metric_value) + z**2 / (4 * n_samples)) / n_samples) / denominator
    
    return max(0, centre - margin), min(1, centre + margin)

def metrics_to_dataframe(results_dict):
    """
    Convert results dictionary to pandas DataFrame
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
    
    Returns:
        DataFrame with model comparison
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'model_name': model_name}
        row.update(metrics)
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('model_name')