"""
Model evaluation utilities.

💡 KEY CONCEPT: How do you know if your model is any good?
You can't just check accuracy — if 85% of players DON'T score,
a model that always says "no goal" would be 85% accurate but
completely useless.

Instead, we use metrics designed for imbalanced binary prediction:
- ROC AUC: How well does the model separate scorers from non-scorers?
  (1.0 = perfect, 0.5 = coin flip)
- Precision: When the model says "goal", how often is it right?
- Recall: Of all actual goals, how many did the model catch?
- Brier Score: Are the predicted probabilities well-calibrated?
  (lower is better)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    classification_report,
)


def evaluate_model(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Evaluate a binary classification model.

    Args:
        y_true: Actual labels (0 or 1)
        y_prob: Predicted probabilities (0.0 to 1.0)
        threshold: Probability cutoff for positive prediction

    Returns:
        Dict of evaluation metrics.

    💡 CONCEPT: The "threshold" is where you draw the line.
    At 0.5, the model predicts "goal" if probability > 50%.
    Lower thresholds catch more goals but have more false alarms.
    Higher thresholds are more conservative but miss more goals.
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "total_samples": len(y_true),
        "actual_positives": int(y_true.sum()),
        "predicted_positives": int(y_pred.sum()),
    }

    return metrics


def print_evaluation(metrics: dict, model_name: str = "Model"):
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Output from evaluate_model()
        model_name: Label for display
    """
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  ROC AUC:    {metrics['roc_auc']:.3f}  (1.0=perfect, 0.5=coin flip)")
    print(f"  Brier:      {metrics['brier_score']:.3f}  (lower is better)")
    print(f"  Precision:  {metrics['precision']:.3f}  (when it says 'goal', how often correct?)")
    print(f"  Recall:     {metrics['recall']:.3f}  (of all goals, how many did it catch?)")
    print(f"  F1 Score:   {metrics['f1']:.3f}  (balance of precision & recall)")
    print(f"  Threshold:  {metrics['threshold']}")
    print(f"  Samples:    {metrics['total_samples']} total, "
          f"{metrics['actual_positives']} actual goals, "
          f"{metrics['predicted_positives']} predicted goals")
    print(f"{'='*50}")
