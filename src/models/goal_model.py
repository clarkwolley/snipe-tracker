"""
Goal Scorer Prediction Model.

💡 KEY CONCEPT: This is a BINARY CLASSIFICATION problem.
For each player in each game, we predict: "Will this player score?"
The answer is yes (1) or no (0), but we output a PROBABILITY
(e.g., 0.35 = 35% chance of scoring).

We train two models and compare them:
1. Logistic Regression — simple, interpretable, fast
2. Gradient Boosting — more powerful, captures complex patterns

Why two? Always start simple. If the simple model works well,
you don't need the complex one. If it doesn't, the complex one
shows you what you're leaving on the table.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TEST_SIZE, RANDOM_STATE
from src.features.player_features import (
    build_player_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.models.evaluate import evaluate_model, print_evaluation


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _ensure_model_dir():
    """Create models/ directory if it doesn't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_training_data(game_log: pd.DataFrame) -> tuple:
    """
    Prepare features and target from raw game log.

    Args:
        game_log: Raw game log DataFrame

    Returns:
        Tuple of (X, y, full_featured_df)
        - X: Feature matrix (just the model input columns)
        - y: Target vector (scored: 0 or 1)
        - df: Full DataFrame with all columns (for analysis)

    💡 CONCEPT: We separate the FEATURES (inputs) from the
    TARGET (what we're predicting). The model learns the
    relationship: "given these features → probability of goal."

    We also drop rows where rolling averages are NaN — these
    are first-appearances where we have no history to reference.
    """
    df = build_player_features(game_log)

    # Drop rows with missing features
    df = df.dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y, df


def train_goal_model(game_log: pd.DataFrame) -> dict:
    """
    Train and evaluate goal scorer models.

    This is the main training pipeline:
    1. Build features from raw game log
    2. Split into train/test sets
    3. Train logistic regression (baseline)
    4. Train gradient boosting (advanced)
    5. Evaluate both and pick the winner
    6. Save the best model

    Args:
        game_log: Raw game log DataFrame

    Returns:
        Dict with training results, metrics, and model artifacts.

    💡 KEY CONCEPT — Train/Test Split:
    We hold out 20% of the data and NEVER train on it.
    The model learns from the 80% (training set), then we
    check its predictions on the 20% it's never seen (test set).
    This tells us if the model has learned real patterns vs.
    just memorizing the training data ("overfitting").
    """
    print("🏗️  Preparing features...")
    X, y, df = prepare_training_data(game_log)
    print(f"   {len(X)} samples, {len(FEATURE_COLUMNS)} features")
    print(f"   Scoring rate: {y.mean()*100:.1f}% of player-games")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Model 1: Logistic Regression (baseline) ---
    print("\n🔵 Training Logistic Regression (baseline)...")
    lr_model = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight="balanced",  # handles imbalanced classes
        max_iter=1000,
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_probs, threshold=0.20)
    print_evaluation(lr_metrics, "Logistic Regression")
    results["logistic_regression"] = lr_metrics

    # Show feature importance for logistic regression
    print("\n   Feature weights (logistic regression):")
    for feat, coef in sorted(
        zip(FEATURE_COLUMNS, lr_model.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        direction = "↑" if coef > 0 else "↓"
        print(f"     {direction} {feat}: {coef:+.3f}")

    # --- Model 2: Gradient Boosting (advanced) ---
    print("\n🟢 Training Gradient Boosting (advanced)...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb_model.fit(X_train, y_train)  # GB doesn't need scaling
    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate_model(y_test, gb_probs, threshold=0.20)
    print_evaluation(gb_metrics, "Gradient Boosting")
    results["gradient_boosting"] = gb_metrics

    # Show feature importance for gradient boosting
    print("\n   Feature importance (gradient boosting):")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, gb_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "█" * int(imp * 50)
        print(f"     {feat:25s} {imp:.3f} {bar}")

    # --- Pick the winner ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = lr_model if best_name == "logistic_regression" else gb_model
    best_scaler = scaler if best_name == "logistic_regression" else None

    print(f"\n🏆 Winner: {best_name} (AUC: {results[best_name]['roc_auc']:.3f})")

    # Save the best model
    _ensure_model_dir()
    model_path = os.path.join(MODEL_DIR, "goal_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "goal_scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "goal_model_meta.pkl")

    joblib.dump(best_model, model_path)
    if best_scaler is not None:
        joblib.dump(best_scaler, scaler_path)
    joblib.dump(
        {"model_type": best_name, "metrics": results[best_name],
         "feature_columns": FEATURE_COLUMNS, "needs_scaling": best_scaler is not None},
        meta_path,
    )

    print(f"💾 Model saved to {model_path}")

    return {
        "best_model_name": best_name,
        "results": results,
        "model": best_model,
        "scaler": best_scaler,
    }


def load_goal_model() -> tuple:
    """
    Load a previously trained goal model.

    Returns:
        Tuple of (model, scaler_or_None, metadata_dict)

    Raises:
        FileNotFoundError: If no trained model exists.
    """
    model_path = os.path.join(MODEL_DIR, "goal_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "goal_scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "goal_model_meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train_goal_model() first!"
        )

    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler, meta


def predict_goal_probability(model, scaler, player_features: pd.DataFrame) -> np.ndarray:
    """
    Predict goal-scoring probability for players.

    Args:
        model: Trained sklearn model
        scaler: StandardScaler (or None if not needed)
        player_features: DataFrame with FEATURE_COLUMNS

    Returns:
        Array of probabilities (0.0 to 1.0)

    💡 CONCEPT: The model outputs a probability, not a yes/no.
    "0.35" means "35% chance this player scores tonight."
    This is more useful than a binary prediction because
    you can rank players by likelihood and set your own
    threshold for what counts as "likely."
    """
    X = player_features[FEATURE_COLUMNS].copy()

    if scaler is not None:
        X = scaler.transform(X)

    return model.predict_proba(X)[:, 1]
