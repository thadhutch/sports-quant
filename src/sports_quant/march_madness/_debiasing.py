"""Debiasing algorithm for March Madness predictions.

Addresses Team1 positional bias by predicting on both original and
column-swapped data, then averaging the results.

Supports both raw 36-column features (column swap) and difference
21-column features (simple negation).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sports_quant.march_madness._features import DIFF_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def swap_team_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Swap Team1 and Team2 feature columns to address model bias.

    For raw 36-column features only. For difference features, use
    swap_difference_features() instead.

    Args:
        X: DataFrame containing raw team features.

    Returns:
        New DataFrame with Team1 and Team2 columns swapped.
    """
    # Build unique (col1, col2) pairs — each pair swapped exactly once.
    # Read from the original X to avoid overwrite-then-read bugs.
    pairs: list[tuple[str, str]] = []
    for column in X.columns:
        if column.endswith("_Team2"):
            base = column[: -len("_Team2")]
            if base in X.columns:
                pairs.append((base, column))

    swapped_data: dict[str, pd.Series] = {}
    for col1, col2 in pairs:
        swapped_data[col1] = X[col2]
        swapped_data[col2] = X[col1]

    return X.assign(**swapped_data)


def swap_difference_features(X: pd.DataFrame) -> pd.DataFrame:
    """Swap teams by negating all difference features.

    For difference features, swapping Team1 and Team2 is simply
    negation: diff(A,B) = -diff(B,A). This is much simpler than
    the raw column swap approach.

    Args:
        X: DataFrame with difference feature columns.

    Returns:
        New DataFrame with all values negated.
    """
    return X * -1


def _is_difference_features(X: pd.DataFrame) -> bool:
    """Check if a DataFrame uses difference feature columns."""
    return set(X.columns) == set(DIFF_FEATURE_COLUMNS)


def run_debiased_prediction(
    models: list,
    X_original: pd.DataFrame,
    equal_weights: bool = True,
) -> np.ndarray:
    """Run predictions on original and swapped data, then combine.

    Automatically detects whether X_original uses difference features
    or raw features and dispatches to the correct swap function.

    Args:
        models: List of trained models to use for prediction.
        X_original: Original feature data.
        equal_weights: Whether to weight predictions equally.

    Returns:
        Combined probability predictions as numpy array.
    """
    if _is_difference_features(X_original):
        X_swapped = swap_difference_features(X_original)
    else:
        X_swapped = swap_team_columns(X_original)

    # Predictions on original data
    original_probs = [
        model.predict_proba(X_original)[:, 1] for model in models
    ]
    avg_original_probs = np.mean(original_probs, axis=0)

    # Predictions on swapped data (invert probabilities)
    swapped_probs = [
        1 - model.predict_proba(X_swapped)[:, 1] for model in models
    ]
    avg_swapped_probs = np.mean(swapped_probs, axis=0)

    # Combine with equal weights
    combined_probs = (avg_original_probs + avg_swapped_probs) / 2
    return combined_probs


def evaluate_debiased_predictions(
    debiased_probs: np.ndarray,
    y_true: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate debiased predictions and return metrics.

    Args:
        debiased_probs: Combined probability predictions.
        y_true: True labels.
        threshold: Probability threshold for binary classification.

    Returns:
        Tuple of (metrics dict, binary predictions array).
    """
    y_pred = (debiased_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    return metrics, y_pred


def process_backtest_results(
    base_dir: str,
    backtest_year: int,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply debiased algorithm to backtest results for a given year.

    Loads all trained models from the year directory, runs debiased
    predictions using the full ensemble, and saves results.

    Args:
        base_dir: Base directory containing backtest results.
        backtest_year: Year being backtested.
        top_n: Deprecated, ignored. All available models are loaded.

    Returns:
        Tuple of (results DataFrame, metrics dict).
    """
    from pathlib import Path
    from sports_quant.march_madness._config import load_models

    year_dir = Path(base_dir) / str(backtest_year)
    models_dir = year_dir / "models"

    models = load_models(models_dir)
    if not models:
        raise ValueError(f"No models found in {models_dir}")

    # Load backtest data
    X_backtest = pd.read_csv(year_dir / "X_backtest.csv")
    y_backtest = pd.read_csv(year_dir / "y_backtest.csv")["Team1_Win"]
    backtest_teams = pd.read_csv(year_dir / "backtest_teams.csv")

    # Run debiased prediction
    debiased_probs = run_debiased_prediction(models, X_backtest)
    metrics, debiased_preds = evaluate_debiased_predictions(
        debiased_probs, y_backtest
    )

    # Build results
    results = backtest_teams.copy()
    results["Debiased_Prob"] = debiased_probs
    results["Debiased_Pred"] = debiased_preds
    results["Correct_Prediction"] = results["Team1_Win"] == results["Debiased_Pred"]
    results.to_csv(year_dir / "debiased_results.csv", index=False)

    # Analyze upsets
    from sports_quant.march_madness._upsets import analyze_upsets

    upset_analysis = analyze_upsets(
        results["Team1_Win"], results["Debiased_Pred"], results
    )

    # Mark upset predictions
    all_predictions = results.copy()
    all_predictions["Is_Upset_Predicted"] = False
    for i, row in all_predictions.iterrows():
        if (row["Seed1"] < row["Seed2"] and row["Debiased_Pred"] == 0) or (
            row["Seed1"] > row["Seed2"] and row["Debiased_Pred"] == 1
        ):
            all_predictions.at[i, "Is_Upset_Predicted"] = True
    all_predictions.to_csv(year_dir / "debiased_all_predictions.csv", index=False)

    # Save upset predictions
    upset_predictions = all_predictions[
        all_predictions["Is_Upset_Predicted"]
    ].copy()
    upset_predictions["Seed_Difference"] = 0
    for i, row in upset_predictions.iterrows():
        if row["Seed1"] < row["Seed2"]:
            upset_predictions.at[i, "Seed_Difference"] = (
                row["Seed2"] - row["Seed1"]
            )
        else:
            upset_predictions.at[i, "Seed_Difference"] = (
                row["Seed1"] - row["Seed2"]
            )
    upset_predictions = upset_predictions.sort_values(
        "Seed_Difference", ascending=False
    )
    upset_predictions.to_csv(
        year_dir / "debiased_upset_predictions.csv", index=False
    )

    # Save upset analysis text
    with open(year_dir / "debiased_upset_analysis.txt", "w") as f:
        f.write(f"Debiased Algorithm Upset Analysis - Year {backtest_year}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Games: {upset_analysis['total_games']}\n")
        f.write(f"Total Actual Upsets: {upset_analysis['total_upsets_actual']}\n")
        f.write(
            f"Total Predicted Upsets: {upset_analysis['total_upsets_predicted']}\n"
        )
        f.write(
            f"Correctly Predicted Upsets: {upset_analysis['correct_upset_predictions']}\n"
        )
        f.write(
            f"Upset Seed Difference Sum: {upset_analysis['upset_seed_diff_sum']}\n"
        )
        f.write(
            f"Biggest Upset Seed Difference: "
            f"{upset_analysis['biggest_upset_seed_diff']}\n\n"
        )

        if upset_analysis["biggest_upset_details"]:
            bd = upset_analysis["biggest_upset_details"]
            f.write("Biggest Upset Details:\n")
            f.write(f"  Year: {bd['year']}\n")
            f.write(f"  Underdog: {bd['underdog']} (Seed {bd['underdog_seed']})\n")
            f.write(f"  Favorite: {bd['favorite']} (Seed {bd['favorite_seed']})\n")
            f.write(f"  Correctly Predicted: {bd['correctly_predicted']}\n\n")

        f.write("All Predicted Upsets:\n")
        f.write("-" * 50 + "\n")
        for j, upset in enumerate(upset_analysis["upsets"]):
            f.write(
                f"{j+1}. {upset['year']} - {upset['underdog']} "
                f"(Seed {upset['underdog_seed']}) vs. "
                f"{upset['favorite']} (Seed {upset['favorite_seed']}), "
                f"Correctly Predicted: {upset['correctly_predicted']}\n"
            )

    # Save metrics
    with open(year_dir / "debiased_metrics.txt", "w") as f:
        f.write(f"Debiased Algorithm Metrics - Year {backtest_year}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(
            f"Total Predicted Upsets: "
            f"{upset_analysis['total_upsets_predicted']}\n"
        )
        f.write(
            f"Correctly Predicted Upsets: "
            f"{upset_analysis['correct_upset_predictions']}\n"
        )
        correct = upset_analysis["correct_upset_predictions"]
        total = max(1, upset_analysis["total_upsets_predicted"])
        f.write(f"Upset Prediction Accuracy: {correct / total:.4f}\n")

    logger.info(
        "Debiased results for year %d: accuracy=%.4f, predicted_upsets=%d",
        backtest_year,
        metrics["accuracy"],
        upset_analysis["total_upsets_predicted"],
    )

    return results, metrics
