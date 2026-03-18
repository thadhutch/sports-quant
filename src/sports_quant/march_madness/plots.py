"""Plotting functions for March Madness models."""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


def plot_learning_curve(
    results: dict,
    rank: int,
    out_path: str,
    year: int | None = None,
) -> None:
    """Plot training/validation learning curves for a model."""
    title = f"Learning Curve - Top Model #{rank}"
    if year:
        title += f" - Year {year}"

    plt.figure(figsize=(10, 6))
    for metric in ["binary_logloss", "auc"]:
        plt.plot(results["training"][metric], label=f"Training {metric}")
        plt.plot(results["valid_1"][metric], label=f"Validation {metric}")
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: list[str] | np.ndarray,
    rank: int,
    out_path: str,
    year: int | None = None,
) -> None:
    """Plot feature importance bar chart for a model."""
    title = f"Feature Importance - Top Model #{rank}"
    if year:
        title += f" - Year {year}"

    sorted_idx = np.argsort(feature_importances)[::-1]
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(12, 6))
    plt.bar(pos, feature_importances[sorted_idx], align="center")
    plt.xticks(pos, np.array(feature_names)[sorted_idx], rotation=90)
    plt.ylabel("Feature Importance")
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rank: int,
    year: int,
    out_path: str,
) -> None:
    """Plot confusion matrix for a model's backtest predictions."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Top Model #{rank} - Year {year}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Team2 Win", "Team1 Win"])
    plt.yticks(tick_marks, ["Team2 Win", "Team1 Win"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_distribution(
    accuracies: list[float],
    out_path: str,
) -> None:
    """Plot histogram of model accuracies."""
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, alpha=0.7)
    plt.axvline(
        np.mean(accuracies), color="red", linestyle="dashed", linewidth=2
    )
    plt.title("Distribution of Model Accuracies")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_average_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | np.ndarray,
    out_path: str,
) -> None:
    """Plot average feature importance across all models."""
    sorted_idx = np.argsort(importances)[::-1]
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(12, 6))
    plt.bar(pos, importances[sorted_idx], align="center")
    plt.xticks(pos, np.array(feature_names)[sorted_idx], rotation=90)
    plt.ylabel("Average Feature Importance")
    plt.title("Average Feature Importance Across All Models")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_popular_upsets(
    popular_upsets: list[dict],
    year: int,
    num_models: int,
    out_path: str,
    max_display: int = 15,
) -> None:
    """Plot bar chart of most popular upset predictions."""
    if not popular_upsets:
        return

    top_upsets = popular_upsets[: min(max_display, len(popular_upsets))]

    plt.figure(figsize=(12, 8))

    labels = [
        f"{u['underdog']} (S{u['underdog_seed']}) over "
        f"{u['favorite']} (S{u['favorite_seed']})"
        for u in top_upsets
    ]
    counts = [u["count"] for u in top_upsets]
    percentages = [(u["count"] / num_models) * 100 for u in top_upsets]
    colors = ["green" if u["is_actual_upset"] else "red" for u in top_upsets]

    bars = plt.barh(range(len(labels)), percentages, align="center", color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Percentage of Models (%)")
    plt.title(f"Most Popular Upset Predictions - Year {year}")

    for bar, count, upset in zip(bars, counts, top_upsets):
        label = f"{count}/{num_models}"
        label += " \u2713" if upset["is_actual_upset"] else " \u2717"
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
        )

    legend_elements = [
        Patch(facecolor="green", label="Correct Upset Prediction"),
        Patch(facecolor="red", label="Incorrect Upset Prediction"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_validation_f1_distribution(
    f1_scores: list[float],
    year: int,
    out_path: str,
) -> None:
    """Plot distribution of validation F1 scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(f1_scores, bins=20, alpha=0.7)
    plt.axvline(np.mean(f1_scores), color="red", linestyle="dashed", linewidth=2)
    plt.title(f"Distribution of Validation F1 Scores - Year {year}")
    plt.xlabel("Validation F1")
    plt.ylabel("Frequency")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_validation_vs_backtest(
    f1_scores: list[float],
    accuracies: list[float],
    year: int,
    out_path: str,
) -> None:
    """Plot validation F1 vs backtest accuracy scatter."""
    plt.figure(figsize=(10, 6))
    plt.scatter(f1_scores, accuracies, alpha=0.7)
    plt.title(f"Validation F1 vs Backtest Accuracy - Year {year}")
    plt.xlabel("Validation F1")
    plt.ylabel("Backtest Accuracy")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_ensemble_accuracy_by_year(
    years: list[int],
    accuracies: list[float],
    out_path: str,
) -> None:
    """Plot ensemble model accuracy across backtest years."""
    plt.figure(figsize=(10, 6))
    plt.plot(years, accuracies, marker="o", linestyle="-")
    plt.title("Ensemble Model Performance by Year")
    plt.xlabel("Year")
    plt.ylabel("Ensemble Accuracy")
    plt.xticks(years)
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_ensemble_vs_debiased(
    years: list[int],
    ensemble_acc: list[float],
    debiased_acc: list[float],
    out_path: str,
) -> None:
    """Plot ensemble vs debiased accuracy by year."""
    plt.figure(figsize=(10, 6))
    plt.plot(years, ensemble_acc, marker="o", linestyle="-", label="Ensemble")
    plt.plot(years, debiased_acc, marker="x", linestyle="--", label="Debiased")
    plt.title("Model Performance by Year")
    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.xticks(years)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_round(
    X_train, X_test, y_train, y_test, model,
    round_column: str = "CURRENT ROUND",
    save_path: str | None = None,
) -> None:
    """Train a model and plot accuracy by tournament round."""
    unique_rounds = np.sort(X_train[round_column].unique())
    accuracies = []

    for current_round in unique_rounds:
        X_train_round = X_train[X_train[round_column] == current_round]
        y_train_round = y_train[X_train[round_column] == current_round]
        X_test_round = X_test[X_test[round_column] == current_round]
        y_test_round = y_test[X_test[round_column] == current_round]

        if X_train_round.empty or X_test_round.empty:
            logger.debug("Skipping round %s due to insufficient data.", current_round)
            accuracies.append(np.nan)
            continue

        model.fit(X_train_round, y_train_round)
        y_pred_round = model.predict(X_test_round)
        accuracy = accuracy_score(y_test_round, y_pred_round)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(unique_rounds, accuracies, marker="o")
    plt.xlabel("Current Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Current Round")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
