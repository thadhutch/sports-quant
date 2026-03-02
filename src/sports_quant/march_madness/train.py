"""March Madness model training orchestrator.

Trains N LightGBM models with different random seeds, selects the top models
by F1 score, and saves models, plots, and metrics.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sports_quant import _config as config
from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._data import load_and_prepare
from sports_quant.march_madness import plots

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load the 'march_madness' section from model_config.yaml."""
    with open(config.MODEL_CONFIG_FILE) as f:
        full = yaml.safe_load(f)
    return full["march_madness"]


def run_training() -> None:
    """Train an ensemble of LightGBM models for March Madness prediction.

    1. Loads training data and config.
    2. Trains N models with different random seeds.
    3. Selects top models by validation F1 score.
    4. Saves models, plots, and aggregate metrics.
    """
    cfg = _load_config()
    model_version = cfg["model_version"]
    num_models = cfg["models_to_train"]
    top_n = cfg["top_models"]
    test_size = cfg["train"]["test_size"]
    hyperparams = cfg["hyperparameters"]

    # Setup directories
    plots_dir = Path(mm_config.MM_PLOTS_DIR / model_version)
    models_dir = Path(mm_config.MM_MODELS_DIR / model_version)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_and_prepare()
    X, y = data.X, data.y

    logger.info(
        "Training %d models (version %s) on %d samples with %d features",
        num_models, model_version, len(X), X.shape[1],
    )

    # Train models
    all_model_data = []
    for model_num in range(num_models):
        random_seed = np.random.randint(1, 10000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )

        model = LGBMClassifier(
            objective=hyperparams["objective"],
            metric=hyperparams["metric"],
        )

        evalset = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=evalset,
            eval_metric=["binary_logloss", "auc"],
        )

        results = model.evals_result_
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        model_data = {
            "model_num": model_num,
            "seed": random_seed,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "model": model,
            "results": results,
            "feature_importances": model.feature_importances_,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }
        all_model_data.append(model_data)

        logger.info(
            "Model %d/%d - Acc: %.4f, F1: %.4f",
            model_num + 1, num_models,
            model_data["accuracy"], model_data["f1"],
        )

    # Select top models by F1
    top_models = sorted(all_model_data, key=lambda x: x["f1"], reverse=True)[
        :top_n
    ]

    # Save top models and generate plots
    for i, md in enumerate(top_models):
        rank = i + 1
        model_path = models_dir / f"top_model_{rank}.joblib"
        joblib.dump(md["model"], model_path)
        logger.info("Saved model #%d to %s", rank, model_path)

        plots.plot_learning_curve(
            md["results"], rank, str(plots_dir / f"learning_curve_top_{rank}.png")
        )
        plots.plot_feature_importance(
            md["feature_importances"],
            md["X_train"].columns.tolist(),
            rank,
            str(plots_dir / f"feature_importance_top_{rank}.png"),
        )

    # Aggregate metrics
    aggregate_results = pd.DataFrame([
        {
            "Model": i + 1,
            "Seed": d["seed"],
            "Accuracy": d["accuracy"],
            "Precision": d["precision"],
            "Recall": d["recall"],
            "F1": d["f1"],
        }
        for i, d in enumerate(all_model_data)
    ])
    aggregate_results.to_csv(plots_dir / "all_model_metrics.csv", index=False)

    # Top models metrics
    top_models_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Original Model #": d["model_num"] + 1,
            "Seed": d["seed"],
            "Accuracy": d["accuracy"],
            "Precision": d["precision"],
            "Recall": d["recall"],
            "F1": d["f1"],
        }
        for i, d in enumerate(top_models)
    ])
    top_models_df.to_csv(plots_dir / "top_models_metrics.csv", index=False)

    # Batch statistics
    with open(plots_dir / "batch_stats.txt", "w") as f:
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Number of Models: {num_models}\n\n")
        f.write("Overall Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Accuracy: {np.mean([d['accuracy'] for d in all_model_data]):.4f}\n")
        f.write(f"Std Dev Accuracy: {np.std([d['accuracy'] for d in all_model_data]):.4f}\n")
        f.write(f"Mean Precision: {np.mean([d['precision'] for d in all_model_data]):.4f}\n")
        f.write(f"Mean Recall: {np.mean([d['recall'] for d in all_model_data]):.4f}\n")
        f.write(f"Mean F1 Score: {np.mean([d['f1'] for d in all_model_data]):.4f}\n\n")
        f.write(f"Top {top_n} Models:\n")
        f.write("-" * 50 + "\n")
        for i, md in enumerate(top_models):
            f.write(f"Rank {i+1} (Original Model #{md['model_num']+1}):\n")
            f.write(f"  Seed: {md['seed']}\n")
            f.write(f"  Accuracy: {md['accuracy']:.4f}\n")
            f.write(f"  Precision: {md['precision']:.4f}\n")
            f.write(f"  Recall: {md['recall']:.4f}\n")
            f.write(f"  F1 Score: {md['f1']:.4f}\n\n")

    # Accuracy distribution plot
    plots.plot_accuracy_distribution(
        [d["accuracy"] for d in all_model_data],
        str(plots_dir / "accuracy_distribution.png"),
    )

    # Average feature importance
    avg_importance = np.mean(
        [d["feature_importances"] for d in all_model_data], axis=0
    )
    plots.plot_average_feature_importance(
        avg_importance,
        X.columns.tolist(),
        str(plots_dir / "average_feature_importance.png"),
    )

    mean_acc = np.mean([d["accuracy"] for d in all_model_data])
    std_acc = np.std([d["accuracy"] for d in all_model_data])
    logger.info(
        "Training complete! Avg accuracy: %.4f +/- %.4f. Results in %s",
        mean_acc, std_acc, plots_dir,
    )


if __name__ == "__main__":
    run_training()
