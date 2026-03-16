"""March Madness sequential year-by-year backtesting orchestrator.

For each backtest year, trains N models on all previous years' data, selects
the top models by validation F1, creates an ensemble, analyzes upsets, and
applies the debiasing algorithm.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from sports_quant.march_madness._config import (
    MM_BACKTEST_DIR,
    build_lgbm,
    load_mm_config,
)
from sports_quant.march_madness._data import (
    compute_difference_features,
    load_and_prepare,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import (
    DIFF_FEATURE_COLUMNS,
    DROP_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
)
from sports_quant.march_madness._upsets import analyze_upsets, track_popular_upsets
from sports_quant.march_madness._debiasing import process_backtest_results
from sports_quant.march_madness import plots

logger = logging.getLogger(__name__)


def _get_val_year(
    backtest_year: int, available_years: list[int],
) -> int | None:
    """Find the most recent year before backtest_year that has data.

    Handles gaps (e.g., COVID 2020 with no tournament data).

    Returns:
        The validation year, or None if no prior year is available.
    """
    candidates = [y for y in available_years if y < backtest_year]
    return max(candidates) if candidates else None


def _prepare_features(
    df: pd.DataFrame, feature_mode: str,
) -> pd.DataFrame:
    """Extract feature matrix from a raw matchup DataFrame.

    Args:
        df: Raw matchup data with all columns.
        feature_mode: "difference" or "raw".

    Returns:
        Feature-only DataFrame.
    """
    if feature_mode == "difference":
        return compute_difference_features(df)

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    filtered = df.drop(columns=drop_cols)
    return filtered.drop(columns=[TARGET_COLUMN], errors="ignore")


def run_backtest() -> None:
    """Run sequential year-by-year backtesting with ensemble and debiasing.

    For each backtest year:
    1. Train N models on all data from prior years.
    2. Select top models by validation F1 score.
    3. Evaluate on the held-out year.
    4. Build an ensemble and analyze upsets.
    5. Apply debiased algorithm.
    6. Save all results, plots, and metrics.
    """
    cfg = load_mm_config()
    model_version = cfg["model_version"]
    num_models = cfg["models_to_train"]
    top_n = cfg["top_models"]
    backtest_years = sorted(cfg["backtest_years"])
    hyperparams = cfg["hyperparameters"]
    feature_mode = cfg.get("feature_mode", "raw")
    do_symmetrize = cfg.get("train", {}).get("symmetrize", False)
    early_stop_rounds = cfg.get("train", {}).get("early_stopping_rounds", 50)

    base_dir = MM_BACKTEST_DIR / model_version
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset (raw — we'll compute features per-split)
    data = load_and_prepare(feature_mode="raw")
    matchups = data.df

    # All available years for temporal split lookups
    all_years = sorted(matchups[YEAR_COLUMN].unique().tolist())

    logger.info(
        "Full dataset: %d games. Backtesting years: %s (feature_mode=%s)",
        len(matchups), backtest_years, feature_mode,
    )

    # Aggregate results across years
    all_years_results: dict[str, list] = {
        "years": [],
        "ensemble_accuracies": [],
        "ensemble_log_losses": [],
        "ensemble_brier_scores": [],
        "debiased_accuracies": [],
    }

    for backtest_year in backtest_years:
        logger.info("=" * 50)
        logger.info("Processing backtest for year %d", backtest_year)

        year_dir = base_dir / str(backtest_year)
        plots_dir = year_dir / "plots"
        models_dir = year_dir / "models"
        plots_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Split data: train on prior years, test on backtest year
        all_prior = matchups[matchups[YEAR_COLUMN] < backtest_year]
        backtest_data = matchups[matchups[YEAR_COLUMN] == backtest_year]

        # Temporal validation split: use most recent available year as val
        val_year = _get_val_year(backtest_year, all_years)
        if val_year is not None:
            train_data = all_prior[all_prior[YEAR_COLUMN] < val_year]
            val_data = all_prior[all_prior[YEAR_COLUMN] == val_year]
        else:
            # Fallback: no prior year available — use all prior data
            train_data = all_prior
            val_data = pd.DataFrame(columns=all_prior.columns)

        logger.info(
            "Training: %d games (years < %d), Val: %d games (year %s), "
            "Backtest: %d games (year %d)",
            len(train_data), val_year or backtest_year,
            len(val_data), val_year,
            len(backtest_data), backtest_year,
        )

        # Save backtest team info
        backtest_teams = backtest_data[
            ["YEAR", "Team1", "Seed1", "Team2", "Seed2", "Team1_Win"]
        ].copy()
        backtest_teams.to_csv(year_dir / "backtest_teams.csv", index=False)

        # Prepare features for each split
        X_train_full = _prepare_features(train_data, feature_mode)
        y_train_full = train_data[TARGET_COLUMN].reset_index(drop=True)

        X_val = _prepare_features(val_data, feature_mode) if len(val_data) > 0 else None
        y_val = val_data[TARGET_COLUMN].reset_index(drop=True) if len(val_data) > 0 else None

        X_backtest = _prepare_features(backtest_data, feature_mode)
        y_backtest = backtest_data[TARGET_COLUMN].reset_index(drop=True)

        # Apply symmetrization to training data only
        if do_symmetrize and feature_mode == "difference":
            X_train_full, y_train_full = symmetrize_training_data(
                X_train_full, y_train_full,
            )
            logger.info(
                "Symmetrized training data: %d rows", len(X_train_full),
            )

        X_backtest.to_csv(year_dir / "X_backtest.csv", index=False)
        pd.DataFrame(y_backtest).to_csv(
            year_dir / "y_backtest.csv", index=False,
        )

        # Train N models
        has_val = X_val is not None and len(X_val) > 0
        all_model_data = []

        for model_num in range(num_models):
            random_seed = np.random.randint(1, 10000)

            model = build_lgbm(hyperparams, random_seed)

            if has_val:
                eval_set = [(X_train_full, y_train_full), (X_val, y_val)]
                callbacks = [early_stopping(early_stop_rounds)]
            else:
                eval_set = [(X_train_full, y_train_full)]
                callbacks = None

            model.fit(
                X_train_full, y_train_full,
                eval_set=eval_set,
                eval_metric=["binary_logloss", "auc"],
                callbacks=callbacks,
            )

            results = model.evals_result_

            # Validation metrics
            if has_val:
                y_val_pred = model.predict(X_val)
                y_val_proba = model.predict_proba(X_val)[:, 1]
                val_f1 = f1_score(y_val, y_val_pred)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_log_loss = log_loss(y_val, y_val_proba)
            else:
                val_f1 = 0.0
                val_accuracy = 0.0
                val_log_loss = float("inf")

            # Backtest metrics
            y_backtest_pred = model.predict(X_backtest)
            y_backtest_proba = model.predict_proba(X_backtest)[:, 1]

            model_data = {
                "model_num": model_num,
                "seed": random_seed,
                "val_accuracy": val_accuracy,
                "val_precision": precision_score(y_val, y_val_pred) if has_val else 0.0,
                "val_recall": recall_score(y_val, y_val_pred) if has_val else 0.0,
                "val_f1": val_f1,
                "val_log_loss": val_log_loss,
                "backtest_accuracy": accuracy_score(y_backtest, y_backtest_pred),
                "backtest_precision": precision_score(y_backtest, y_backtest_pred),
                "backtest_recall": recall_score(y_backtest, y_backtest_pred),
                "backtest_f1": f1_score(y_backtest, y_backtest_pred),
                "backtest_log_loss": log_loss(y_backtest, y_backtest_proba),
                "backtest_brier": brier_score_loss(y_backtest, y_backtest_proba),
                "model": model,
                "results": results,
                "feature_importances": model.feature_importances_,
                "y_backtest_pred": y_backtest_pred,
                "y_backtest_proba": y_backtest_proba,
            }
            all_model_data.append(model_data)

            logger.info(
                "Model %d/%d (year %d) - Val F1: %.4f, "
                "Backtest Acc: %.4f, Log Loss: %.4f",
                model_num + 1, num_models, backtest_year,
                model_data["val_f1"], model_data["backtest_accuracy"],
                model_data["backtest_log_loss"],
            )

        # Select top models by validation F1
        top_models = sorted(
            all_model_data, key=lambda x: x["val_f1"], reverse=True
        )[:top_n]

        all_years_results["years"].append(backtest_year)

        # Save top models, results, and plots
        for i, md in enumerate(top_models):
            rank = i + 1

            model_path = models_dir / f"top_model_{rank}.joblib"
            joblib.dump(md["model"], model_path)

            # Backtest results for this model
            bt_results = backtest_teams.copy()
            bt_results["Predicted_Win"] = md["y_backtest_pred"]
            bt_results["Win_Probability"] = md["y_backtest_proba"]
            bt_results["Correct_Prediction"] = (
                bt_results["Team1_Win"] == bt_results["Predicted_Win"]
            )
            bt_results.to_csv(
                year_dir / f"backtest_results_top_{rank}.csv", index=False
            )

            # Upset analysis
            upset_analysis = analyze_upsets(
                bt_results["Team1_Win"], bt_results["Predicted_Win"], bt_results
            )

            with open(year_dir / f"upset_analysis_top_{rank}.txt", "w") as f:
                f.write(
                    f"Upset Analysis for Model #{rank} - Year {backtest_year}\n"
                )
                f.write("=" * 50 + "\n\n")
                f.write(f"Validation F1 Score: {md['val_f1']:.4f}\n")
                f.write(f"Backtest Accuracy: {md['backtest_accuracy']:.4f}\n\n")
                f.write(f"Total Games: {upset_analysis['total_games']}\n")
                f.write(
                    f"Total Actual Upsets: "
                    f"{upset_analysis['total_upsets_actual']}\n"
                )
                f.write(
                    f"Total Predicted Upsets: "
                    f"{upset_analysis['total_upsets_predicted']}\n"
                )
                f.write(
                    f"Correctly Predicted Upsets: "
                    f"{upset_analysis['correct_upset_predictions']}\n"
                )

                if upset_analysis["biggest_upset_details"]:
                    bd = upset_analysis["biggest_upset_details"]
                    f.write(f"\nBiggest Upset Details:\n")
                    f.write(f"  Year: {bd['year']}\n")
                    f.write(
                        f"  Underdog: {bd['underdog']} "
                        f"(Seed {bd['underdog_seed']})\n"
                    )
                    f.write(
                        f"  Favorite: {bd['favorite']} "
                        f"(Seed {bd['favorite_seed']})\n"
                    )

                f.write("\nAll Predicted Upsets:\n")
                f.write("-" * 50 + "\n")
                for j, upset in enumerate(upset_analysis["upsets"]):
                    f.write(
                        f"{j+1}. {upset['year']} - {upset['underdog']} "
                        f"(Seed {upset['underdog_seed']}) vs. "
                        f"{upset['favorite']} (Seed {upset['favorite_seed']}), "
                        f"Correct: {upset['correctly_predicted']}\n"
                    )

            # Plots for this model
            plots.plot_learning_curve(
                md["results"], rank,
                str(plots_dir / f"learning_curve_top_{rank}.png"),
                year=backtest_year,
            )
            feature_names = list(
                DIFF_FEATURE_COLUMNS
                if feature_mode == "difference"
                else X_train_full.columns
            )
            plots.plot_feature_importance(
                md["feature_importances"],
                feature_names,
                rank,
                str(plots_dir / f"feature_importance_top_{rank}.png"),
                year=backtest_year,
            )
            plots.plot_confusion_matrix(
                y_backtest, md["y_backtest_pred"],
                rank, backtest_year,
                str(plots_dir / f"confusion_matrix_top_{rank}.png"),
            )

        # Aggregate model metrics
        agg = pd.DataFrame([
            {
                "Model": i + 1,
                "Seed": d["seed"],
                "Validation_Accuracy": d["val_accuracy"],
                "Validation_F1": d["val_f1"],
                "Validation_LogLoss": d["val_log_loss"],
                "Backtest_Accuracy": d["backtest_accuracy"],
                "Backtest_Precision": d["backtest_precision"],
                "Backtest_Recall": d["backtest_recall"],
                "Backtest_F1": d["backtest_f1"],
                "Backtest_LogLoss": d["backtest_log_loss"],
                "Backtest_Brier": d["backtest_brier"],
            }
            for i, d in enumerate(all_model_data)
        ])
        agg.to_csv(year_dir / "all_model_metrics.csv", index=False)

        top_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Original Model #": d["model_num"] + 1,
                "Seed": d["seed"],
                "Validation_F1": d["val_f1"],
                "Backtest_Accuracy": d["backtest_accuracy"],
                "Backtest_LogLoss": d["backtest_log_loss"],
            }
            for i, d in enumerate(top_models)
        ])
        top_df.to_csv(year_dir / "top_models_metrics.csv", index=False)

        # Ensemble predictions
        ensemble_preds = np.mean(
            [md["y_backtest_proba"] for md in top_models], axis=0
        )
        ensemble_binary = (ensemble_preds > 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_backtest, ensemble_binary)
        ensemble_precision = precision_score(y_backtest, ensemble_binary)
        ensemble_recall = recall_score(y_backtest, ensemble_binary)
        ensemble_f1 = f1_score(y_backtest, ensemble_binary)
        ensemble_log_loss = log_loss(y_backtest, ensemble_preds)
        ensemble_brier = brier_score_loss(y_backtest, ensemble_preds)

        all_years_results["ensemble_accuracies"].append(ensemble_accuracy)
        all_years_results["ensemble_log_losses"].append(ensemble_log_loss)
        all_years_results["ensemble_brier_scores"].append(ensemble_brier)

        ensemble_results = backtest_teams.copy()
        ensemble_results["Ensemble_Pred"] = ensemble_binary
        ensemble_results["Ensemble_Prob"] = ensemble_preds
        ensemble_results["Correct_Prediction"] = (
            ensemble_results["Team1_Win"] == ensemble_results["Ensemble_Pred"]
        )
        ensemble_results.to_csv(year_dir / "ensemble_results.csv", index=False)

        # Ensemble upset analysis
        ensemble_upset = analyze_upsets(
            ensemble_results["Team1_Win"],
            ensemble_results["Ensemble_Pred"],
            ensemble_results,
        )

        with open(year_dir / "ensemble_upset_analysis.txt", "w") as f:
            f.write(
                f"Ensemble Upset Analysis (Top {top_n} by Val F1) "
                f"- Year {backtest_year}\n"
            )
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Games: {ensemble_upset['total_games']}\n")
            f.write(
                f"Total Actual Upsets: {ensemble_upset['total_upsets_actual']}\n"
            )
            f.write(
                f"Total Predicted Upsets: "
                f"{ensemble_upset['total_upsets_predicted']}\n"
            )
            f.write(
                f"Correctly Predicted Upsets: "
                f"{ensemble_upset['correct_upset_predictions']}\n"
            )

            if ensemble_upset["biggest_upset_details"]:
                bd = ensemble_upset["biggest_upset_details"]
                f.write(f"\nBiggest Upset:\n")
                f.write(
                    f"  {bd['underdog']} (Seed {bd['underdog_seed']}) vs. "
                    f"{bd['favorite']} (Seed {bd['favorite_seed']})\n"
                )

            f.write("\nAll Predicted Upsets:\n")
            f.write("-" * 50 + "\n")
            for j, upset in enumerate(ensemble_upset["upsets"]):
                f.write(
                    f"{j+1}. {upset['underdog']} "
                    f"(Seed {upset['underdog_seed']}) vs. "
                    f"{upset['favorite']} (Seed {upset['favorite_seed']}), "
                    f"Correct: {upset['correctly_predicted']}\n"
                )

        # Backtest stats summary
        with open(year_dir / "backtest_stats.txt", "w") as f:
            f.write(f"Backtest Statistics - Year {backtest_year}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Version: {model_version}\n")
            f.write(f"Feature Mode: {feature_mode}\n")
            f.write(f"Symmetrized: {do_symmetrize}\n")
            f.write(f"Training on years up to: {backtest_year - 1}\n")
            f.write(f"Validation year: {val_year}\n")
            f.write(f"Number of Models: {num_models}\n")
            f.write(f"Models selected by: Validation F1 Score\n\n")

            f.write("Overall Statistics:\n")
            f.write("-" * 50 + "\n")
            mean_f1 = np.mean([d["val_f1"] for d in all_model_data])
            mean_bt_acc = np.mean(
                [d["backtest_accuracy"] for d in all_model_data]
            )
            mean_bt_ll = np.mean(
                [d["backtest_log_loss"] for d in all_model_data]
            )
            f.write(f"Mean Validation F1: {mean_f1:.4f}\n")
            f.write(f"Mean Backtest Accuracy: {mean_bt_acc:.4f}\n")
            f.write(f"Mean Backtest Log Loss: {mean_bt_ll:.4f}\n\n")

            f.write(f"Ensemble Accuracy: {ensemble_accuracy:.4f}\n")
            f.write(f"Ensemble Precision: {ensemble_precision:.4f}\n")
            f.write(f"Ensemble Recall: {ensemble_recall:.4f}\n")
            f.write(f"Ensemble F1: {ensemble_f1:.4f}\n")
            f.write(f"Ensemble Log Loss: {ensemble_log_loss:.4f}\n")
            f.write(f"Ensemble Brier Score: {ensemble_brier:.4f}\n\n")

            # Favorites benchmark
            favorites_correct = sum(
                (backtest_teams["Seed1"] <= backtest_teams["Seed2"])
                & (backtest_teams["Team1_Win"] == 1)
                | (backtest_teams["Seed1"] > backtest_teams["Seed2"])
                & (backtest_teams["Team1_Win"] == 0)
            )
            favorites_accuracy = favorites_correct / len(backtest_teams)
            f.write("Benchmark:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Always pick favorites: {favorites_accuracy:.4f}\n")
            f.write(
                f"Ensemble improvement: "
                f"{ensemble_accuracy - favorites_accuracy:+.4f}\n"
            )

        # Popular upsets analysis
        popular_upsets = track_popular_upsets(all_model_data, backtest_teams)

        with open(year_dir / "popular_upsets.txt", "w") as f:
            f.write(
                f"Most Popular Upset Predictions - Year {backtest_year}\n"
            )
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of Models: {num_models}\n")
            f.write(f"Unique Upsets Predicted: {len(popular_upsets)}\n\n")

            for i, upset in enumerate(popular_upsets[:30]):
                pct = (upset["count"] / num_models) * 100
                status = (
                    "CORRECT" if upset["is_actual_upset"] else "INCORRECT"
                )
                f.write(
                    f"{i+1}. {upset['underdog']} "
                    f"(Seed {upset['underdog_seed']}) over "
                    f"{upset['favorite']} (Seed {upset['favorite_seed']}) "
                    f"- {status}\n"
                )
                f.write(f"   Picked by {upset['count']}/{num_models} ({pct:.1f}%)\n\n")

        plots.plot_popular_upsets(
            popular_upsets, backtest_year, num_models,
            str(plots_dir / "popular_upsets.png"),
        )

        # Distribution plots
        plots.plot_validation_f1_distribution(
            [d["val_f1"] for d in all_model_data],
            backtest_year,
            str(plots_dir / "validation_f1_distribution.png"),
        )
        plots.plot_validation_vs_backtest(
            [d["val_f1"] for d in all_model_data],
            [d["backtest_accuracy"] for d in all_model_data],
            backtest_year,
            str(plots_dir / "validation_f1_vs_backtest_accuracy.png"),
        )

        logger.info(
            "Year %d complete. Ensemble acc: %.4f, log_loss: %.4f, "
            "brier: %.4f",
            backtest_year, ensemble_accuracy, ensemble_log_loss,
            ensemble_brier,
        )

    # Multi-year summary
    with open(base_dir / "multi_year_summary.txt", "w") as f:
        f.write("Multi-Year Backtest Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Feature Mode: {feature_mode}\n")
        f.write(f"Symmetrized: {do_symmetrize}\n")
        f.write(f"Backtest Years: {backtest_years}\n\n")

        f.write("Ensemble Performance By Year:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Year':<8}{'Accuracy':<12}{'Log Loss':<12}{'Brier':<12}\n")
        f.write("-" * 60 + "\n")
        for i, year in enumerate(all_years_results["years"]):
            acc = all_years_results["ensemble_accuracies"][i]
            ll = all_years_results["ensemble_log_losses"][i]
            bs = all_years_results["ensemble_brier_scores"][i]
            f.write(f"{year:<8}{acc:<12.4f}{ll:<12.4f}{bs:<12.4f}\n")

        if len(backtest_years) > 1:
            avg_acc = np.mean(all_years_results["ensemble_accuracies"])
            avg_ll = np.mean(all_years_results["ensemble_log_losses"])
            avg_bs = np.mean(all_years_results["ensemble_brier_scores"])
            f.write("-" * 60 + "\n")
            f.write(f"{'Avg':<8}{avg_acc:<12.4f}{avg_ll:<12.4f}{avg_bs:<12.4f}\n")

    # Multi-year plot
    if len(backtest_years) > 1:
        plots.plot_ensemble_accuracy_by_year(
            all_years_results["years"],
            all_years_results["ensemble_accuracies"],
            str(base_dir / "ensemble_accuracy_by_year.png"),
        )

    # Apply debiasing to all years
    logger.info("Applying debiased algorithm...")
    for backtest_year in backtest_years:
        _, debiased_metrics = process_backtest_results(
            str(base_dir), backtest_year, top_n=top_n
        )
        all_years_results["debiased_accuracies"].append(
            debiased_metrics["accuracy"]
        )

    # Update summary with debiased results
    with open(base_dir / "multi_year_summary.txt", "a") as f:
        f.write("\nDebiased Algorithm Performance By Year:\n")
        f.write("-" * 50 + "\n")
        for i, year in enumerate(all_years_results["years"]):
            d_acc = all_years_results["debiased_accuracies"][i]
            e_acc = all_years_results["ensemble_accuracies"][i]
            f.write(
                f"Year {year} - Debiased: {d_acc:.4f} "
                f"(vs Ensemble: {e_acc:.4f})\n"
            )

        if len(backtest_years) > 1:
            avg_d = np.mean(all_years_results["debiased_accuracies"])
            avg_e = np.mean(all_years_results["ensemble_accuracies"])
            f.write(
                f"\nAverage Debiased: {avg_d:.4f} "
                f"(vs Ensemble: {avg_e:.4f})\n"
            )

    # Ensemble vs debiased plot
    if len(backtest_years) > 1:
        plots.plot_ensemble_vs_debiased(
            all_years_results["years"],
            all_years_results["ensemble_accuracies"],
            all_years_results["debiased_accuracies"],
            str(base_dir / "ensemble_vs_debiased_accuracy.png"),
        )

    logger.info(
        "Backtesting complete. Avg ensemble: acc=%.4f, log_loss=%.4f, "
        "brier=%.4f. Avg debiased: %.4f",
        np.mean(all_years_results["ensemble_accuracies"]),
        np.mean(all_years_results["ensemble_log_losses"]),
        np.mean(all_years_results["ensemble_brier_scores"]),
        np.mean(all_years_results["debiased_accuracies"]),
    )


if __name__ == "__main__":
    run_backtest()
