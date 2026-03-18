"""A/B comparison of old vs new Optuna-tuned hyperparameters.

Runs Optuna with expanded CV folds and wider search ranges on the full
881-game dataset, then evaluates both old and new params on the same
temporal CV folds to determine which is better.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import log_loss

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sports_quant.march_madness._config import load_mm_config
from sports_quant.march_madness._data import (
    compute_difference_features,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import TARGET_COLUMN, YEAR_COLUMN
from sports_quant.march_madness._tuning import run_optuna_study, save_best_params
from sports_quant.march_madness import _config as mm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_params_on_folds(
    params: dict,
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    symmetrize: bool = True,
    early_stop_rounds: int = 50,
    n_seeds: int = 5,
) -> dict:
    """Evaluate a set of hyperparameters across temporal CV folds.

    Trains multiple seeds per fold to reduce variance in the estimate.

    Args:
        params: LightGBM hyperparameters to evaluate.
        matchups_df: Full matchup DataFrame.
        cv_folds: List of {train_end, val_year} dicts.
        symmetrize: Whether to symmetrize training data.
        early_stop_rounds: Early stopping patience.
        n_seeds: Number of random seeds per fold for stability.

    Returns:
        Dict with per-fold and aggregate metrics.
    """
    fold_results = []

    for fold in cv_folds:
        train_end = fold["train_end"]
        val_year = fold["val_year"]

        train_raw = matchups_df[matchups_df[YEAR_COLUMN] <= train_end]
        val_raw = matchups_df[matchups_df[YEAR_COLUMN] == val_year]

        if len(val_raw) == 0:
            continue

        X_train = compute_difference_features(train_raw)
        y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
        X_val = compute_difference_features(val_raw)
        y_val = val_raw[TARGET_COLUMN].reset_index(drop=True)

        if symmetrize:
            X_train, y_train = symmetrize_training_data(X_train, y_train)

        seed_losses = []
        for seed in range(n_seeds):
            model = LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                random_state=seed * 42 + 7,
                verbose=-1,
                **params,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=["binary_logloss"],
                callbacks=[early_stopping(early_stop_rounds)],
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            seed_losses.append(log_loss(y_val, y_proba))

        avg_loss = float(np.mean(seed_losses))
        std_loss = float(np.std(seed_losses))

        fold_results.append({
            "fold": f"train≤{train_end}→val{val_year}",
            "train_size": len(train_raw),
            "val_size": len(val_raw),
            "mean_log_loss": avg_loss,
            "std_log_loss": std_loss,
        })

    mean_cv_loss = float(np.mean([r["mean_log_loss"] for r in fold_results]))
    std_cv_loss = float(np.std([r["mean_log_loss"] for r in fold_results]))

    return {
        "folds": fold_results,
        "mean_cv_log_loss": mean_cv_loss,
        "std_cv_log_loss": std_cv_loss,
    }


def main() -> None:
    cfg = load_mm_config()
    cv_folds = cfg["tuning"]["cv_folds"]
    n_trials = cfg["tuning"]["n_trials"]
    do_symmetrize = cfg.get("train", {}).get("symmetrize", False)
    early_stop_rounds = cfg.get("train", {}).get("early_stopping_rounds", 50)

    # Current (old) hyperparameters from config
    old_params = {
        k: v for k, v in cfg["hyperparameters"].items()
        if k not in ("objective", "metric")
    }

    matchups_df = pd.read_csv(mm_config.MM_TRAINING_DATA)
    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())

    print("=" * 70)
    print("HYPERPARAMETER RETUNING COMPARISON")
    print("=" * 70)
    print(f"Training data: {len(matchups_df)} games, years {available_years}")
    print(f"CV folds: {len(cv_folds)}")
    for f in cv_folds:
        print(f"  train≤{f['train_end']} → val {f['val_year']}")
    print(f"Optuna trials: {n_trials}")
    print()

    # Step 1: Evaluate OLD params on the expanded CV folds
    print("-" * 70)
    print("STEP 1: Evaluating CURRENT hyperparameters on expanded CV folds...")
    print("-" * 70)
    old_results = evaluate_params_on_folds(
        old_params, matchups_df, cv_folds,
        symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
    )

    print(f"\nCurrent params CV log loss: {old_results['mean_cv_log_loss']:.4f} "
          f"(±{old_results['std_cv_log_loss']:.4f})")
    for fold in old_results["folds"]:
        print(f"  {fold['fold']}: {fold['mean_log_loss']:.4f} "
              f"(±{fold['std_log_loss']:.4f}, "
              f"train={fold['train_size']}, val={fold['val_size']})")
    print()

    # Step 2: Run Optuna
    print("-" * 70)
    print(f"STEP 2: Running Optuna ({n_trials} trials, {len(cv_folds)} folds)...")
    print("-" * 70)
    new_params, study = run_optuna_study(
        matchups_df=matchups_df,
        n_trials=n_trials,
        return_study=True,
    )

    print(f"\nOptuna best trial log loss: {study.best_value:.4f}")
    print(f"Best params: {new_params}")
    print()

    # Step 3: Evaluate NEW params with multi-seed stability
    print("-" * 70)
    print("STEP 3: Evaluating NEW hyperparameters with multi-seed stability...")
    print("-" * 70)
    new_results = evaluate_params_on_folds(
        new_params, matchups_df, cv_folds,
        symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
    )

    print(f"\nNew params CV log loss: {new_results['mean_cv_log_loss']:.4f} "
          f"(±{new_results['std_cv_log_loss']:.4f})")
    for fold in new_results["folds"]:
        print(f"  {fold['fold']}: {fold['mean_log_loss']:.4f} "
              f"(±{fold['std_log_loss']:.4f}, "
              f"train={fold['train_size']}, val={fold['val_size']})")
    print()

    # Step 4: Compare
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    delta = old_results["mean_cv_log_loss"] - new_results["mean_cv_log_loss"]
    pct_improvement = (delta / old_results["mean_cv_log_loss"]) * 100

    print(f"Old params CV log loss: {old_results['mean_cv_log_loss']:.4f}")
    print(f"New params CV log loss: {new_results['mean_cv_log_loss']:.4f}")
    print(f"Improvement: {delta:+.4f} ({pct_improvement:+.1f}%)")
    print()

    # Per-fold comparison
    print("Per-fold breakdown:")
    print(f"{'Fold':<30} {'Old':>10} {'New':>10} {'Delta':>10}")
    print("-" * 60)
    for old_f, new_f in zip(old_results["folds"], new_results["folds"]):
        d = old_f["mean_log_loss"] - new_f["mean_log_loss"]
        winner = "NEW" if d > 0 else "OLD"
        print(f"{old_f['fold']:<30} {old_f['mean_log_loss']:>10.4f} "
              f"{new_f['mean_log_loss']:>10.4f} {d:>+10.4f} {winner}")
    print()

    # Param comparison
    print("Parameter changes:")
    print(f"{'Param':<25} {'Old':>12} {'New':>12} {'Change':>12}")
    print("-" * 65)
    for key in sorted(new_params.keys()):
        old_val = old_params.get(key, "N/A")
        new_val = new_params[key]
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            change = new_val - old_val
            print(f"{key:<25} {old_val:>12.4f} {new_val:>12.4f} {change:>+12.4f}")
        else:
            print(f"{key:<25} {str(old_val):>12} {str(new_val):>12}")
    print()

    # Decision
    if delta > 0:
        print("NEW params WIN. Saving to model_config.yaml...")
        save_best_params(new_params)
        print("Done! New hyperparameters saved.")
    else:
        print("OLD params still better. No changes saved.")
        print("Consider: the old params may have been well-tuned already,")
        print("or the expanded data doesn't change the optimal region much.")


if __name__ == "__main__":
    main()
