"""A/B comparison of v6 vs e2e-tuned meta-learner hyperparameters.

1. Evaluate current v6 params (LightGBM + meta-learner defaults) across CV folds.
2. Run Phase 1 (LightGBM through meta-learner) + Phase 2 (meta-learner params).
3. Evaluate new params across same folds.
4. Print per-fold comparison table + aggregate delta.
5. Save new params if improvement > 0.005 log loss.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sports_quant.march_madness._config import load_mm_config
from sports_quant.march_madness._features import YEAR_COLUMN
from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._tuning_e2e import (
    evaluate_e2e_on_folds,
    run_phase1_lgbm_tuning,
    run_phase2_meta_tuning,
    save_e2e_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMPROVEMENT_THRESHOLD = 0.005


def main() -> None:
    cfg = load_mm_config()
    cv_folds = cfg["tuning"]["cv_folds"]
    feature_mode = cfg.get("feature_mode", "combined")
    do_symmetrize = cfg.get("train", {}).get("symmetrize", False)
    early_stop_rounds = cfg.get("train", {}).get("early_stopping_rounds", 50)
    meta_cfg = cfg.get("meta_learner", {})

    # Tuning trial counts from config (with defaults)
    tuning_meta = cfg.get("tuning", {}).get("meta_learner", {})
    phase1_trials = tuning_meta.get("phase1_trials", 150)
    phase2_trials = tuning_meta.get("phase2_trials", 100)

    # Current LightGBM hyperparameters
    old_lgbm = dict(cfg["hyperparameters"])
    old_meta = dict(meta_cfg)

    matchups_df = pd.read_csv(mm_config.MM_TRAINING_DATA)
    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())

    print("=" * 70)
    print("END-TO-END META-LEARNER HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Training data: {len(matchups_df)} games, years {available_years}")
    print(f"CV folds: {len(cv_folds)}")
    for f in cv_folds:
        print(f"  train≤{f['train_end']} → val {f['val_year']}")
    print(f"Phase 1 trials (LightGBM): {phase1_trials}")
    print(f"Phase 2 trials (meta-learner): {phase2_trials}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Evaluate CURRENT (v6) params through meta-learner pipeline
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("STEP 1: Evaluating CURRENT v6 params through meta-learner pipeline")
    print("-" * 70)

    old_results = evaluate_e2e_on_folds(
        matchups_df=matchups_df,
        cv_folds=cv_folds,
        lgbm_params=old_lgbm,
        meta_cfg=old_meta,
        feature_mode=feature_mode,
        do_symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
    )

    print(
        f"\nv6 meta-learner CV log loss: "
        f"{old_results['mean_meta_log_loss']:.4f} "
        f"(±{old_results['std_meta_log_loss']:.4f})"
    )
    for fold in old_results["folds"]:
        print(f"  {fold['fold']}: {fold['meta_log_loss']:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Phase 1 — Re-tune LightGBM through meta-learner
    # -----------------------------------------------------------------------
    print("-" * 70)
    print(f"STEP 2: Phase 1 — Tuning LightGBM ({phase1_trials} trials)")
    print("-" * 70)

    new_lgbm = run_phase1_lgbm_tuning(
        matchups_df=matchups_df,
        cv_folds=cv_folds,
        meta_cfg=old_meta,
        feature_mode=feature_mode,
        do_symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
        n_trials=phase1_trials,
    )
    print(f"\nPhase 1 best LightGBM params: {new_lgbm}")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Phase 2 — Tune meta-learner params
    # -----------------------------------------------------------------------
    print("-" * 70)
    print(f"STEP 3: Phase 2 — Tuning meta-learner ({phase2_trials} trials)")
    print("-" * 70)

    new_meta = run_phase2_meta_tuning(
        matchups_df=matchups_df,
        cv_folds=cv_folds,
        frozen_lgbm_params=new_lgbm,
        base_meta_cfg=old_meta,
        feature_mode=feature_mode,
        do_symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
        n_trials=phase2_trials,
    )
    print(f"\nPhase 2 best meta-learner config: {new_meta}")
    print()

    # -----------------------------------------------------------------------
    # Step 4: Evaluate NEW params through full meta-learner pipeline
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("STEP 4: Evaluating NEW tuned params through meta-learner pipeline")
    print("-" * 70)

    new_results = evaluate_e2e_on_folds(
        matchups_df=matchups_df,
        cv_folds=cv_folds,
        lgbm_params=new_lgbm,
        meta_cfg=new_meta,
        feature_mode=feature_mode,
        do_symmetrize=do_symmetrize,
        early_stop_rounds=early_stop_rounds,
    )

    print(
        f"\nNew meta-learner CV log loss: "
        f"{new_results['mean_meta_log_loss']:.4f} "
        f"(±{new_results['std_meta_log_loss']:.4f})"
    )
    for fold in new_results["folds"]:
        print(f"  {fold['fold']}: {fold['meta_log_loss']:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Step 5: Compare
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("COMPARISON: v6 vs e2e-tuned")
    print("=" * 70)

    delta = (
        old_results["mean_meta_log_loss"] - new_results["mean_meta_log_loss"]
    )
    pct = (delta / old_results["mean_meta_log_loss"]) * 100

    print(f"v6 meta log loss:  {old_results['mean_meta_log_loss']:.4f}")
    print(f"New meta log loss: {new_results['mean_meta_log_loss']:.4f}")
    print(f"Improvement:       {delta:+.4f} ({pct:+.1f}%)")
    print()

    # Per-fold comparison
    print("Per-fold breakdown:")
    print(f"{'Fold':<30} {'v6':>10} {'New':>10} {'Delta':>10}")
    print("-" * 60)
    for old_f, new_f in zip(old_results["folds"], new_results["folds"]):
        d = old_f["meta_log_loss"] - new_f["meta_log_loss"]
        winner = "NEW" if d > 0 else "v6"
        print(
            f"{old_f['fold']:<30} {old_f['meta_log_loss']:>10.4f} "
            f"{new_f['meta_log_loss']:>10.4f} {d:>+10.4f} {winner}"
        )
    print()

    # LightGBM param comparison
    print("LightGBM parameter changes:")
    print(f"{'Param':<25} {'v6':>12} {'New':>12} {'Change':>12}")
    print("-" * 65)
    skip_keys = {"objective", "metric"}
    all_keys = sorted(set(old_lgbm.keys()) | set(new_lgbm.keys()))
    for key in all_keys:
        if key in skip_keys:
            continue
        old_val = old_lgbm.get(key, "N/A")
        new_val = new_lgbm.get(key, "N/A")
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            change = new_val - old_val
            print(
                f"{key:<25} {old_val:>12.4f} {new_val:>12.4f} "
                f"{change:>+12.4f}"
            )
        else:
            print(f"{key:<25} {str(old_val):>12} {str(new_val):>12}")
    print()

    # Meta-learner param comparison
    print("Meta-learner parameter changes:")
    meta_keys = [
        "lr_C", "rf_n_estimators", "rf_max_depth",
        "rf_min_samples_leaf", "meta_C", "lgbm_oof_ensemble_size",
    ]
    print(f"{'Param':<25} {'v6':>12} {'New':>12} {'Change':>12}")
    print("-" * 65)
    for key in meta_keys:
        old_val = old_meta.get(key, "N/A")
        new_val = new_meta.get(key, "N/A")
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            change = new_val - old_val
            print(
                f"{key:<25} {old_val:>12.4f} {new_val:>12.4f} "
                f"{change:>+12.4f}"
            )
        else:
            print(f"{key:<25} {str(old_val):>12} {str(new_val):>12}")
    print()

    # Decision
    if delta > IMPROVEMENT_THRESHOLD:
        print(
            f"NEW params WIN (improvement {delta:+.4f} > "
            f"threshold {IMPROVEMENT_THRESHOLD}). "
            f"Saving to model_config.yaml..."
        )
        save_e2e_params(new_lgbm, new_meta)
        print("Done! New e2e-tuned hyperparameters saved.")
    elif delta > 0:
        print(
            f"Marginal improvement ({delta:+.4f} < threshold "
            f"{IMPROVEMENT_THRESHOLD}). NOT saving — improvement too small."
        )
    else:
        print("v6 params still better. No changes saved.")


if __name__ == "__main__":
    main()
