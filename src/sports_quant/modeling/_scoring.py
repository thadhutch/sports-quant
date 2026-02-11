"""Season weighting, confidence scoring, consensus, and model selection."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sports_quant.modeling._training import (
    CONF_BINS,
    CONF_LABELS,
    TrainedModel,
)

logger = logging.getLogger(__name__)


def compute_season_progress(current_date) -> tuple[float, float]:
    """Return (weight_current_season, weight_last_season) based on NFL calendar.

    The NFL season runs roughly Sep 1 → Jan 15.  The fraction of elapsed
    days determines how much to weight the *current* season versus the
    *previous* season.  Early in the year the model relies more on last
    season's accuracy; by playoffs it relies mostly on the current season.
    """
    if isinstance(current_date, (np.datetime64, pd.Timestamp)):
        current_date = pd.Timestamp(current_date).to_pydatetime()

    year = current_date.year
    if current_date.month >= 9:
        season_start = datetime(year, 9, 1)
        season_end = datetime(year + 1, 1, 15)
    else:
        season_start = datetime(year - 1, 9, 1)
        season_end = datetime(year, 1, 15)

    total_days = (season_end - season_start).days
    elapsed = (current_date - season_start).days
    pct = max(0.0, min(1.0, elapsed / total_days))

    return pct, 1.0 - pct


def compute_weighted_accuracy(
    models: list[TrainedModel],
    weight_current: float,
    weight_last: float,
) -> list[dict]:
    """Attach ``weighted_accuracy`` to each model dict (in-place-style).

    Returns a list of dicts compatible with downstream selection/sorting.
    """
    out: list[dict] = []
    for m in models:
        cur = m.current_season_accuracy if not np.isnan(m.current_season_accuracy) else 0.0
        last = m.last_season_accuracy if not np.isnan(m.last_season_accuracy) else 0.0
        weighted = weight_current * cur + weight_last * last
        out.append(
            {
                "trained_model": m,
                "weighted_accuracy": weighted,
            }
        )
    return out


def select_top_models(
    scored: list[dict],
    top_n: int = 3,
) -> list[dict]:
    """Return the *top_n* models sorted by weighted accuracy descending."""
    return sorted(scored, key=lambda x: x["weighted_accuracy"], reverse=True)[:top_n]


def predict_with_consensus(
    top_models: list[dict],
    test_df: pd.DataFrame,
    weight_current: float,
    weight_last: float,
    model_weights: list[float],
) -> pd.DataFrame | None:
    """Run top models on the test set, keep only consensus picks, score them.

    A *consensus pick* is one where **all** top models predict the same class.
    The final algorithm score is the weighted combination of per-model
    adjusted scores (season-weighted confidence-bin accuracy).

    Returns a DataFrame of consensus picks or ``None`` if no consensus.
    """
    from sports_quant.modeling._data import DATE_COLUMN, TARGET_COLUMN
    from sports_quant.modeling._features import ALL_FEATURES

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN]
    seasons_test = test_df["season"].values

    all_picks: list[pd.DataFrame] = []

    for idx, entry in enumerate(top_models):
        tm: TrainedModel = entry["trained_model"]
        clf = tm.model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        confidences = np.max(y_proba, axis=1)

        adjusted_scores = []
        for i in range(len(y_pred)):
            conf_bin = pd.cut(
                [confidences[i]], bins=CONF_BINS, labels=CONF_LABELS, include_lowest=True
            )[0]

            cur_season = seasons_test[i]
            last_season = cur_season - 1

            acc_cur = _lookup_bin_accuracy(tm.confidence_accuracy, conf_bin, cur_season)
            acc_last = _lookup_bin_accuracy(tm.confidence_accuracy, conf_bin, last_season)

            adjusted_scores.append(weight_current * acc_cur + weight_last * acc_last)

        picks_df = pd.DataFrame(
            {
                "Date": test_df[DATE_COLUMN].values,
                "Season": seasons_test,
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Confidence": confidences,
                "Confidence Bin": pd.cut(
                    confidences, bins=CONF_BINS, labels=CONF_LABELS, include_lowest=True
                ),
                f"Adjusted Score Model {idx + 1}": adjusted_scores,
            }
        )
        all_picks.append(picks_df)

    # Build combined frame
    combined = pd.DataFrame(
        {
            "Date": test_df[DATE_COLUMN].values,
            "Season": seasons_test,
            "Actual": y_test.values,
            "Confidence Bin": all_picks[0]["Confidence Bin"],
            "Confidence": all_picks[0]["Confidence"],
        }
    )

    for idx, picks_df in enumerate(all_picks):
        combined[f"Predicted_Model_{idx + 1}"] = picks_df["Predicted"]
        combined[f"Adjusted Score Model {idx + 1}"] = picks_df[
            f"Adjusted Score Model {idx + 1}"
        ]

    # Consensus: all models agree
    n_top = len(top_models)
    pred_cols = [f"Predicted_Model_{i + 1}" for i in range(n_top)]
    consensus_mask = pd.Series(True, index=combined.index)
    for i in range(1, n_top):
        consensus_mask &= combined[pred_cols[0]] == combined[pred_cols[i]]

    consensus = combined[consensus_mask].copy()
    if consensus.empty:
        return None

    consensus["Predicted"] = consensus[pred_cols[0]]

    # Weighted final algorithm score
    consensus["Final Algorithm Score"] = 0.0
    for idx, w in enumerate(model_weights):
        consensus["Final Algorithm Score"] += (
            consensus[f"Adjusted Score Model {idx + 1}"] * w
        )

    return consensus[
        [
            "Date",
            "Season",
            "Actual",
            "Predicted",
            "Confidence",
            "Confidence Bin",
            "Final Algorithm Score",
        ]
    ]


def _lookup_bin_accuracy(
    conf_acc_df: pd.DataFrame, conf_bin, season: float
) -> float:
    """Look up accuracy for a given confidence bin and season."""
    match = conf_acc_df[
        (conf_acc_df["Confidence Bin"] == conf_bin)
        & (conf_acc_df["Season"] == season)
    ]["Accuracy"]
    return match.values[0] if not match.empty else 0.0
