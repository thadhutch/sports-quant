"""Generate a feature importance chart for O/U prediction using XGBoost."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sports_quant import _config as config
from sports_quant.modeling._features import ALL_FEATURES, DISPLAY_NAMES

logger = logging.getLogger(__name__)

_ALL_FEATURES = ALL_FEATURES
_DISPLAY_NAMES = DISPLAY_NAMES


def _load_training_data() -> tuple[pd.DataFrame, pd.Series, int, int, int]:
    """Load ranked dataset and prepare features/target for training.

    Returns (X, y, n_games, min_season, max_season).
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d rows from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games (no prior PFF data)
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Drop rows with NaN in any feature or target column
    cols_needed = _ALL_FEATURES + ["total"]
    df = df.dropna(subset=cols_needed)
    logger.info("After filtering: %d games", len(df))

    X = df[_ALL_FEATURES]
    y = df["total"].astype(int)

    seasons = df["season"].dropna().unique()
    return X, y, len(df), int(min(seasons)), int(max(seasons))


def _train_ensemble(
    X: pd.DataFrame, y: pd.Series, n_models: int = 20
) -> tuple[pd.Series, float]:
    """Train an ensemble of XGBoost models and average gain-based importances.

    Returns (importance_series indexed by feature name, mean_accuracy).
    """
    all_importances = []
    all_accuracies = []

    for i in range(n_models):
        seed = 42 + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        all_accuracies.append(acc)

        # Gain-based importance
        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        feat_names = X.columns.tolist()
        imp = pd.Series(gain_scores, dtype=float)
        # Reindex to ensure all features present (some may have 0 importance)
        imp = imp.reindex(feat_names, fill_value=0.0)
        all_importances.append(imp)

    avg_importance = pd.concat(all_importances, axis=1).mean(axis=1)
    mean_accuracy = np.mean(all_accuracies)

    logger.info(
        "Trained %d models — mean accuracy: %.1f%%", n_models, mean_accuracy * 100
    )
    return avg_importance, mean_accuracy


def _render_chart(
    importances: pd.Series,
    mean_accuracy: float,
    n_games: int,
    min_season: int,
    max_season: int,
) -> None:
    """Render and save the horizontal bar chart."""
    bg_color = "#0e1117"
    text_color = "#e0e0e0"

    # Sort ascending so highest importance appears at top of horizontal bar chart
    importances = importances.sort_values(ascending=True)
    display_labels = [_DISPLAY_NAMES.get(f, f) for f in importances.index]

    # Normalize for color mapping
    norm = plt.Normalize(vmin=importances.min(), vmax=importances.max())
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in importances.values]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    bars = ax.barh(display_labels, importances.values, color=colors, edgecolor="none")

    # Display gain values to the right of each bar
    max_val = importances.max()
    for bar, val in zip(bars, importances.values):
        ax.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color=text_color,
        )

    ax.set_xlabel("Gain (avg loss reduction per split)", fontsize=10, color=text_color)
    ax.tick_params(axis="y", colors=text_color, labelsize=9)
    ax.tick_params(axis="x", colors=text_color, labelsize=8)
    ax.set_xlim(0, max_val * 1.12)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title and subtitle
    season_label = (
        f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    )
    fig.suptitle(
        "Feature Importance for O/U Prediction",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"XGBoost gain \u00b7 {season_label} \u00b7 {n_games:,} games "
        f"\u00b7 mean accuracy {mean_accuracy:.1%}",
        fontsize=10,
        color="#888888",
        pad=14,
    )

    # Footer
    fig.text(
        0.5,
        0.005,
        "Source: PFF grades + PFR/Vegas lines \u00b7 20-model ensemble",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.GRADES_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.FEATURE_IMPORTANCE_CHART,
        dpi=200,
        bbox_inches="tight",
        facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.FEATURE_IMPORTANCE_CHART)
    plt.close(fig)


def generate_feature_importance() -> None:
    """Public entry point: load data, train ensemble, render chart."""
    X, y, n_games, min_season, max_season = _load_training_data()
    importances, mean_accuracy = _train_ensemble(X, y)
    _render_chart(importances, mean_accuracy, n_games, min_season, max_season)


if __name__ == "__main__":
    generate_feature_importance()
