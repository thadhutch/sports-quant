"""Upset analysis for March Madness predictions.

An upset is when a lower seed (higher number) beats a higher seed (lower number).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_upsets(
    actual_results: np.ndarray | pd.Series,
    predicted_results: np.ndarray | pd.Series,
    team_data: pd.DataFrame,
) -> dict:
    """Analyze upsets in predictions.

    Args:
        actual_results: Actual game outcomes (1=Team1 wins, 0=Team2 wins).
        predicted_results: Predicted game outcomes.
        team_data: DataFrame with Seed1, Seed2, Team1, Team2, YEAR columns.

    Returns:
        Dictionary with upset analysis results.
    """
    results = {
        "total_games": len(actual_results),
        "total_upsets_actual": 0,
        "total_upsets_predicted": 0,
        "correct_upset_predictions": 0,
        "upset_seed_diff_sum": 0,
        "biggest_upset_seed_diff": 0,
        "biggest_upset_details": None,
        "upsets": [],
    }

    for i, (actual, pred) in enumerate(zip(actual_results, predicted_results)):
        team1_seed = team_data.iloc[i]["Seed1"]
        team2_seed = team_data.iloc[i]["Seed2"]
        team1_name = team_data.iloc[i]["Team1"]
        team2_name = team_data.iloc[i]["Team2"]
        year = team_data.iloc[i]["YEAR"]

        # Check if this is an actual upset
        is_upset_actual = False
        if team1_seed < team2_seed and actual == 0:
            is_upset_actual = True
            results["total_upsets_actual"] += 1
        elif team2_seed < team1_seed and actual == 1:
            is_upset_actual = True
            results["total_upsets_actual"] += 1

        # Check if this is a predicted upset
        is_upset_predicted = False
        if team1_seed < team2_seed and pred == 0:
            is_upset_predicted = True
            seed_diff = team2_seed - team1_seed
            results["total_upsets_predicted"] += 1
            results["upset_seed_diff_sum"] += seed_diff

            if seed_diff > results["biggest_upset_seed_diff"]:
                results["biggest_upset_seed_diff"] = seed_diff
                results["biggest_upset_details"] = {
                    "year": year,
                    "underdog": team2_name,
                    "underdog_seed": team2_seed,
                    "favorite": team1_name,
                    "favorite_seed": team1_seed,
                    "correctly_predicted": (pred == actual),
                }

            results["upsets"].append({
                "year": year,
                "underdog": team2_name,
                "underdog_seed": team2_seed,
                "favorite": team1_name,
                "favorite_seed": team1_seed,
                "correctly_predicted": (pred == actual),
            })

        elif team2_seed < team1_seed and pred == 1:
            is_upset_predicted = True
            seed_diff = team1_seed - team2_seed
            results["total_upsets_predicted"] += 1
            results["upset_seed_diff_sum"] += seed_diff

            if seed_diff > results["biggest_upset_seed_diff"]:
                results["biggest_upset_seed_diff"] = seed_diff
                results["biggest_upset_details"] = {
                    "year": year,
                    "underdog": team1_name,
                    "underdog_seed": team1_seed,
                    "favorite": team2_name,
                    "favorite_seed": team2_seed,
                    "correctly_predicted": (pred == actual),
                }

            results["upsets"].append({
                "year": year,
                "underdog": team1_name,
                "underdog_seed": team1_seed,
                "favorite": team2_name,
                "favorite_seed": team2_seed,
                "correctly_predicted": (pred == actual),
            })

        if is_upset_actual and is_upset_predicted and pred == actual:
            results["correct_upset_predictions"] += 1

    # Sort upsets by seed difference
    results["upsets"] = sorted(
        results["upsets"],
        key=lambda x: x["underdog_seed"] - x["favorite_seed"],
        reverse=True,
    )

    return results


def track_popular_upsets(
    all_model_data: list[dict],
    backtest_teams: pd.DataFrame,
) -> list[dict]:
    """Track the most commonly predicted upsets across all models.

    Args:
        all_model_data: List of dicts with 'y_backtest_pred' key per model.
        backtest_teams: DataFrame with team info and actual results.

    Returns:
        List of upset dicts sorted by prediction count (descending).
    """
    upset_counts: dict[str, dict] = {}

    for model_idx, model_data in enumerate(all_model_data):
        y_pred = model_data["y_backtest_pred"]

        for pred, (_, team_row) in zip(y_pred, backtest_teams.iterrows()):
            team1_seed = team_row["Seed1"]
            team2_seed = team_row["Seed2"]
            team1_name = team_row["Team1"]
            team2_name = team_row["Team2"]
            year = team_row["YEAR"]
            actual_result = team_row["Team1_Win"]

            upset_key = None
            correct_prediction = False

            if team1_seed < team2_seed and pred == 0:
                upset_key = (
                    f"{year}_{team2_name}_{team2_seed}_over_"
                    f"{team1_name}_{team1_seed}"
                )
                correct_prediction = actual_result == pred
            elif team2_seed < team1_seed and pred == 1:
                upset_key = (
                    f"{year}_{team1_name}_{team1_seed}_over_"
                    f"{team2_name}_{team2_seed}"
                )
                correct_prediction = actual_result == pred

            if upset_key:
                if upset_key not in upset_counts:
                    is_actual_upset = (
                        (team1_seed < team2_seed and actual_result == 0)
                        or (team2_seed < team1_seed and actual_result == 1)
                    )
                    upset_counts[upset_key] = {
                        "year": year,
                        "underdog": (
                            team2_name if team1_seed < team2_seed else team1_name
                        ),
                        "underdog_seed": (
                            team2_seed if team1_seed < team2_seed else team1_seed
                        ),
                        "favorite": (
                            team1_name if team1_seed < team2_seed else team2_name
                        ),
                        "favorite_seed": (
                            team1_seed if team1_seed < team2_seed else team2_seed
                        ),
                        "count": 0,
                        "models": [],
                        "is_actual_upset": is_actual_upset,
                        "correct_predictions": 0,
                    }

                upset_counts[upset_key]["count"] += 1
                upset_counts[upset_key]["models"].append(model_idx + 1)
                if correct_prediction:
                    upset_counts[upset_key]["correct_predictions"] += 1

    # Build sorted list
    popular_upsets = []
    for data in upset_counts.values():
        data["accuracy"] = (
            data["correct_predictions"] / data["count"]
            if data["count"] > 0
            else 0
        )
        popular_upsets.append(data)

    popular_upsets.sort(key=lambda x: x["count"], reverse=True)
    return popular_upsets
