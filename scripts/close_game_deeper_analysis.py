"""Deeper analysis: where is the model actually wrong on close games?

The gap_consumed approach didn't work because the model isn't compressing
probabilities near 0.50 for most close-game upsets. It's confidently wrong.

This script looks at the actual failure modes to find actionable patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED_MATCHUP_PRIORS: dict[tuple[int, int], float] = {
    (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.20,
    (5, 12): 0.35, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.48,
}

BASE = Path(
    "/Users/thadhutcheson/Documents/GitHub/sports-quant"
    "/data/march-madness/backtest/v6b"
)
YEARS = [2019, 2021, 2022, 2023, 2024, 2025]

frames = []
for year in YEARS:
    df = pd.read_csv(BASE / str(year) / "ensemble_results.csv")
    frames.append(df)
all_games = pd.concat(frames, ignore_index=True)

# Annotate
all_games["fav_is_team1"] = all_games["Seed1"] < all_games["Seed2"]
# Handle same-seed: treat Team1 as "favorite" arbitrarily
all_games.loc[all_games["Seed1"] == all_games["Seed2"], "fav_is_team1"] = True

all_games["fav_seed"] = all_games[["Seed1", "Seed2"]].min(axis=1)
all_games["dog_seed"] = all_games[["Seed1", "Seed2"]].max(axis=1)
all_games["seed_gap"] = abs(all_games["Seed1"] - all_games["Seed2"])
all_games["seed_matchup"] = (
    all_games["fav_seed"].astype(str) + "v" + all_games["dog_seed"].astype(str)
)

all_games["fav_win_prob"] = np.where(
    all_games["fav_is_team1"],
    all_games["Ensemble_Prob"],
    1.0 - all_games["Ensemble_Prob"],
)
all_games["fav_won"] = np.where(
    all_games["fav_is_team1"],
    all_games["Team1_Win"] == 1,
    all_games["Team1_Win"] == 0,
)
all_games["model_picked_fav"] = all_games["fav_win_prob"] > 0.50
all_games["correct"] = all_games["model_picked_fav"] == all_games["fav_won"]

# R64 canonical pairings only
r64 = all_games[
    (all_games["Seed1"] + all_games["Seed2"] == 17)
    & (all_games["Seed1"] != all_games["Seed2"])
].copy()

# -----------------------------------------------------------------------
# Part 1: Failure mode analysis — when the model is wrong on close games
# -----------------------------------------------------------------------
print("=" * 75)
print("PART 1: FAILURE MODES ON CLOSE-SEED R64 GAMES (5v12, 6v11, 7v10, 8v9)")
print("=" * 75)
print()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = r64[r64["seed_matchup"] == matchup]
    upsets = subset[~subset["fav_won"]]
    chalks = subset[subset["fav_won"]]

    n_upset = len(upsets)
    n_chalk = len(chalks)
    n = len(subset)
    upset_rate = n_upset / n if n > 0 else 0

    # How often does the model pick an upset correctly?
    model_called_upsets = subset[~subset["model_picked_fav"]]
    correct_upset_calls = model_called_upsets[~model_called_upsets["fav_won"]]

    # How often does model pick chalk correctly?
    model_called_chalk = subset[subset["model_picked_fav"]]
    correct_chalk_calls = model_called_chalk[model_called_chalk["fav_won"]]

    # Missed upsets: model picked fav but upset happened
    missed_upsets = subset[subset["model_picked_fav"] & ~subset["fav_won"]]

    # False upsets: model picked dog but fav won
    false_upsets = subset[~subset["model_picked_fav"] & subset["fav_won"]]

    print(f"--- {matchup}: {n} games, {n_upset} upsets ({upset_rate:.0%}), "
          f"{n_chalk} chalk ---")
    print(f"  Model accuracy:     {subset['correct'].sum()}/{n} = "
          f"{subset['correct'].mean():.0%}")
    print(f"  Correct upset calls: {len(correct_upset_calls)}/{n_upset} upsets caught")
    print(f"  Missed upsets:       {len(missed_upsets)} (model picked fav, dog won)")
    print(f"  False upsets:        {len(false_upsets)} (model picked dog, fav won)")
    print()

    if len(missed_upsets) > 0:
        print(f"  MISSED UPSETS (model confidently picked favorite but upset happened):")
        for _, row in missed_upsets.sort_values("fav_win_prob", ascending=False).iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            print(f"    {row['YEAR']} {dog}({row['dog_seed']}) beat "
                  f"{fav}({row['fav_seed']}): model gave fav {row['fav_win_prob']:.1%}")
        print()

    if len(false_upsets) > 0:
        print(f"  FALSE UPSETS (model picked dog but favorite won):")
        for _, row in false_upsets.sort_values("fav_win_prob").iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            print(f"    {row['YEAR']} {fav}({row['fav_seed']}) beat "
                  f"{dog}({row['dog_seed']}): model gave fav {row['fav_win_prob']:.1%}")
        print()

# -----------------------------------------------------------------------
# Part 2: Probability distribution when model is wrong vs right
# -----------------------------------------------------------------------
print("=" * 75)
print("PART 2: PROBABILITY CONFIDENCE WHEN MODEL IS RIGHT vs WRONG")
print("=" * 75)
print()

close = r64[r64["seed_gap"] <= 3].copy()
close["confidence"] = abs(close["fav_win_prob"] - 0.50)

correct_games = close[close["correct"]]
wrong_games = close[~close["correct"]]

print(f"Close-seed R64 games: {len(close)}")
print(f"  Correct predictions: {len(correct_games)} ({len(correct_games)/len(close):.0%})")
print(f"  Wrong predictions:   {len(wrong_games)} ({len(wrong_games)/len(close):.0%})")
print()
print(f"  Avg confidence (|P-0.50|) when CORRECT: {correct_games['confidence'].mean():.3f}")
print(f"  Avg confidence (|P-0.50|) when WRONG:   {wrong_games['confidence'].mean():.3f}")
print()

# Bucket by confidence level
print("  Accuracy by confidence bucket:")
for lo, hi, label in [
    (0.00, 0.05, "  toss-up (0-5%)"),
    (0.05, 0.10, "  lean    (5-10%)"),
    (0.10, 0.20, "  moderate(10-20%)"),
    (0.20, 0.50, "  strong  (20%+)"),
]:
    bucket = close[(close["confidence"] >= lo) & (close["confidence"] < hi)]
    if len(bucket) > 0:
        acc = bucket["correct"].mean()
        print(f"    {label}: {bucket['correct'].sum()}/{len(bucket)} "
              f"= {acc:.0%}")
    else:
        print(f"    {label}: no games")

# -----------------------------------------------------------------------
# Part 3: What if we just flip ALL close-game picks to the dog?
# (extreme test — is "always pick upset" better than the model?)
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("PART 3: NAIVE STRATEGIES FOR CLOSE GAMES")
print("=" * 75)
print()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = r64[r64["seed_matchup"] == matchup]
    n = len(subset)
    n_upsets = (~subset["fav_won"]).sum()

    model_acc = subset["correct"].mean()
    always_fav_acc = subset["fav_won"].mean()
    always_dog_acc = (~subset["fav_won"]).mean()

    print(f"{matchup} ({n} games, {n_upsets} upsets):")
    print(f"  Model:        {subset['correct'].sum()}/{n} = {model_acc:.0%}")
    print(f"  Always fav:   {subset['fav_won'].sum()}/{n} = {always_fav_acc:.0%}")
    print(f"  Always dog:   {n_upsets}/{n} = {always_dog_acc:.0%}")
    print()

# -----------------------------------------------------------------------
# Part 4: The REAL question — can we find a probability cutoff
# below which "pick the dog" beats "pick the fav"?
# -----------------------------------------------------------------------
print("=" * 75)
print("PART 4: CONDITIONAL UPSET RATE BY MODEL PROBABILITY")
print("=" * 75)
print()
print("When the model gives the favorite X% chance, how often does")
print("the upset actually happen? (close-seed R64 games only)")
print()

close_r64 = r64[r64["seed_gap"] <= 3].copy()

# Only look at games where model picks favorite (P > 0.50)
fav_picks = close_r64[close_r64["model_picked_fav"]].copy()

prob_bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
             (0.70, 0.75), (0.75, 0.80), (0.80, 0.85)]

print(f"{'P(fav) range':<16} {'Games':<8} {'Fav won':<10} {'Dog won':<10} "
      f"{'Upset rate':<12} {'Pick dog wins?'}")
print("-" * 75)

for lo, hi in prob_bins:
    bucket = fav_picks[
        (fav_picks["fav_win_prob"] >= lo) & (fav_picks["fav_win_prob"] < hi)
    ]
    if len(bucket) == 0:
        continue
    n = len(bucket)
    fav_won_n = bucket["fav_won"].sum()
    dog_won_n = n - fav_won_n
    upset_rate = dog_won_n / n
    better = "YES" if upset_rate > 0.50 else "no"
    print(f"  {lo:.0%}-{hi:.0%}       {n:<8} {fav_won_n:<10} {dog_won_n:<10} "
          f"{upset_rate:<12.0%} {better}")

# Same thing but per matchup type
print()
print("Same breakdown by matchup:")
for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = fav_picks[fav_picks["seed_matchup"] == matchup]
    if len(subset) == 0:
        continue
    print(f"\n  {matchup}:")
    for lo, hi in prob_bins:
        bucket = subset[
            (subset["fav_win_prob"] >= lo) & (subset["fav_win_prob"] < hi)
        ]
        if len(bucket) == 0:
            continue
        n = len(bucket)
        dog_won_n = n - bucket["fav_won"].sum()
        upset_rate = dog_won_n / n
        marker = " <<<" if upset_rate > 0.50 else ""
        print(f"    {lo:.0%}-{hi:.0%}: {n} games, upset rate={upset_rate:.0%}{marker}")

# -----------------------------------------------------------------------
# Part 5: Combine model probability WITH seed prior for a blended signal
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("PART 5: BAYESIAN BLEND — MODEL PROB + SEED PRIOR")
print("=" * 75)
print()
print("Instead of thresholding model output directly, blend model output")
print("with the seed prior using different weights for the model.")
print()

r64_close = r64[r64["seed_gap"] <= 3].copy()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = r64_close[r64_close["seed_matchup"] == matchup].copy()
    if len(subset) == 0:
        continue

    higher = int(matchup.split("v")[0])
    lower = int(matchup.split("v")[1])
    prior_fav = 1.0 - SEED_MATCHUP_PRIORS[(higher, lower)]

    n = len(subset)
    n_upsets = (~subset["fav_won"]).sum()
    baseline_acc = subset["correct"].sum()

    print(f"{matchup} ({n} games, {n_upsets} upsets, prior={prior_fav:.0%}):")
    print(f"  {'model_weight':<14} {'Accuracy':<10} {'Upsets caught':<16} "
          f"{'False upsets'}")
    print(f"  {'-'*55}")

    for model_w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        prior_w = 1.0 - model_w
        blended = model_w * subset["fav_win_prob"] + prior_w * prior_fav
        picks_fav = blended > 0.50
        correct = (picks_fav.values == subset["fav_won"].values).sum()

        # Upsets caught: model picks dog AND dog actually won
        upsets_caught = ((~picks_fav) & (~subset["fav_won"])).sum()
        false_upsets = ((~picks_fav) & (subset["fav_won"])).sum()

        marker = " <-- baseline" if model_w == 1.0 else ""
        print(f"  {model_w:<14.1f} {correct}/{n:<7} {upsets_caught}/{n_upsets:<13} "
              f"{false_upsets}{marker}")
    print()

# -----------------------------------------------------------------------
# Part 6: What about using model confidence as an upset detector?
# If model gives fav LESS than the prior, that's an "upset lean"
# -----------------------------------------------------------------------
print("=" * 75)
print("PART 6: PRIOR-RELATIVE UPSET DETECTION")
print("=" * 75)
print()
print("When model gives favorite LESS than the seed prior expects,")
print("is that actually predictive of upsets?")
print()

r64_close = r64[r64["seed_gap"] <= 3].copy()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = r64_close[r64_close["seed_matchup"] == matchup].copy()
    if len(subset) == 0:
        continue

    higher = int(matchup.split("v")[0])
    lower = int(matchup.split("v")[1])
    prior_fav = 1.0 - SEED_MATCHUP_PRIORS[(higher, lower)]

    subset = subset.copy()
    subset["model_below_prior"] = subset["fav_win_prob"] < prior_fav

    below = subset[subset["model_below_prior"]]
    above = subset[~subset["model_below_prior"]]

    n_below = len(below)
    n_above = len(above)

    if n_below > 0:
        upset_rate_below = (~below["fav_won"]).mean()
    else:
        upset_rate_below = 0
    if n_above > 0:
        upset_rate_above = (~above["fav_won"]).mean()
    else:
        upset_rate_above = 0

    print(f"{matchup} (prior={prior_fav:.0%}):")
    print(f"  Model BELOW prior ({n_below} games): "
          f"actual upset rate = {upset_rate_below:.0%}")
    print(f"  Model ABOVE prior ({n_above} games): "
          f"actual upset rate = {upset_rate_above:.0%}")

    if n_below > 0:
        print(f"  Games where model < prior:")
        for _, row in below.sort_values("fav_win_prob").iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            result = "UPSET" if not row["fav_won"] else "chalk"
            print(f"    {row['YEAR']} {fav}({row['fav_seed']}) vs "
                  f"{dog}({row['dog_seed']}): P(fav)={row['fav_win_prob']:.1%} "
                  f"[prior={prior_fav:.0%}] {result}")
    print()
