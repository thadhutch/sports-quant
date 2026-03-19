"""Analyze prior-relative upset threshold on v6b backtest results.

Tests whether using the gap between model probability and seed prior
as the decision signal (instead of flat 0.50 threshold) improves
close-game predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed priors (from _features.py)
# ---------------------------------------------------------------------------

SEED_MATCHUP_PRIORS: dict[tuple[int, int], float] = {
    (1, 16): 0.01,
    (2, 15): 0.06,
    (3, 14): 0.15,
    (4, 13): 0.20,
    (5, 12): 0.35,
    (6, 11): 0.37,
    (7, 10): 0.39,
    (8, 9): 0.48,
}


def favorite_win_prior(seed1: int, seed2: int) -> float:
    """Return P(favorite wins) from the historical seed prior."""
    if seed1 == seed2:
        return 0.50
    higher = min(seed1, seed2)
    lower = max(seed1, seed2)
    upset_rate = SEED_MATCHUP_PRIORS.get((higher, lower), 0.50)
    return 1.0 - upset_rate


def seed_gap_bucket(seed1: int, seed2: int) -> str:
    gap = abs(seed1 - seed2)
    if gap <= 3:
        return "close (gap<=3)"
    elif gap <= 7:
        return "moderate (4-7)"
    else:
        return "chalk (8+)"


# ---------------------------------------------------------------------------
# Load v6b data
# ---------------------------------------------------------------------------

BASE = Path("/Users/thadhutcheson/Documents/GitHub/sports-quant/data/march-madness/backtest/v6b")
YEARS = [2019, 2021, 2022, 2023, 2024, 2025]

frames = []
for year in YEARS:
    df = pd.read_csv(BASE / str(year) / "ensemble_results.csv")
    frames.append(df)

all_games = pd.concat(frames, ignore_index=True)

# Only R64 games (first-round seed matchups) — these are the canonical
# seed pairings. Later rounds have arbitrary seed combos.
r64 = all_games[
    (all_games["Seed1"] + all_games["Seed2"] == 17)
    & (all_games["Seed1"] != all_games["Seed2"])
].copy()

print(f"Total games across {len(YEARS)} years: {len(all_games)}")
print(f"R64 canonical seed matchups (seed1+seed2=17): {len(r64)}")
print()

# ---------------------------------------------------------------------------
# Annotate each R64 game
# ---------------------------------------------------------------------------

r64 = r64.copy()

# Who is the favorite (lower seed number)?
r64["fav_is_team1"] = r64["Seed1"] < r64["Seed2"]
r64["fav_seed"] = r64[["Seed1", "Seed2"]].min(axis=1)
r64["dog_seed"] = r64[["Seed1", "Seed2"]].max(axis=1)

# Model's probability the favorite wins
r64["fav_win_prob"] = np.where(
    r64["fav_is_team1"],
    r64["Ensemble_Prob"],
    1.0 - r64["Ensemble_Prob"],
)

# Did the favorite actually win?
r64["fav_actually_won"] = np.where(
    r64["fav_is_team1"],
    r64["Team1_Win"] == 1,
    r64["Team1_Win"] == 0,
)

# Seed prior
r64["fav_prior"] = r64.apply(
    lambda row: favorite_win_prior(int(row["Seed1"]), int(row["Seed2"])),
    axis=1,
)

# Gap consumed: how far the model moved from prior toward 0.50
# prior=0.65, model=0.53 → consumed = (0.65-0.53)/(0.65-0.50) = 0.80
r64["gap_consumed"] = np.where(
    r64["fav_prior"] > 0.50,
    (r64["fav_prior"] - r64["fav_win_prob"]) / (r64["fav_prior"] - 0.50),
    0.0,
)

r64["seed_matchup"] = r64["fav_seed"].astype(str) + "v" + r64["dog_seed"].astype(str)
r64["bucket"] = r64.apply(
    lambda row: seed_gap_bucket(int(row["Seed1"]), int(row["Seed2"])),
    axis=1,
)

# Current prediction (flat 0.50 threshold)
r64["current_picks_fav"] = r64["fav_win_prob"] > 0.50
r64["current_correct"] = r64["current_picks_fav"] == r64["fav_actually_won"]

# ---------------------------------------------------------------------------
# Section 1: Show what the model sees for close games
# ---------------------------------------------------------------------------

print("=" * 70)
print("SECTION 1: MODEL PROBABILITIES vs SEED PRIORS (R64 only)")
print("=" * 70)
print()

for matchup in ["8v9", "7v10", "6v11", "5v12"]:
    subset = r64[r64["seed_matchup"] == matchup]
    n = len(subset)
    if n == 0:
        continue

    actual_upset_rate = (~subset["fav_actually_won"]).mean()
    model_avg_prob = subset["fav_win_prob"].mean()
    prior = subset["fav_prior"].iloc[0]
    current_acc = subset["current_correct"].mean()

    print(f"--- {matchup} ({n} games across {len(YEARS)} years) ---")
    print(f"  Historical prior (fav wins):  {prior:.0%}")
    print(f"  Model avg P(fav wins):        {model_avg_prob:.1%}")
    print(f"  Actual fav win rate:           {1-actual_upset_rate:.0%}")
    print(f"  Current accuracy (>0.50):      {current_acc:.0%}")
    print()

    # Show individual games where model is close
    close_calls = subset[subset["fav_win_prob"] < 0.70].sort_values("fav_win_prob")
    if len(close_calls) > 0:
        print(f"  Games where model gave favorite < 70%:")
        for _, row in close_calls.iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            upset_happened = "UPSET" if not row["fav_actually_won"] else "chalk"
            picked = "picked fav" if row["current_picks_fav"] else "PICKED DOG"
            gap_pct = row["gap_consumed"]
            print(
                f"    {row['YEAR']} {fav}({row['fav_seed']}) vs {dog}({row['dog_seed']}): "
                f"P(fav)={row['fav_win_prob']:.1%}  gap_consumed={gap_pct:.0%}  "
                f"{upset_happened}  {picked}"
            )
        print()

# ---------------------------------------------------------------------------
# Section 2: Test gap_consumed thresholds
# ---------------------------------------------------------------------------

print("=" * 70)
print("SECTION 2: PRIOR-RELATIVE THRESHOLD SWEEP (close games only)")
print("=" * 70)
print()

close_games = r64[r64["bucket"] == "close (gap<=3)"].copy()
print(f"Close-seed R64 games (5v12, 6v11, 7v10, 8v9): {len(close_games)}")
print()

# Current baseline
baseline_correct = close_games["current_correct"].sum()
baseline_total = len(close_games)
print(f"Baseline (flat 0.50 threshold): {baseline_correct}/{baseline_total} "
      f"= {baseline_correct/baseline_total:.1%}")
print()

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

print(f"{'Threshold':<12} {'Correct':<10} {'Accuracy':<10} {'Flips':<8} "
      f"{'Flip→Right':<12} {'Flip→Wrong':<12}")
print("-" * 70)

for thresh in thresholds:
    # New logic: if gap_consumed > threshold, flip to upset pick
    picks_fav = close_games["fav_win_prob"] > 0.50  # start with baseline
    flipped = close_games["gap_consumed"] > thresh

    # For flipped games, override: pick the underdog instead
    new_picks_fav = picks_fav & ~flipped

    # Also handle cases where baseline already picked the dog
    # (fav_win_prob < 0.50) — don't flip those back
    new_picks_fav = np.where(
        flipped & picks_fav,  # was picking fav, now flip to dog
        False,
        picks_fav,
    )

    correct = (new_picks_fav == close_games["fav_actually_won"].values).sum()

    # How many games flipped from fav→dog?
    n_flips = (flipped & picks_fav).sum()
    flip_mask = flipped & picks_fav
    flips_correct = (
        (~close_games["fav_actually_won"].values[flip_mask.values]).sum()
        if n_flips > 0
        else 0
    )
    flips_wrong = n_flips - flips_correct

    print(
        f"{thresh:<12.2f} {correct:<10} {correct/baseline_total:<10.1%} "
        f"{n_flips:<8} {flips_correct:<12} {flips_wrong:<12}"
    )

# ---------------------------------------------------------------------------
# Section 3: Per-matchup breakdown at a few key thresholds
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("SECTION 3: MATCHUP-LEVEL DETAIL AT KEY THRESHOLDS")
print("=" * 70)

for thresh in [0.65, 0.70, 0.75]:
    print(f"\n--- gap_consumed threshold = {thresh:.0%} ---")
    for matchup in ["5v12", "6v11", "7v10", "8v9"]:
        subset = close_games[close_games["seed_matchup"] == matchup]
        if len(subset) == 0:
            continue

        picks_fav_base = subset["fav_win_prob"] > 0.50
        flipped = subset["gap_consumed"] > thresh
        would_flip = flipped & picks_fav_base

        new_picks_fav = picks_fav_base & ~flipped
        new_correct = (new_picks_fav.values == subset["fav_actually_won"].values).sum()
        base_correct = subset["current_correct"].sum()
        n = len(subset)

        print(
            f"  {matchup}: baseline {base_correct}/{n} → "
            f"new {new_correct}/{n}  "
            f"(flips: {would_flip.sum()})"
        )

        # Show the flipped games
        if would_flip.sum() > 0:
            for _, row in subset[would_flip].iterrows():
                fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
                dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
                result = "UPSET" if not row["fav_actually_won"] else "chalk"
                flip_outcome = "GOOD FLIP" if not row["fav_actually_won"] else "BAD FLIP"
                print(
                    f"      {row['YEAR']} {dog}({row['dog_seed']}) over "
                    f"{fav}({row['fav_seed']}): P(fav)={row['fav_win_prob']:.1%} "
                    f"gap={row['gap_consumed']:.0%}  {result} → {flip_outcome}"
                )

# ---------------------------------------------------------------------------
# Section 4: Full-bracket impact (all games, not just R64)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("SECTION 4: FULL-BRACKET IMPACT (all 63 games per year)")
print("=" * 70)
print()

# For later-round games, we need a more general prior lookup
# Only apply the threshold to games with a known seed prior
all_annotated = all_games.copy()
all_annotated["fav_is_team1"] = all_annotated["Seed1"] <= all_annotated["Seed2"]
all_annotated["fav_seed"] = all_annotated[["Seed1", "Seed2"]].min(axis=1)
all_annotated["dog_seed"] = all_annotated[["Seed1", "Seed2"]].max(axis=1)
all_annotated["seed_gap"] = abs(all_annotated["Seed1"] - all_annotated["Seed2"])

all_annotated["fav_win_prob"] = np.where(
    all_annotated["fav_is_team1"],
    all_annotated["Ensemble_Prob"],
    1.0 - all_annotated["Ensemble_Prob"],
)
all_annotated["fav_actually_won"] = np.where(
    all_annotated["fav_is_team1"],
    all_annotated["Team1_Win"] == 1,
    all_annotated["Team1_Win"] == 0,
)

# For later rounds, use the actual seed gap to compute a rough prior
# Only apply threshold logic to close games (gap <= 3)
all_annotated["fav_prior"] = all_annotated.apply(
    lambda row: favorite_win_prior(int(row["Seed1"]), int(row["Seed2"])),
    axis=1,
)
all_annotated["gap_consumed"] = np.where(
    all_annotated["fav_prior"] > 0.50,
    (all_annotated["fav_prior"] - all_annotated["fav_win_prob"])
    / (all_annotated["fav_prior"] - 0.50),
    0.0,
)
all_annotated["is_close"] = all_annotated["seed_gap"] <= 3

baseline_correct_all = (all_annotated["Correct_Prediction"]).sum()
print(f"Baseline full-bracket accuracy: {baseline_correct_all}/{len(all_annotated)} "
      f"= {baseline_correct_all/len(all_annotated):.1%}")
print()

for thresh in [0.65, 0.70, 0.75]:
    # Only flip close games where we currently pick the favorite
    picks_fav = all_annotated["fav_win_prob"] > 0.50
    should_flip = (
        all_annotated["is_close"]
        & picks_fav
        & (all_annotated["gap_consumed"] > thresh)
    )

    # New prediction: flip close games, keep everything else
    new_picks_fav = picks_fav.copy()
    new_picks_fav[should_flip] = False

    new_correct = (new_picks_fav.values == all_annotated["fav_actually_won"].values).sum()
    delta = new_correct - baseline_correct_all

    n_flips = should_flip.sum()
    if n_flips > 0:
        flip_results = ~all_annotated.loc[should_flip, "fav_actually_won"]
        good_flips = flip_results.sum()
        bad_flips = n_flips - good_flips
    else:
        good_flips = bad_flips = 0

    print(
        f"  thresh={thresh:.2f}: {new_correct}/{len(all_annotated)} "
        f"= {new_correct/len(all_annotated):.1%}  "
        f"(delta={delta:+d})  flips={n_flips} "
        f"(good={good_flips}, bad={bad_flips})"
    )

# ---------------------------------------------------------------------------
# Section 5: Year-by-year breakdown at best threshold
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("SECTION 5: YEAR-BY-YEAR AT THRESHOLD = 0.70")
print("=" * 70)
print()

THRESH = 0.70

print(f"{'Year':<8} {'Baseline':<12} {'New':<12} {'Delta':<8} {'Flips':<8} {'Detail'}")
print("-" * 80)

for year in YEARS:
    year_df = all_annotated[all_annotated["YEAR"] == year]
    base = year_df["Correct_Prediction"].sum()
    total = len(year_df)

    picks_fav = year_df["fav_win_prob"] > 0.50
    should_flip = (
        year_df["is_close"]
        & picks_fav
        & (year_df["gap_consumed"] > THRESH)
    )

    new_picks_fav = picks_fav.copy()
    new_picks_fav[should_flip] = False
    new_correct = (new_picks_fav.values == year_df["fav_actually_won"].values).sum()
    delta = new_correct - base

    flip_details = []
    for idx in year_df[should_flip].index:
        row = year_df.loc[idx]
        fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
        dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
        outcome = "GOOD" if not row["fav_actually_won"] else "BAD"
        flip_details.append(f"{dog}>{fav}({outcome})")

    detail_str = ", ".join(flip_details) if flip_details else "no flips"
    print(
        f"{year:<8} {base}/{total:<9} {new_correct}/{total:<9} "
        f"{delta:<+8} {should_flip.sum():<8} {detail_str}"
    )
