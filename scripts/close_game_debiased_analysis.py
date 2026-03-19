"""Rerun close-game analysis using DEBIASED results (what the simulation actually uses).

Previous analysis used ensemble_results.csv (raw average).
The simulation SVG uses debiased probabilities — which are meaningfully different
for close games.
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

# Load both ensemble and debiased for comparison
ens_frames = []
deb_frames = []
for year in YEARS:
    ens = pd.read_csv(BASE / str(year) / "ensemble_results.csv")
    deb = pd.read_csv(BASE / str(year) / "debiased_results.csv")
    ens_frames.append(ens)
    deb_frames.append(deb)

all_ens = pd.concat(ens_frames, ignore_index=True)
all_deb = pd.concat(deb_frames, ignore_index=True)

print(f"Loaded {len(all_ens)} ensemble games, {len(all_deb)} debiased games")
print()

# Quick sanity: where do ensemble and debiased disagree?
merged = all_ens[["YEAR", "Team1", "Team2", "Seed1", "Seed2", "Team1_Win",
                   "Ensemble_Prob", "Ensemble_Pred", "Correct_Prediction"]].copy()
merged["Debiased_Prob"] = all_deb["Debiased_Prob"]
merged["Debiased_Pred"] = all_deb["Debiased_Pred"]
merged["Debiased_Correct"] = all_deb["Correct_Prediction"]

disagreements = merged[merged["Ensemble_Pred"] != merged["Debiased_Pred"]]
print(f"Games where ensemble and debiased DISAGREE: {len(disagreements)}/{len(merged)}")
print()

ens_correct = merged["Correct_Prediction"].sum()
deb_correct = merged["Debiased_Correct"].sum()
print(f"Overall accuracy - Ensemble: {ens_correct}/{len(merged)} = "
      f"{ens_correct/len(merged):.1%}")
print(f"Overall accuracy - Debiased: {deb_correct}/{len(merged)} = "
      f"{deb_correct/len(merged):.1%}")
print()

# Show all disagreements
print("=" * 80)
print("ALL DISAGREEMENTS (ensemble vs debiased)")
print("=" * 80)
print()

for _, row in disagreements.sort_values(["YEAR", "Seed1"]).iterrows():
    seed_gap = abs(row["Seed1"] - row["Seed2"])
    ens_winner = row["Team1"] if row["Ensemble_Pred"] == 1 else row["Team2"]
    deb_winner = row["Team1"] if row["Debiased_Pred"] == 1 else row["Team2"]
    actual_winner = row["Team1"] if row["Team1_Win"] == 1 else row["Team2"]

    ens_right = "OK" if row["Correct_Prediction"] else "WRONG"
    deb_right = "OK" if row["Debiased_Correct"] else "WRONG"

    print(f"  {row['YEAR']} {row['Team1']}({row['Seed1']}) vs "
          f"{row['Team2']}({row['Seed2']})  gap={seed_gap}")
    print(f"    Ensemble: {ens_winner} (P={row['Ensemble_Prob']:.3f}) {ens_right}")
    print(f"    Debiased: {deb_winner} (P={row['Debiased_Prob']:.3f}) {deb_right}")
    print(f"    Actual:   {actual_winner}")
    print()

# Now redo the close-game analysis with debiased data
print("=" * 80)
print("CLOSE-GAME ANALYSIS WITH DEBIASED PROBABILITIES")
print("=" * 80)
print()

# Annotate with debiased data
df = merged.copy()
df["fav_is_team1"] = df["Seed1"] < df["Seed2"]
df.loc[df["Seed1"] == df["Seed2"], "fav_is_team1"] = True
df["fav_seed"] = df[["Seed1", "Seed2"]].min(axis=1)
df["dog_seed"] = df[["Seed1", "Seed2"]].max(axis=1)
df["seed_gap"] = abs(df["Seed1"] - df["Seed2"])
df["seed_matchup"] = df["fav_seed"].astype(str) + "v" + df["dog_seed"].astype(str)

# Debiased probability from favorite's perspective
df["deb_fav_prob"] = np.where(
    df["fav_is_team1"],
    df["Debiased_Prob"],
    1.0 - df["Debiased_Prob"],
)
df["ens_fav_prob"] = np.where(
    df["fav_is_team1"],
    df["Ensemble_Prob"],
    1.0 - df["Ensemble_Prob"],
)
df["fav_won"] = np.where(
    df["fav_is_team1"],
    df["Team1_Win"] == 1,
    df["Team1_Win"] == 0,
)

# R64 canonical pairings
r64 = df[
    (df["Seed1"] + df["Seed2"] == 17)
    & (df["Seed1"] != df["Seed2"])
].copy()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = r64[r64["seed_matchup"] == matchup]
    n = len(subset)
    upsets = (~subset["fav_won"]).sum()

    ens_picks_fav = subset["ens_fav_prob"] > 0.50
    deb_picks_fav = subset["deb_fav_prob"] > 0.50

    ens_correct = (ens_picks_fav.values == subset["fav_won"].values).sum()
    deb_correct = (deb_picks_fav.values == subset["fav_won"].values).sum()

    # Upsets caught
    ens_upsets_caught = ((~ens_picks_fav) & (~subset["fav_won"])).sum()
    deb_upsets_caught = ((~deb_picks_fav) & (~subset["fav_won"])).sum()

    print(f"--- {matchup} ({n} games, {upsets} actual upsets) ---")
    print(f"  Ensemble:  {ens_correct}/{n} = {ens_correct/n:.0%}  "
          f"(caught {ens_upsets_caught}/{upsets} upsets)")
    print(f"  Debiased:  {deb_correct}/{n} = {deb_correct/n:.0%}  "
          f"(caught {deb_upsets_caught}/{upsets} upsets)")
    print()

    # Show games where debiased differs from ensemble
    diffs = subset[ens_picks_fav != deb_picks_fav]
    if len(diffs) > 0:
        print(f"  Games where debiasing changed the pick:")
        for _, row in diffs.iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            result = "UPSET" if not row["fav_won"] else "chalk"
            ens_pick = "fav" if row["ens_fav_prob"] > 0.50 else "DOG"
            deb_pick = "fav" if row["deb_fav_prob"] > 0.50 else "DOG"
            print(f"    {row['YEAR']} {fav}({row['fav_seed']}) vs "
                  f"{dog}({row['dog_seed']}): "
                  f"ens={row['ens_fav_prob']:.1%}→{ens_pick}  "
                  f"deb={row['deb_fav_prob']:.1%}→{deb_pick}  "
                  f"actual={result}")
        print()

    # Show ALL games sorted by debiased probability
    print(f"  All {matchup} games sorted by debiased P(fav):")
    for _, row in subset.sort_values("deb_fav_prob").iterrows():
        fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
        dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
        result = "UPSET" if not row["fav_won"] else "chalk"
        deb_pick = "picked fav" if row["deb_fav_prob"] > 0.50 else "PICKED DOG"
        delta = row["deb_fav_prob"] - row["ens_fav_prob"]
        print(f"    {row['YEAR']} {fav}({row['fav_seed']}) vs "
              f"{dog}({row['dog_seed']}): "
              f"ens={row['ens_fav_prob']:.1%} deb={row['deb_fav_prob']:.1%} "
              f"(delta={delta:+.1%})  {result}  {deb_pick}")
    print()

# Summary: how much does debiasing help on close games specifically?
print("=" * 80)
print("SUMMARY: DEBIASING IMPACT ON CLOSE GAMES")
print("=" * 80)
print()

close_r64 = r64[r64["seed_gap"] <= 3]
ens_acc = (
    (close_r64["ens_fav_prob"] > 0.50).values == close_r64["fav_won"].values
).sum()
deb_acc = (
    (close_r64["deb_fav_prob"] > 0.50).values == close_r64["fav_won"].values
).sum()
n = len(close_r64)

print(f"Close-seed R64 (5v12, 6v11, 7v10, 8v9): {n} games")
print(f"  Ensemble accuracy: {ens_acc}/{n} = {ens_acc/n:.1%}")
print(f"  Debiased accuracy: {deb_acc}/{n} = {deb_acc/n:.1%}")
print(f"  Delta: {deb_acc - ens_acc:+d} games")
print()

# Now: what about the gap_consumed approach on DEBIASED probabilities?
print("=" * 80)
print("GAP CONSUMED ANALYSIS ON DEBIASED PROBABILITIES")
print("=" * 80)
print()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    higher = int(matchup.split("v")[0])
    lower = int(matchup.split("v")[1])
    prior_fav = 1.0 - SEED_MATCHUP_PRIORS[(higher, lower)]

    subset = r64[r64["seed_matchup"] == matchup].copy()
    if len(subset) == 0:
        continue

    subset = subset.copy()
    subset["gap_consumed_deb"] = np.where(
        prior_fav > 0.50,
        (prior_fav - subset["deb_fav_prob"]) / (prior_fav - 0.50),
        0.0,
    )

    n = len(subset)
    n_upsets = (~subset["fav_won"]).sum()

    # How many games have gap_consumed > various thresholds?
    print(f"{matchup} (prior={prior_fav:.0%}, {n} games, {n_upsets} upsets):")
    for thresh in [0.0, 0.25, 0.50, 0.75, 1.0]:
        above = subset[subset["gap_consumed_deb"] > thresh]
        n_above = len(above)
        if n_above > 0:
            upset_rate = (~above["fav_won"]).mean()
            print(f"  gap_consumed > {thresh:.0%}: {n_above} games, "
                  f"upset rate = {upset_rate:.0%}")
    print()

    # Show games with positive gap_consumed
    positives = subset[subset["gap_consumed_deb"] > 0].sort_values(
        "gap_consumed_deb", ascending=False
    )
    if len(positives) > 0:
        print(f"  Games where debiased P(fav) < prior ({prior_fav:.0%}):")
        for _, row in positives.iterrows():
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            result = "UPSET" if not row["fav_won"] else "chalk"
            print(f"    {row['YEAR']} {fav} vs {dog}: "
                  f"deb={row['deb_fav_prob']:.1%} "
                  f"gap_consumed={row['gap_consumed_deb']:.0%}  {result}")
        print()
