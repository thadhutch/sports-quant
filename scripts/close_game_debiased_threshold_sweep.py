"""Threshold sweep on DEBIASED probabilities for close-game upset detection.

Tests whether gap_consumed (how far the debiased probability drops below
the seed prior) is predictive of upsets — and whether we can use it to
flip picks profitably.
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
    deb = pd.read_csv(BASE / str(year) / "debiased_results.csv")
    frames.append(deb)
all_deb = pd.concat(frames, ignore_index=True)

# Annotate
all_deb["fav_is_team1"] = all_deb["Seed1"] < all_deb["Seed2"]
all_deb.loc[all_deb["Seed1"] == all_deb["Seed2"], "fav_is_team1"] = True
all_deb["fav_seed"] = all_deb[["Seed1", "Seed2"]].min(axis=1)
all_deb["dog_seed"] = all_deb[["Seed1", "Seed2"]].max(axis=1)
all_deb["seed_gap"] = abs(all_deb["Seed1"] - all_deb["Seed2"])
all_deb["seed_matchup"] = (
    all_deb["fav_seed"].astype(str) + "v" + all_deb["dog_seed"].astype(str)
)

all_deb["deb_fav_prob"] = np.where(
    all_deb["fav_is_team1"],
    all_deb["Debiased_Prob"],
    1.0 - all_deb["Debiased_Prob"],
)
all_deb["fav_won"] = np.where(
    all_deb["fav_is_team1"],
    all_deb["Team1_Win"] == 1,
    all_deb["Team1_Win"] == 0,
)

def favorite_win_prior(seed1: int, seed2: int) -> float:
    if seed1 == seed2:
        return 0.50
    higher = min(seed1, seed2)
    lower = max(seed1, seed2)
    upset_rate = SEED_MATCHUP_PRIORS.get((higher, lower), 0.50)
    return 1.0 - upset_rate

all_deb["fav_prior"] = all_deb.apply(
    lambda row: favorite_win_prior(int(row["Seed1"]), int(row["Seed2"])),
    axis=1,
)
all_deb["gap_consumed"] = np.where(
    all_deb["fav_prior"] > 0.50,
    (all_deb["fav_prior"] - all_deb["deb_fav_prob"]) / (all_deb["fav_prior"] - 0.50),
    0.0,
)

all_deb["deb_picks_fav"] = all_deb["deb_fav_prob"] > 0.50
all_deb["deb_correct"] = all_deb["deb_picks_fav"] == all_deb["fav_won"]

# R64 canonical pairings
r64 = all_deb[
    (all_deb["Seed1"] + all_deb["Seed2"] == 17)
    & (all_deb["Seed1"] != all_deb["Seed2"])
].copy()

# "Close" = the historically competitive R64 matchups (5v12, 6v11, 7v10, 8v9)
CLOSE_MATCHUPS = {"5v12", "6v11", "7v10", "8v9"}
close = r64[r64["seed_matchup"].isin(CLOSE_MATCHUPS)].copy()

# -----------------------------------------------------------------------
# Section 1: Baseline
# -----------------------------------------------------------------------
print("=" * 75)
print("SECTION 1: DEBIASED BASELINE (close-seed R64 games)")
print("=" * 75)
print()

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = close[close["seed_matchup"] == matchup]
    n = len(subset)
    upsets = (~subset["fav_won"]).sum()
    acc = subset["deb_correct"].sum()
    upsets_caught = ((~subset["deb_picks_fav"]) & (~subset["fav_won"])).sum()
    print(f"{matchup}: {acc}/{n} = {acc/n:.0%}  "
          f"(caught {upsets_caught}/{upsets} upsets)")

total_correct = close["deb_correct"].sum()
total = len(close)
print(f"\nTotal close-seed R64: {total_correct}/{total} = {total_correct/total:.1%}")

# -----------------------------------------------------------------------
# Section 2: Gap consumed threshold sweep per matchup
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("SECTION 2: GAP CONSUMED THRESHOLD SWEEP (per matchup)")
print("=" * 75)
print()
print("Logic: if debiased P(fav) has dropped more than X% of the way from")
print("the seed prior toward 0.50, flip the pick to the underdog.")
print()

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50]

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = close[close["seed_matchup"] == matchup].copy()
    n = len(subset)
    n_upsets = (~subset["fav_won"]).sum()
    baseline = subset["deb_correct"].sum()

    print(f"--- {matchup} ({n} games, {n_upsets} upsets, baseline={baseline}/{n}) ---")
    print(f"  {'Threshold':<12} {'Acc':<8} {'Delta':<8} {'Flips':<8} "
          f"{'Good':<8} {'Bad':<8} {'Upset recall'}")
    print(f"  {'-'*68}")

    for thresh in thresholds:
        # Start with debiased picks
        picks_fav = subset["deb_fav_prob"].values > 0.50

        # Flip: if gap_consumed > threshold AND currently picking fav
        should_flip = (subset["gap_consumed"].values > thresh) & picks_fav
        new_picks_fav = picks_fav.copy()
        new_picks_fav[should_flip] = False

        correct = (new_picks_fav == subset["fav_won"].values).sum()
        delta = correct - baseline

        n_flips = should_flip.sum()
        good = 0
        bad = 0
        if n_flips > 0:
            good = (~subset["fav_won"].values[should_flip]).sum()
            bad = n_flips - good

        # Upset recall: how many of the actual upsets do we now catch?
        upsets_caught = ((~new_picks_fav) & (~subset["fav_won"].values)).sum()

        print(f"  {thresh:<12.2f} {correct}/{n:<5} {delta:<+8} {n_flips:<8} "
              f"{good:<8} {bad:<8} {upsets_caught}/{n_upsets}")

    print()

# -----------------------------------------------------------------------
# Section 3: Combined close-game sweep
# -----------------------------------------------------------------------
print("=" * 75)
print("SECTION 3: COMBINED CLOSE-GAME SWEEP (all 5v12 + 6v11 + 7v10 + 8v9)")
print("=" * 75)
print()

n = len(close)
baseline = close["deb_correct"].sum()
n_upsets = (~close["fav_won"]).sum()
print(f"Total: {n} games, {n_upsets} upsets, baseline={baseline}/{n} = {baseline/n:.1%}")
print()

print(f"{'Threshold':<12} {'Acc':<10} {'Pct':<8} {'Delta':<8} {'Flips':<8} "
      f"{'Good':<8} {'Bad':<8} {'Upset recall'}")
print("-" * 80)

for thresh in thresholds:
    picks_fav = close["deb_fav_prob"].values > 0.50
    should_flip = (close["gap_consumed"].values > thresh) & picks_fav
    new_picks_fav = picks_fav.copy()
    new_picks_fav[should_flip] = False

    correct = (new_picks_fav == close["fav_won"].values).sum()
    delta = correct - baseline
    n_flips = should_flip.sum()
    good = (~close["fav_won"].values[should_flip]).sum() if n_flips > 0 else 0
    bad = n_flips - good
    upsets_caught = ((~new_picks_fav) & (~close["fav_won"].values)).sum()

    print(f"{thresh:<12.2f} {correct}/{n:<7} {correct/n:<8.1%} {delta:<+8} "
          f"{n_flips:<8} {good:<8} {bad:<8} {upsets_caught}/{n_upsets}")

# -----------------------------------------------------------------------
# Section 4: Full bracket impact with debiased data
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("SECTION 4: FULL BRACKET IMPACT (all 63 games × 6 years)")
print("=" * 75)
print()

all_deb["is_close"] = all_deb["seed_matchup"].isin(CLOSE_MATCHUPS)

baseline_all = all_deb["deb_correct"].sum()
total_all = len(all_deb)
print(f"Full bracket baseline: {baseline_all}/{total_all} = "
      f"{baseline_all/total_all:.1%}")
print()

# Also add: only apply to R64 canonical pairings (most reliable priors)
r64_canonical_mask = (
    (all_deb["Seed1"] + all_deb["Seed2"] == 17)
    & (all_deb["Seed1"] != all_deb["Seed2"])
    & (all_deb["seed_matchup"].isin(CLOSE_MATCHUPS))
)

print(f"{'Threshold':<12} {'CloseOnly':<20} {'R64CanonOnly':<20} {'AllClose':<20}")
print("-" * 75)

for thresh in thresholds:
    # Strategy A: only flip R64 canonical close pairings
    picks_fav_a = all_deb["deb_fav_prob"].values > 0.50
    should_flip_a = (
        r64_canonical_mask.values
        & picks_fav_a
        & (all_deb["gap_consumed"].values > thresh)
    )
    new_a = picks_fav_a.copy()
    new_a[should_flip_a] = False
    correct_a = (new_a == all_deb["fav_won"].values).sum()
    delta_a = correct_a - baseline_all

    # Strategy B: flip ALL close games (including later rounds)
    picks_fav_b = all_deb["deb_fav_prob"].values > 0.50
    should_flip_b = (
        all_deb["is_close"].values
        & picks_fav_b
        & (all_deb["gap_consumed"].values > thresh)
    )
    new_b = picks_fav_b.copy()
    new_b[should_flip_b] = False
    correct_b = (new_b == all_deb["fav_won"].values).sum()
    delta_b = correct_b - baseline_all

    print(f"{thresh:<12.2f} "
          f"{correct_a}/{total_all} ({delta_a:+d}, {should_flip_a.sum()} flips)   "
          f"{correct_b}/{total_all} ({delta_b:+d}, {should_flip_b.sum()} flips)")

# -----------------------------------------------------------------------
# Section 5: Year-by-year at a few key thresholds (R64 canonical only)
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("SECTION 5: YEAR-BY-YEAR (R64 canonical close games only)")
print("=" * 75)

for thresh in [0.50, 0.70, 0.90]:
    print(f"\n--- gap_consumed threshold = {thresh:.0%} ---")
    print(f"{'Year':<8} {'Baseline':<12} {'New':<12} {'Delta':<8} {'Flips (detail)'}")
    print("-" * 80)

    for year in YEARS:
        year_all = all_deb[all_deb["YEAR"] == year]
        year_r64_close = year_all[
            (year_all["Seed1"] + year_all["Seed2"] == 17)
            & (year_all["Seed1"] != year_all["Seed2"])
            & (year_all["seed_matchup"].isin(CLOSE_MATCHUPS))
        ]

        base = year_all["deb_correct"].sum()
        total = len(year_all)

        # Apply flips only to R64 canonical close games
        picks_fav = year_all["deb_fav_prob"].values > 0.50
        is_target = (
            (year_all["Seed1"] + year_all["Seed2"] == 17).values
            & (year_all["Seed1"] != year_all["Seed2"]).values
            & (year_all["seed_matchup"].isin(CLOSE_MATCHUPS)).values
        )
        should_flip = is_target & picks_fav & (year_all["gap_consumed"].values > thresh)

        new_picks = picks_fav.copy()
        new_picks[should_flip] = False
        new_correct = (new_picks == year_all["fav_won"].values).sum()
        delta = new_correct - base

        flip_details = []
        for idx in year_all[should_flip].index:
            row = year_all.loc[idx]
            fav = row["Team1"] if row["fav_is_team1"] else row["Team2"]
            dog = row["Team2"] if row["fav_is_team1"] else row["Team1"]
            outcome = "GOOD" if not row["fav_won"] else "BAD"
            print_prob = f"{row['deb_fav_prob']:.0%}"
            flip_details.append(f"{dog}>{fav}[{print_prob}]({outcome})")

        detail = ", ".join(flip_details) if flip_details else "no flips"
        print(f"{year:<8} {base}/{total:<9} {new_correct}/{total:<9} "
              f"{delta:<+8} {detail}")

# -----------------------------------------------------------------------
# Section 6: Matchup-specific optimal thresholds
# -----------------------------------------------------------------------
print()
print("=" * 75)
print("SECTION 6: OPTIMAL THRESHOLD PER MATCHUP (leave-one-year-out)")
print("=" * 75)
print()
print("For each year, find the best threshold on the OTHER years,")
print("then apply it to the held-out year. This avoids overfitting.")
print()

fine_thresholds = np.arange(0.0, 2.01, 0.05)

for matchup in ["5v12", "6v11", "7v10", "8v9"]:
    subset = close[close["seed_matchup"] == matchup].copy()
    n = len(subset)
    n_upsets = (~subset["fav_won"]).sum()
    baseline = subset["deb_correct"].sum()

    print(f"--- {matchup} ({n} games, {n_upsets} upsets, baseline={baseline}/{n}) ---")

    loo_correct_total = 0
    loo_baseline_total = 0

    for held_out_year in YEARS:
        train = subset[subset["YEAR"] != held_out_year]
        test = subset[subset["YEAR"] == held_out_year]

        if len(test) == 0:
            continue

        # Find best threshold on training years
        best_thresh = 999.0
        best_acc = -1
        for t in fine_thresholds:
            pf = train["deb_fav_prob"].values > 0.50
            sf = (train["gap_consumed"].values > t) & pf
            npf = pf.copy()
            npf[sf] = False
            acc = (npf == train["fav_won"].values).sum()
            if acc > best_acc or (acc == best_acc and t > best_thresh):
                best_acc = acc
                best_thresh = t

        # Apply to test year
        pf_test = test["deb_fav_prob"].values > 0.50
        sf_test = (test["gap_consumed"].values > best_thresh) & pf_test
        npf_test = pf_test.copy()
        npf_test[sf_test] = False
        test_correct = (npf_test == test["fav_won"].values).sum()
        test_baseline = test["deb_correct"].sum()

        loo_correct_total += test_correct
        loo_baseline_total += test_baseline

        n_test = len(test)
        delta = test_correct - test_baseline
        if delta != 0:
            print(f"  {held_out_year}: best_thresh={best_thresh:.2f} → "
                  f"{test_correct}/{n_test} (delta={delta:+d})")

    loo_delta = loo_correct_total - loo_baseline_total
    print(f"  LOO total: {loo_correct_total}/{n} vs baseline {loo_baseline_total}/{n} "
          f"(delta={loo_delta:+d})")
    print()
