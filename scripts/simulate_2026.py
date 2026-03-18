"""Forward-simulate the 2026 NCAA tournament bracket.

Fetches R64 matchups from the ESPN API (with region info),
resolves First Four results, orders games in canonical bracket
order, and runs a deterministic simulation using the v6b ensemble.

Outputs an SVG bracket to the backtest plots directory.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sports_quant.march_madness._bracket import (
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
    determine_upset,
)
from sports_quant.march_madness._bracket_builder import (
    _CANONICAL_SEED_PAIR_POS,
    _assign_region,
)
from sports_quant.march_madness._config import (
    MM_BACKTEST_DIR,
    MM_BARTTORVIK_DATA,
    MM_KENPOM_DATA,
    load_mm_config,
    load_models,
)
from sports_quant.march_madness._feature_builder import FeatureLookup, TeamStats
from sports_quant.march_madness._features import standardize_team_name
from sports_quant.march_madness.bracket_plots import render_bracket
from sports_quant.march_madness.simulate import _predict_game

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESPN helpers
# ---------------------------------------------------------------------------

_ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
    "?dates={date}&groups=100&limit=50"
)

# Map ESPN names → FeatureLookup index names.
# These must match what standardize_team_name(KenPom "Team" column) produces.
_ESPN_NAME_OVERRIDES: dict[str, str] = {
    "UConn": "Connecticut",
    "Ole Miss": "Mississippi",
    "Long Island University": "LIU Brooklyn",
    "SIU-Edwardsville": "SIUE",
    "Miami (OH)": "Miami OH",
    "California Baptist": "Cal Baptist",
    "Queens University": "Queens",
    "Hawai'i": "Hawaii",
    "Kennesaw State": "Kennesaw St.",
    "Tennessee State": "Tennessee St.",
    "Wright State": "Wright St.",
    "North Dakota State": "North Dakota St.",
    "South Dakota State": "South Dakota St.",
    "Prairie View A&M": "Prairie View",
    "Pennsylvania": "Penn",
    "Utah State": "Utah St.",
    "Miami": "Miami FL",
}


def _espn_to_standard(name: str) -> str:
    """Map ESPN display name to FeatureLookup index name."""
    if name in _ESPN_NAME_OVERRIDES:
        return _ESPN_NAME_OVERRIDES[name]
    return standardize_team_name(name)


def _fetch_espn(date_str: str) -> dict:
    url = _ESPN_URL.format(date=date_str)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Region ordering — canonical region order for the bracket
# ---------------------------------------------------------------------------

# The standard bracket layout: regions are displayed in a specific order.
# We assign region index based on alphabetical ordering of region names
# from the ESPN data, which gives a stable ordering.
_REGION_ORDER = ["East Region", "Midwest Region", "South Region", "West Region"]


def _fetch_first_four_results() -> dict[str, dict]:
    """Fetch First Four results to resolve TBD entries.

    Returns dict keyed by region+seed_type with winner info:
    {
        ("Midwest Region", 11): {"name": "SMU", "seed": 11},
        ("South Region", 16): {"name": "Lehigh", "seed": 16},
        ...
    }
    """
    results: dict[tuple[str, int], dict] = {}

    for date_str in ["20260317", "20260318"]:
        data = _fetch_espn(date_str)
        for event in data.get("events", []):
            comp = event["competitions"][0]
            notes = comp.get("notes", [])

            # Only First Four games
            is_first_four = any(
                "First Four" in n.get("headline", "") for n in notes
            )
            if not is_first_four:
                continue

            # Extract region from notes
            region = ""
            for n in notes:
                h = n.get("headline", "")
                for r in _REGION_ORDER:
                    if r in h:
                        region = r
                        break

            teams = comp["competitors"]
            if len(teams) != 2:
                continue

            t1 = teams[0]
            t2 = teams[1]
            seed = t1.get("curatedRank", {}).get("current", 0)

            status = comp.get("status", {}).get("type", {})
            completed = status.get("completed", False)

            if completed:
                winner_idx = 0 if t1.get("winner", False) else 1
                winner_team = teams[winner_idx]
                name = winner_team["team"].get(
                    "location",
                    winner_team["team"].get("displayName", ""),
                )
                results[(region, seed)] = {
                    "name": _espn_to_standard(name),
                    "seed": seed,
                }
                logger.info(
                    "First Four result: %s seed %d → %s",
                    region, seed, name,
                )
            else:
                # Game not completed — use the higher-ranked team as placeholder
                # (ESPN lists home team first which is often the favorite)
                name1 = t1["team"].get(
                    "location", t1["team"].get("displayName", ""),
                )
                name2 = t2["team"].get(
                    "location", t2["team"].get("displayName", ""),
                )
                logger.warning(
                    "First Four not completed: %s vs %s (%s seed %d). "
                    "Using %s as placeholder.",
                    name1, name2, region, seed, name1,
                )
                results[(region, seed)] = {
                    "name": _espn_to_standard(name1),
                    "seed": seed,
                }

        time.sleep(0.5)

    return results


def _fetch_r64_matchups() -> list[dict]:
    """Fetch all 32 R64 matchups with region info from ESPN.

    Returns list of dicts with: team1, team2, seed1, seed2, region.
    """
    first_four = _fetch_first_four_results()
    games: list[dict] = []

    for date_str in ["20260319", "20260320"]:
        data = _fetch_espn(date_str)
        for event in data.get("events", []):
            comp = event["competitions"][0]
            notes = comp.get("notes", [])

            # Only 1st Round games
            is_r64 = any(
                "1st Round" in n.get("headline", "") for n in notes
            )
            if not is_r64:
                continue

            # Extract region
            region = ""
            for n in notes:
                h = n.get("headline", "")
                for r in _REGION_ORDER:
                    if r in h:
                        region = r
                        break

            teams = comp["competitors"]
            if len(teams) != 2:
                continue

            t1 = teams[0]
            t2 = teams[1]
            name1 = t1["team"].get(
                "location", t1["team"].get("displayName", ""),
            )
            name2 = t2["team"].get(
                "location", t2["team"].get("displayName", ""),
            )
            seed1 = t1.get("curatedRank", {}).get("current", 0)
            seed2 = t2.get("curatedRank", {}).get("current", 0)

            # Resolve TBD entries using First Four results.
            # The known opponent's seed tells us the TBD seed type:
            # - If opponent is seed 6, TBD is an 11-seed (First Four play-in)
            # - If opponent is seed 1, TBD is a 16-seed (First Four play-in)
            if name1 == "TBD" or seed1 == 99:
                # Infer what seed the TBD should be from bracket structure
                tbd_seed = {1: 16, 6: 11}.get(seed2, 0)
                ff_key = (region, tbd_seed)
                if ff_key in first_four:
                    name1 = first_four[ff_key]["name"]
                    seed1 = first_four[ff_key]["seed"]
                    logger.info("Resolved TBD → %s (seed %d)", name1, seed1)

            if name2 == "TBD" or seed2 == 99:
                tbd_seed = {1: 16, 6: 11}.get(seed1, 0)
                ff_key = (region, tbd_seed)
                if ff_key in first_four:
                    name2 = first_four[ff_key]["name"]
                    seed2 = first_four[ff_key]["seed"]
                    logger.info("Resolved TBD → %s (seed %d)", name2, seed2)

            games.append({
                "team1": _espn_to_standard(name1),
                "team2": _espn_to_standard(name2),
                "seed1": seed1,
                "seed2": seed2,
                "region": region,
            })

        time.sleep(0.5)

    logger.info("Fetched %d R64 games", len(games))
    return games


def _order_r64_canonical(games: list[dict]) -> list[dict]:
    """Order R64 games in canonical bracket order.

    Groups by region (in _REGION_ORDER), then sorts within each region
    by the canonical seed-pair position (1v16, 8v9, 5v12, ...).
    """
    ordered: list[dict] = []

    for region in _REGION_ORDER:
        region_games = [g for g in games if g["region"] == region]

        if len(region_games) != 8:
            logger.warning(
                "Region %s has %d games (expected 8)",
                region, len(region_games),
            )

        # Sort by canonical seed-pair position
        def sort_key(g: dict) -> int:
            pair = (min(g["seed1"], g["seed2"]), max(g["seed1"], g["seed2"]))
            return _CANONICAL_SEED_PAIR_POS.get(pair, 99)

        region_games.sort(key=sort_key)
        ordered.extend(region_games)

    return ordered


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _simulate_forward(
    r64_ordered: list[dict],
    models: list,
    feature_lookup: FeatureLookup,
    feature_mode: str,
) -> list[BracketGame]:
    """Simulate a full bracket from R64 through NCG.

    Takes R64 matchups in canonical bracket order and predicts
    each game deterministically (higher probability wins).
    """
    # Build TeamStats for all R64 teams
    matchups: list[tuple[TeamStats, TeamStats]] = []
    for g in r64_ordered:
        t1 = feature_lookup.get_team(g["team1"], 2026, g["seed1"])
        t2 = feature_lookup.get_team(g["team2"], 2026, g["seed2"])
        matchups.append((t1, t2))

    games: list[BracketGame] = []
    current_matchups = list(matchups)

    for round_name in ROUND_ORDER:
        winners: list[TeamStats] = []

        for game_idx, (t1, t2) in enumerate(current_matchups):
            prob = _predict_game(
                t1, t2, models, feature_lookup, feature_mode,
            )

            team1_wins = prob > 0.5
            winner_stats = t1 if team1_wins else t2
            winners.append(winner_stats)

            slot1 = BracketSlot(team=t1.team, seed=t1.seed)
            slot2 = BracketSlot(team=t2.team, seed=t2.seed)

            game = BracketGame(
                round_name=round_name,
                region=_assign_region(round_name, game_idx),
                game_index=game_idx,
                team1=slot1,
                team2=slot2,
                winner=slot1 if team1_wins else slot2,
                win_probability=prob,
                is_upset=determine_upset(t1.seed, t2.seed, team1_wins),
                is_correct=None,  # No actuals to compare against
            )
            games.append(game)

        # Pair adjacent winners for next round
        if len(winners) >= 2:
            current_matchups = [
                (winners[i], winners[i + 1])
                for i in range(0, len(winners), 2)
            ]

    return games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    year = 2026
    version = "v6b"

    # 1. Fetch R64 matchups from ESPN
    logger.info("Fetching 2026 R64 matchups from ESPN...")
    raw_games = _fetch_r64_matchups()
    r64_ordered = _order_r64_canonical(raw_games)

    logger.info("Bracket order:")
    for i, g in enumerate(r64_ordered):
        region_idx = i // 8
        region = _REGION_ORDER[region_idx]
        logger.info(
            "  %s: (%d) %s vs (%d) %s",
            region, g["seed1"], g["team1"], g["seed2"], g["team2"],
        )

    # 2. Load models
    # Can override with MODELS_YEAR env var for comparison
    import os
    models_year = os.environ.get("MODELS_YEAR", str(year))
    models_dir = MM_BACKTEST_DIR / version / models_year / "models"
    models = load_models(models_dir)
    if not models:
        raise FileNotFoundError(f"No models found in {models_dir}")
    logger.info("Loaded %d models from %s", len(models), models_dir)

    # 3. Build feature lookup
    cfg = load_mm_config()
    feature_mode = cfg.get("feature_mode", "combined")

    kenpom_df = pd.read_csv(MM_KENPOM_DATA)
    barttorvik_df = None
    if MM_BARTTORVIK_DATA.exists() and feature_mode == "combined":
        barttorvik_df = pd.read_csv(MM_BARTTORVIK_DATA)
    feature_lookup = FeatureLookup(kenpom_df, barttorvik_df=barttorvik_df)

    # 4. Run deterministic simulation
    logger.info("Running deterministic simulation...")
    games = _simulate_forward(r64_ordered, models, feature_lookup, feature_mode)

    bracket = Bracket(year=year, source="simulation", games=tuple(games))

    # 5. Print results
    champion = [g for g in games if g.round_name == "NCG"][0]
    print(f"\n{'=' * 60}")
    print(f"  2026 NCAA Tournament Simulation (v6b)")
    print(f"  Champion: ({champion.winner.seed}) {champion.winner.team}")
    print(f"{'=' * 60}\n")

    for round_name in ROUND_ORDER:
        round_games = [g for g in games if g.round_name == round_name]
        print(f"{round_name}:")
        for g in round_games:
            upset = " *UPSET*" if g.is_upset else ""
            prob = g.win_probability
            winner_prob = prob if g.winner == g.team1 else 1 - prob
            print(
                f"  ({g.team1.seed}) {g.team1.team:25s} vs "
                f"({g.team2.seed}) {g.team2.team:25s} "
                f"→ {g.winner.team} ({winner_prob:.0%}){upset}",
            )
        print()

    # 6. Render SVG
    output_dir = MM_BACKTEST_DIR / version / str(year) / "plots" / "brackets"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_models{models_year}" if models_year != str(year) else ""
    svg_path = render_bracket(bracket, output_dir / f"{year}_simulation{suffix}.svg")
    print(f"SVG saved to: {svg_path}")


if __name__ == "__main__":
    main()
