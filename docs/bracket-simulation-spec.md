# Bracket Simulation & Survivor Pool — Specification

## Problem Statement

The current backtest system evaluates the model on **every actual game that occurred** in a tournament year. For later rounds (R32+), this means the model is handed the real matchup pairings — it never faces the consequences of its own earlier predictions.

**Example:** If the model predicts a #16 seed upsets a #1 seed in R64, the current backtest still evaluates the model on the actual R32 matchup where the #1 seed appears (because they actually won). The model's wrong R64 pick has zero cascading impact.

This spec covers **two evaluation modes** that use model predictions differently:

| Mode | Matchups | Prediction Style | Error Model |
|------|----------|-----------------|-------------|
| **Bracket Simulation** | Model-generated (forward sim) | Predict every game, advance winners | Cascading — wrong R64 pick creates wrong R32 matchups |
| **Survivor Pool** | Actual (real tournament matchups) | Pick 1 team per round to win, no reuse | Non-cascading — wrong pick = eliminated, but matchups stay real |

The current backtest's per-game predictions feed directly into survivor mode. The bracket simulation requires a new forward simulation engine.

## Architecture Overview

```
                    ┌─────────────────────────┐
                    │    KenPom Stats Lookup   │
                    │  (per-team-per-year)     │
                    └────────────┬────────────┘
                                 │
┌──────────────┐    ┌────────────▼────────────┐    ┌─────────────────┐
│ Restructured │───▶│   Bracket Simulator     │───▶│ SimulatedBracket│
│  Matchups    │    │                         │    │   (63 games)    │
│ (R64 seeds)  │    │  For each round:        │    └────────┬────────┘
└──────────────┘    │  1. Build feature vecs  │             │
                    │  2. Ensemble predict    │    ┌────────▼────────┐
┌──────────────┐    │  3. Debias             │    │   Evaluation    │
│ Trained      │───▶│  4. Advance winners    │    │  vs Actual      │
│ Models       │    │     OR sample from     │    │  (flat W/L per  │
│ (top K)      │    │     probabilities      │    │   round)        │
└──────────────┘    └─────────────────────────┘    └─────────────────┘
```

## Data Availability Confirmation

**KenPom data** (`kenpom_data.csv`) stores one row per team per year. We can look up any team's stats by `(Team, Year)` and construct feature vectors for arbitrary matchup pairings — not just the actual games that occurred.

**Restructured matchups** (`restructured_matchups.csv`) has `Seed1`, `Seed2`, `Team1`, `Team2`, `CURRENT ROUND`, and `YEAR` columns. We can extract R64 matchups (where `CURRENT ROUND == 64`) and know the canonical bracket structure from seeds.

**Bracket structure** is deterministic from seeding via `_CANONICAL_SEED_PAIR_POS`:
```
(1, 16): slot 0    (8, 9): slot 1    (5, 12): slot 2    (4, 13): slot 3
(6, 11): slot 4    (3, 14): slot 5    (7, 10): slot 6    (2, 15): slot 7
```

Adjacent pairs feed into the next round: slots 0+1 → R32 game 0, slots 2+3 → R32 game 1, etc.

## Implementation Plan

### New Files

| File | Purpose |
|------|---------|
| `_feature_builder.py` | Build feature vectors for arbitrary team pairings from KenPom lookup |
| `simulate.py` | Forward simulation engine (deterministic + Monte Carlo) |
| `survivor.py` | Survivor pool optimizer (greedy + optimal strategies) |

### Modified Files

| File | Change |
|------|--------|
| `_bracket.py` | Add `SimulationResult` dataclass |
| `_config.py` | Add `MM_SIMULATION_DIR` path |
| `model_config.yaml` | Add `simulation` and `survivor` config sections |
| `backtest.py` | Add simulation + survivor integration after training |
| `_bracket_cli.py` | Add simulation bracket rendering |

---

### Phase 1: Feature Builder (`_feature_builder.py`)

**Purpose:** Given two teams and a year, construct the feature vector the model expects.

**Why a separate module:** The current merge in `merge_matchups_stats.py` operates on DataFrames of actual matchups. We need to construct features for *hypothetical* matchups — teams that may never have actually played in a given round.

```python
@dataclass(frozen=True)
class TeamStats:
    """Pre-tournament KenPom statistics for a single team."""
    team: str
    year: int
    seed: int
    features: dict[str, float]  # column_name -> value


class FeatureLookup:
    """Immutable lookup for constructing matchup feature vectors."""

    def __init__(self, kenpom_df: pd.DataFrame) -> None:
        # Index: (standardized_team_name, year) -> row of KenPom stats
        ...

    def build_matchup_features(
        self, team1: TeamStats, team2: TeamStats,
    ) -> pd.DataFrame:
        """Build a single-row DataFrame with Team1 and Team2 features.

        Column layout matches what load_and_prepare() produces after
        DROP_COLUMNS are removed — i.e., the exact format the trained
        models expect.
        """
        ...
```

**Key details:**
- Must produce the exact same column set and ordering as `training_data.csv` after dropping non-feature columns
- Applies `standardize_team_name()` for lookup consistency
- Team1 features are base columns (`Rank`, `AdjEM`, ...); Team2 features are suffixed (`Rank_Team2`, `AdjEM_Team2`, ...)
- Returns a single-row `pd.DataFrame` (or a multi-row one for batch predictions)

---

### Phase 2: Forward Simulation Engine (`simulate.py`)

#### 2a. Extract R64 Matchups

```python
def _extract_r64_matchups(
    matchups_df: pd.DataFrame,
    year: int,
) -> list[tuple[TeamStats, TeamStats]]:
    """Extract the 32 Round-of-64 matchups from restructured_matchups.csv.

    Filters to CURRENT ROUND == 64 for the given year, then orders
    by canonical seed-pair position within each region (4 regions × 8 games).

    Returns:
        32 (team1, team2) tuples in canonical bracket order.
    """
```

We derive the 32 R64 matchups from actual data because they are **known before the tournament starts** (determined by Selection Sunday seeding). This is not data leakage — these matchups are public information.

Region assignment: the first 8 R64 games in canonical order belong to region 0, next 8 to region 1, etc. We get regions from the actual matchup data's grouping.

#### 2b. Deterministic Simulation

```python
@dataclass(frozen=True)
class SimulationResult:
    """Result of a single deterministic bracket simulation."""
    year: int
    bracket: Bracket           # The 63-game predicted bracket
    accuracy_by_round: dict[str, tuple[int, int]]  # round -> (correct, total)
    overall_accuracy: tuple[int, int]               # (correct, total)


def simulate_bracket_deterministic(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
) -> SimulationResult:
    """Simulate a full bracket by always picking the higher-probability team.

    Algorithm:
        1. Extract 32 R64 matchups (known from seeding)
        2. For each round (R64 → R32 → S16 → E8 → F4 → NCG):
           a. For each game in the round:
              - Build feature vector from FeatureLookup
              - Run ensemble prediction (avg of model probabilities)
              - Apply debiasing (swap columns, average)
              - Pick the team with probability > 0.5
           b. Advance winners to form next round's matchups
              - Winners from adjacent games (0+1, 2+3, ...) become
                the next round's matchup pairs
        3. Build a Bracket object with all 63 BracketGames
        4. Compare against actual_bracket for accuracy

    Returns:
        SimulationResult with the full predicted bracket and per-round accuracy.
    """
```

**Winner advancement logic:**
```
R64 games:  [g0, g1, g2, g3, g4, g5, g6, g7, ...]  (32 games)
                ↓       ↓       ↓       ↓
R32 games:  [winner(g0) vs winner(g1), winner(g2) vs winner(g3), ...]  (16 games)
                    ↓                       ↓
S16 games:  [winner(r32_0) vs winner(r32_1), ...]  (8 games)
... and so on through NCG
```

**Evaluation (flat per-round W/L):**
For each of the 63 predicted games, check if the predicted winner matches the actual winner of that same game slot. But note: if a team was wrongly advanced, the "actual" comparator is the team that *should* be there — so we compare by bracket position, not by team name.

Actually, the simpler and more standard evaluation: for each round, count how many of the model's predicted winners **actually won that round in reality**. A team counts as a correct pick if:
1. The model predicted them to reach that round, AND
2. They actually reached that round and won

This is the standard bracket scoring approach — "did you pick the right team to win in each slot?"

```python
def _evaluate_simulation(
    simulated: Bracket,
    actual: Bracket,
) -> dict[str, tuple[int, int]]:
    """Compare simulation against actual results.

    For each round, count how many teams the model correctly
    predicted to win. A pick is correct if:
    - The model's predicted winner for a bracket slot matches
      the actual winner of that same bracket slot.

    Returns:
        dict of round_name -> (num_correct, num_games)
    """
```

#### 2c. Monte Carlo Simulation

```python
@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated results from N Monte Carlo bracket simulations."""
    year: int
    n_simulations: int
    team_round_rates: dict[str, dict[str, float]]
    # team_name -> {round_name -> fraction of sims where team won in that round}
    accuracy_distribution: dict[str, list[int]]
    # round_name -> list of N correct counts (one per sim)
    mean_accuracy_by_round: dict[str, float]
    # round_name -> mean(correct / total) across sims
    champion_distribution: dict[str, float]
    # team_name -> fraction of sims where team won NCG


def simulate_bracket_monte_carlo(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> MonteCarloResult:
    """Run N bracket simulations, sampling outcomes from model probabilities.

    Same as deterministic, except instead of always picking prob > 0.5,
    each game outcome is sampled: if the model gives Team A a 72% chance,
    Team A advances in ~72% of simulations.

    Uses numpy random generator seeded with rng_seed for reproducibility.

    Key outputs:
    - How often each team makes each round across N sims
    - Distribution of bracket accuracy scores
    - Championship probability for each team
    """
```

**Shared prediction logic:** Both deterministic and Monte Carlo call the same `_predict_game()` function that returns a probability. The only difference is whether the winner is determined by `prob > 0.5` (deterministic) or `rng.random() < prob` (Monte Carlo).

```python
def _predict_game(
    team1: TeamStats,
    team2: TeamStats,
    models: list,
    feature_lookup: FeatureLookup,
) -> float:
    """Predict Team1 win probability with debiasing.

    1. Build feature vector for (team1, team2)
    2. Get probability from each model
    3. Swap team1/team2 features
    4. Get swapped probability from each model
    5. Average original and (1 - swapped) across all models

    Returns:
        Debiased probability that team1 wins.
    """
```

---

### Phase 3: Evaluation & Output

**Per-round flat W/L** — for each round, how many picks were correct.

Output structure:
```
data/march-madness/backtest/v1/{year}/simulation/
├── deterministic_bracket.csv      # 63 rows, one per game
├── deterministic_accuracy.csv     # per-round correct/total
├── monte_carlo_team_rates.csv     # team × round advance rates
├── monte_carlo_accuracy_dist.csv  # per-sim accuracy scores
├── monte_carlo_champions.csv      # team → championship probability
└── simulation_summary.txt         # human-readable summary
```

**CSV schemas:**

`deterministic_bracket.csv`:
```
Round, GameIndex, Region, Team1, Seed1, Team2, Seed2, PredictedWinner, PredictedWinnerSeed, WinProbability, ActualWinner, IsCorrect
```

`monte_carlo_team_rates.csv`:
```
Team, Seed, R64, R32, S16, E8, F4, NCG, Champion
```
(Values are fractions 0.0-1.0 representing how often that team reached/won each round)

`monte_carlo_champions.csv`:
```
Team, Seed, ChampionshipProbability
```
(Sorted descending by probability)

---

### Phase 4: Integration

#### Config additions (`model_config.yaml`):
```yaml
march_madness:
  simulation:
    n_monte_carlo: 1000
    rng_seed: 42
  survivor:
    n_monte_carlo: 1000
    rng_seed: 42
```

#### Backtest integration (`backtest.py`):
After training models and running ensemble/debiased predictions for a year, optionally run the forward simulation. Add a `run_simulation` flag or just always run it.

```python
# After ensemble + debiased predictions for this year...
from sports_quant.march_madness.simulate import (
    simulate_bracket_deterministic,
    simulate_bracket_monte_carlo,
)

sim_result = simulate_bracket_deterministic(
    year=backtest_year,
    models=[md["model"] for md in top_models],
    feature_lookup=feature_lookup,
    matchups_df=matchups_raw,
    actual_bracket=actual_bracket,
)

mc_result = simulate_bracket_monte_carlo(
    year=backtest_year,
    models=[md["model"] for md in top_models],
    feature_lookup=feature_lookup,
    matchups_df=matchups_raw,
    actual_bracket=actual_bracket,
    n_simulations=cfg["simulation"]["n_monte_carlo"],
    rng_seed=cfg["simulation"]["rng_seed"],
)
```

#### CLI integration (`_bracket_cli.py`):
Add simulation bracket rendering alongside the existing actual/ensemble/debiased brackets. The deterministic simulation bracket can reuse the existing SVG rendering by producing a standard `Bracket` object.

---

---

## Survivor Pool (`survivor.py`)

### Overview

In a survivor pool you make **6 picks** — one team per round to win their game. Once you've used a team, it's **burned for the rest of the tournament**. If your pick loses, you're eliminated.

Unlike bracket simulation, survivor uses **actual matchups** (you see who's really playing each round). This means the existing backtest output (debiased probabilities for all 63 actual games) is the exact input the survivor optimizer needs — no new model or forward simulation required.

### Data Flow

```
Existing backtest output                Survivor optimizer
(debiased_results.csv)
  |-- Team1, Team2, Seed1, Seed2   --> run_survivor_greedy()
  |-- CURRENT ROUND                    run_survivor_optimal()
  |-- Debiased_Prob                         |
  +-- Team1_Win (actual)                    v
                                       SurvivorResult
                                       (6 picks + outcomes)
```

### Why It's an Optimization Problem

**Naive approach:** Each round, pick the team with the highest win probability that hasn't been used. This is suboptimal:

- **R64 has 32 games** — many near-100% favorites. You don't need to burn a #1 seed here.
- **E8/F4/NCG have 4/2/1 games** — fewer options, tougher matchups. You might wish you'd saved that dominant team.

**Example:** Kansas has a 99% win prob in R64 but faces a tough E8 matchup where they're the best available option at 68%. You should pick a different safe team in R64 (say, a #2 seed at 95%) and save Kansas for E8 where it matters more.

### Data Structures

```python
@dataclass(frozen=True)
class SurvivorPick:
    """A single survivor pool pick."""
    round_name: str
    team: str
    seed: int
    opponent: str
    opponent_seed: int
    win_probability: float
    actual_winner: str
    survived: bool  # did the pick actually win?


@dataclass(frozen=True)
class SurvivorResult:
    """Complete survivor pool result for one tournament year."""
    year: int
    strategy: str  # "greedy", "optimal"
    picks: tuple[SurvivorPick, ...]  # 6 picks, one per round
    survived_all: bool
    rounds_survived: int  # 0-6
    survival_probability: float  # product of all pick win probs
```

### Strategies

#### Greedy Strategy

```python
def run_survivor_greedy(
    year: int,
    game_probabilities: pd.DataFrame,
) -> SurvivorResult:
    """Pick highest win-prob unused team each round.

    Args:
        year: Tournament year.
        game_probabilities: DataFrame with columns:
            YEAR, Team1, Seed1, Team2, Seed2, CURRENT ROUND,
            Debiased_Prob (Team1 win probability), Team1_Win (actual)

    Algorithm:
        For each round (R64 -> R32 -> S16 -> E8 -> F4 -> NCG):
            1. Get all games in this round
            2. For each game, compute win probability for both teams:
               - Team1 win prob: Debiased_Prob
               - Team2 win prob: 1 - Debiased_Prob
            3. Among all teams not yet picked, select the one
               with the highest win probability
            4. Record the pick and whether it survived
    """
```

#### Optimal Strategy (backtest only)

```python
def run_survivor_optimal(
    year: int,
    game_probabilities: pd.DataFrame,
) -> SurvivorResult:
    """Find the pick sequence that maximizes P(survive all 6 rounds).

    Since all matchups are known in backtesting, we can find the
    pick sequence with the highest product of win probabilities.

    Uses branch-and-bound pruning: abandon any partial sequence
    whose probability product is already below the current best
    complete sequence.

    Search space: at most ~64 x 32 x 16 x 8 x 4 x 2 candidates
    per level, heavily prunable. Runs in milliseconds.
    """
```

The optimal strategy answers: "given perfect knowledge of all matchups, what's the best possible survivor run?" This is the upper bound for any real-time strategy.

#### Monte Carlo Survivor

```python
@dataclass(frozen=True)
class SurvivorMonteCarloResult:
    """Aggregated survivor results from N Monte Carlo simulations."""
    year: int
    strategy: str  # "greedy_mc", "optimal_mc"
    n_simulations: int
    survival_rate: float  # fraction of sims that survived all 6 rounds
    mean_rounds_survived: float
    rounds_survived_distribution: dict[int, float]
    # rounds_survived (0-6) -> fraction of sims
    team_pick_rates: dict[str, dict[str, float]]
    # round_name -> {team_name -> fraction of sims where that team was picked}
    elimination_round_distribution: dict[str, float]
    # round_name -> fraction of sims eliminated in that round


def run_survivor_monte_carlo(
    year: int,
    game_probabilities: pd.DataFrame,
    strategy: str = "greedy",
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> SurvivorMonteCarloResult:
    """Run N survivor simulations with stochastic game outcomes.

    For each simulation:
        1. For each round, resolve all game outcomes by sampling
           from model probabilities (72% favorite wins ~72% of the time)
        2. Run the specified strategy (greedy or optimal) on the
           simulated outcomes to select a pick
        3. Check if the pick survived (based on sampled outcome)
        4. Record how many rounds survived

    This answers: "Given the model's uncertainty in game outcomes,
    how often does each strategy survive the full tournament?"

    The greedy strategy is fast per sim (just pick max prob each round).
    The optimal strategy is also fast since the search space is small.
    1000 sims should complete in under a second.
    """
```

Monte Carlo survivor is different from backtest survivor in a key way: **backtest uses actual outcomes** to check survival, while **MC samples outcomes from model probabilities**. This tells you how robust each strategy is to the model's uncertainty — a greedy pick with 95% probability still fails 5% of the time, and over 6 rounds those failures compound.

#### Live Prediction Mode

```python
@dataclass(frozen=True)
class LiveSurvivorState:
    """State of a survivor pool in progress."""
    year: int
    completed_rounds: tuple[str, ...]  # rounds already played
    picks_made: tuple[SurvivorPick, ...]  # picks from completed rounds
    teams_used: frozenset[str]  # teams already burned
    still_alive: bool  # have all picks survived so far?


def run_survivor_live(
    state: LiveSurvivorState,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    known_results: pd.DataFrame,
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Recommend a survivor pick for the next round of a live tournament.

    Combines known results (completed rounds) with Monte Carlo forward
    simulation (future rounds) to evaluate pick candidates.

    Algorithm:
        1. Identify the next round to pick for
        2. Get actual matchups for that round (known once bracket updates)
        3. For each candidate team (not yet used):
           a. Assume we pick this team for the current round
           b. Run N Monte Carlo forward sims of ALL remaining rounds:
              - Current round: does our pick survive? (sample from prob)
              - Future rounds: simulate game outcomes, then run greedy
                strategy on simulated matchups with remaining unused teams
           c. Record: fraction of sims where we survive ALL remaining rounds
        4. Return: team_name -> P(survive rest of tournament | pick this team)

    Args:
        state: Current survivor pool state (picks made, teams used).
        models: Trained models for predicting future matchups.
        feature_lookup: For building feature vectors for future matchups.
        matchups_df: Known matchups (at least current round must be known).
        known_results: Actual results for completed rounds.
        n_simulations: Number of Monte Carlo forward simulations.
        rng_seed: Random seed for reproducibility.

    Returns:
        Dict of team_name -> probability of surviving all remaining rounds.
        The recommended pick is the team with the highest value.
    """
```

**How live mode bridges the two systems:**

```
Known results (completed rounds)    Future rounds (unknown)
         |                                  |
    Actual matchups                 Forward simulation
    Actual outcomes                 (from simulate.py)
         |                                  |
         +--- LiveSurvivorState ---+        |
                    |                       |
              Candidate picks          MC forward sim
              for next round       of remaining tournament
                    |                       |
                    +-------+-------+-------+
                            |
                   P(survive all remaining)
                   per candidate team
```

This is where the bracket simulation engine and survivor pool converge: live survivor needs `_predict_game()` and the forward simulation loop from `simulate.py` to estimate what matchups will look like in future rounds, then runs the survivor optimizer on those simulated futures.

### Survivor Output

```
data/march-madness/backtest/v1/{year}/survivor/
|-- greedy_picks.csv             # 6 rows (one per round)
|-- optimal_picks.csv            # 6 rows (one per round)
|-- greedy_mc_summary.csv        # MC survival stats for greedy
|-- optimal_mc_summary.csv       # MC survival stats for optimal
|-- mc_pick_rates.csv            # how often each team is picked per round across sims
|-- mc_elimination_dist.csv      # which round eliminations happen in
+-- survivor_summary.txt         # strategy comparison (all modes)
```

**CSV schema (`greedy_picks.csv` / `optimal_picks.csv`):**
```
Round, Team, Seed, Opponent, OpponentSeed, WinProbability, ActualWinner, Survived
```

**CSV schema (`greedy_mc_summary.csv` / `optimal_mc_summary.csv`):**
```
Metric, Value
survival_rate, 0.312
mean_rounds_survived, 4.7
rounds_0, 0.001
rounds_1, 0.012
rounds_2, 0.045
rounds_3, 0.123
rounds_4, 0.234
rounds_5, 0.273
rounds_6, 0.312
```

**`survivor_summary.txt` example:**
```
Survivor Pool Results — 2024

Greedy Strategy:
  R64: (1) UConn vs (16) Stetson — Win Prob: 0.98 — SURVIVED
  R32: (1) Houston vs (9) Texas A&M — Win Prob: 0.91 — SURVIVED
  S16: (1) Purdue vs (5) Gonzaga — Win Prob: 0.74 — SURVIVED
  E8:  (2) Tennessee vs (3) Creighton — Win Prob: 0.63 — ELIMINATED
  Rounds survived: 3/6
  Survival probability: 0.98 × 0.91 × 0.74 × 0.63 = 0.416

Optimal Strategy:
  R64: (2) Marquette vs (15) Western Kentucky — Win Prob: 0.96 — SURVIVED
  R32: (1) Houston vs (9) Texas A&M — Win Prob: 0.91 — SURVIVED
  S16: (3) Illinois vs (11) Duquesne — Win Prob: 0.81 — SURVIVED
  E8:  (1) UConn vs (5) San Diego St. — Win Prob: 0.78 — SURVIVED
  F4:  (1) Purdue vs (1) Alabama — Win Prob: 0.55 — ELIMINATED
  Rounds survived: 4/6
  Survival probability: 0.96 × 0.91 × 0.81 × 0.78 × 0.55 = 0.301
```

### Integration with Backtest

Survivor runs after debiasing, consuming the same results:

```python
# In backtest.py, after process_backtest_results()...
from sports_quant.march_madness.survivor import (
    run_survivor_greedy,
    run_survivor_optimal,
    run_survivor_monte_carlo,
)

# Build game_probabilities from debiased_results + matchup round info
greedy = run_survivor_greedy(backtest_year, game_probs)
optimal = run_survivor_optimal(backtest_year, game_probs)

# Monte Carlo: how robust is each strategy to outcome uncertainty?
greedy_mc = run_survivor_monte_carlo(
    backtest_year, game_probs, strategy="greedy",
    n_simulations=cfg["survivor"]["n_monte_carlo"],
)
optimal_mc = run_survivor_monte_carlo(
    backtest_year, game_probs, strategy="optimal",
    n_simulations=cfg["survivor"]["n_monte_carlo"],
)
```

### Integration with Live Prediction

Live mode is invoked separately (not as part of backtest) when a tournament is in progress:

```python
from sports_quant.march_madness.survivor import run_survivor_live

# After R64 and R32 have been played, picking for S16:
state = LiveSurvivorState(
    year=2025,
    completed_rounds=("R64", "R32"),
    picks_made=(r64_pick, r32_pick),
    teams_used=frozenset({"Duke", "Houston"}),
    still_alive=True,
)

recommendations = run_survivor_live(
    state=state,
    models=loaded_models,
    feature_lookup=feature_lookup,
    matchups_df=matchups,
    known_results=results_so_far,
)
# recommendations = {"UConn": 0.47, "Purdue": 0.41, "Tennessee": 0.38, ...}
# Pick UConn for S16 — highest probability of surviving S16 through NCG
```

---

## Edge Cases & Considerations

1. **Play-in games:** Some seeds (especially 11 and 16) have play-in games. By the time R64 starts, play-in winners are known. Our matchups data should already reflect post-play-in matchups since R64 is `CURRENT ROUND == 64`.

2. **Team name consistency:** `FeatureLookup` must apply `standardize_team_name()` identically to how `merge_matchups_stats.py` does, or feature lookups will fail silently (NaN features).

3. **Feature column ordering:** The model expects features in the exact column order they were trained on. `FeatureLookup.build_matchup_features()` must match the column order from `training_data.csv` after dropping non-feature columns.

4. **F4/NCG region assignment:** When building `BracketGame` objects for the simulation bracket, F4 and NCG games should use `region = -1` to match existing convention.

5. **Missing KenPom data:** If a team appears in matchups but has no KenPom entry (unlikely but possible for play-in teams from very small conferences), we need to handle this gracefully — log a warning and skip that simulation year, or fall back to seed-based prediction.

6. **Year 2020:** No tournament was played. Ensure it's excluded from backtest years (it already is — config has `[2022, 2023, 2024, 2025]`).

7. **Survivor team availability:** A team can only be picked in a round if they actually appear in that round's matchups. The optimizer must filter candidates to teams present in each round — not all 64 teams are available in R32+.

8. **Survivor with equal seeds:** In F4/NCG, both teams may have the same seed (e.g., two #1 seeds). The optimizer should handle this without special-casing — it just picks based on win probability regardless of seed.

## Comparison: Current vs Simulation Accuracy

The current "oracle matchup" backtest will **always show higher accuracy** than the forward simulation because:
- It gets perfect matchup information for every round
- It never faces cascading errors
- Later-round games are between stronger teams (easier to predict favorites)

The gap between current accuracy and simulation accuracy tells us how much the model benefits from being handed correct matchups vs having to earn them. This gap is the whole point of building the simulation.

## Implementation Order

Two independent tracks after the shared foundation:

```
_feature_builder.py  (shared — Phase 1)
    |-- simulate.py          (bracket track — Phase 2)
    +-- survivor.py          (survivor track — Phase 3)
            |
            +-- live mode    (convergence — Phase 3c, depends on simulate.py)
```

**Phase 1:** `_feature_builder.py` — shared foundation, testable in isolation with known team pairs

**Phase 2 (bracket track):**
1. `simulate.py` — `_predict_game()` first, then deterministic, then Monte Carlo
2. `_bracket.py` updates — `SimulationResult` and `MonteCarloResult` dataclasses

**Phase 3 (survivor track):**
1. `survivor.py` — greedy strategy (backtest, actual outcomes)
2. `survivor.py` — optimal strategy with branch-and-bound (backtest, actual outcomes)
3. `survivor.py` — Monte Carlo survivor (sample outcomes from probabilities)
4. `survivor.py` — live prediction mode (depends on `simulate.py` for forward sim of future rounds)

**Phase 4 (integration — both tracks):**
1. `_config.py` + `model_config.yaml` — path and config additions
2. `backtest.py` — wire simulation + survivor into the backtest loop
3. `_bracket_cli.py` — render simulation brackets

**Dependency note:** Phase 3 steps 1-3 are independent of Phase 2. Phase 3 step 4 (live mode) depends on Phase 2 because it uses `_predict_game()` and the forward simulation loop to estimate future matchups. Build live mode last.

Phases 2 and 3 (steps 1-3) are **independent** and can be built in either order or in parallel. Each phase is independently testable before moving to the next.
