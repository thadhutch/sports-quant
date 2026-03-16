# Injury Adjustment System — Specification

## Problem Statement

The current March Madness prediction pipeline uses **team-level KenPom efficiency stats** (AdjEM, AdjO, AdjD, etc.) that reflect full-season performance. These stats cannot account for **late-season player absences** — a star player's torn ACL in the conference tournament, a key contributor suspended before Selection Sunday, or a starting point guard ruled out for the first weekend.

KenPom ratings are slow-moving aggregates. When a team loses a player who accounts for 30% of their offensive production, KenPom AdjO does not meaningfully adjust before the tournament. The model sees the same team-level stats regardless of roster availability.

**Example:** If a #3 seed loses its leading scorer and best defender to injury the week before the tournament, the model still predicts based on full-strength KenPom ratings. This overvalues the injured team and undervalues their opponent — exactly the kind of information edge that a bracket predictor should exploit.

### Why This Matters for March Madness Specifically

- Conference tournament injuries often occur 1-2 weeks before the NCAA tournament — KenPom barely adjusts
- College basketball rosters are shallow (8-9 rotation players); losing one starter has outsized impact compared to pro sports
- Tournament upsets frequently correlate with key player absences (these are the "why did a 12 beat a 5?" explanations that box-score features miss)
- Injury information is publicly available but not captured in any feature the model currently uses

### Design Principle

Use the LLM as a **data preprocessing tool**, not as a predictor. The injury system adjusts KenPom stats before they enter the existing model pipeline. The model itself is unchanged — it still sees 36-column feature vectors of team efficiency stats. The stats are just more accurate because they reflect actual roster availability.

## Architecture Overview

```
                                    ┌──────────────────┐
                                    │  ESPN Injury Page │
                                    │  + Manual Input   │
                                    │  + LLM Parser     │
                                    └────────┬─────────┘
                                             │
                                             ▼
┌──────────────────┐    ┌──────────────────────────────┐
│ Sports-Reference │    │     Injury Reports           │
│ Player Stats     │    │  (player, team, status)      │
│ (min, usage, BPM)│    └────────────┬─────────────────┘
└────────┬─────────┘                 │
         │                           │
         ▼                           ▼
┌──────────────────┐    ┌──────────────────────────────┐
│ Player Importance│    │   Adjustment Calculator      │
│ Scoring          │───▶│                              │
│ (per-team, sums  │    │  adjusted_AdjO = AdjO *      │
│  to 1.0)         │    │    (1 - lost_frac * degrade) │
└──────────────────┘    └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │   Adjusted KenPom Stats      │
                        │  (drop-in replacement)       │
                        └────────────┬─────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             FeatureLookup    simulate.py      predict.py
             (existing)       (existing)       (existing)
```

The injury system produces **adjusted stat dictionaries** that replace raw KenPom values inside `FeatureLookup`. Every downstream consumer — simulation, prediction, survivor pool — gets injury-aware features without any code changes.

## Data Sources

### Player Stats: sports-reference.com/cbb

**URL pattern:** `https://www.sports-reference.com/cbb/schools/{slug}/{year}.html`

Each team page contains two tables we need:

| Table | Key Columns |
|-------|-------------|
| Per Game | Player, G, MP (minutes/game), PTS, TRB, AST, STL, BLK |
| Advanced | Player, MP, USG% (usage rate), ORtg, DRtg, BPM (Box Plus/Minus), OBPM, DBPM |

**Why this source:**
- Free, stable, scrapable (the project already uses BeautifulSoup + requests for KenPom)
- Richest advanced stats available for college basketball without a paid API
- Predictable URL structure makes scraping straightforward
- Rate limit: ~20 requests/minute (use 3-second delays between requests)

**Scope:** Scrape only the 68 tournament teams for the target year. That is 68 HTTP requests — manageable with rate limiting.

**Computed field:** `minutes_pct = player_minutes_per_game / sum(all_players_minutes_per_game)` — the fraction of total team minutes this player plays.

### Injury Reports: ESPN + Manual Input

**Primary source:** ESPN college basketball injuries page
- URL: `https://www.espn.com/mens-college-basketball/injuries`
- Contains: player name, team, status (Out / Day-To-Day / etc.), injury description, date

**Secondary source:** Manual CSV input for injuries ESPN misses
- College basketball injury reporting is less standardized than NBA/NFL
- Coach press conferences may reveal an absence before ESPN updates
- Format: simple CSV with columns `player_name, team, status, injury_description, report_date`

**Tertiary source (Phase 5):** Claude API for parsing unstructured text
- Paste in a tweet, news blurb, or coach quote
- Claude extracts structured `InjuryReport` data
- Fuzzy-match extracted names against scraped roster to validate

### Team Name Matching

This is the highest-risk integration point. Three data sources use different team name conventions:

| Source | Example |
|--------|---------|
| KenPom | `"North Carolina St."` |
| Sports-Reference slug | `"north-carolina-state"` |
| ESPN | `"NC State Wolfpack"` |

A dedicated `_name_matching.py` module provides bidirectional mappings. Start with the 68 tournament teams — extend to full D1 (~360 teams) only if needed.

## Data Models

All models are frozen dataclasses following the existing pattern (`TeamStats`, `BracketGame`, `SimulationResult`).

```python
class InjuryStatus(Enum):
    """Player availability status from injury reports."""
    OUT = "out"                 # Will not play
    DOUBTFUL = "doubtful"       # Unlikely to play
    QUESTIONABLE = "questionable"  # May or may not play
    PROBABLE = "probable"       # Expected to play
    HEALTHY = "healthy"         # No injury designation


@dataclass(frozen=True)
class PlayerStats:
    """Season-level statistics for a single player."""
    player_name: str
    team: str                   # Standardized team name (KenPom convention)
    year: int
    games: int
    minutes_per_game: float
    minutes_pct: float          # Fraction of team's total minutes
    usage_rate: float           # USG% — fraction of team possessions used
    offensive_rating: float     # ORtg — points per 100 possessions
    defensive_rating: float     # DRtg — points allowed per 100 possessions
    box_plus_minus: float       # BPM — overall impact estimate
    offensive_bpm: float        # OBPM
    defensive_bpm: float        # DBPM
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float
    steals_per_game: float
    blocks_per_game: float


@dataclass(frozen=True)
class InjuryReport:
    """Injury status for a single player."""
    player_name: str
    team: str                   # Standardized team name
    status: InjuryStatus
    injury_description: str
    report_date: str            # ISO format: "2025-03-15"
    source: str                 # "espn", "manual", "llm"


@dataclass(frozen=True)
class PlayerImpact:
    """Quantified contribution of a player to their team's efficiency."""
    player_name: str
    team: str
    year: int
    importance_score: float     # 0.0 to 1.0, sums to 1.0 across roster
    adj_o_contribution: float   # Offensive importance (0.0 to 1.0)
    adj_d_contribution: float   # Defensive importance (0.0 to 1.0)


@dataclass(frozen=True)
class AdjustedTeamStats:
    """Team stats after applying injury adjustments."""
    team: str
    year: int
    original_stats: dict[str, float]    # Original KenPom values
    adjusted_stats: dict[str, float]    # Adjusted KenPom values
    adjustments_applied: tuple[PlayerImpact, ...]  # Which players caused adjustments
    adjustment_source: str              # Description of what triggered the adjustment
```

## Player Importance Scoring

### Algorithm

Each player receives a composite importance score from three signals:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| **Minutes share** | 0.35 | Players who are on the court more have more impact. Simple, reliable. |
| **Usage-weighted minutes** | 0.30 | A player with 30% usage who plays 80% of minutes drives the offense far more than a player with 15% usage playing the same minutes. |
| **BPM contribution** | 0.35 | BPM is the best single advanced stat for overall player impact. Weighted by minutes to account for sample size — a bench player with a high BPM on small minutes is not as impactful. |

```python
def score_player_importance(
    players: list[PlayerStats],
) -> list[PlayerImpact]:
    """Score each player's contribution to team efficiency.

    Algorithm:
        For each player:
            minutes_component = player.minutes_pct
            usage_component = (player.usage_rate / 100) * player.minutes_pct
            bpm_component = player.box_plus_minus * player.minutes_pct

        Normalize each component across the roster so they sum to 1.0.
        Final score = 0.35 * minutes_norm + 0.30 * usage_norm + 0.35 * bpm_norm

        For offensive/defensive split:
            adj_o_contribution uses OBPM instead of BPM
            adj_d_contribution uses DBPM instead of BPM

    Returns:
        List of PlayerImpact, one per player, scores summing to 1.0.
    """
```

### Edge Cases

- **Negative BPM:** Some players have negative BPM (below replacement level). Shift all BPM values by `abs(min_BPM) + 0.1` before computing contributions so all values are positive. This preserves relative ordering while avoiding negative importance scores.
- **Single dominant player:** If one player has >50% importance, cap at 0.60 and redistribute the excess proportionally. This prevents a single-player adjustment from moving AdjEM by an unrealistic amount.
- **Players with very few games:** Players with fewer than 5 games played are excluded from importance scoring (too small a sample to be meaningful).

## Injury Adjustment Calculation

### Core Formula

When a key player is absent, the team's efficiency degrades. The adjustment is:

```python
# Status multipliers — how much of the player's absence to count
STATUS_MULTIPLIER = {
    InjuryStatus.OUT: 1.0,          # Definitely not playing
    InjuryStatus.DOUBTFUL: 0.75,    # Very unlikely to play
    InjuryStatus.QUESTIONABLE: 0.25,  # Coin flip — discount heavily
    InjuryStatus.PROBABLE: 0.05,    # Almost certainly playing
    InjuryStatus.HEALTHY: 0.0,      # No adjustment
}

# Degradation factor — accounts for replacement player absorption
# 0.5 means losing a player who accounts for 30% of production
# degrades the stat by 15% (30% * 0.5)
DEFAULT_DEGRADATION_FACTOR = 0.5
```

```python
def compute_adjustments(
    team: str,
    year: int,
    original_stats: dict[str, float],
    player_impacts: list[PlayerImpact],
    injuries: list[InjuryReport],
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> AdjustedTeamStats:
    """Compute adjusted KenPom stats based on player absences.

    Algorithm:
        1. For each injured player with status != HEALTHY:
           - Look up their PlayerImpact
           - lost_o = adj_o_contribution * status_multiplier * degradation_factor
           - lost_d = adj_d_contribution * status_multiplier * degradation_factor

        2. Sum lost_o and lost_d across all injured players

        3. Apply to stats:
           - adjusted_AdjO = original_AdjO * (1 - total_lost_o)
           - adjusted_AdjD = original_AdjD * (1 + total_lost_d)
             (defense gets WORSE, i.e., allows more points, so value increases)
           - adjusted_AdjEM = adjusted_AdjO - adjusted_AdjD

    Stats NOT adjusted (with rationale):
        - AdjustT: Tempo is a coaching scheme choice, not a single-player effect
        - Luck: Random variance metric, not player-dependent
        - SOS / NCSOS: Strength of schedule is fixed by the games already played
        - All Rank columns: Ordinal positions cannot be recomputed without
          adjusting all ~360 teams simultaneously. Left at original values.

    Returns:
        AdjustedTeamStats with both original and adjusted stat dicts.
        If no injuries affect the team, adjusted_stats == original_stats.
    """
```

### Why This Formula

The degradation factor is the key design choice. Alternatives considered:

| Approach | Pros | Cons |
|----------|------|------|
| **Proportional (factor=1.0)** | Simple | Unrealistic — losing 30% of production does not make a team 30% worse. Bench players absorb minutes. |
| **Historical with/without** | Most accurate | Requires play-by-play data with on/off splits. Not freely available at scale for college basketball. |
| **Regression-based** | Data-driven | Would need a large labeled dataset of (player-absent, team-performance-change) pairs. Not available. |
| **Fixed factor (ours, factor=0.5)** | Simple, tunable, defensible | The 0.5 is a guess. Calibrated in Phase 6. |

A degradation factor of 0.5 is the conservative middle ground. Empirically, NBA research suggests team performance degrades at roughly 40-60% of the star player's marginal contribution when they're absent (replacement players absorb 40-60% of the production). College basketball may be closer to the higher end (less roster depth), but 0.5 is a reasonable starting point.

### Adjustment Bounds

To prevent unrealistic adjustments:
- Maximum total offensive degradation: 25% (`total_lost_o` capped at 0.25)
- Maximum total defensive degradation: 25% (`total_lost_d` capped at 0.25)
- These caps trigger when multiple starters are simultaneously absent — at that point the team has bigger problems than our model can capture

## Pipeline Integration

### Integration Point: FeatureLookup

The `FeatureLookup` class is the single point where KenPom stats become feature vectors. This is the natural integration point.

```python
class FeatureLookup:
    """Immutable lookup for constructing matchup feature vectors."""

    def __init__(
        self,
        kenpom_df: pd.DataFrame,
        adjustments: dict[tuple[str, int], dict[str, float]] | None = None,
    ) -> None:
        """Initialize lookup index.

        Args:
            kenpom_df: Raw KenPom stats DataFrame.
            adjustments: Optional dict of (team_name, year) -> adjusted stat values.
                If provided, these values override the corresponding KenPom stats
                for the specified teams. Other teams are unaffected.
        """
        # Build index as before, but overlay adjustments where provided
        ...
```

This is a **backwards-compatible** change. Existing callers that do not pass `adjustments` get identical behavior.

### Orchestrator

```python
# In src/sports_quant/march_madness/injuries/__init__.py

def build_injury_adjustments(
    year: int,
    tournament_teams: list[str] | None = None,
    injury_overrides: list[InjuryReport] | None = None,
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> dict[tuple[str, int], dict[str, float]]:
    """Build injury adjustments for all tournament teams.

    Steps:
        1. Load player stats for the year (from CSV or scrape if missing)
        2. Load injury reports (from CSV or scrape if missing)
        3. Merge any manual injury_overrides
        4. Compute importance scores for all teams
        5. Compute adjustments for teams with injured players
        6. Return adjustments dict in the format FeatureLookup expects

    Args:
        year: Tournament year.
        tournament_teams: List of team names to process. If None, processes
            all teams with available player stats.
        injury_overrides: Manual injury reports to merge with scraped data.
            These take precedence over scraped reports for the same player.
        degradation_factor: How much of the lost production translates to
            team performance degradation. Default 0.5.

    Returns:
        Dict of (team_name, year) -> adjusted_stats_dict.
        Only includes teams that have injured players.
        Teams with no injuries are omitted (FeatureLookup uses raw stats).
    """
```

### Usage in Simulation

```python
# In simulate.py — no structural changes, just an optional parameter

def simulate_bracket_deterministic(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,  # May contain adjustments
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
) -> SimulationResult:
    # Unchanged — FeatureLookup already has adjusted stats baked in
    ...
```

The caller constructs `FeatureLookup` with adjustments before passing it in:

```python
# Example usage
adjustments = build_injury_adjustments(year=2025)
feature_lookup = FeatureLookup(kenpom_df, adjustments=adjustments)
result = simulate_bracket_deterministic(year, models, feature_lookup, ...)
```

## LLM-Powered Injury Parsing (Phase 5)

### Purpose

Tournament injury news often breaks in unstructured formats — tweets, coach press conference quotes, beat reporter articles. The Claude API parses these into structured `InjuryReport` objects.

### Interface

```python
def parse_injury_text(
    raw_text: str,
    known_teams: list[str] | None = None,
) -> list[InjuryReport]:
    """Parse unstructured text into structured injury reports.

    Uses Claude API with a structured extraction prompt.
    Validates extracted player names against scraped rosters.
    Rejects extractions where the player cannot be fuzzy-matched
    to a known player on the identified team.

    Args:
        raw_text: Unstructured injury text (tweet, article, quote).
        known_teams: If provided, constrains team matching to these teams.

    Returns:
        List of validated InjuryReport objects. Empty if no injuries
        could be reliably extracted.
    """
```

### Prompt Strategy

```
System: You are a sports data extraction tool. Given unstructured text about
college basketball player injuries, extract structured injury information.

For each injury mentioned, extract:
- player_name: Full name of the player
- team: College team name
- status: One of "out", "doubtful", "questionable", "probable"
- injury_description: Brief description of the injury
- expected_return: When the player might return (if mentioned)

Return a JSON array. If no injuries are mentioned, return [].
Do not infer or guess — only extract what is explicitly stated.
```

### Validation Pipeline

1. Claude extracts raw JSON from text
2. Parse JSON into candidate `InjuryReport` objects
3. For each candidate:
   a. Fuzzy-match `team` against known team names via `_name_matching.py`
   b. Fuzzy-match `player_name` against that team's scraped roster
   c. If either match fails (below similarity threshold), discard the candidate and log a warning
4. Return only validated reports

This prevents hallucinated player names or team names from entering the adjustment pipeline.

## File Structure

### New Files

```
src/sports_quant/march_madness/injuries/
    __init__.py                 # Orchestrator: build_injury_adjustments()
    _models.py                  # Frozen dataclasses: PlayerStats, InjuryReport, etc.
    _config.py                  # Paths: INJURIES_DIR, PLAYER_STATS_DIR, etc.
    _player_scraper.py          # Scrape sports-reference player stats
    _injury_scraper.py          # Scrape ESPN injury reports
    _player_importance.py       # Composite importance scoring
    _adjustment.py              # Compute adjusted KenPom stats
    _name_matching.py           # Cross-source team/player name mapping
    _llm_parser.py              # (Phase 5) Claude API injury text parser

data/march-madness/injuries/
    player_stats/               # Per-year player stat CSVs
        player_stats_2025.csv
    injury_reports/             # Scraped + manual injury data
        injuries_2025.csv
    adjustments/                # Computed team adjustments (for inspection)
        adjusted_kenpom_2025.csv
```

### Modified Files

| File | Change |
|------|--------|
| `_feature_builder.py` | Add optional `adjustments` parameter to `FeatureLookup.__init__()` |

### Unmodified Files

| File | Why Unchanged |
|------|---------------|
| `train.py` | Model trains on historical data where injuries are baked into outcomes |
| `predict.py` | Consumes `FeatureLookup` — gets adjustments automatically |
| `_data.py` | Data loading is unchanged; adjustments are a separate layer |
| `_features.py` | Feature column definitions are unchanged |
| `simulate.py` | Consumes `FeatureLookup` — gets adjustments automatically |
| `survivor.py` | Consumes simulation results — gets adjustments automatically |

## Implementation Phases

### Phase 1: Data Models + Player Stats Collection

**Deliverable:** Player stat CSVs for all 68 tournament teams.

| Step | File | Description |
|------|------|-------------|
| 1 | `injuries/_models.py` | Define all frozen dataclasses and enums |
| 2 | `injuries/_config.py` | Define data directory paths |
| 3 | `injuries/_name_matching.py` | Build team name mapping (68 tournament teams) |
| 4 | `injuries/_player_scraper.py` | Scrape sports-reference player stats |
| 5 | `tests/test_player_scraper.py` | Test with saved HTML fixtures |

**Verification:** Inspect `player_stats_2025.csv` — every tournament team should have 8-15 players with reasonable stat values.

### Phase 2: Player Importance Scoring

**Deliverable:** Importance scores for every player on every tournament team.

| Step | File | Description |
|------|------|-------------|
| 6 | `injuries/_player_importance.py` | Composite importance scoring algorithm |
| 7 | `tests/test_player_importance.py` | Test scoring with synthetic data |

**Verification:** For each team, scores sum to 1.0. The team's leading scorer / highest-minutes player should have the highest score. Bench players should have scores < 0.05.

### Phase 3: Injury Collection + Adjustment Calculation

**Deliverable:** Adjusted KenPom stats for teams with injured players.

| Step | File | Description |
|------|------|-------------|
| 8 | `injuries/_injury_scraper.py` | Scrape ESPN injury page |
| 9 | `injuries/_adjustment.py` | Core adjustment formula |
| 10 | `tests/test_adjustment.py` | Test adjustment scenarios |

**Verification:** For a team with its best player OUT, AdjEM should drop by 3-8 points (roughly). AdjO drops, AdjD rises. A team with no injuries has identical original and adjusted stats.

### Phase 4: Pipeline Integration

**Deliverable:** Bracket simulations run with injury-adjusted stats.

| Step | File | Description |
|------|------|-------------|
| 11 | `_feature_builder.py` | Add `adjustments` parameter to `FeatureLookup` |
| 12 | `injuries/__init__.py` | Top-level orchestrator function |
| 13 | `simulate.py` | Pass adjustments through (optional param) |
| 14 | `tests/test_injury_integration.py` | End-to-end flow test |

**Verification:** Run two bracket simulations — one with adjustments, one without. The simulation with a major injury adjustment should produce different win probabilities for the affected team (and its opponents).

### Phase 5: LLM-Powered Injury Parsing

**Deliverable:** Paste unstructured text, get structured injury reports.

| Step | File | Description |
|------|------|-------------|
| 15 | `injuries/_llm_parser.py` | Claude API extraction + roster validation |
| 16 | `pyproject.toml` | Add `anthropic` as optional dependency |

**Verification:** Parse a known injury tweet. The extracted player name and team should match the scraped roster. Fabricated player names should be rejected.

### Phase 6: Calibration + Backtesting

**Deliverable:** Calibrated degradation factor, evidence that adjustments help.

| Step | File | Description |
|------|------|-------------|
| 17 | `injuries/_backtest.py` | Historical injury backtest (10-20 known cases) |
| 18 | `injuries/_calibration.py` | Grid search over degradation factor values |

**Verification:** Plot bracket accuracy vs degradation factor. Identify whether adjustments improve predictions for historically injured teams.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sports-reference blocks scraping | Low | High | Rate limit aggressively (3s delays). Cache all responses. Fall back to Barttorvik if blocked. |
| Team name mismatches between sources | High | Medium | Dedicated name matching module. Log all unmatched names. Fail loudly. |
| Degradation factor is poorly calibrated | Medium | Medium | Conservative default (0.5). Phase 6 calibration. Adjustments are always optional. |
| Injury data is incomplete or late | High | Low | Support manual overrides. System degrades gracefully — no injury data means no adjustments (original stats used). |
| College basketball injury reporting is sparse | High | Low | Focus only on high-impact absences (starters, >25% minutes share). Minor bench injuries produce negligible adjustments. |
| Adjusting rank columns creates inconsistency | Medium | Low | Don't adjust rank columns. Document this. Raw efficiency values carry more model signal than ordinal ranks. |
| LLM hallucinations in injury parsing | Medium | Medium | Validate all LLM extractions against scraped roster. Reject unmatched names. |

## Success Criteria

- [ ] Player stats scraped for all 68 tournament teams (2025)
- [ ] Importance scores sum to 1.0 per team, star players score highest
- [ ] Injury adjustments produce directionally correct stat changes
- [ ] `FeatureLookup` integration is backwards compatible
- [ ] Bracket simulation produces different results with vs without adjustments for injured teams
- [ ] Unit test coverage for importance scoring and adjustment calculation at 80%+
- [ ] Name matching handles all 68 tournament teams without unmatched names
- [ ] (Phase 5) LLM parser extracts injuries from unstructured text with roster validation
- [ ] (Phase 6) Degradation factor calibrated against historical injuries
