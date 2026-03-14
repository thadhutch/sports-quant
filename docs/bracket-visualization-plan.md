# Bracket Visualization Plan

## New Files (3 source + 2 test)

| File | Purpose | ~Lines |
|------|---------|--------|
| `_bracket.py` | Frozen dataclasses: `BracketSlot`, `BracketGame`, `Bracket` | ~200 |
| `_bracket_builder.py` | Build `Bracket` objects from matchup CSVs and backtest result CSVs | ~250 |
| `bracket_plots.py` | Matplotlib rendering — single bracket, comparison, per-round accuracy | ~400 |
| `test_march_madness_bracket.py` | Unit tests for data structures + builder logic | ~200 |
| `test_march_madness_bracket_plots.py` | Smoke tests for rendering (files created, no errors) | ~150 |

## Data Structures (Immutable)

```python
@dataclass(frozen=True)
class BracketSlot:        # team + seed
class BracketGame:        # two slots, winner, round, region, probability, upset/correct flags
class Bracket:            # year, source label, tuple of 63 BracketGames
```

## Three Visualization Modes

1. **Single bracket** — Full 64-team bracket with seeds, team names, and confidence. Color-coded
   by upset status. Left two regions flow right, right two flow left, Final Four in center.

2. **Comparison bracket** — Predicted vs actual side-by-side. Green = correct pick, red = wrong.
   Immediately shows where the model went off the rails by round.

3. **Accuracy-by-round bar chart** — Simple and analytically powerful. Shows what % of picks were
   correct in R64, R32, S16, E8, F4, NCG. This is the key analytical view.

## Key Technical Challenge: Recovering Round Info

Backtest results (`ensemble_results.csv`) don't have round numbers — they were dropped as
non-feature columns. The builder joins back to `restructured_matchups.csv` on
`(Year, Team1, Team2)` to recover which round each game belongs to.

## Region Assignment

Games are ordered in groups of 8 in R64 which map to regions. We'll use generic labels
("Region 1-4") initially, with an optional per-year config override for actual region names.

## CLI Integration

```bash
sports-quant march-madness bracket --year 2024 --source debiased --compare
```

Reads from existing backtest output dirs, renders to
`data/march-madness/backtest/{version}/{year}/plots/bracket.png`.

## Implementation Phases

### Phase 1 — Data structures & builder (TDD)

- Tests first for dataclasses and bracket construction
- `build_bracket_from_matchups()` for ground truth
- `build_bracket_from_backtest()` for predictions (joins to recover rounds)
- Upset detection and correctness flagging

### Phase 2 — Rendering

- Smoke tests first
- Coordinate system: fixed x per round, y computed recursively (parent centered between children)
- `matplotlib.patches.FancyBboxPatch` for team boxes
- Color coding: green (correct), red (incorrect), white (no ground truth)
- Confidence shown via opacity or annotation

### Phase 3 — CLI & integration

- Add `bracket` subcommand
- Wire to backtest output paths
- Validate with all 4 backtest years (2022-2025)

### Phase 4 (future) — Forward simulation

- `simulate_bracket()` to generate a full predicted bracket from just R64 matchups
- Needed for current-year predictions where later rounds aren't known yet
- Deferred — backtest analysis is the immediate priority

## Out of Scope

- Web/HTML rendering (matplotlib only, matches existing stack)
- Interactive features (static PNGs for now)
- Forward bracket simulation (Phase 4)
- Modifying `backtest.py` or `predict.py`
