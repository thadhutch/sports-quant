# Bracket Visualization Plan

## New Files

### Phase 1 — Data (complete)

| File | Purpose | ~Lines |
|------|---------|--------|
| `_bracket.py` | Frozen dataclasses: `BracketSlot`, `BracketGame`, `Bracket` | ~200 |
| `_bracket_builder.py` | Build `Bracket` objects from matchup CSVs and backtest result CSVs | ~250 |
| `test_march_madness_bracket.py` | Unit tests for data structures + builder logic | ~200 |

### Phase 2 — Rendering

| File | Purpose | ~Lines |
|------|---------|--------|
| `_bracket_theme.py` | Frozen theme dataclass — colors, fonts, dimensions, spacing constants | ~80 |
| `_bracket_layout.py` | Pure coordinate math — slot positions, connector paths, recursive y-centering | ~150 |
| `_bracket_render_svg.py` | SVG generation using `drawsvg` — team boxes, connectors, headers, legend | ~300 |
| `_bracket_charts.py` | Plotly accuracy-by-round + round-over-round comparison charts | ~150 |
| `bracket_plots.py` | Public API — `render_bracket()`, `render_comparison()`, `render_accuracy_chart()` | ~100 |
| `test_march_madness_bracket_plots.py` | Smoke tests + layout math unit tests | ~200 |

## Data Structures (Immutable)

```python
@dataclass(frozen=True)
class BracketSlot:        # team + seed
class BracketGame:        # two slots, winner, round, region, probability, upset/correct flags
class Bracket:            # year, source label, tuple of 63 BracketGames
```

## Three Visualization Modes

1. **Single bracket** — Full 64-team SVG bracket. Left regions flow right, right regions flow
   left, Final Four centered. Each team box shows `[seed] Team Name` with confidence bar.
   Upsets highlighted with gold accent border.

2. **Comparison bracket** — Two-tone overlay. Correct picks get green fill, incorrect get red
   with strikethrough on the wrong team. Shows predicted winner vs actual winner side-by-side
   within each game box.

3. **Accuracy-by-round chart** — Plotly grouped bar chart. Shows correct/total per round with
   percentage labels. Optional year-over-year overlay. Interactive HTML output + static PNG.

## Rendering Stack

### Why not matplotlib?

Matplotlib's `FancyBboxPatch` approach produces charts that look like charts. Brackets need
precise layout control, clean typography, and crisp vector output. Matplotlib fights you on
all three.

### Libraries

| Library | Role | Why |
|---------|------|-----|
| **`drawsvg`** | Bracket rendering | Purpose-built for SVG — precise coordinate control, clean vector output, PNG/PDF export via `cairosvg`. Produces ESPN/CBS-quality bracket layouts. |
| **`plotly`** | Accuracy-by-round chart | Modern, interactive, publication-quality analytical charts. HTML + static PNG export. |
| **`cairosvg`** _(optional)_ | SVG → PNG rasterization | High-quality rasterization of SVG brackets. Requires system Cairo lib. Falls back to SVG when unavailable. |

### New dependencies

```toml
# Required (pure Python, no system deps)
drawsvg = "^2.3"
plotly = "^5.0"

# Optional (requires system Cairo lib: `brew install cairo` on macOS)
# Enables PNG and PDF export from SVG brackets
# cairosvg = "^2.7"
```

### Design System

- **Color palette**: Dark slate background (`#1a1a2e`) with white text, green (`#00c853`) for
  correct picks, red (`#ff1744`) for wrong, gold accent (`#ffd700`) for champions/upsets
- **Typography**: Clean sans-serif (system fonts), seed numbers in bold monospace
- **Team boxes**: Rounded rectangles with subtle drop shadows, seed badges on the left edge
- **Connector lines**: Smooth bezier curves between rounds (not harsh right angles)
- **Confidence**: Gradient fill intensity on team boxes (higher confidence = more saturated)
- **Round headers**: Styled column headers with game counts

### Rendering Pipeline

```
Bracket (data) → Layout (coordinates) → Render (SVG) → Export (PNG/PDF/SVG)
```

Each step is a pure function operating on immutable data:

1. **Layout** — Takes a `Bracket` + `BracketTheme`, returns a frozen `BracketLayout` with
   computed `(x, y)` for every game, connector paths, and bounding box
2. **Render** — Takes `BracketLayout` + `Bracket` + `BracketTheme`, returns an SVG `Drawing`
3. **Export** — `Drawing.save_svg()` or `cairosvg.svg2png()` for rasterization

### Output Formats

- **SVG** — Primary output. Scalable, embeddable, archivable.
- **PNG** — Via `cairosvg`. For quick sharing, README embedding.
- **HTML** — Plotly charts only. Interactive exploration.
- **PDF** — Via `cairosvg`. Print-quality for reports.

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
`data/march-madness/backtest/{version}/{year}/plots/bracket.{svg,png}`.

## Implementation Phases

### Phase 1 — Data structures & builder (TDD) ✓

- Tests first for dataclasses and bracket construction
- `build_bracket_from_matchups()` for ground truth
- `build_bracket_from_backtest()` for predictions (joins to recover rounds)
- Upset detection and correctness flagging

### Phase 2 — Professional rendering (SVG + Plotly)

- TDD: unit tests for `_bracket_layout.py` coordinate math (deterministic, pure functions)
- TDD: smoke tests for SVG rendering (file created, valid XML, expected elements present)
- `_bracket_theme.py` — frozen theme dataclass with full design system constants
- `_bracket_layout.py` — pure coordinate math, fixed x per round, recursive y-centering
- `_bracket_render_svg.py` — `drawsvg`-based SVG generation with team boxes, bezier
  connectors, round headers, legend, and confidence visualization
- `_bracket_charts.py` — Plotly accuracy-by-round grouped bar chart with interactive HTML
  and static PNG export
- `bracket_plots.py` — public API surface: `render_bracket()`, `render_comparison()`,
  `render_accuracy_chart()`
- Visual regression baseline (save reference PNGs, compare dimensions)

### Phase 3 — CLI & integration ✓

- Add `bracket` subcommand
- Wire to backtest output paths
- Validate with all 4 backtest years (2022-2025)

### Phase 4 (future) — Forward simulation

- `simulate_bracket()` to generate a full predicted bracket from just R64 matchups
- Needed for current-year predictions where later rounds aren't known yet
- Deferred — backtest analysis is the immediate priority

## Out of Scope

- Interactive bracket features (click-to-expand, hover tooltips — future consideration)
- Forward bracket simulation (Phase 4)
- Modifying `backtest.py` or `predict.py`
