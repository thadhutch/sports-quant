# Seed Slot Debiasing — Specification

## Problem Statement

The LightGBM model exhibits **positional bias in R32+** that the existing column-swap debiasing does not fully eliminate.

### Root Cause

The canonical bracket structure always places the team from the **stronger seed path** in position 0 (Team1):

```
R64:  (1) vs (16)  →  winner enters R32 as Team1
      (8) vs (9)   →  winner enters R32 as Team2

R32:  (1-seed path winner) vs (8/9-seed path winner)
       ↑ always Team1          ↑ always Team2
```

This pattern propagates through every round. In training data, Team1 in R32+ is the team from the structurally stronger seed path **~95%+ of the time**. The model's gradient-boosted trees learn asymmetric, non-linear splits on Team1 vs Team2 feature columns that encode this positional signal — effectively learning `position 0 features ≈ stronger team` as a latent prior.

### Why Column-Swap Averaging Is Insufficient

The existing debiasing in `_debiasing.py` computes:

```
p_debiased = (p_original + (1 - p_swapped)) / 2
```

This assumes the model's positional bias is **symmetric and linear** — that `f(A, B)` and `1 - f(B, A)` average to the true probability. But LightGBM learns **non-linear feature interactions** via decision tree splits. The bias is baked into the tree structure itself (e.g., a split on `Rank < 10` in position 0 carries different meaning than `Rank_Team2 < 10`), so the column-swap average may not fully cancel it.

### Empirical Evidence

Analysis of backtest results revealed **large variance in R32+ between position 0 and position 1 predictions** — the model systematically overvalues teams in position 0 and undervalues teams in position 1. This is consistent with learned seed-slot association.

In 2024, swapping team positions after R64 led to **improved Cinderella identification** — the model gave more realistic (higher) probabilities to lower-seeded teams when they occupied the structurally favorable position 0.

## Proposed Solution

**Swap the team ordering** when constructing matchups for R32+ in the simulation engine. Instead of:

```python
# Current: higher-seed-path winner always in position 0
current_matchups = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
```

Use:

```python
# Proposed: lower-seed-path winner in position 0 for R32+
current_matchups = [(winners[i + 1], winners[i]) for i in range(0, len(winners), 2)]
```

This forces the model to evaluate the Cinderella team with the "favorable" position, counteracting the learned bias. The existing column-swap debiasing in `_predict_game` still runs on top of this — the two mechanisms address different layers of the same bias.

### Why This Works

| Layer | What it addresses | Mechanism |
|-------|-------------------|-----------|
| **Structural swap** (this spec) | Bracket tree ordering always puts favorites in position 0 | Reverse the structural ordering so underdogs get position 0 |
| **Column swap** (existing) | Model may favor whichever features are in the Team1 columns | Predict both ways, average |

The structural swap is a stronger intervention than column swapping alone because it changes **which team's features the model sees first** at the input level, before any tree splitting occurs. Combined with column-swap averaging, this creates a more balanced estimate.

### R64 Is Unaffected

R64 matchups are constructed from known seeding (Selection Sunday). The canonical ordering (1v16, 8v9, etc.) matches the model's expectations and the training data distribution. Swapping here would be counterproductive.

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `simulate.py` | Add `swap_after_r64` parameter to simulation functions |

### Changes to `simulate.py`

#### `_simulate_single_bracket`

Add `swap_after_r64: bool = False` parameter. When enabled, reverse the team order in each matchup tuple when pairing winners for R32+:

```python
def _simulate_single_bracket(
    year: int,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    models: list,
    feature_lookup: FeatureLookup,
    rng: np.random.Generator | None,
    swap_after_r64: bool = False,           # NEW
) -> tuple[list[BracketGame], dict[str, list[float]]]:
```

Winner pairing logic changes from:

```python
if len(winners) >= 2:
    current_matchups = [
        (winners[i], winners[i + 1])
        for i in range(0, len(winners), 2)
    ]
```

To:

```python
if len(winners) >= 2:
    if swap_after_r64:
        current_matchups = [
            (winners[i + 1], winners[i])
            for i in range(0, len(winners), 2)
        ]
    else:
        current_matchups = [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]
```

Note: This swap applies to **every** round after R64 (R32, S16, E8, F4, NCG). The swap reverses the structural ordering of the winner pairs, not the ordering within a single game's Team1/Team2 features.

#### `_simulate_from_precomputed`

Same change — add `swap_after_r64` parameter and reverse winner pairing when enabled. The precomputed probability lookup already handles both orderings via:

```python
probs[(t2.team, t1.team)] = 1.0 - p
```

So swapping the matchup tuple order automatically looks up the correct (inverted) probability. No additional probability computation is needed.

#### `simulate_bracket_deterministic`

Thread the parameter through:

```python
def simulate_bracket_deterministic(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    swap_after_r64: bool = False,           # NEW
) -> SimulationResult:
```

Pass to `_simulate_single_bracket`.

#### `simulate_bracket_monte_carlo`

Thread the parameter through:

```python
def simulate_bracket_monte_carlo(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    n_simulations: int = 1000,
    rng_seed: int = 42,
    swap_after_r64: bool = False,           # NEW
) -> MonteCarloResult:
```

Pass to `_simulate_from_precomputed`.

## Testing Plan

### Unit Tests

1. **Swap mechanics**: Verify that with `swap_after_r64=True`, the R32 matchup `(A, B)` becomes `(B, A)` compared to the default
2. **R64 unaffected**: Verify that R64 matchups are identical regardless of the flag
3. **Probability symmetry**: With precomputed probs, verify `prob(A, B) = 1 - prob(B, A)` so the swap produces correct probabilities

### Backtest Comparison

Run full backtests for all available years with both settings:

```
swap_after_r64=False  (current behavior, baseline)
swap_after_r64=True   (proposed debiasing)
```

Compare:
- **R32+ accuracy** (primary metric — does swapping improve later-round picks?)
- **Upset detection rate** (secondary — does swapping better identify Cinderellas?)
- **R64 accuracy** (sanity check — should be identical between the two)
- **Overall bracket score** (holistic — ESPN/CBS scoring if applicable)
- **Champion prediction accuracy** (MC mode — does the right team appear more often?)

### Expected Results

Based on 2024 empirical results:
- R64 accuracy: **unchanged** (swap is not applied)
- R32+ accuracy: **improved**, especially for games involving lower seeds
- Cinderella identification: **improved** — lower seeds get more realistic probabilities
- Overall accuracy: **modest improvement** (most games are still favorites winning)

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Over-correction in R32 where the higher seed genuinely dominates | LOW | Column-swap debiasing still runs, providing a check. Also, the model's features (KenPom stats) still reflect true team quality — only the positional signal is affected |
| Degraded R64 performance if flag is accidentally applied globally | LOW | Flag defaults to `False`; R64 logic is separate from winner-pairing logic |
| Precomputed MC probabilities become incorrect | NONE | Precomputed probs already store both orderings symmetrically |

## Configuration

The `swap_after_r64` parameter defaults to `False` to preserve backward compatibility. It can be enabled:

1. **Programmatically** via function arguments
2. **Via config** if added to `model_config.yaml` under `simulation.swap_after_r64`
3. **Via CLI** if a `--swap-after-r64` flag is added to the bracket simulation command

For initial validation, programmatic usage in backtest comparisons is sufficient. Config/CLI integration can follow once the approach is validated.

## Complexity

**LOW** — ~20 lines of new code across 2 functions, plus parameter threading through 2 public API functions. All changes are additive with backward-compatible defaults.
