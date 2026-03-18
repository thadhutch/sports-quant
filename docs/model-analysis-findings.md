# March Madness Model — Analysis & Findings

## Strengths

### 1. Solid Engineering Foundation

Frozen dataclasses everywhere, clean separation of concerns (features, training, debiasing, simulation, survivor, visualization). The codebase is genuinely well-structured for iterative model improvement.

### 2. Honest Backtesting Methodology

Training only on years < Y and testing on year Y prevents data leakage. This is the right approach for temporal data — many March Madness models cheat here.

### 3. Forward Simulation Is the Right Evaluation Framework

Evaluating by simulating the full bracket (where wrong R64 picks cascade into wrong R32 matchups) is far more realistic than the naive "oracle matchup" approach most people use. This alone puts you ahead of most amateur bracket models.

### 4. Column-Swap Debiasing Is Clever and Correct

Averaging `p_orig` and `1 - p_swapped` is a principled way to handle LightGBM's asymmetric tree structure. The spec for seed-slot debiasing correctly identifies that this is necessary but insufficient.

### 5. Survivor Pool Optimizer

Branch-and-bound for optimal pick sequences is elegant. The live mode with MC forward simulation is a genuinely useful tool.

### 6. Pre-Computed Pairwise Probabilities for MC

Computing all 2016 pairings once and reusing them is the right optimization — avoids redundant inference during sampling.

---

## Biggest Shortfalls

### 1. Feature Poverty — KenPom-Only Is a Ceiling

18 KenPom stats per team (36 total features) are excellent efficiency metrics, but they are **end-of-regular-season snapshots** that miss crucial tournament-relevant signals:

- **No seed differential or matchup-type features.** The model has no explicit concept of "this is a 5v12 game." It has to infer seed matchup dynamics purely from stat deltas, which is lossy. A simple `seed_diff` feature would help, and interaction terms like `seed1 * adjEM_diff` would let trees learn that stat gaps matter differently for 1v16 vs 5v12.
- **No momentum / trajectory features.** A team that went 15-3 in conference play but lost their last 3 games is very different from one that won their last 10. KenPom end-of-season doesn't capture this.
- **No tournament experience features.** Coach tournament appearances, program historical seed performance, recent Final Four runs — these are known predictors of tournament success that KenPom doesn't encode.
- **No conference strength signal beyond SOS.** Conference tournament performance, number of bids, NET rankings by conference — all informative.

### 2. LightGBM Is Used with Essentially No Tuning

In `train.py:72-76`, only `objective` and `metric` are set — everything else is defaults:

```python
model = LGBMClassifier(
    objective=hyperparams["objective"],
    metric=hyperparams["metric"],
)
```

This means default `num_leaves=31`, `learning_rate=0.1`, `n_estimators=100`, `max_depth=-1`, `min_child_samples=20`, etc. For a dataset this small (~1500-2000 games across 20+ years), these defaults are almost certainly overfitting. Needed:

- **Reduce model complexity** — `num_leaves=15-20`, `max_depth=4-6`, `min_child_samples=50+`
- **Add regularization** — `reg_alpha`, `reg_lambda`, `min_gain_to_split`
- **Bayesian hyperparameter optimization** (Optuna) with the backtest loop as the evaluation
- **Reduce `n_estimators` with early stopping** — `eval_set` is present but no `callbacks` for early stopping

### 3. Random Train/Val Split Loses Temporal Structure

In `backtest.py:119`, training data (all years < Y) is split into train/val randomly:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=test_size, random_state=random_seed,
)
```

A 2023 game can end up in training while a 2018 game is in validation — val score doesn't reflect forward-looking generalization. The val split should be temporal (e.g., most recent year before the backtest year as val).

### 4. Model Selection by Random Seed Is Noisy

Training 50 models with random seeds and picking top 3 by val F1 (`backtest.py:167-169`) is a form of overfitting to the val set. With a small dataset, val F1 variance across seeds is largely noise. More stable alternatives:

- Cross-validation within the temporal training fold
- Bagging (bootstrap aggregation) rather than seed-lottery
- Averaging all 50 models instead of cherry-picking 3

### 5. No Calibration of Probabilities

The model outputs raw LightGBM probabilities, which are not well-calibrated by default. For both the survivor pool optimizer and MC simulations, **calibration matters enormously** — a predicted 70% that's actually 60% will systematically overvalue favorites and undervalue upsets. Needed:

- Platt scaling or isotonic regression as a calibration layer
- Evaluate calibration explicitly (reliability diagrams, Brier score) alongside accuracy
- The survivor optimizer's "optimal" picks are only optimal if the probabilities are accurate

### 6. The Debiasing Spec Is Good but Doesn't Go Far Enough

The seed-slot debiasing spec correctly identifies the R32+ positional bias problem. But the proposed fix (swap team ordering) is still a band-aid on a structural issue. Deeper fixes:

- **Train on symmetrized data** — for each game (A, B, outcome=1), also include (B, A, outcome=0) in training. This forces the model to learn symmetric features from the start.
- **Use difference/ratio features instead of raw team columns** — `adjEM_diff = team1_adjEM - team2_adjEM` instead of separate `team1_adjEM` and `team2_adjEM`. This makes the model inherently position-invariant.

### 7. No ESPN/Bracket Scoring Evaluation

Accuracy (correct/total per round) is tracked, but bracket pools score games with escalating points (10, 20, 40, 80, 160, 320). A model that nails R64 but whiffs on the Final Four loses to one that gets later rounds right. Bracket pool scoring should be a first-class metric.

---

## What to Incorporate Next (Prioritized by Expected Impact)

### High Impact, Low Effort

1. **Difference features** (`adjEM_team1 - adjEM_team2`, `rank_diff`, `seed_diff`) — eliminates positional bias at the feature level, likely obsoletes both debiasing mechanisms
2. **Hyperparameter tuning with Optuna** — 30 minutes of tuning will likely beat default params significantly
3. **Early stopping** in LightGBM training — prevents overfitting, free improvement
4. **Probability calibration** (Platt scaling) — critical for survivor and MC accuracy

### High Impact, Medium Effort

5. **Training data augmentation via symmetrization** — double training data, eliminate positional bias structurally
6. **ESPN-style bracket scoring** as a primary backtest metric
7. **Temporal validation split** instead of random split within the training fold
8. **Ensemble all 50 models** instead of picking top 3 — more robust, less variance

### Medium Impact, Higher Effort

9. **Additional data sources** — Barttorvik (T-Rank), Evan Miya ratings, Sagarin, Massey for ensemble-of-data-sources approach
10. **Matchup-specific features** — seed matchup type (5v12, 3v14), region strength, path difficulty
11. **Conference tournament results** as late-breaking momentum signal
12. **Historical seed performance priors** — empirical P(12-seed beats 5-seed) as a Bayesian prior blended with model output

### Exploratory / Research

13. **Neural network approach** (small MLP) — could learn feature interactions that trees miss, especially with difference features
14. **Stacking ensemble** — LightGBM + logistic regression + random forest, meta-learner combines them
15. **Player-level data** (transfer portal impact, injury status) — harder to systematize but high signal in individual years

---

## Key Takeaway

The single highest-ROI change is **switching to difference features**. It simultaneously fixes the positional bias problem, reduces feature dimensionality from 36 to 18, and makes the model's job fundamentally easier — it just needs to learn "how much better is Team A than Team B" rather than separately modeling both teams' absolute quality and implicitly computing the gap.
