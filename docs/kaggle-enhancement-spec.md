# March Madness Prediction — Kaggle-Informed Enhancement Spec

## Context

Research into consistently winning Kaggle March Machine Learning Mania solutions
(2019-2026) reveals a clear pattern: top solutions share a common architecture of
**composite power ratings + gradient boosted trees + probability calibration**.
The theoretical accuracy ceiling is ~75% (per Georgia Tech's Joel Sokol), and top
Kaggle log loss scores land around 0.47-0.50.

Our current system uses 36 raw KenPom features with a minimally-tuned LightGBM
ensemble. This spec describes the enhancements needed to close the gap with
Kaggle gold-medal solutions, organized into phases by ROI.

See `model-analysis-findings.md` for the internal analysis that preceded this spec.

---

## Phase 1: Fix the Foundation (High Impact, Low Effort)

These are structural issues in the current pipeline that limit our ceiling
regardless of what features or models we add.

### 1.1 Difference Features

**What:** Replace 36 raw team columns with 18 difference columns
(`team1_stat - team2_stat`).

**Why:** Every top Kaggle solution uses pairwise differences. Raw columns force
the model to independently learn Team1 and Team2 strength and then implicitly
compute the gap — lossy and position-dependent. Difference features make the model
inherently position-invariant, likely obsoleting both debiasing mechanisms.

**Features to compute:**
- `adjEM_diff` (single strongest predictor per research)
- `adjO_diff`, `adjD_diff`
- `adjT_diff`
- `rank_diff`
- `seed_diff` (new — currently not a feature at all)
- `luck_diff`
- `sos_adjEM_diff`, `sos_oppO_diff`, `sos_oppD_diff`
- `ncsos_adjEM_diff`

**Bonus derived features:**
- `efficiency_ratio_diff` = `(adjO / adjD)_team1 - (adjO / adjD)_team2`
- `seed_x_adjEM_interaction` = `seed_diff * adjEM_diff` (lets trees learn that
  stat gaps mean different things in 1v16 vs 5v12 matchups)

**Acceptance criteria:**
- [ ] `_features.py` defines the new difference feature set
- [ ] `_feature_builder.py` computes differences for any team pair
- [ ] Backtest shows log loss improvement over raw features
- [ ] Both debiasing layers still work but are expected to have minimal effect

### 1.2 Training Data Symmetrization

**What:** For each game (A, B, outcome=1), also include (B, A, outcome=0).

**Why:** Doubles training data from ~1500 to ~3000 games. With difference
features, the symmetrized row is simply the negation of all features with a
flipped label — this is mathematically consistent and forces the model to learn
no positional preference.

**Acceptance criteria:**
- [ ] Training pipeline generates symmetrized rows
- [ ] Original and symmetrized rows have consistent labels
- [ ] No data leakage (symmetrization happens within each temporal fold)

### 1.3 Hyperparameter Tuning with Optuna

**What:** Bayesian optimization over LightGBM hyperparameters using the temporal
backtest loop as the objective.

**Why:** Current model uses pure defaults (`num_leaves=31`, `lr=0.1`,
`n_estimators=100`). For ~1500-3000 training games, this almost certainly
overfits. Top Kaggle solutions use Optuna with 100 trials.

**Parameters to tune:**
- `num_leaves`: 8-32
- `max_depth`: 3-8
- `learning_rate`: 0.01-0.3
- `n_estimators`: 100-2000 (with early stopping)
- `min_child_samples`: 20-100
- `reg_alpha`: 0-10
- `reg_lambda`: 0-10
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0
- `min_gain_to_split`: 0-1.0

**Cross-validation strategy:** Rolling forward validation (mirrors Kaggle winners):
- Fold 1: Train 2003-2018, validate 2019
- Fold 2: Train 2003-2019, validate 2020
- Fold 3: Train 2003-2021, validate 2022 (skip 2020 COVID)
- Fold 4: Train 2003-2022, validate 2023
- Fold 5: Train 2003-2023, validate 2024

Objective: minimize mean log loss across folds.

**Acceptance criteria:**
- [ ] Optuna study runs with temporal CV
- [ ] Best params stored in `model_config.yaml`
- [ ] Backtest log loss improves over defaults
- [ ] Early stopping callback prevents overfitting

### 1.4 Temporal Validation Split

**What:** Replace random train/val split with temporal split. Use the most recent
year before the backtest year as validation.

**Why:** Random splits leak temporal information. A 2023 game in training while a
2018 game is in validation doesn't reflect real forward-looking generalization.

**Acceptance criteria:**
- [ ] `backtest.py` uses temporal split (year Y-1 as val, years < Y-1 as train)
- [ ] Validation metrics better reflect true out-of-sample performance

### 1.5 Probability Calibration

**What:** Apply isotonic regression on out-of-fold predictions to calibrate
raw model probabilities.

**Why:** LightGBM probabilities are systematically overconfident at extremes.
The survivor pool optimizer and MC simulations are only as good as the
probability estimates. Top Kaggle solutions universally calibrate. A predicted
70% that's actually 60% systematically overvalues favorites.

**Implementation:**
- Fit `sklearn.isotonic.IsotonicRegression` on out-of-fold predictions
  vs actual outcomes from the temporal CV folds
- Apply as a post-processing step before any downstream use
- **Clip final probabilities to [0.025, 0.975]** — a 16-seed beat a 1-seed in
  2018; predicting P=0 yields infinite log loss

**Acceptance criteria:**
- [ ] Calibration layer fits on OOF predictions
- [ ] Reliability diagram shows improved calibration
- [ ] Brier score tracked alongside log loss and accuracy
- [ ] No predicted probability is ever 0.0 or 1.0

---

## Phase 2: Richer Features (High Impact, Medium Effort)

### 2.1 Custom Elo Rating System

**What:** Build a game-by-game Elo rating system for all D1 teams.

**Why:** Elo captures trajectory and recency that end-of-season KenPom snapshots
miss. FiveThirtyEight uses Elo as one of six rating systems. The Odds Gods model
showed Elo features (`elo_last`, `elo_trend`, `elo_sos`) alongside KenPom
metrics produce ~0.54 log loss.

**Specification:**
- Initial rating: 1500 per team
- Season-to-season regression: `0.85 * prev_elo + 0.15 * league_mean`
- Home court adjustment: ±50 points
- K-factor schedule:
  - Early season (days 1-50): K=50 (fast adaptation)
  - Mid season (days 50-100): K=40
  - Late season + conf tournament (days 100+): K=15 (stable)
- Cross-conference multiplier: 1.75x (inter-conference games more informative)
- Quality multiplier: scale K by average Elo of both teams

**Derived features (as differences):**
- `elo_diff` — current Elo gap between teams
- `elo_trend_diff` — Elo change over last 10 games (momentum proxy)
- `elo_sos_diff` — average opponent Elo faced (schedule strength)

**Data required:** Game-by-game results with dates, scores, locations.
Available from Kaggle competition datasets or Sports Reference.

**Acceptance criteria:**
- [ ] Elo ratings computed for all teams from 2003-present
- [ ] Three derived difference features integrated into feature set
- [ ] Backtest confirms additive value over KenPom-only features

### 2.2 Composite Power Ratings

**What:** Ingest multiple third-party rating systems and average them.

**Why:** FiveThirtyEight's key insight: averaging 4-6 independent rating systems
consistently outperforms any single one. This is the highest-signal feature
engineering approach across Kaggle winners.

**Rating systems to ingest (via Massey Composite or direct scraping):**
- KenPom (already have)
- Sagarin
- Bart Torvik (T-Rank)
- Massey
- NET rankings (available 2019+, handle missing with model's native support)
- ESPN BPI

**Implementation:**
- Compute `composite_rating = mean(available_ratings)` per team per season
- `composite_rating_diff` as a single powerful feature
- Also keep individual rating diffs for the model to weight

**Acceptance criteria:**
- [ ] At least 3 additional rating systems ingested
- [ ] Composite rating computed and integrated
- [ ] Missing ratings handled gracefully (LightGBM native NaN support)

### 2.3 Dean Oliver's Four Factors

**What:** Add the four factors of basketball as explicit features.

**Why:** The four factors explain 98% of variance in offensive efficiency.
While KenPom AdjO/AdjD implicitly encode some of this, the raw four factors
give the model direct access to the components.

**Features (all as differences):**
- `efg_pct_diff` — Effective Field Goal % (most important factor)
- `tov_pct_diff` — Turnover Rate
- `oreb_pct_diff` — Offensive Rebound %
- `ft_rate_diff` — Free Throw Rate (FTA/FGA)

**Data source:** Box score aggregates from Kaggle datasets or Sports Reference.

**Acceptance criteria:**
- [ ] Four factors computed as season averages per team
- [ ] Difference features integrated into feature set
- [ ] Feature importance analysis confirms additive value

### 2.4 Tournament-Specific Features

**What:** Add features that specifically capture tournament dynamics.

**Why:** Regular season stats miss tournament-specific signals. Coach experience,
seed matchup history, and conference tournament momentum all have known
predictive power.

**Features:**
- `seed_diff` — (if not already added in Phase 1)
- `seed_matchup_historical_winrate` — empirical P(better seed wins) for this
  specific seed pairing (e.g., 5v12 → 0.65). Acts as a Bayesian prior.
- `coach_tourney_appearances_diff` — experience under tournament pressure
- `conf_tourney_result` — won conf tournament (1), lost final (0.5),
  lost semifinal (0.25), etc. — late-breaking momentum signal
- `days_since_last_game` — rest advantage
- `travel_distance` — teams perform worse traveling far (FiveThirtyEight adjusts
  for this)

**Acceptance criteria:**
- [ ] At least `seed_matchup_historical_winrate` and `coach_tourney_appearances_diff`
      are implemented
- [ ] Features integrated and backtest shows impact

### 2.5 Recent Form / Momentum

**What:** Capture team trajectory in the weeks before the tournament.

**Why:** A team peaking at the right time vs one that lost 3 of their last 5
games is a meaningful signal that end-of-season stats flatten out.

**Features:**
- `last_5_margin_diff` — average scoring margin over last 5 games
- `last_10_win_pct_diff` — win rate over last 10 games
- `elo_trend_diff` — (covered in 2.1, Elo slope over recent window)

**Acceptance criteria:**
- [ ] Recent form features computed from game-by-game data
- [ ] Integrated as difference features

---

## Phase 3: Model Architecture (Medium Impact, Medium Effort)

### 3.1 Sample Weighting by Game Type

**What:** Weight training samples by game importance.

**Why:** The Odds Gods model (0.54 log loss, 77.6% NCAA accuracy) uses:
- Regular season early (days 1-100): weight = 1
- Regular season late + conference tournament: weight = 2
- **NCAA tournament games: weight = 6**

Tournament games are the distribution we're predicting on. Upweighting them
teaches the model tournament-specific patterns.

**Acceptance criteria:**
- [ ] Sample weights computed by game type and season phase
- [ ] Passed to LightGBM via `sample_weight` parameter
- [ ] Backtest confirms improvement (especially in later tournament rounds)

### 3.2 Ensemble All Models (Not Cherry-Pick 3)

**What:** Average predictions from all 50 trained models instead of selecting
the top 3 by validation F1.

**Why:** Picking top 3 by val F1 is overfitting to the validation set. With a
small dataset, val F1 variance across seeds is largely noise. Averaging all
models is more robust and reduces variance — this is standard bagging.

**Alternative:** If some models are genuinely bad, use a soft threshold
(e.g., average all models with val F1 > median) rather than hard top-3.

**Acceptance criteria:**
- [ ] Ensemble uses all models (or soft-filtered subset)
- [ ] Backtest shows reduced variance across years

### 3.3 Stacking Ensemble

**What:** Train a multi-model stack: Layer 1 = LightGBM + Logistic Regression +
Random Forest. Layer 2 = meta-learner combining their OOF predictions.

**Why:** Top Kaggle solutions consistently use stacking. Different model families
capture different patterns — trees find interactions, logistic regression finds
linear trends, random forests average over different feature subsets.

**Architecture:**
```
Layer 1 (base models, trained on temporal CV folds):
  ├── LightGBM (tuned per Phase 1.3)
  ├── Logistic Regression (on difference features)
  ├── Random Forest (moderate depth, high n_estimators)
  └── Optional: Small MLP (2 hidden layers, 32-16 units)

Layer 2 (meta-learner, trained on OOF predictions from Layer 1):
  └── Logistic Regression or Ridge Regression
```

**Acceptance criteria:**
- [ ] At least 3 base models trained with OOF predictions
- [ ] Meta-learner trained on OOF predictions
- [ ] Stacked ensemble outperforms best single model on backtest

### 3.4 Bradley-Terry / Bayesian Model as Ensemble Member

**What:** Fit a Bayesian Bradley-Terry model estimating latent team strengths,
and include its win probabilities as a feature or ensemble member.

**Why:** Bradley-Terry naturally models pairwise comparisons and provides
principled uncertainty quantification via posterior distributions. It captures
a fundamentally different signal than tree-based models.

**Implementation options:**
- Stan/PyMC for full Bayesian inference (more principled, slower)
- Maximum likelihood via `scipy.optimize` (faster, less uncertainty info)
- Use as a Layer 1 base model in the stacking ensemble

**Acceptance criteria:**
- [ ] Bradley-Terry model fitted on regular season outcomes
- [ ] Win probabilities generated for all tournament matchups
- [ ] Integrated as ensemble member or meta-feature

---

## Phase 4: Evaluation & Scoring (Medium Impact, Low Effort)

### 4.1 ESPN-Style Bracket Scoring

**What:** Add bracket pool scoring (10/20/40/80/160/320 points per round) as a
first-class evaluation metric.

**Why:** Accuracy treats all rounds equally, but bracket pools heavily reward
later rounds. A model that nails R64 but whiffs the Final Four loses to one
that gets later rounds right. Our optimization target should match how brackets
are actually scored.

**Acceptance criteria:**
- [ ] ESPN scoring computed alongside accuracy in backtest
- [ ] Round-weighted metrics visible in versioning comparison

### 4.2 Log Loss + Brier Score as Primary Metrics

**What:** Track log loss and Brier score as the primary optimization targets,
matching the Kaggle competition metrics.

**Why:** Accuracy is a threshold metric that throws away probability information.
Log loss and Brier score reward well-calibrated probabilities, which is what
the MC simulator and survivor optimizer actually need.

**Acceptance criteria:**
- [ ] Log loss and Brier score computed per backtest year
- [ ] Displayed alongside accuracy in all reporting
- [ ] Optuna optimizes for log loss (not F1)

### 4.3 Reliability Diagrams

**What:** Plot predicted probability vs actual win rate in bins.

**Why:** Visual calibration check. If the model says 70% for a set of games,
~70% of those games should be wins. This directly validates whether the
calibration layer (Phase 1.5) is working.

**Acceptance criteria:**
- [ ] Reliability diagram generated per backtest year
- [ ] Saved alongside existing bracket visualizations

---

## Phase 5: Data Infrastructure (Lower Priority, Higher Effort)

### 5.1 Game-by-Game Data Pipeline

**What:** Ingest individual game results (not just season aggregates) with
dates, scores, locations, and box scores.

**Why:** Required for Elo computation (Phase 2.1), recent form features
(Phase 2.5), and conference tournament results (Phase 2.4). Currently we only
have end-of-season KenPom snapshots.

**Data sources:**
- Kaggle competition datasets (provided free with the competition)
- Sports Reference / College Basketball Reference
- NCAA stats API

**Acceptance criteria:**
- [ ] Game-by-game results for 2003-present ingested and cleaned
- [ ] Schema includes: date, team1, team2, score1, score2, location, game_type
- [ ] Pipeline is repeatable for new seasons

### 5.2 Massey Composite Ratings Pipeline

**What:** Automated ingestion of Massey Composite rankings (aggregates 100+
rating systems into one dataset).

**Why:** Required for Phase 2.2 composite power ratings. Massey Composite is
the easiest single source for multiple rating systems.

**Acceptance criteria:**
- [ ] Massey Composite data ingested for 2003-present
- [ ] Individual rating systems (KenPom, Sagarin, Torvik, etc.) extracted
- [ ] Updated annually before tournament

---

## Priority Order & Dependencies

```
Phase 1 (do first — fixes fundamental issues):
  1.1 Difference Features ← no dependencies
  1.2 Symmetrization ← depends on 1.1
  1.3 Optuna Tuning ← depends on 1.1, 1.2
  1.4 Temporal Val Split ← no dependencies
  1.5 Calibration ← depends on 1.3

Phase 2 (do second — richer signal):
  2.1 Elo System ← depends on 5.1 (game-by-game data)
  2.2 Composite Ratings ← depends on 5.2 (Massey data)
  2.3 Four Factors ← depends on 5.1
  2.4 Tournament Features ← partial dependency on 5.1
  2.5 Recent Form ← depends on 5.1

Phase 3 (do third — better models):
  3.1 Sample Weighting ← depends on 5.1 (need game type labels)
  3.2 Ensemble All Models ← no dependencies
  3.3 Stacking ← depends on 1.3 (need tuned base models)
  3.4 Bradley-Terry ← depends on 5.1

Phase 4 (do anytime — better evaluation):
  4.1 ESPN Scoring ← no dependencies
  4.2 Log Loss + Brier ← no dependencies
  4.3 Reliability Diagrams ← depends on 1.5

Phase 5 (do early if targeting Phase 2):
  5.1 Game-by-Game Data ← no dependencies
  5.2 Massey Composite ← no dependencies
```

**Recommended execution order:**
1. Phase 1.1 + 1.4 + 4.1 + 4.2 (parallel, no dependencies)
2. Phase 1.2 + 1.3 (sequential, build on 1.1)
3. Phase 1.5 + 4.3 (calibration)
4. Phase 3.2 (quick win — ensemble all models)
5. Phase 5.1 + 5.2 (data infrastructure, unblocks Phase 2)
6. Phase 2.1-2.5 (feature expansion)
7. Phase 3.1 + 3.3 + 3.4 (model architecture)

---

## Expected Impact

| Milestone | Expected Log Loss | Expected Accuracy |
|-----------|------------------|-------------------|
| Current (raw KenPom, default LightGBM) | ~0.58-0.60 | ~68% |
| After Phase 1 (diff features, tuning, calibration) | ~0.53-0.55 | ~71-72% |
| After Phase 2 (Elo, composite ratings, four factors) | ~0.50-0.52 | ~73-74% |
| After Phase 3 (stacking, sample weighting) | ~0.48-0.50 | ~74-75% |
| Kaggle gold medal range | ~0.47-0.49 | ~75% |

---

## References

- FiveThirtyEight March Madness methodology (composite ratings + Elo + travel)
- Odds Gods technical methodology (LightGBM + Elo + isotonic calibration, 0.54 log loss)
- Kaggle March ML Mania 2023 gold solution (XGBoost + RFE + Brier score)
- Bradley-Terry Bayesian model for Kaggle 2019 (Stan/MCMC pairwise inference)
- Conor Dewey's ensemble approach (composite of SAG/POM/MOR/WLK + Elo + logistic regression)
- Dean Oliver's Four Factors (eFG%, TOV%, OREB%, FT rate — 98% of efficiency variance)
