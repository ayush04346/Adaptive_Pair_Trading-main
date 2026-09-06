# Adaptive Pair Trading — Statistical Arbitrage on Indian Equities

**Ayush Arora | MQMS2404 | Quantitative Finance Research**

A research project on **cointegration-based pairs trading** over a 79-stock subset of
the NIFTY 100 (2015–2024). It runs an end-to-end pipeline — universe scan →
cointegration tests → spread/Z-score → volatility model → ML filter → adaptive hedge →
walk-forward validation → multi-pair portfolio — and reports what the pipeline actually
produces, including where it does **not** work.

> **Honest status (read this first).** On the data in this repo the strategy does
> **not** have a demonstrated edge. The universe scan yields **3** tradable pairs, not
> a deep book. Out-of-sample Sharpe is positive but its bootstrap confidence interval
> **includes zero** at every cost level. The **Kalman adaptive hedge — the feature the
> project is named after — underperforms a plain static OLS hedge** on every pair
> tested, because at the configured adaptation speed it churns. The value of the repo
> is the *method and the diagnostics*, not the P&L. See **Limitations** below.

---

## What the pipeline does

Identify stock pairs whose prices share a long-run equilibrium (cointegration),
build the mean-reverting spread, and trade deviations: short the rich leg / long the
cheap leg when the spread Z-score exceeds ±2σ, close inside ±0.5σ.

"Adaptive" refers to estimating the hedge ratio β with a **Kalman filter** (time-varying
`β_t`, no look-ahead) instead of a single full-sample OLS β. NB07 and NB10 implement and
**test** this; NB07's result is that the static OLS β is the better choice here.

---

## Pipeline

```
NB00  Pair Selection (hierarchical filter — illustrative, metals sector)
NB01  Data Download (NIFTY-100 constituents, yfinance, 2015–2024)
NB02  Cointegration Validation (Engle–Granger + OLS hedge ratio)
NB03  Spread Construction (A − β·B − α) + rolling 252/60 Z-score
NB03A Diagnostics (stationarity, market neutrality, ARCH test)
NB04  GARCH(1,1)-t conditional volatility
NB05  ML Signals (Random Forest / Logistic reversion probability, OOS only)
NB06  Backtesting (baseline vs ML-filtered vs vol-scaled)
NB07  Kalman Adaptive Hedge  (_07_Kalman_Adaptive_Hedge.ipynb)
  │
  ├── NB08  Universe Scan — C(79,2) = 3,081 candidate pairs → data/top_pairs.csv
  ├── NB09  Walk-Forward Validation — reads top_pairs.csv, 16 OOS windows per pair
  └── NB10  Multi-Pair Portfolio — reads top_pairs.csv, Kalman hedge + ERC weights
```

`config.py` is the single source of truth for parameters **and** the shared
implementations: `rolling_zscore`, `pair_backtest`, `sharpe_ratio`, `max_drawdown`,
`stationary_bootstrap`, `kalman_hedge`. No notebook re-implements them.

---

## Key Results (from the committed notebook outputs)

### NB08 — universe scan

All 79 tickers in `data/prices.csv` are sector-mapped, so the scan covers the full
**C(79,2) = 3,081** pairs (NB08 asserts this at runtime). Five-stage funnel:

| Stage | Test | Pairs |
|-------|------|------:|
| 0 | Candidate pairs `C(79,2)` | 3081 |
| 1 | Return correlation `r ≥ 0.40` | 226 |
| 2 | Engle–Granger raw `p < 0.05` | 34 |
| 3 | Survive Benjamini–Hochberg FDR (α = 0.05) | **3** |
| 4 | Also pass Johansen trace (95%) | 3 |
| 5 | Spread quality (ADF `p<0.05`, Hurst `<0.5`, half-life 5–90d) | **3** |

BH correction does its job (34 raw hits → 3 after FDR control), but only **3 pairs**
survive: `DABUR/HINDUNILVR`, `GRASIM/HINDALCO`, `HINDALCO/TATASTEEL` (half-lives
26 / 35 / 44 d, Hurst 0.34 / 0.35 / 0.45). Two of the three share the HINDALCO leg.

The `hurst_exponent` function is the **variance-of-lagged-differences** (structure-function)
estimator — it regresses the log dispersion of the lag-τ differences `xₜ₊τ − xₜ` on `log τ`
and rescales the slope to H — not R/S analysis, despite the docstring's label. The maths is
a valid Hurst estimator; only the name is wrong.

### NB09 — walk-forward out-of-sample validation

16 rolling windows per pair (24-month train / 6-month test, 2017–2024). P&L is a
**return on deployed capital** (`config.pair_backtest`), Sharpe is over the **full**
daily series, costs are swept at **10 / 30 / 60 bps round trip**. Significance is a
**stationary block bootstrap** (mean block 20 d) on the pooled daily series — a t-test
would assume the pair-windows are independent, which they are not.

Full-period OOS Sharpe (all 16 windows stitched):

| Pair | 10 bps | 30 bps | 60 bps |
|------|------:|------:|------:|
| DABUR / HINDUNILVR | 0.41 | 0.36 | 0.28 |
| GRASIM / HINDALCO | 0.42 | 0.39 | 0.33 |
| HINDALCO / TATASTEEL | 0.23 | 0.19 | 0.13 |

Pooled equal-weight (secondary — see the dependence caveat above):

| Cost | Ann. return | Sharpe | Sharpe 95% bootstrap CI | P(Sharpe > 0) |
|------|------:|------:|:--:|------:|
| 10 bps | +3.1% | 0.56 | [−0.18, 1.26] | 0.93 |
| 30 bps | +2.7% | 0.50 | [−0.25, 1.20] | 0.91 |
| 60 bps | +2.1% | 0.40 | [−0.35, 1.10] | 0.87 |

**Every pooled CI includes zero.** In-sample per-window Sharpe (0.84) is well above
OOS (0.16 mean per window) — the usual overfitting gap. 8 of 48 pair-windows never
traded (kept as a defined zero return, not dropped).

### NB07 — Kalman vs static hedge (TATASTEEL / HINDALCO, in-sample)

| | Static OLS β | Kalman `β_t` |
|--|------:|------:|
| Sharpe @ 30 bps | **+0.72** | **−0.90** |
| Ann. return @ 30 bps | +7.5% | −3.7% |
| Trades | 21 | **131** |
| Time in market | 40% | 9% |

Kalman β range `[0.12, 0.30]` (static β = 0.245). The drift is real but small, and
tracking it at `KF_DELTA = 1e-4` makes the innovation series close to white noise, so
the Z-score whipsaws across the bands — 131 round trips at a ~1-day hold, which is
pure transaction cost. The Kalman innovation is *more* stationary by ADF (−9.8 vs −4.6)
precisely because it is closer to noise, which is the wrong property for this strategy.

### NB10 — multi-pair portfolio (in-sample, full 2015–2024)

Only 3 pairs are available (NB08 `TOP_N_PAIRS = 10`, `PORT_TOP_PAIRS = 5`). ERC on a
3×3 covariance:

| @ 30 bps round trip | Ann. return | Sharpe | Max DD |
|--|------:|------:|------:|
| Equal-weight portfolio | +0.9% | 0.18 | −15% |
| ERC portfolio (capped) | +0.4% | 0.11 | −12% |

Diversification ratio ≈ **1.6** at every cost level (portfolio vol below the
weighted-average leg vol) — the one clean positive. At 60 bps both portfolios are
negative.

**Diagnostic 3a — Kalman turnover.** `KF_DELTA = 1e-4` is too fast for
`DABUR/HINDUNILVR`: its β moves ~5× more per day than the metals pairs (0.96%/day vs
0.18%), its innovation autocorrelation collapses to +0.02 (vs +0.96), and it trades 98
times at a 1.5-day hold. Its Sharpe goes 0.33 → −0.21 → −1.05 across the three cost
levels — the only pair that turns negative on cost. Not re-tuned; a per-pair δ or an
autocorrelation gate would be the fix.

**Diagnostic 3b — ERC convergence.** The optimiser *has* converged (objective ≈ 1e-7,
8 random restarts agree). The residual ~10 pp risk-contribution deviation is the price
of `PORT_MAX_WEIGHT = 0.50`: unconstrained ERC wants `DABUR/HINDUNILVR` at **57%**
(lowest-vol leg), so the cap forces it to carry 23% of risk instead of 33%. The
notebook now reports both the capped and uncapped solutions and labels the result
"capped ERC".

---

## Methodological corrections applied

Beyond the textbook fixes, three passes of structural corrections:

| Issue | Before | After |
|-------|--------|-------|
| ADF test | on raw price difference | on the OLS residual `A − β·B − α` |
| Z-score | full-sample mean/std (look-ahead) | `config.rolling_zscore`, rolling 252/60, one definition everywhere |
| GARCH | Zero-Mean + Normal | Constant-Mean + Student-t |
| ML evaluation | RF threshold sweep read Logit probabilities | correct model, correct probabilities |
| ML in backtest | `np.random.uniform()` fallback | walk-forward OOS probabilities only |
| Multiple testing | none | Benjamini–Hochberg FDR |
| **Pipeline wiring** | NB09 hardcoded one pair; NB10 silently fell back to 5 hardcoded pairs when `top_pairs.csv` was absent | NB08 writes `top_pairs.csv`; NB09/NB10 read it; missing file raises |
| **Universe count** | `SECTORS` mapped 66/79 tickers → scan tested 2,145, README claimed 3,081 | all 79 mapped, scan tests 3,081, NB08 asserts coverage |
| **P&L model** | `spread.diff() / mean|spread|` — no capital base, unbounded, equity curve crossed zero (Max DD printed as −1e13 %) | return on gross exposure `price_A + \|β\|·price_B`, `min(ret) > −1` asserted |
| **Sharpe** | annualised over active days only (overstated by 1/√time-in-market) | full daily series, flat days included; time-in-market reported |
| **Max drawdown** | returned garbage if equity ≤ 0 | raises if equity ≤ 0 |
| **Transaction costs** | 10 bps one-way on one leg (~4× understated for a two-leg round turn) | `COST_BPS_ROUND_TRIP`, swept 10 / 30 / 60 bps, all reported |
| **Walk-forward windows** | 18 of 48 pair-windows silently dropped (`sharpe()` → NaN → `.dropna()`) | every window kept; zero-trade windows counted |
| **Pooled significance** | t-test on non-independent pair-windows + a separate n=3 t-test | stationary block bootstrap; per-pair result is primary |
| **NB07 filename** | `_07_.ipynb` | `_07_Kalman_Adaptive_Hedge.ipynb` |
| **`kalman_hedge`** | near-duplicate copies in NB07 and NB10 | one `config.kalman_hedge` |

---

## Limitations

- **No demonstrated edge.** Pooled OOS Sharpe CIs include zero at 10/30/60 bps; per-pair
  Sharpe is 0.1–0.4 with n = 3 pairs over 8 years.
- **3 pairs, not a portfolio.** The "5-pair ERC portfolio" is 3 pairs, two sharing a leg,
  with the ERC weight cap binding.
- **The adaptive hedge hurts.** NB07 and NB10-3a both show the Kalman hedge losing to
  static OLS at the configured `KF_DELTA`. A single global δ cannot serve both a
  slowly-drifting structural pair and a noisy one.
- **NB10 is in-sample.** Its equity curves and Sharpe use the full 2015–2024 history.
  NB09 is the only out-of-sample read.
- **NB00–NB06 not re-run in the correction passes.** NB03/NB06 still compute their
  Z-score / P&L inline (NB06 still on one-way `TC`); NB09/NB10/NB07 are the corrected
  reference. NB05's ML filter has AUC ≈ 0.61 (RF) / 0.65 (Logit) — weak.
- **Pair-selection look-ahead.** NB08 selects pairs with cointegration tests run over
  the full 2015–2024 sample, and NB09 then validates on walk-forward windows inside
  that same sample. The walk-forward is out-of-sample for the strategy parameters
  (β, Z-score, entry timing) but not for pair selection: the three pairs were chosen
  because they were cointegrated over exactly the period being tested. Removing this
  bias means re-running the universe scan at each window's training cutoff using only
  data available then. This is the largest remaining bias in the project and would
  likely push the measured Sharpe further toward zero.
- **FDR correction applied after a data-dependent screen.** The correlation filter runs
  first on the full sample and Benjamini–Hochberg is applied only to its 226 survivors,
  not to all 3,081 candidates. Screening on the same data that produces the p-values
  weakens the FDR guarantee, so "3 pairs survive FDR control" is optimistic. A clean
  version would carry all 3,081 tests into the BH procedure, or split the screen and
  the test onto different samples.
- **One-sided correlation filter.** The screen keeps only `r ≥ 0.40`. Cointegration
  does not require positive return correlation and a negative hedge ratio is perfectly
  tradeable, so roughly half the candidate space is excluded without justification.
  Fixing it means filtering on `|r|` (or dropping the correlation pre-screen and paying
  for the extra cointegration tests).
- **Engle–Granger asymmetry.** `coint(A, B)` and `coint(B, A)` return different
  p-values, and the pair ordering comes from `itertools.combinations`, so which
  direction is tested is arbitrary. The Johansen filter mitigates this but does not
  remove the direction dependence from the p-values that enter the BH correction.
  Testing both directions and taking the min (with a correspondingly larger correction)
  would make the ranking order-invariant.
- **Costs are a flat bps parameter.** No market impact, no per-name SLB borrow, no lot
  sizes. 30–60 bps is the intended judging range; 10 bps is optimistic.
- **Single data vintage** (`data/prices.csv`, yfinance, adjusted, 2015-01 → 2024-12).
  Survivorship of the constituent list is not controlled.

---

## Setup & Reproduction

```bash
# Python 3.11 (the machine default 3.14 has a broken numpy; use a 3.11 venv)
python3.11 -m venv .venv && . .venv/Scripts/activate      # or .venv/bin/activate
pip install -r requirements.txt

# Execute in dependency order (nbconvert writes outputs back in place):
jupyter nbconvert --to notebook --execute --inplace \
  _01_Data_Download.ipynb _00_Pair_Selection.ipynb _02_Cointegration_Validation.ipynb \
  _03_spread_Construction.ipynb _03A_Diagnostics_Final.ipynb _04_GARCH_Modeling.ipynb \
  _05_ML_Signals_Reframed.ipynb _06_Backtesting_Combined.ipynb \
  _07_Kalman_Adaptive_Hedge.ipynb \
  _08_Universe_Pair_Scan.ipynb _09_Walk_Forward_Validation.ipynb _10_Portfolio_Multi_Pair.ipynb
```

NB08 must run before NB09 and NB10 — they read `data/top_pairs.csv` and raise if it is
missing.

### Directory structure

```
├── config.py                          # parameters + shared implementations
├── requirements.txt
├── _00_Pair_Selection.ipynb
├── _01_Data_Download.ipynb
├── _02_Cointegration_Validation.ipynb
├── _03_spread_Construction.ipynb
├── _03A_Diagnostics_Final.ipynb
├── _04_GARCH_Modeling.ipynb
├── _05_ML_Signals_Reframed.ipynb
├── _06_Backtesting_Combined.ipynb
├── _07_Kalman_Adaptive_Hedge.ipynb    # Kalman adaptive hedge (was _07_.ipynb)
├── _08_Universe_Pair_Scan.ipynb       # full 79-stock scan → top_pairs.csv
├── _09_Walk_Forward_Validation.ipynb  # OOS validation over every top pair
├── _10_Portfolio_Multi_Pair.ipynb     # ERC multi-pair portfolio + KF diagnostics
└── data/
    ├── prices.csv                     # 79 NIFTY-100 tickers, 2015-01 → 2024-12
    ├── top_pairs.csv                  # NB08 output (up to 10 ranked pairs; 3 survived)
    ├── spread_tatasteel_hindalco.csv
    ├── zscore_tatasteel_hindalco.csv
    ├── garch_vol_tatasteel_hindalco.csv
    ├── ml_prob_rf.csv                 # NB05 out-of-sample RF probabilities
    ├── ml_prob_logit.csv             # NB05 out-of-sample Logit probabilities
    └── kalman_adaptive_hedge.csv      # NB07 kf_alpha / kf_beta / kf_spread / kf_zscore
```

---

## Technologies

- **Python 3.11** | pandas, NumPy, SciPy
- **statsmodels** — Engle–Granger, Johansen, ADF, OLS
- **arch** — GARCH(1,1)-t
- **scikit-learn** — Random Forest, Logistic Regression
- **matplotlib / seaborn**

---

## Concepts demonstrated

- Time-series econometrics: cointegration, stationarity, Engle–Granger, Johansen
- State-space models: Kalman filter for time-varying parameters — and its failure mode
- Statistical inference: Benjamini–Hochberg FDR, stationary block bootstrap
- Volatility modelling: GARCH with Student-t innovations
- Machine learning: classification, precision/recall trade-off, walk-forward evaluation
- Portfolio construction: ERC / risk parity, box-constrained optimisation, diversification ratio
- Backtesting done honestly: capital-based P&L, full-series Sharpe, cost sensitivity, IS vs OOS

---

*Data: Yahoo Finance (yfinance), adjusted close | Universe: NIFTY-100 constituents | Period: Jan 2015 – Dec 2024*
