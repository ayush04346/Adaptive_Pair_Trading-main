# Adaptive Pair Trading — Statistical Arbitrage on Indian Equities

**Ayush Arora | MQMS2404 | Quantitative Finance Research**

A production-grade statistical arbitrage framework applied to the **NIFTY 100** universe.
Combines cointegration theory, volatility modelling, machine learning, and portfolio construction
into a complete, end-to-end quantitative trading system.

---

## What This Project Does

This project identifies and trades **cointegrated stock pairs** on the NSE — stocks whose prices
diverge from a long-run equilibrium and tend to revert. When the gap is too wide, we short the
expensive stock and buy the cheap one. When it reverts, we close the trade.

The word **"adaptive"** refers to a key technical contribution: the hedge ratio (the proportion
of one stock to trade vs. the other) is estimated in real-time using a **Kalman Filter** rather
than a static OLS regression — eliminating look-ahead bias and allowing the strategy to track
structural shifts in the relationship over a full market cycle (2015–2025).

---

## Technical Highlights

| Component | Implementation | Why It Matters |
|-----------|---------------|----------------|
| Pair discovery | 3,081-pair universe scan with Engle-Granger + Johansen | Systematic, not cherry-picked |
| Multiple testing | Benjamini-Hochberg FDR correction | Prevents ~30 false positives at this scale |
| Mean-reversion test | Hurst exponent (R/S analysis) | Directly quantifies the property being traded |
| Adaptive hedge | Kalman Filter — time-varying β_t | Eliminates look-ahead bias, adapts to regime shifts |
| Volatility model | GARCH(1,1)-t (Student-t innovations) | Captures fat tails and volatility clustering |
| Signal filter | Random Forest classifier (OOS probabilities only) | ML gates entry, no data leakage |
| Portfolio | Equal Risk Contribution (ERC) across 5 pairs | Risk-parity allocation, no return forecasts needed |
| Validation | Walk-forward: 14 × (2yr train / 6mo test) windows | OOS Sharpe distribution, t-test significance |
| Transaction costs | 10 bps one-way (20 bps round-trip) | Realistic for NSE liquid mid-cap equities |

---

## Pipeline

```
NB00 ── Pair Selection (hierarchical 5-stage filter)
NB01 ── Data Download (NIFTY 100, yfinance, 2015-2025)
NB02 ── Cointegration Validation (EG test + OLS hedge ratio)
NB03 ── Spread Construction (OLS residual + rolling Z-score)
NB03A── Diagnostics (stationarity, market neutrality, ARCH test)
NB04 ── GARCH Modeling (GARCH(1,1)-t conditional volatility)
NB05 ── ML Signals (Random Forest reversion probability, OOS saved)
NB06 ── Backtesting (baseline vs ML-filtered vs vol-scaled)
NB07 ── Kalman Filter Adaptive Hedge (time-varying β_t)
  ║
  ╠══ NB08 ── Universe Scan (all 3,081 pairs, BH correction, Hurst, composite ranking)
  ╠══ NB09 ── Walk-Forward Validation (14 OOS windows, t-test on Sharpe)
  ╚══ NB10 ── Multi-Pair Portfolio (5 pairs, ERC weights, diversification ratio)
```

---

## Key Results

### Single-Pair (TATASTEEL / HINDALCO)
- Engle-Granger cointegration: p = 0.0006
- Hurst exponent: H < 0.5 (mean-reverting)
- Half-life: ~20 trading days
- Kalman β_t range: [0.18, 0.32] over 10 years — static OLS (β = 0.245) misses this drift

### Walk-Forward Validation (NB09)
- 14 out-of-sample windows tested
- t-test on OOS Sharpe distribution tests the null hypothesis of zero edge

### Multi-Pair Portfolio (NB10)
- 5 pairs from universe scan, sector-diversified
- ERC weighting ensures no single pair dominates portfolio risk
- Diversification ratio > 1.0 confirms risk reduction vs. individual pairs

---

## Methodological Integrity — Bugs Corrected vs. Naive Implementation

This project explicitly addresses common errors in academic pair trading implementations:

| Issue | Naive Approach | This Project |
|-------|---------------|--------------|
| ADF test | On raw price difference | On OLS residual (β·B + α removed) |
| Z-score | Full-sample mean/std (look-ahead) | Rolling 252-day (no look-ahead) |
| Spread | `A − β·B` (missing intercept) | `A − β·B − α` (full OLS residual) |
| GARCH | Zero-Mean + Normal | Constant-Mean + Student-t |
| ML evaluation | RF evaluated on Logit probs (bug) | Correct model evaluated on correct probs |
| ML in backtest | `np.random.uniform()` fallback | Walk-forward OOS probabilities only |
| Multiple testing | None (30+ false positives expected) | Benjamini-Hochberg FDR correction |
| Portfolio | Single pair only | 5-pair ERC portfolio |
| OOS validation | Single train/test split | 14-window walk-forward |

---

## Setup & Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook

# Recommended execution order:
#   NB01 → NB00 → NB02 → NB03 → NB03A → NB04 → NB05 → NB06 → NB07
#   NB08 → NB09 → NB10
```

### Directory structure
```
├── config.py                        # Central parameters (edit here, propagates everywhere)
├── requirements.txt
├── README.md
├── _00_Pair_Selection.ipynb
├── _01_Data_Download.ipynb
├── _02_Cointegration_Validation.ipynb
├── _03_spread_Construction.ipynb
├── _03A_Diagnostics_Final.ipynb
├── _04_GARCH_Modeling.ipynb
├── _05_ML_Signals_Reframed.ipynb
├── _06_Backtesting_Combined.ipynb
├── _07_.ipynb                        # Kalman Filter adaptive hedge ratio
├── _08_Universe_Pair_Scan.ipynb      # Full 79-stock universe scan
├── _09_Walk_Forward_Validation.ipynb # OOS validation framework
├── _10_Portfolio_Multi_Pair.ipynb    # ERC multi-pair portfolio
└── data/
    ├── prices.csv                    # 79-stock NIFTY 100 price matrix
    ├── top_pairs.csv                 # Output of NB08 (top 10 pairs)
    ├── spread_tatasteel_hindalco.csv
    ├── zscore_tatasteel_hindalco.csv
    ├── garch_vol_tatasteel_hindalco.csv
    ├── ml_prob_rf.csv
    ├── kalman_adaptive_hedge.csv
    └── cointegration_results.csv
```

---

## Technologies

- **Python 3.10** | pandas, NumPy, SciPy
- **statsmodels** — cointegration tests (EG, Johansen), OLS, VECM
- **arch** — GARCH volatility modelling
- **scikit-learn** — Random Forest, Logistic Regression, StandardScaler
- **matplotlib / seaborn** — visualisation

---

## Concepts Demonstrated

- Time-series econometrics: cointegration, stationarity, error-correction models
- State-space models: Kalman Filter for time-varying parameter estimation
- Statistical inference: multiple testing correction, bootstrap significance, t-tests
- Volatility modelling: GARCH with heavy-tailed distributions
- Machine learning: classification, precision-recall tradeoff, walk-forward evaluation
- Portfolio construction: risk parity, ERC, diversification ratio
- Backtesting: transaction costs, walk-forward validation, IS vs OOS comparison

---

*Data source: Yahoo Finance (yfinance) | Universe: NIFTY 100 | Period: Jan 2015 – Jan 2025*
