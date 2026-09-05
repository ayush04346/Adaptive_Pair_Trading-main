"""
config.py — Central parameter store for Adaptive Pair Trading project.

All notebooks import from here.  Change a value once and it propagates everywhere.
"""

# ── Data ────────────────────────────────────────────────────────────────────
PRICES_FILE   = "data/prices.csv"
START_DATE    = "2015-01-01"
END_DATE      = "2025-01-01"
TOP_PAIRS_FILE = "data/top_pairs.csv"

# ── Pair Selection (NB00, NB08) ──────────────────────────────────────────────
CORR_THRESHOLD   = 0.40   # minimum pairwise return correlation
COINT_ALPHA      = 0.05   # EG cointegration significance level
ADF_ALPHA        = 0.05   # spread stationarity significance
MIN_HALF_LIFE    = 5      # minimum mean-reversion half-life (trading days)
MAX_HALF_LIFE    = 90     # maximum mean-reversion half-life (trading days)
TOP_N_PAIRS      = 10     # pairs forwarded to portfolio stage

# ── Spread / Z-Score (NB03, NB06, NB09, NB10) ──────────────────────────────
ROLL_WINDOW      = 252    # rolling lookback for Z-score (1 trading year)
ROLL_MIN_PERIODS = 60     # minimum observations before Z-score is computed


def rolling_zscore(spread, window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS):
    """The ONE Z-score definition for the whole project — import it, never
    re-derive it inline.

        z_t = (spread_t - mean(spread[t-window+1 : t])) / std(spread[t-window+1 : t])

    Look-ahead free: every point uses only the trailing ``window`` observations.
    Used by NB03 (single pair), NB09 (walk-forward, in- and out-of-sample) and
    NB10 (portfolio). NB06 consumes the saved output of NB03.

    Out-of-sample warm-up: if the test spread is shorter than ``window``, prepend
    the tail of the training spread, call this, then slice back to the test
    index. Do NOT substitute a static (fixed train mean/std) Z-score — that makes
    the in-sample vs out-of-sample comparison measure two different strategies.
    """
    import pandas as pd
    s = pd.Series(spread).astype(float)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std  = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / std


# ── Trading Rules (NB06, NB09, NB10) ───────────────────────────────────────
ENTRY_Z  = 2.0            # enter when |Z| exceeds this
EXIT_Z   = 0.5            # exit when |Z| drops below this
TC_BPS   = 10             # one-way transaction cost in basis points
TC       = TC_BPS / 10_000

# ── GARCH (NB04) ────────────────────────────────────────────────────────────
GARCH_SCALE = 100         # scale returns before fitting (numerical stability)
GARCH_P     = 1
GARCH_Q     = 1
GARCH_DIST  = "t"         # Student-t for fat tails
GARCH_MEAN  = "Constant"  # allows non-zero drift in spread returns

# ── Machine Learning (NB05) ─────────────────────────────────────────────────
ML_HORIZON   = 5          # forward horizon for reversion label (days)
ML_TEST_SIZE = 0.30       # fraction of data held out (time-ordered)
ML_THRESHOLD = 0.50       # RF probability threshold for trade entry

# ── Kalman Filter (NB07, NB10) ──────────────────────────────────────────────
KF_DELTA   = 1e-4         # process noise (higher = faster adaptation)
KF_INIT_P  = 10.0         # initial state covariance (diffuse prior)

# ── Walk-Forward Validation (NB09) ─────────────────────────────────────────
WF_TRAIN_MONTHS = 24      # training window length in months
WF_TEST_MONTHS  = 6       # test window length in months

# ── Portfolio (NB10) ────────────────────────────────────────────────────────
PORT_TOP_PAIRS  = 5       # number of pairs in live portfolio
PORT_MAX_WEIGHT = 0.50    # maximum weight per pair (ERC constraint)
PORT_MIN_WEIGHT = 0.05    # minimum weight per pair

# ── Sector Map ───────────────────────────────────────────────────────────────
# Every ticker in data/prices.csv (79 stocks) must appear here, or NB08 silently
# drops it and the candidate-pair count falls below C(79,2)=3081. NB08 asserts
# full coverage at runtime.
SECTORS = {
    "Metals":     ["TATASTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS"],
    "Financials": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS",
                   "KOTAKBANK.NS", "INDUSINDBK.NS", "YESBANK.NS", "CANBK.NS",
                   "BANKBARODA.NS", "IDFCFIRSTB.NS"],
    "IT":         ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Pharma":     ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS",
                   "CIPLA.NS", "AUROPHARMA.NS", "TORNTPHARM.NS"],
    "Healthcare": ["APOLLOHOSP.NS"],
    "FMCG":       ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
                   "DABUR.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS",
                   "UBL.NS"],
    "Consumer":   ["TITAN.NS"],
    "Cement":     ["ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS", "GRASIM.NS"],
    "Auto":       ["MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
                   "ASHOKLEY.NS", "TVSMOTOR.NS", "EICHERMOT.NS", "BHARATFORG.NS",
                   "BOSCHLTD.NS"],
    "Energy":     ["ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS",
                   "COALINDIA.NS", "TATAPOWER.NS", "NTPC.NS", "POWERGRID.NS",
                   "RELIANCE.NS", "OIL.NS"],
    "Telecom":    ["BHARTIARTL.NS"],
    "CapGoods":   ["LT.NS", "BHEL.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS"],
    "Paints":     ["ASIANPAINT.NS", "BERGEPAINT.NS"],
    "Chemicals":  ["SRF.NS", "PIDILITIND.NS", "TATACHEM.NS"],
    "NBFCs":      ["BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS",
                   "BAJAJHLDNG.NS"],
    "Insurance":  ["ICICIPRULI.NS", "ICICIGI.NS", "HDFCLIFE.NS", "HDFCAMC.NS"],
    "Ports":      ["ADANIENT.NS", "ADANIPORTS.NS"],
}
