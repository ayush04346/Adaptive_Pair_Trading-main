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
TC_BPS   = 10             # legacy one-way cost (NB06 only); superseded below
TC       = TC_BPS / 10_000

# Round-trip transaction cost for a two-leg pair position, expressed in bps of
# the gross capital deployed (both legs, entry + exit). One entry or one exit
# costs half of this. 10 bps is optimistic; 60 bps is a realistic Indian
# long-short round trip once STT, stamp duty, exchange + SEBI turnover fees,
# GST and SLB stock-borrow cost on the short leg are included. Backtests sweep
# all three levels and report every one — never pick the flattering number.
COST_BPS_ROUND_TRIP        = 30
COST_BPS_ROUND_TRIP_LEVELS = (10, 30, 60)


def pair_backtest(price_a, price_b, beta, zscore,
                  entry_z=ENTRY_Z, exit_z=EXIT_Z,
                  cost_bps_round_trip=COST_BPS_ROUND_TRIP):
    """The ONE pair backtester — import it, do not re-implement per notebook.

    Economic position: long 1 unit of A, short ``beta`` units of B, entered when
    ``zscore`` pierces +/- ``entry_z`` and closed when it reverts inside
    +/- ``exit_z``.

    P&L is a return **on deployed capital**, not on the spread level:
        gross exposure per unit  = price_a + |beta| * price_b   (both legs)
        daily currency P&L/unit  = pos.shift(1) * (dA - beta * dB)
        daily return             = currency P&L / gross exposure locked at entry
    so a daily return is a small, bounded fraction and ``(1 + ret).cumprod()``
    can never cross zero. The function asserts ``min(ret) > -1``.

    ``beta`` may be a scalar (OLS, NB09) or a Series (Kalman, NB10); a Series is
    lagged one day so the hedge ratio is point-in-time.

    Costs: |pos.diff()| * (cost_bps_round_trip / 2) bps of capital. An entry and
    an exit each move |pos.diff()| by 1 and each cost half a round trip; both
    legs sit inside ``gross`` so this is a per-leg, notional-weighted charge.

    Returns a DataFrame indexed like the prices with columns:
        pos            position held (-1/0/+1)
        ret            daily return on capital, net of cost   (flat days = 0.0)
        ret_gross      daily return on capital, before cost
        in_market      1.0 on days a position is held
        turnover       |pos.diff()|  (for cost / trade accounting)
    """
    import numpy as np
    import pandas as pd

    price_a = pd.Series(price_a).astype(float)
    price_b = pd.Series(price_b).astype(float)
    idx = price_a.index
    z = pd.Series(zscore).reindex(idx)

    if np.isscalar(beta):
        beta_pnl = float(beta)
        abs_beta = abs(float(beta))
    else:
        beta_pnl = pd.Series(beta).reindex(idx).shift(1)
        abs_beta = beta_pnl.abs()

    # ── position state machine (enter at +/-entry_z, exit inside +/-exit_z) ──
    zv = z.to_numpy(dtype=float)
    pos = np.zeros(len(zv))
    cur = 0.0
    for i in range(len(zv)):
        v = zv[i]
        if np.isnan(v):
            pos[i] = 0.0
            continue
        if cur == 0.0:
            if v > entry_z:
                cur = -1.0
            elif v < -entry_z:
                cur = 1.0
        elif cur == 1.0 and v > -exit_z:
            cur = 0.0
        elif cur == -1.0 and v < exit_z:
            cur = 0.0
        pos[i] = cur
    pos = pd.Series(pos, index=idx)

    d_spread = price_a.diff() - beta_pnl * price_b.diff()
    gross    = price_a + abs_beta * price_b

    in_market = pos.ne(0)
    held      = pos.shift(1).ne(0)                 # carried a position into day t
    entry_day = in_market & ~held
    # gross exposure locked at entry, carried forward. Kept valid on the exit day
    # too (held is True there) so the last day's move is not dropped; P&L on
    # genuinely flat days is zeroed by pos.shift(1) == 0, not by nulling this.
    entry_gross = gross.where(entry_day).ffill().where(in_market | held)

    ret_gross = (pos.shift(1) * d_spread / entry_gross).fillna(0.0)

    turnover = pos.diff().abs().fillna(pos.abs())
    cost     = turnover * (cost_bps_round_trip / 2.0) / 10_000.0
    ret      = ret_gross - cost

    if float(ret.min()) <= -1.0:
        raise AssertionError(
            f'pair_backtest produced a daily return <= -100% (min={ret.min():.4f}). '
            'The capital model is broken — refusing to report performance on it.'
        )

    return pd.DataFrame({
        'pos':        pos,
        'ret':        ret,
        'ret_gross':  ret_gross,
        'in_market':  in_market.astype(float),
        'turnover':   turnover,
    })


def sharpe_ratio(ret, periods_per_year=252):
    """Annualised Sharpe over the FULL daily return series, including flat days.

    Dropping zero-P&L days and still scaling by sqrt(252) overstates Sharpe by
    1/sqrt(time-in-market). Pass the whole series; flat days belong in it.
    """
    import numpy as np
    import pandas as pd
    r = pd.Series(ret).astype(float)
    if len(r) < 2 or r.std(ddof=1) == 0 or np.isnan(r.std(ddof=1)):
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(ret):
    """Max drawdown (%) of the compounded equity curve of ``ret``.

    Raises if the equity curve touches zero — that only happens when a daily
    return is <= -100%, i.e. the P&L model is broken, and a broken model must
    not be reported as a number.
    """
    import numpy as np
    import pandas as pd
    r = pd.Series(ret).astype(float)
    cum = (1.0 + r).cumprod()
    if (cum <= 0).any():
        raise AssertionError(
            'Equity curve crossed zero (daily return <= -100%). P&L model broken.'
        )
    peak = cum.cummax()
    return float(((cum - peak) / peak).min() * 100.0)


def stationary_bootstrap(x, n_boot=5000, mean_block=20, seed=0):
    """Politis-Romano stationary block bootstrap resamples of a 1-D series.

    Returns an (n_boot, len(x)) array of resampled series. Geometric block
    lengths with mean ``mean_block``; wraps around the end. Use for dependent
    daily P&L where an ordinary i.i.d. bootstrap or a plain t-test would
    understate the standard error.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    fresh   = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = fresh[:, 0]
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], fresh[:, t], cont)
    return x[idx]


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
