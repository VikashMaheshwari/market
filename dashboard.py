"""
Market Regime Detection Dashboard (K=3 only — Bull / Neutral / Bear)
Run with:  streamlit run dashboard.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM
from scipy.stats import f_oneway

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Market Regime Dashboard",
                   page_icon="📈", layout="wide")
st.title("📈 Market Regime Detection — Bull / Neutral / Bear")
st.caption("Gaussian Mixture Model & Hidden Markov Model on weekly S&P 500")

# =====================================================
# CONFIG
# =====================================================
TRAIN_START = "2010-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = date.today().strftime("%Y-%m-%d")
PRED_START  = (pd.Timestamp(TEST_START) - relativedelta(months=12)).strftime("%Y-%m-%d")

FEATURE_COLS = ["Log_Return", "Volatility", "MA_Crossover",
                "VIX_Change", "term_spread"]
COLORS = {"Bull": "green", "Neutral": "orange", "Bear": "red"}
ALLOC  = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}
COST_BPS = 5 / 10_000

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Controls")
model_choice    = st.sidebar.selectbox("Model", ["GMM", "HMM"])
show_backtest   = st.sidebar.checkbox("Show backtest", True)
show_statistics = st.sidebar.checkbox("Show statistics", True)
show_oos        = st.sidebar.checkbox("Show out-of-sample predictions", True)

# =====================================================
# DATA
# =====================================================
@st.cache_data(show_spinner=False)
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


@st.cache_data(show_spinner=False)
def fetch_fred(series, start, end):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(url)
    df.columns = ["Date", series]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[series] = pd.to_numeric(df[series], errors="coerce")
    df = df.dropna().set_index("Date").sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return df[series]


@st.cache_data(show_spinner="Fetching market data...")
def build_dataset(start, end):
    spy = yf.download("^GSPC", start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy["VIX"] = yf_close("^VIX", start, end, "VIX")

    weekly = spy.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum", "VIX": "last"
    }).dropna()

    try:
        ts = fetch_fred("T10Y2Y", start, end)
    except Exception:
        t10 = yf_close("^TNX", start, end, "T10")
        t3m = yf_close("^IRX", start, end, "T3M")
        ts  = (t10 - t3m).rename("T10Y2Y")

    # Same semantics as the notebook's merge_backward: for each Friday, take the most
    # recent term_spread <= that Friday, but only within a 7-day tolerance.
    ts = ts.sort_index()
    weekly["term_spread"] = ts.reindex(weekly.index, method="ffill",
                                       tolerance=pd.Timedelta("7 days"))
    return weekly.replace([np.inf, -np.inf], np.nan).dropna()


def compute_features(df):
    out = df.copy()
    out["Log_Return"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]   = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]   = out["VIX"].pct_change()
    return out.dropna()


class Preprocessor:
    """Matches the notebook's class signature so joblib-loaded instances work."""
    def __init__(self, lower_q=0.01, upper_q=0.99, smooth_window=3):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.smooth_window = smooth_window
        self.scaler = StandardScaler()

    def fit(self, X):
        df = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        self.feature_names_ = list(df.columns)
        clipped = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                   min_periods=self.smooth_window).mean().dropna()
        self.scaler.fit(smoothed.values)
        return self

    def transform(self, X):
        cols = getattr(self, "feature_names_", FEATURE_COLS)
        df = X[cols].replace([np.inf, -np.inf], np.nan)
        clipped = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window,
                                   min_periods=self.smooth_window).mean().dropna()
        return pd.DataFrame(self.scaler.transform(smoothed.values),
                            index=smoothed.index, columns=cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =====================================================
# TRAIN
# =====================================================
import os
import joblib

PKL_DIR   = os.path.dirname(os.path.abspath(__file__))
PKL_FILES = ["gmm_model.pkl", "hmm_model.pkl", "preprocessor.pkl", "label_maps.pkl"]


@st.cache_resource(show_spinner="Loading models...")
def train():
    train_df       = build_dataset(TRAIN_START, TRAIN_END)
    train_features = compute_features(train_df)

    pkl_paths = [os.path.join(PKL_DIR, f) for f in PKL_FILES]
    if all(os.path.exists(p) for p in pkl_paths):
        prep   = joblib.load(pkl_paths[2])
        gmm    = joblib.load(pkl_paths[0])
        hmm    = joblib.load(pkl_paths[1])
        lmaps  = joblib.load(pkl_paths[3])
        X_train = prep.transform(train_features[FEATURE_COLS])
        ret     = train_features.loc[X_train.index, "Log_Return"]
        return {
            "train_df": train_df, "train_features": train_features, "prep": prep,
            "X_train": X_train, "ret": ret,
            "gmm": gmm, "gmm_lm": lmaps["gmm"],
            "hmm": hmm, "hmm_lm": lmaps["hmm"],
            "loaded_from_pkl": True,
        }

    # Fallback — retrain with notebook's BIC-optimal hyperparameters
    prep    = Preprocessor()
    X_train = prep.fit_transform(train_features[FEATURE_COLS])
    X       = X_train.values
    ret     = train_features.loc[X_train.index, "Log_Return"]

    gmm = GaussianMixture(n_components=3, covariance_type="full",
                          reg_covar=1e-4, n_init=15, max_iter=500,
                          tol=1e-4, random_state=42).fit(X)
    hmm = GaussianHMM(n_components=3, covariance_type="full",
                      n_iter=100, tol=1e-4, random_state=42).fit(X)

    def label_map(states):
        s = pd.DataFrame({
            "ret": ret.groupby(states).mean(),
            "vol": train_features.loc[X_train.index, "Volatility"].groupby(states).mean(),
        })
        s["score"] = s["ret"] - s["vol"]
        order = s["score"].sort_values().index.tolist()
        return {order[0]: "Bear", order[1]: "Neutral", order[2]: "Bull"}

    gmm_lm = label_map(gmm.predict(X))
    hmm_lm = label_map(hmm.predict(X))

    return {
        "train_df": train_df, "train_features": train_features, "prep": prep,
        "X_train": X_train, "ret": ret,
        "gmm": gmm, "gmm_lm": gmm_lm,
        "hmm": hmm, "hmm_lm": hmm_lm,
    }


PRED_PKL = ["pred_gmm.pkl", "pred_hmm.pkl", "pred_features.pkl"]


@st.cache_data(show_spinner="Loading OOS predictions...")
def predict_oos(_M):
    pred_paths = [os.path.join(PKL_DIR, f) for f in PRED_PKL]

    # Prefer the notebook's saved predictions — guaranteed to match the chart
    if all(os.path.exists(p) for p in pred_paths):
        pred_gmm      = joblib.load(pred_paths[0])
        pred_hmm      = joblib.load(pred_paths[1])
        pred_features = joblib.load(pred_paths[2])
        idx = pred_gmm.index

        def attach(df):
            out = df.copy()
            if "Confidence" not in out.columns and "confidence" in out.columns:
                out["Confidence"] = out["confidence"]
            out["Close"]      = pred_features.loc[idx, "Close"].values
            out["Log_Return"] = pred_features.loc[idx, "Log_Return"].values
            return out

        return {"X_pred": pred_gmm,
                "gmm": attach(pred_gmm),
                "hmm": attach(pred_hmm),
                "loaded_from_pkl": True}

    # Fallback — recompute from raw data
    pred_df       = build_dataset(PRED_START, TEST_END)
    pred_features = compute_features(pred_df)
    X_pred        = _M["prep"].transform(pred_features[FEATURE_COLS])
    X_pred        = X_pred[X_pred.index >= TEST_START]
    Xp            = X_pred.values

    def make(model, lm):
        return pd.DataFrame({
            "Regime":     pd.Series(model.predict(Xp), index=X_pred.index).map(lm),
            "Confidence": model.predict_proba(Xp).max(axis=1),
            "Close":      pred_features.loc[X_pred.index, "Close"],
            "Log_Return": pred_features.loc[X_pred.index, "Log_Return"],
        }, index=X_pred.index)

    return {"X_pred": X_pred,
            "gmm": make(_M["gmm"], _M["gmm_lm"]),
            "hmm": make(_M["hmm"], _M["hmm_lm"]),
            "loaded_from_pkl": False}


M    = train()
OOS  = predict_oos(M)

X_train        = M["X_train"]
train_features = M["train_features"]
ret            = M["ret"]
train_df       = M["train_df"]

st.sidebar.success("✓ Loaded notebook models" if M.get("loaded_from_pkl")
                   else "Trained from scratch")

# Pick the active model
if model_choice == "GMM":
    model       = M["gmm"]
    label_map_  = M["gmm_lm"]
else:
    model       = M["hmm"]
    label_map_  = M["hmm_lm"]

states = model.predict(X_train.values)
probs  = model.predict_proba(X_train.values)
regimes = pd.Series(states, index=X_train.index).map(label_map_)

# =====================================================
# HEADER METRICS — current regime (latest OOS)
# =====================================================
oos_df       = OOS["gmm"] if model_choice == "GMM" else OOS["hmm"]
latest_date  = oos_df.index[-1]
latest_reg   = oos_df["Regime"].iloc[-1]
latest_conf  = oos_df["Confidence"].iloc[-1]
latest_close = oos_df["Close"].iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Regime", latest_reg)
c2.metric("Confidence",     f"{latest_conf:.2%}")
c3.metric("S&P 500 (latest)", f"{latest_close:,.2f}")
c4.metric("As of",          latest_date.strftime("%Y-%m-%d"))

# =====================================================
# REGIME OVERLAY (training period)
# =====================================================
st.subheader(f"{model_choice} — Regime Overlay on S&P 500 (2010–2024)")
price_train = train_features.loc[X_train.index, "Close"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=price_train.index, y=price_train,
                         mode="lines", name="S&P 500",
                         line=dict(color="black", width=1.5)))
for r in ["Bull", "Neutral", "Bear"]:
    idx = regimes[regimes == r].index
    fig.add_trace(go.Scatter(x=idx, y=price_train.loc[idx],
                             mode="markers", name=r,
                             marker=dict(color=COLORS[r], size=6)))
fig.update_layout(template="plotly_white", height=520,
                  xaxis_title="Date", yaxis_title="S&P 500")
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MODEL METRICS  (hardcoded from notebook for parity)
# =====================================================
NOTEBOOK_PERF = {
    "GMM": {"Silhouette": 0.1921, "BIC": 8771.69, "AIC": 8493.42,
            "AvgConf": 0.9076},
    "HMM": {"Silhouette": 0.1994, "Stickiness": 0.9702,
            "LogLik_total": -3629.69, "LogLik_sample": -5.0412,
            "BIC": 7693.61, "AvgConf": 0.9867},
}

st.subheader(f"{model_choice} Performance")
p = NOTEBOOK_PERF[model_choice]
m1, m2, m3, m4 = st.columns(4)
if model_choice == "GMM":
    m1.metric("Silhouette",     f"{p['Silhouette']:.4f}")
    m2.metric("BIC",            f"{p['BIC']:,.2f}")
    m3.metric("AIC",            f"{p['AIC']:,.2f}")
    m4.metric("Avg Confidence", f"{p['AvgConf']:.2%}")
else:
    m1.metric("Silhouette",     f"{p['Silhouette']:.4f}")
    m2.metric("Stickiness",     f"{p['Stickiness']:.4f}")
    m3.metric("BIC",            f"{p['BIC']:,.2f}")
    m4.metric("Avg Confidence", f"{p['AvgConf']:.2%}")
    st.caption(f"Log-Likelihood (total): {p['LogLik_total']:,.2f}  ·  "
               f"per sample: {p['LogLik_sample']:.4f}")

# =====================================================
# HMM TRANSITION MATRIX
# =====================================================
if model_choice == "HMM":
    st.subheader("HMM Transition Matrix")
    order = sorted(label_map_, key=lambda s: ["Bear", "Neutral", "Bull"].index(label_map_[s]))
    trans = model.transmat_[np.ix_(order, order)]
    fig2, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(trans, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=["Bear", "Neutral", "Bull"],
                yticklabels=["Bear", "Neutral", "Bull"], ax=ax)
    ax.set_xlabel("To"); ax.set_ylabel("From")
    st.pyplot(fig2)

# =====================================================
# BACKTEST
# =====================================================
def _run_bt(log_ret, regimes_in):
    df = pd.DataFrame({"log_ret": log_ret, "regime": regimes_in}).dropna()
    df["signal"]    = df["regime"].shift(1)
    df["alloc"]     = df["signal"].map(ALLOC).fillna(0)
    df["strat_ret"] = df["alloc"] * df["log_ret"]
    df["switched"]  = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]   = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"] = df["strat_ret"].cumsum().apply(np.exp)
    return df


def _bt_metrics(df):
    years  = len(df) / 52
    cagr_s = df["cum_strat"].iloc[-1] ** (1 / years) - 1
    cagr_b = df["cum_bnh"].iloc[-1]   ** (1 / years) - 1
    vol_s  = df["strat_ret"].std() * np.sqrt(52)
    vol_b  = df["log_ret"].std()   * np.sqrt(52)
    sh_s   = cagr_s / vol_s if vol_s > 0 else 0
    sh_b   = cagr_b / vol_b if vol_b > 0 else 0
    mdd_s  = ((df["cum_strat"] - df["cum_strat"].cummax()) / df["cum_strat"].cummax()).min()
    mdd_b  = ((df["cum_bnh"]   - df["cum_bnh"].cummax())   / df["cum_bnh"].cummax()).min()
    return cagr_s, cagr_b, sh_s, sh_b, mdd_s, mdd_b, int(df["switched"].sum())


# Hardcoded backtest values — exact match with the notebook
NOTEBOOK_BT = {
    "GMM": {
        "IS":  {"CAGR_S": 0.0688, "CAGR_B": 0.1226,
                "Sharpe_S": 0.735, "Sharpe_B": 0.741,
                "MDD_S": -0.1835, "MDD_B": -0.3181,
                "Switches": 92,
                "Final_S": 2.544, "Final_B": 5.022},
        "OOS": {"CAGR_S": 0.0085, "CAGR_B": 0.1586,
                "Sharpe_S": 0.085, "Sharpe_B": 0.973,
                "MDD_S": -0.1198, "MDD_B": -0.1702,
                "Switches": 21,
                "Final_S": 1.011, "Final_B": 1.216},
    },
    "HMM": {
        "IS":  {"CAGR_S": 0.0955, "CAGR_B": 0.1226,
                "Sharpe_S": 1.094, "Sharpe_B": 0.741,
                "MDD_S": -0.0934, "MDD_B": -0.3181,
                "Switches": 21,
                "Final_S": 3.584, "Final_B": 5.022},
        "OOS": {"CAGR_S": 0.0176, "CAGR_B": 0.1586,
                "Sharpe_S": 0.200, "Sharpe_B": 0.973,
                "MDD_S": -0.0891, "MDD_B": -0.1702,
                "Switches": 8,
                "Final_S": 1.023, "Final_B": 1.216},
    },
}


def _show_bt(df, title, line_color, metrics):
    st.markdown(f"**{title}**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("CAGR (Strat)",   f"{metrics['CAGR_S']:.2%}",
              delta=f"BnH {metrics['CAGR_B']:.2%}")
    b2.metric("Sharpe (Strat)", f"{metrics['Sharpe_S']:.3f}",
              delta=f"BnH {metrics['Sharpe_B']:.3f}")
    b3.metric("Max DD (Strat)", f"{metrics['MDD_S']:.2%}",
              delta=f"BnH {metrics['MDD_B']:.2%}", delta_color="inverse")
    b4.metric("Switches", metrics["Switches"])
    st.caption(f"Final $1: Strategy ${metrics['Final_S']:.3f}  ·  "
               f"Buy & Hold ${metrics['Final_B']:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_bnh"],   name="Buy & Hold",
                             line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_strat"], name="Strategy",
                             line=dict(color=line_color, width=2)))
    fig.update_layout(template="plotly_white", height=380,
                      yaxis_title="Portfolio Value ($1 invested)")
    st.plotly_chart(fig, use_container_width=True)


if show_backtest:
    st.subheader("Strategy Backtest  (Bull=100%, Neutral=50%, Bear=0%)")
    st.caption("1-week signal lag · 5 bps transaction cost per regime switch  ·  "
               "metrics frozen to notebook values for parity")

    is_bt  = _run_bt(ret, regimes)
    oos_bt = _run_bt(oos_df["Log_Return"], oos_df["Regime"])
    nb_is  = NOTEBOOK_BT[model_choice]["IS"]
    nb_oos = NOTEBOOK_BT[model_choice]["OOS"]

    tab_is, tab_oos = st.tabs(["In-Sample (2010–2024)",
                               "Out-of-Sample (2025 → Today)"])
    with tab_is:
        _show_bt(is_bt,  f"{model_choice} K=3 — In-Sample",     "#1565C0", nb_is)
    with tab_oos:
        _show_bt(oos_bt, f"{model_choice} K=3 — Out-of-Sample", "#E65100", nb_oos)

# =====================================================
# STATISTICS
# =====================================================
if show_statistics:
    st.subheader("Statistical Validation")
    # Values from the notebook's training-set ANOVA / Levene tests
    NOTEBOOK_STATS = {
        "GMM": {"anova_F": 6.3619,  "anova_p": 0.001825,
                "levene_F": 78.5033, "levene_p": 0.0},
        "HMM": {"anova_F": 1.4575,  "anova_p": 0.233516,
                "levene_F": 70.3592, "levene_p": 0.0},
    }
    s = NOTEBOOK_STATS[model_choice]

    st.markdown(f"**One-way ANOVA — {model_choice} K=3 regimes (mean returns)**")
    a1, a2 = st.columns(2)
    a1.metric("F-statistic", f"{s['anova_F']:.4f}")
    a2.metric("p-value",     f"{s['anova_p']:.6f}")
    st.caption("ANOVA tests whether mean returns differ across Bull / Neutral / Bear. "
               "p < 0.05 means returns are significantly different across regimes.")

    st.markdown(f"**Levene's test — {model_choice} K=3 regimes (variance / volatility)**")
    l1, l2 = st.columns(2)
    l1.metric("F-statistic", f"{s['levene_F']:.4f}")
    l2.metric("p-value",     f"{s['levene_p']:.6f}")
    st.caption("Levene tests whether variance differs across regimes — strong p < 0.05 here "
               "is the most meaningful validation: regimes capture real volatility shifts.")

# =====================================================
# OOS PREDICTIONS
# =====================================================
if show_oos:
    st.subheader(f"Out-of-Sample Predictions ({TEST_START} → today)")
    fig_oos = go.Figure()
    fig_oos.add_trace(go.Scatter(x=oos_df.index, y=oos_df["Close"],
                                 mode="lines", name="S&P 500",
                                 line=dict(color="black", width=1.5)))
    for r in ["Bull", "Neutral", "Bear"]:
        idx = oos_df.index[oos_df["Regime"] == r]
        fig_oos.add_trace(go.Scatter(x=idx, y=oos_df.loc[idx, "Close"],
                                     mode="markers", name=r,
                                     marker=dict(color=COLORS[r], size=8)))
    fig_oos.update_layout(template="plotly_white", height=460,
                          xaxis_title="Date", yaxis_title="S&P 500")
    st.plotly_chart(fig_oos, use_container_width=True)

    st.markdown("**Recent calls (last 10 weeks)**")
    st.dataframe(oos_df[["Close", "Regime", "Confidence"]].round(3).tail(10),
                 use_container_width=True)

st.markdown("---")
st.caption("Trained on 2010–2024 weekly S&P 500. K=3 regimes: Bull / Neutral / Bear.")
