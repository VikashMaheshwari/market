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
show_features   = st.sidebar.checkbox("Show feature charts", True)
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
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, X):
        df = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        self.lower = df.quantile(0.01)
        self.upper = df.quantile(0.99)
        clipped = df.clip(self.lower, self.upper, axis=1)
        smoothed = clipped.rolling(3, min_periods=3).mean().dropna()
        self.scaler.fit(smoothed.values)
        return self
    def transform(self, X):
        df = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        clipped = df.clip(self.lower, self.upper, axis=1)
        smoothed = clipped.rolling(3, min_periods=3).mean().dropna()
        return pd.DataFrame(self.scaler.transform(smoothed.values),
                            index=smoothed.index, columns=FEATURE_COLS)
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
# MODEL METRICS
# =====================================================
st.subheader(f"{model_choice} Performance")
m1, m2, m3, m4 = st.columns(4)
sil = silhouette_score(X_train.values, states)
if model_choice == "GMM":
    m1.metric("Silhouette",     f"{sil:.4f}")
    m2.metric("BIC",            f"{model.bic(X_train.values):,.0f}")
    m3.metric("AIC",            f"{model.aic(X_train.values):,.0f}")
    m4.metric("Avg Confidence", f"{probs.max(axis=1).mean():.2%}")
else:
    stick = float(np.mean(np.diag(model.transmat_)))
    m1.metric("Silhouette",     f"{sil:.4f}")
    m2.metric("Stickiness",     f"{stick:.3f}")
    m3.metric("Log Likelihood", f"{model.score(X_train.values):,.0f}")
    m4.metric("Avg Confidence", f"{probs.max(axis=1).mean():.2%}")

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
# FEATURE CHARTS
# =====================================================
if show_features:
    st.subheader("Feature Analysis")
    sel = st.selectbox("Choose feature", FEATURE_COLS)
    fig3 = px.line(x=X_train.index, y=X_train[sel],
                   labels={"x": "Date", "y": sel},
                   title=f"{sel} (standardised)")
    fig3.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# BACKTEST
# =====================================================
if show_backtest:
    st.subheader("Strategy Backtest (Bull=100%, Neutral=50%, Bear=0%)")
    df = pd.DataFrame({"log_ret": ret, "regime": regimes}).dropna()
    df["signal"]    = df["regime"].shift(1)
    df["alloc"]     = df["signal"].map(ALLOC).fillna(0)
    df["strat_ret"] = df["alloc"] * df["log_ret"]
    df["switched"]  = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]   = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"] = df["strat_ret"].cumsum().apply(np.exp)

    years   = len(df) / 52
    cagr_s  = df["cum_strat"].iloc[-1] ** (1 / years) - 1
    cagr_b  = df["cum_bnh"].iloc[-1]   ** (1 / years) - 1
    sh_s    = (cagr_s / (df["strat_ret"].std() * np.sqrt(52))) if df["strat_ret"].std() > 0 else 0
    sh_b    = (cagr_b / (df["log_ret"].std()   * np.sqrt(52))) if df["log_ret"].std() > 0 else 0
    mdd_s   = ((df["cum_strat"] - df["cum_strat"].cummax()) / df["cum_strat"].cummax()).min()
    mdd_b   = ((df["cum_bnh"]   - df["cum_bnh"].cummax())   / df["cum_bnh"].cummax()).min()

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("CAGR (Strat)",  f"{cagr_s:.2%}", delta=f"BnH {cagr_b:.2%}")
    b2.metric("Sharpe (Strat)", f"{sh_s:.3f}",  delta=f"BnH {sh_b:.3f}")
    b3.metric("Max DD (Strat)", f"{mdd_s:.2%}", delta=f"BnH {mdd_b:.2%}", delta_color="inverse")
    b4.metric("Switches",       int(df["switched"].sum()))

    bt = go.Figure()
    bt.add_trace(go.Scatter(x=df.index, y=df["cum_bnh"],   name="Buy & Hold",
                            line=dict(color="gray", dash="dash")))
    bt.add_trace(go.Scatter(x=df.index, y=df["cum_strat"], name="Strategy",
                            line=dict(color="#1565C0", width=2)))
    bt.update_layout(template="plotly_white", height=420,
                     yaxis_title="Portfolio Value ($1 invested)")
    st.plotly_chart(bt, use_container_width=True)

# =====================================================
# STATISTICS
# =====================================================
if show_statistics:
    st.subheader("Statistical Validation (one-way ANOVA)")
    bull = ret[regimes == "Bull"]
    neu  = ret[regimes == "Neutral"]
    bear = ret[regimes == "Bear"]
    f, p = f_oneway(bull, neu, bear)
    s1, s2 = st.columns(2)
    s1.metric("ANOVA F",  f"{f:.4f}")
    s2.metric("p-value",  f"{p:.2e}")
    st.caption("p < 0.05 → returns differ significantly across regimes (model captures real signal).")

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
