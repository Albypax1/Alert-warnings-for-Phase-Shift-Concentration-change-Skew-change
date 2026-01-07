# !pip install streamlit scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import optimize
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# -------------------- Constants & helpers --------------------
TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719  # Kimberley
TIMEZONE = "Africa/Johannesburg"

def wrap_angle(x):
    return np.mod(x, TWO_PI)

def doy_today():
    today = pd.Timestamp.today(tz=TIMEZONE)
    return int(today.dayofyear)

# -------------------- SS-GvM core --------------------

def normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu, n_grid=4096):
    grid = np.linspace(0, TWO_PI, n_grid, endpoint=False)
    expo = (
        k1*np.cos(grid-mu1)
        + k2*np.cos(2*(grid-mu2))
        + np.log(np.maximum(1 + eta*np.sin(grid-nu), 1e-12))
    )
    m = np.max(expo)
    return (TWO_PI/n_grid)*np.exp(m)*np.sum(np.exp(expo-m))


def logpdf_ssgvm(x, params):
    mu1, mu2, k1, k2, eta, nu = params
    Z = normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu)
    xw = wrap_angle(x)
    return (
        k1*np.cos(xw-mu1)
        + k2*np.cos(2*(xw-mu2))
        + np.log(np.maximum(1 + eta*np.sin(xw-nu), 1e-12))
        - np.log(Z)
    )


def fit_ssgvm_mle_all_starts(data, n_starts=30):
    data = wrap_angle(np.asarray(data))
    def neg_ll(p):
        val = np.sum(logpdf_ssgvm(data, p))
        return -val if np.isfinite(val) else 1e12
    bounds = optimize.Bounds([0,0,0,0,-0.999,0],[TWO_PI,TWO_PI,35.0,35.0,0.999,TWO_PI])
    best_x, best_fun = None, np.inf
    for _ in range(n_starts):
        init = np.array([
            np.random.rand()*TWO_PI, np.random.rand()*TWO_PI,
            np.random.gamma(2.0,1.0), np.random.gamma(2.0,1.0),
            np.tanh(np.random.randn()), np.random.rand()*TWO_PI
        ])
        res = optimize.minimize(neg_ll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 6000})
        if res.fun < best_fun:
            best_fun, best_x = res.fun, res.x
    return best_x, -best_fun

# -------------------- Data ingestion --------------------

def fetch_hotday_phases(start_date: str, end_date: str, q=0.90):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": start_date, "end_date": end_date,
            "daily": ["temperature_2m_max"],
            "timezone": TIMEZONE
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()["daily"]
        df = pd.DataFrame({"date": data["time"], "tmax": data["temperature_2m_max"]})
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear
        thr = np.quantile(df["tmax"].dropna(), q)
        df["hot"] = (df["tmax"] >= thr).astype(int)
        phi = TWO_PI * (df[df["hot"]==1]["doy"].values - 1) / 365.0
        return phi, df
    except Exception:
        # Fallback: synthetic empty frame
        dates = pd.date_range(start_date, end_date, freq='D')
        df = pd.DataFrame({'date': dates})
        df['doy'] = df['date'].dayofyear
        df['tmax'] = np.nan
        df['hot'] = 0
        phi = np.array([])
        return phi, df

# -------------------- Streamlit UI --------------------
st.title("Hybrid SS‑GvM + ML Hot‑Day Simulation (7‑day horizon)")

st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Data start", date(2019,1,1))
end_date   = st.sidebar.date_input("Data end", date.today())
hot_q      = st.sidebar.slider("Hot‑day quantile", 0.80, 0.99, 0.90)
n_starts   = st.sidebar.slider("MLE multi‑starts", 10, 60, 30, step=5)
n_sims     = st.sidebar.slider("Monte Carlo simulations", 500, 10000, 3000, step=500)
blend_w    = st.sidebar.slider("Blend weight SS‑GvM vs ML", 0.0, 1.0, 0.5)
model_type = st.sidebar.selectbox("ML model", ["GradientBoosting", "RandomForest"])

# -------------------- Fit SS‑GvM --------------------
phi_samples, df_temp = fetch_hotday_phases(str(start_date), str(end_date), q=hot_q)
st.write(f"Hot‑day samples: {len(phi_samples)}")
params, ll = fit_ssgvm_mle_all_starts(phi_samples, n_starts=n_starts)
mu1, mu2, k1, k2, eta, nu = params

# SS‑GvM seasonal probabilities for all DOYs
all_days = np.arange(1, 366)
theta = 2*np.pi*all_days/365.0
Z = normalizer_ssgvm(*params)
density = np.exp(np.clip(k1*np.cos(theta-mu1) + k2*np.cos(2*(theta-mu2)), -700, 700)) \
          * np.maximum(1.0 + eta*np.sin(theta - nu), 1e-12) / Z
hotprob_ssgvm = density/density.sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(all_days, hotprob_ssgvm, color="crimson", lw=2, label="SS‑GvM HotProb")
ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability")
ax.set_title("SS‑GvM daily hot‑day probability (Kimberley)")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# -------------------- Feature engineering for ML --------------------
df_temp_ml = df_temp.dropna(subset=['tmax']).copy()
if len(df_temp_ml) >= 30 and df_temp_ml['hot'].nunique() > 1:
    df_temp_ml['sin_doy'] = np.sin(2*np.pi*df_temp_ml['doy']/365.0)
    df_temp_ml['cos_doy'] = np.cos(2*np.pi*df_temp_ml['doy']/365.0)
    df_temp_ml['tmax_roll3'] = df_temp_ml['tmax'].rolling(3, min_periods=1).mean()
    df_temp_ml['tmax_roll7'] = df_temp_ml['tmax'].rolling(7, min_periods=1).mean()
    df_temp_ml['tmax_diff1'] = df_temp_ml['tmax'].diff().fillna(0.0)
    # Leap-year safe indexing for SS‑GvM feature
    doy_idx = (df_temp_ml['doy'].values - 1) % 365
    df_temp_ml['hotprob_ssgvm'] = hotprob_ssgvm[doy_idx]

    features = ['sin_doy','cos_doy','tmax','tmax_roll3','tmax_roll7','tmax_diff1','hotprob_ssgvm']
    X = df_temp_ml[features].values
    y = df_temp_ml['hot'].values.astype(int)

    if model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
else:
    model = None

# -------------------- Monte Carlo: simulate next 7 days temps & hot days --------------------
doy0 = doy_today()
future_doys = np.array([((doy0 + i - 1) % 365) + 1 for i in range(1, 8)])
ssgvm_probs_7 = hotprob_ssgvm[future_doys - 1]

# Build climatology by DOY from historical tmax
clim_by_doy = df_temp.groupby('doy')['tmax'].agg(['mean','std']).reindex(all_days).fillna(method='ffill').fillna(method='bfill')
mu_clim = clim_by_doy['mean'].values
sd_clim = np.clip(clim_by_doy['std'].values, 0.5, None)  # avoid zeros

# Simulate n_sims paths: sample temps from Normal(mean_DOY, sd_DOY), compute ML proba (if model), blend with SS-GvM, sample hot indicators
sim_probs_daily = np.zeros((n_sims, 7))
sim_hot_daily   = np.zeros((n_sims, 7), dtype=bool)

for s in range(n_sims):
    temps7 = np.random.normal(mu_clim[future_doys - 1], sd_clim[future_doys - 1])
    sin_d = np.sin(2*np.pi*future_doys/365.0)
    cos_d = np.cos(2*np.pi*future_doys/365.0)
    t_roll3 = pd.Series(temps7).rolling(3, min_periods=1).mean().values
    t_roll7 = pd.Series(temps7).rolling(7, min_periods=1).mean().values
    t_diff1 = np.diff(np.concatenate([[temps7[0]], temps7]))
    X7 = np.column_stack([sin_d, cos_d, temps7, t_roll3, t_roll7, t_diff1, ssgvm_probs_7])
    if model is not None:
        ml_probs7 = model.predict_proba(X7)[:,1]
    else:
        # fallback: climatology from hot frequency
        hot_freq = df_temp.groupby('doy')['hot'].mean().reindex(all_days).fillna(df_temp['hot'].mean()).values
        ml_probs7 = hot_freq[future_doys - 1]
    hybrid_p = np.clip(blend_w*ssgvm_probs_7 + (1.0 - blend_w)*ml_probs7, 0.0, 1.0)
    sim_probs_daily[s] = hybrid_p
    sim_hot_daily[s]   = np.random.rand(7) < hybrid_p

# Aggregate statistics
p_daily_mean = sim_probs_daily.mean(axis=0)
p_daily_med  = np.median(sim_probs_daily, axis=0)
p_daily_lo, p_daily_q1, p_daily_q3, p_daily_hi = np.percentile(sim_probs_daily, [5,25,75,95], axis=0)
count_hot = sim_hot_daily.sum(axis=1)
count_hist = np.bincount(count_hot, minlength=8)

# Fan chart of daily hot probability
fig2, ax2 = plt.subplots(figsize=(10,4))
x = np.arange(1, 8)
ax2.fill_between(x, p_daily_lo, p_daily_hi, color='orange', alpha=0.20, label='5–95% interval')
ax2.fill_between(x, p_daily_q1, p_daily_q3, color='orange', alpha=0.35, label='25–75% (IQR)')
ax2.plot(x, p_daily_med, color='darkred', lw=2, label='Median')
ax2.plot(x, p_daily_mean, color='saddlebrown', lw=1.5, ls='--', label='Mean')
ax2.set_xlabel("Days ahead"); ax2.set_ylabel("Probability of hot day")
ax2.set_title("Simulated hot‑day probability (7 days)")
ax2.legend(); ax2.grid(alpha=0.3)
st.pyplot(fig2)

# Distribution of total hot days in 7‑day horizon
fig3, ax3 = plt.subplots(figsize=(8,3.5))
ax3.bar(np.arange(0,8), count_hist/count_hist.sum(), color='teal', alpha=0.7)
ax3.set_xlabel("Number of hot days in 7 days"); ax3.set_ylabel("Probability")
ax3.set_title("Distribution of hot‑day counts (Monte Carlo)")
ax3.grid(alpha=0.3, axis='y')
st.pyplot(fig3)

# Summary table
calendar_doy = future_doys
summary_df = pd.DataFrame({
    'DayAhead': x,
    'CalendarDOY': calendar_doy,
    'SSGvM_Prob': np.round(ssgvm_probs_7, 4),
    'HybridProb_mean': np.round(p_daily_mean, 4),
    'HybridProb_median': np.round(p_daily_med, 4),
    'HybridProb_p05': np.round(p_daily_lo, 4),
    'HybridProb_p25': np.round(p_daily_q1, 4),
    'HybridProb_p75': np.round(p_daily_q3, 4),
    'HybridProb_p95': np.round(p_daily_hi, 4),
})
st.dataframe(summary_df)

# -------------------- Downloads --------------------
st.subheader("Downloads")
csv_summary = summary_df.to_csv(index=False).encode('utf-8')
st.download_button("Download hot‑day probability summary (CSV)", data=csv_summary, file_name="hotday_prob_summary_7d.csv", mime="text/csv")
raw_paths = pd.DataFrame(sim_hot_daily[:50].astype(int), columns=[f"Day{i}" for i in range(1,8)])
csv_paths = raw_paths.to_csv(index=False).encode('utf-8')
st.download_button("Download raw hot‑day paths (CSV)", data=csv_paths, file_name="hotday_paths_7d.csv", mime="text/csv")
