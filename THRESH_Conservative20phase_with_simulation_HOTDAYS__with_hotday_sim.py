
# -*- coding: utf-8 -*-
# SS-GvM Monitoring + Simulation — Kimberley (ML-enhanced: Option A)
# Author: M365 Copilot
# Date: 2026-01-07

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from scipy import optimize
import requests
from sklearn.ensemble import GradientBoostingRegressor

TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719
TIMEZONE = "Africa/Johannesburg"
ML_RANDOM_STATE_DEFAULT = 20260107
RNG_SEED_DEFAULT = 20260107

def wrap_angle(x): return np.mod(x, TWO_PI)

def circ_diff(a, b):
    d = np.abs(a - b)
    return np.minimum(d, TWO_PI - d)

def angle_to_days(angle): return (angle / TWO_PI) * 365.0

def safe_se(x, default=0.08):
    try:
        xv = float(x)
        if not np.isfinite(xv) or xv <= 0: return float(default)
        return xv
    except Exception:
        return float(default)

def normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu, n_grid=4096):
    grid = np.linspace(0, TWO_PI, n_grid, endpoint=False)
    expo = (
        k1*np.cos(grid-mu1)
        + k2*np.cos(2*(grid-mu2))
        + np.log(np.maximum(1+eta*np.sin(grid-nu), 1e-12))
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
        + np.log(np.maximum(1+eta*np.sin(xw-nu), 1e-12))
        - np.log(Z)
    )

def fit_ssgvm_mle_all_starts(data, n_starts=30):
    data = wrap_angle(np.asarray(data))
    def neg_ll(p):
        val = np.sum(logpdf_ssgvm(data, p))
        return -val if np.isfinite(val) else 1e12
    bounds = optimize.Bounds([0,0,0,0,-0.999,0],[TWO_PI,TWO_PI,35.0,35.0,0.999,TWO_PI])
    best = {'x': None, 'fun': np.inf}
    for _ in range(n_starts):
        init = np.array([
            np.random.rand()*TWO_PI, np.random.rand()*TWO_PI,
            np.random.gamma(2.0,1.0), np.random.gamma(2.0,1.0),
            np.tanh(np.random.randn()), np.random.rand()*TWO_PI
        ])
        res = optimize.minimize(neg_ll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 6000})
        if res.fun < best['fun']: best = res
    return best.x, -best.fun

def numerical_hessian(func, x, eps_rel=1e-4, eps_abs=1e-6):
    x = np.asarray(x, dtype=float); n = x.size
    H = np.zeros((n, n)); f0 = func(x)
    for i in range(n):
        h_i = eps_abs + eps_rel * max(1.0, abs(x[i]))
        x_i_plus = x.copy(); x_i_plus[i] += h_i
        x_i_minus = x.copy(); x_i_minus[i] -= h_i
        f_i_plus = func(x_i_plus); f_i_minus = func(x_i_minus)
        H[i, i] = (f_i_plus - 2*f0 + f_i_minus) / (h_i**2)
        for j in range(i+1, n):
            h_j = eps_abs + eps_rel * max(1.0, abs(x[j]))
            x_pp = x.copy(); x_pp[i] += h_i; x_pp[j] += h_j
            x_pm = x.copy(); x_pm[i] += h_i; x_pm[j] -= h_j
            x_mp = x.copy(); x_mp[i] -= h_i; x_mp[j] += h_j
            x_mm = x.copy(); x_mm[i] -= h_i; x_mm[j] -= h_j
            f_pp = func(x_pp); f_pm = func(x_pm); f_mp = func(x_mp); f_mm = func(x_mm)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4*h_i*h_j)
            H[i, j] = H_ij; H[j, i] = H_ij
    return H

def se_from_hessian(neg_ll_func, params):
    try:
        H = numerical_hessian(neg_ll_func, np.array(params))
        cov = np.linalg.pinv(H + 1e-8*np.eye(len(params)))
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        return se, cov
    except Exception:
        return np.full(len(params), np.nan), np.full((len(params), len(params)), np.nan)

@st.cache_data(show_spinner=False)
def fetch_hotday_phases(start_date: str, end_date: str, q=0.90):
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
    df["doy"]  = df["date"].dt.dayofyear
    thr = np.quantile(df["tmax"].dropna(), q)
    hot = df[df["tmax"] >= thr]
    phi = TWO_PI * (hot["doy"].values - 1) / 365.0
    return phi, df

@st.cache_data(show_spinner=False)
def build_monthly_param_series(df_daily: pd.DataFrame, q=0.90, min_samples: int = 20, n_starts: int = 15):
    if df_daily.empty:
        return pd.DataFrame(columns=["month_start","mu1","mu2","k1","k2","eta","nu","hot_n"])
    thr = np.quantile(df_daily["tmax"].dropna(), q)
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"]) 
    df["doy"]  = df["date"].dt.dayofyear
    df["ym"]   = df["date"].dt.to_period("M")
    rows = []
    for ym, dfg in df.groupby("ym"):
        hot = dfg[dfg["tmax"] >= thr]
        n = len(hot)
        if n < min_samples: continue
        phi = TWO_PI * (hot["doy"].values - 1) / 365.0
        try:
            params, _ = fit_ssgvm_mle_all_starts(phi, n_starts=n_starts)
            rows.append({
                "month_start": ym.to_timestamp(),
                "mu1": float(params[0]), "mu2": float(params[1]),
                "k1": float(params[2]), "k2": float(params[3]),
                "eta": float(params[4]), "nu": float(params[5]),
                "hot_n": int(n)
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)

def add_seasonal_features(dfm: pd.DataFrame):
    df = dfm.copy(); df["month"] = df["month_start"].dt.month
    ang = 2*np.pi*(df["month"]-1)/12.0
    df["m_sin"], df["m_cos"] = np.sin(ang), np.cos(ang)
    return df

def make_lagged(dfm: pd.DataFrame, L: int = 6):
    df = dfm.copy()
    for col in ["mu1","mu2","k1","k2","eta","nu","hot_n"]:
        for l in range(1, L+1): df[f"{col}_lag{l}"] = df[col].shift(l)
    df = add_seasonal_features(df)
    return df.dropna().reset_index(drop=True)

def fit_angle_models(df_lagged: pd.DataFrame, target_col: str, random_state: int):
    y = df_lagged[target_col].values
    y_sin, y_cos = np.sin(y), np.cos(y)
    feature_cols = [c for c in df_lagged.columns if (
        c.startswith("mu1_") or c.startswith("mu2_") or c.startswith("k1_") or
        c.startswith("k2_") or c.startswith("eta_") or c.startswith("nu_") or
        c.startswith("hot_n_") or c in ["m_sin","m_cos"])]
    X = df_lagged[feature_cols].values
    gbr_sin = GradientBoostingRegressor(loss="squared_error", random_state=random_state)
    gbr_cos = GradientBoostingRegressor(loss="squared_error", random_state=random_state)
    gbr_sin.fit(X, y_sin); gbr_cos.fit(X, y_cos)
    ang_pred = np.arctan2(gbr_sin.predict(X), gbr_cos.predict(X))
    res = circ_diff(ang_pred, y)
    sigma_ang = float(np.median(res)) if np.isfinite(np.median(res)) else 0.08
    if sigma_ang <= 0: sigma_ang = 0.08
    return {"feature_cols": feature_cols, "sin_model": gbr_sin, "cos_model": gbr_cos, "sigma": sigma_ang}

def fit_scalar_models(df_lagged: pd.DataFrame, target_col: str, random_state: int):
    feature_cols = [c for c in df_lagged.columns if (
        c.startswith("mu1_") or c.startswith("mu2_") or c.startswith("k1_") or
        c.startswith("k2_") or c.startswith("eta_") or c.startswith("nu_") or
        c.startswith("hot_n_") or c in ["m_sin","m_cos"])]
    X = df_lagged[feature_cols].values; y = df_lagged[target_col].values
    gbr_med = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=random_state)
    gbr_lo  = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=random_state)
    gbr_hi  = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=random_state)
    gbr_med.fit(X, y); gbr_lo.fit(X, y); gbr_hi.fit(X, y)
    spread = np.median(np.maximum(gbr_hi.predict(X) - gbr_lo.predict(X), 1e-9))
    sigma = float(spread / 2.563) if np.isfinite(spread) else 0.10
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(y - gbr_med.predict(X))) if X.shape[0] else 0.10
    if not np.isfinite(sigma) or sigma <= 0: sigma = 0.10
    return {"feature_cols": feature_cols, "med_model": gbr_med, "lo_model": gbr_lo, "hi_model": gbr_hi, "sigma": sigma}

@st.cache_data(show_spinner=True)
def fit_ml_all(df_monthly: pd.DataFrame, random_state: int):
    df_lagged = make_lagged(df_monthly, L=6)
    if df_lagged.empty: return None
    models = {}
    for tcol in ["mu1","mu2","nu"]: models[tcol] = fit_angle_models(df_lagged, tcol, random_state=random_state)
    for tcol in ["k1","k2","eta"]: models[tcol] = fit_scalar_models(df_lagged, tcol, random_state=random_state)
    models["df_lagged"] = df_lagged
    return models

def forecast_next_month(x_row: pd.Series, models: dict):
    res = {}
    for tcol in ["mu1","mu2","nu"]:
        m = models[tcol]; X = x_row[m["feature_cols"]].values.reshape(1,-1)
        sin_pred = m["sin_model"].predict(X)[0]; cos_pred = m["cos_model"].predict(X)[0]
        ang = float(np.arctan2(sin_pred, cos_pred))
        res[tcol] = {"mean": wrap_angle(ang), "sigma": float(m["sigma"]) }
    for tcol in ["k1","k2","eta"]:
        m = models[tcol]; X = x_row[m["feature_cols"]].values.reshape(1,-1)
        med = float(m["med_model"].predict(X)[0])
        lo  = float(m["lo_model"].predict(X)[0])
        hi  = float(m["hi_model"].predict(X)[0])
        if tcol == "eta":
            med = float(np.clip(med, -0.999, 0.999))
            lo  = float(np.clip(lo,  -0.999, 0.999))
            hi  = float(np.clip(hi,  -0.999, 0.999))
        res[tcol] = {"mean": med, "q10": lo, "q90": hi, "sigma": float(m["sigma"]) }
    return res

def recursive_forecast_12(df_monthly: pd.DataFrame, models: dict):
    df_hist = df_monthly.copy(); out_rows = []
    df_lagged = make_lagged(df_hist, L=6)
    if df_lagged.empty: return pd.DataFrame()
    last = df_lagged.iloc[-1].copy(); current_month = df_hist["month_start"].iloc[-1]
    for h in range(1, 13):
        fc = forecast_next_month(last, models)
        next_month = (current_month + pd.offsets.MonthBegin(1))
        out_rows.append({
            "month_start": next_month,
            "mu1_mean": fc["mu1"]["mean"], "mu2_mean": fc["mu2"]["mean"], "nu_mean": fc["nu"]["mean"],
            "mu1_sigma": fc["mu1"]["sigma"], "mu2_sigma": fc["mu2"]["sigma"], "nu_sigma": fc["nu"]["sigma"],
            "k1_mean": fc["k1"]["mean"], "k1_q10": fc["k1"]["q10"], "k1_q90": fc["k1"]["q90"], "k1_sigma": fc["k1"]["sigma"],
            "k2_mean": fc["k2"]["mean"], "k2_q10": fc["k2"]["q10"], "k2_q90": fc["k2"]["q90"], "k2_sigma": fc["k2"]["sigma"],
            "eta_mean": fc["eta"]["mean"], "eta_q10": fc["eta"]["q10"], "eta_q90": fc["eta"]["q90"], "eta_sigma": fc["eta"]["sigma"]
        })
        for col in ["mu1","mu2","k1","k2","eta","nu","hot_n"]:
            for L in range(6,1,-1): last[f"{col}_lag{L}"] = last.get(f"{col}_lag{L-1}", np.nan)
        last["mu1_lag1"], last["mu2_lag1"], last["nu_lag1"] = out_rows[-1]["mu1_mean"], out_rows[-1]["mu2_mean"], out_rows[-1]["nu_mean"]
        last["k1_lag1"],  last["k2_lag1"],  last["eta_lag1"] = out_rows[-1]["k1_mean"], out_rows[-1]["k2_mean"], out_rows[-1]["eta_mean"]
        last["hot_n_lag1"] = last.get("hot_n_lag1", 30)
        m = int(next_month.month); ang = 2*np.pi*(m-1)/12.0
        last["m_sin"], last["m_cos"] = np.sin(ang), np.cos(ang)
        current_month = next_month
    return pd.DataFrame(out_rows)

def simulate_angles_ml(mean_monthly, sigma_monthly, mu_base, step_days, n_paths, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n_months = len(mean_monthly); grid_days = np.arange(1, n_months+1) * int(round(30.4))
    paths = np.empty((n_paths, n_months), dtype=float)
    for t in range(n_months):
        noise = rng.normal(0.0, sigma_monthly[t], size=n_paths)
        mu_sim = wrap_angle(mean_monthly[t] + noise)
        paths[:, t] = angle_to_days(circ_diff(mu_sim, mu_base))
    return grid_days, paths

def simulate_scalar_ou_ml(mean_monthly, sigma_monthly, init_val, transform: str, step_days, n_paths, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n_months = len(mean_monthly); grid_days = np.arange(1, n_months+1) * int(round(30.4))
    paths = np.empty((n_paths, n_months), dtype=float)
    def fwd(x):
        if transform == "log":   return np.log(np.maximum(x, 1e-6))
        if transform == "atanh": return np.arctanh(np.clip(x, -0.999, 0.999))
        return x
    def inv(y):
        if transform == "log":   return np.exp(y)
        if transform == "atanh": return np.tanh(y)
        return y
    x = np.full(n_paths, fwd(init_val)); alpha = 0.25
    for t in range(n_months):
        m_t = fwd(mean_monthly[t]); sigma_t = sigma_monthly[t]
        noise = rng.normal(0.0, sigma_t, size=n_paths)
        x = x + alpha*(m_t - x) + noise
        paths[:, t] = inv(x)
    return grid_days, paths

# --------------------------- UI ---------------------------
st.title("SS-GvM Monitoring + Simulation — Kimberley (ML-enhanced)")
st.sidebar.header("Configuration")
baseline_start = st.sidebar.date_input("Baseline start", date(2019,1,1))
baseline_end   = st.sidebar.date_input("Baseline end",   date(2023,12,31))
monitor_start  = st.sidebar.date_input("Monitor start",  date(2024,1,1))
monitor_end    = st.sidebar.date_input("Monitor end",    date.today())
quantile_q     = st.sidebar.slider("Hot-day threshold quantile", 0.80, 0.99, 0.90)
n_starts       = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

st.sidebar.header("Reproducibility")
ML_RANDOM_STATE = st.sidebar.number_input("ML random_state", value=ML_RANDOM_STATE_DEFAULT, step=1)
RNG_SEED        = st.sidebar.number_input("Monte Carlo RNG seed", value=RNG_SEED_DEFAULT, step=1)

st.sidebar.header("Alert thresholds")
theta_mu1_days = st.sidebar.number_input("Phase shift threshold mu1 (days)", value=10.0, min_value=0.0)
theta_mu2_days = st.sidebar.number_input("Phase shift threshold mu2 (days)", value=7.0,  min_value=0.0)
theta_eta      = st.sidebar.number_input("Skew change threshold (eta)",     value=0.05, min_value=0.0)
theta_nu_days  = st.sidebar.number_input("Skew orientation threshold nu (days)", value=10.0, min_value=0.0)
cl_factor      = st.sidebar.selectbox("Control limit width (SE multiples)", [1.0, 1.5, 2.0, 2.5], index=2)

st.sidebar.header("Simulation settings")
step_days    = st.sidebar.selectbox("Time step (days)", [7, 14, 30], index=0)
n_paths      = st.sidebar.slider("Monte Carlo paths", 1000, 20000, 5000, step=1000)

labels = ["mu1","mu2","k1","k2","eta","nu"]

try:
    phi_base, df_base = fetch_hotday_phases(str(baseline_start), str(baseline_end), q=quantile_q)
    phi_mon,  df_mon  = fetch_hotday_phases(str(monitor_start),  str(monitor_end),  q=quantile_q)
    # FIXED: avoid unterminated f-string by keeping it on one line
    st.write(f"Baseline hot-day samples: {len(phi_base)}  \nMonitoring hot-day samples: {len(phi_mon)}")

    params_base, ll_base = fit_ssgvm_mle_all_starts(phi_base, n_starts=n_starts)
    params_mon,  ll_mon  = fit_ssgvm_mle_all_starts(phi_mon,  n_starts=n_starts)

    base_negll = lambda p: -np.sum(logpdf_ssgvm(phi_base, p))
    se_base, cov_base = se_from_hessian(base_negll, params_base)
    base_df = pd.DataFrame({"Param": labels, "Baseline": params_base, "SE": se_base})
    mon_df  = pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    st.markdown("**Baseline parameters (±SE):**"); st.dataframe(base_df)
    st.markdown("**Monitoring parameters:**");     st.dataframe(mon_df)

    mu1_base, mu1_mon = params_base[0], params_mon[0]
    mu2_base, mu2_mon = params_base[1], params_mon[1]
    k1_base,  k2_base = params_base[2], params_base[3]
    k1_mon,   k2_mon  = params_mon[2],  params_mon[3]
    eta_base, eta_mon = params_base[4], params_mon[4]
    nu_base,  nu_mon  = params_base[5], params_mon[5]

    delta_mu1_days = angle_to_days(circ_diff(mu1_mon, mu1_base))
    delta_mu2_days = angle_to_days(circ_diff(mu2_mon, mu2_base))
    delta_eta      = abs(eta_mon - eta_base)
    delta_nu_days  = angle_to_days(circ_diff(nu_mon,  nu_base))

    k1_se, k2_se = se_base[2], se_base[3]
    k1_upper = k1_base + cl_factor * k1_se
    k1_lower = max(0.0, k1_base - cl_factor * k1_se)
    k2_upper = k2_base + cl_factor * k2_se
    k2_lower = max(0.0, k2_base - cl_factor * k2_se)

    alerts = []
    if delta_mu1_days > theta_mu1_days:
        alerts.append(f"Phase shift alert (mu1): Delta = {delta_mu1_days:.1f} days > theta_mu1 = {theta_mu1_days:.1f}")
    if delta_mu2_days > theta_mu2_days:
        alerts.append(f"Secondary phase alert (mu2): Delta = {delta_mu2_days:.1f} days > theta_mu2 = {theta_mu2_days:.1f}")
    if (k1_mon > k1_upper) or (k1_mon < k1_lower):
        alerts.append(f"Concentration change alert (k1): {k1_mon:.2f} outside [{k1_lower:.2f}, {k1_upper:.2f}]")
    if (k2_mon > k2_upper) or (k2_mon < k2_lower):
        alerts.append(f"Concentration change alert (k2): {k2_mon:.2f} outside [{k2_lower:.2f}, {k2_upper:.2f}]")
    if delta_eta > theta_eta:
        alerts.append(f"Skew change alert (eta): Delta = {delta_eta:.3f} > theta_eta = {theta_eta:.3f}")
    if delta_nu_days > theta_nu_days:
        alerts.append(f"Skew orientation alert (nu): Delta = {delta_nu_days:.1f} days > theta_nu = {theta_nu_days:.1f}")

    st.subheader("Alerts")
    if alerts:
        for a in alerts: st.error(a)
    else:
        st.success("No alerts triggered under current thresholds.")

    st.subheader("Baseline vs Monitoring density")
    days = np.arange(1, 366); theta_grid = 2*np.pi*days/365.0
    def ssgvm_density(params, theta):
        Z = normalizer_ssgvm(*params)
        return np.exp(
            np.clip(params[2]*np.cos(theta-params[0]) + params[3]*np.cos(2*(theta-params[1])), -700, 700)
        ) * np.maximum(1.0 + params[4]*np.sin(theta-params[5]), 1e-12) / Z
    dens_base = ssgvm_density(params_base, theta_grid)
    dens_mon  = ssgvm_density(params_mon,  theta_grid)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(days, dens_base, label="Baseline",   color="steelblue", lw=2)
    ax.plot(days, dens_mon,  label="Monitoring", color="crimson",   lw=2)
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
    ax.set_title("SS-GvM density: baseline vs monitoring")
    ax.legend(); ax.grid(alpha=0.3); st.pyplot(fig)

    st.subheader("ML-based monthly drift & uncertainty (Option A)")
    with st.spinner("Estimating monthly SS-GvM parameters and training ML models..."):
        df_all = pd.concat([df_base, df_mon], axis=0).sort_values("date").reset_index(drop=True)
        df_all = df_all[(df_all["date"].dt.date >= baseline_start) & (df_all["date"].dt.date <= monitor_end)]
        df_monthly = build_monthly_param_series(df_all, q=quantile_q, min_samples=20, n_starts=max(10, n_starts//2))
        ml_models = fit_ml_all(df_monthly, random_state=int(ML_RANDOM_STATE)) if (not df_monthly.empty and len(df_monthly) >= 24) else None
        if ml_models is None:
            st.warning("Insufficient monthly samples for ML. Falling back to OU-only simulation.")

    if ml_models is None:
        grid_days, paths_mu1 = simulate_angles_ml([params_mon[0]]*12, [safe_se(se_base[0],0.08)]*12, params_base[0], step_days, n_paths, int(RNG_SEED)+1)
        _,         paths_mu2 = simulate_angles_ml([params_mon[1]]*12, [safe_se(se_base[1],0.08)]*12, params_base[1], step_days, n_paths, int(RNG_SEED)+2)
        grid_days_k1, paths_k1 = simulate_scalar_ou_ml([params_mon[2]]*12, [safe_se(se_base[2],0.10)]*12, params_mon[2], "log",   step_days, n_paths, int(RNG_SEED)+3)
        grid_days_k2, paths_k2 = simulate_scalar_ou_ml([params_mon[3]]*12, [safe_se(se_base[3],0.10)]*12, params_mon[3], "log",   step_days, n_paths, int(RNG_SEED)+4)
        grid_days_eta, paths_eta = simulate_scalar_ou_ml([params_mon[4]]*12, [safe_se(se_base[4],0.10)]*12, params_mon[4], "atanh", step_days, n_paths, int(RNG_SEED)+5)
        source = "OU-only"
    else:
        fc12 = recursive_forecast_12(df_monthly, ml_models)
        st.write("Forecasted monthly means & sigma/quantiles (first rows):")
        st.dataframe(fc12.head(6))
        grid_days, paths_mu1 = simulate_angles_ml(fc12["mu1_mean"].tolist(), fc12["mu1_sigma"].tolist(), params_base[0], step_days, n_paths, int(RNG_SEED)+11)
        _,         paths_mu2 = simulate_angles_ml(fc12["mu2_mean"].tolist(), fc12["mu2_sigma"].tolist(), params_base[1], step_days, n_paths, int(RNG_SEED)+12)
        grid_days_k1, paths_k1 = simulate_scalar_ou_ml(fc12["k1_mean"].tolist(),  fc12["k1_sigma"].tolist(),  params_mon[2], "log",   step_days, n_paths, int(RNG_SEED)+13)
        grid_days_k2, paths_k2 = simulate_scalar_ou_ml(fc12["k2_mean"].tolist(),  fc12["k2_sigma"].tolist(),  params_mon[3], "log",   step_days, n_paths, int(RNG_SEED)+14)
        grid_days_eta, paths_eta = simulate_scalar_ou_ml(fc12["eta_mean"].tolist(), fc12["eta_sigma"].tolist(), params_mon[4], "atanh", step_days, n_paths, int(RNG_SEED)+15)
        source = "ML + OU"

    prob_mu1 = (paths_mu1 > theta_mu1_days).mean(axis=0)
    prob_mu2 = (paths_mu2 > theta_mu2_days).mean(axis=0)
    prob_k1  = ((paths_k1 > k1_upper) | (paths_k1 < k1_lower)).mean(axis=0)
    prob_k2  = ((paths_k2 > k2_upper) | (paths_k2 < k2_lower)).mean(axis=0)
    prob_eta = (np.abs(paths_eta - eta_base) > theta_eta).mean(axis=0)

    st.subheader("Monthly forward risk (probabilities)")
    summary_rows = []
    for m_idx in range(1, 13):
        i = m_idx-1
        summary_rows.append({
            "MonthAhead": m_idx,
            f"P(Δ mu1 > {theta_mu1_days:.1f}d)": prob_mu1[i],
            f"P(Δ mu2 > {theta_mu2_days:.1f}d)": prob_mu2[i],
            "P(k1 outside CL)": prob_k1[i],
            "P(k2 outside CL)": prob_k2[i],
            f"P(|Δ eta| > {theta_eta:.2f})": prob_eta[i],
            "Source": source
        })
    monthly_summary = pd.DataFrame(summary_rows)
    st.dataframe(monthly_summary)

    fig_prob, axpr = plt.subplots(figsize=(10,5))
    axpr.plot(grid_days,     prob_mu1, label=f"P(Δ mu1 > {theta_mu1_days:.1f}d)", color="navy", marker="o")
    axpr.plot(grid_days,     prob_mu2, label=f"P(Δ mu2 > {theta_mu2_days:.1f}d)", color="purple", marker="o")
    axpr.plot(grid_days_k1,  prob_k1,  label="P(k1 outside CL)", color="teal", marker="o")
    axpr.plot(grid_days_k2,  prob_k2,  label="P(k2 outside CL)", color="brown", marker="o")
    axpr.plot(grid_days_eta, prob_eta, label=f"P(|Δ eta| > {theta_eta:.2f})", color="darkgreen", marker="o")
    axpr.set_ylim(0,1); axpr.set_xlabel("Days ahead (monthly anchors)"); axpr.set_ylabel("Probability")
    axpr.grid(alpha=0.3); axpr.legend(ncol=2); st.pyplot(fig_prob)

    horizons = [90,180,270,360]; rows = []
    for H in horizons:
        idx = min(np.searchsorted(grid_days, H), len(grid_days)-1)
        rows.append({
            "Horizon_days": H,
            "mu1_P5":  np.percentile(paths_mu1[:, idx], 5),
            "mu1_Med": np.percentile(paths_mu1[:, idx], 50),
            "mu1_P95": np.percentile(paths_mu1[:, idx], 95),
            "mu2_P5":  np.percentile(paths_mu2[:, idx], 5),
            "mu2_Med": np.percentile(paths_mu2[:, idx], 50),
            "mu2_P95": np.percentile(paths_mu2[:, idx], 95),
            f"P(Δ mu1 > {theta_mu1_days:.1f}d)": prob_mu1[idx],
            f"P(Δ mu2 > {theta_mu2_days:.1f}d)": prob_mu2[idx]
        })
    quarterly = pd.DataFrame(rows)
    st.download_button("Download monthly probabilities (CSV)", data=monthly_summary.to_csv(index=False).encode("utf-8"), file_name="monthly_forward_risk_probabilities_ML.csv", mime="text/csv")
    st.download_button("Download quarterly percentiles & probs (CSV)", data=quarterly.to_csv(index=False).encode("utf-8"), file_name="quarterly_simulation_summary_ML.csv", mime="text/csv")

    report = pd.concat([
        pd.DataFrame({"Param": labels, "Baseline": params_base, "Baseline_SE": se_base}),
        pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    ], axis=1)
    st.download_button("Download parameter report (CSV)", data=report.to_csv(index=False).encode("utf-8"), file_name="kimberley_ssgvm_parameter_monitoring_ML.csv", mime="text/csv")

except Exception as e:
    st.error(f"Data, fitting, or simulation failed: {e}")
    st.info("Tip: Ensure sufficient hot-day samples, adjust quantile or windows, and verify network access.")
