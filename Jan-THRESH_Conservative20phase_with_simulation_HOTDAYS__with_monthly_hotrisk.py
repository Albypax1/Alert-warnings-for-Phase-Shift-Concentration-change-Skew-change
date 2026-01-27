
# -*- coding: utf-8 -*-
# SS-GvM Monitoring + Simulation — Arid and semi-arid regions (Updated: SE fallbacks, mu2 drift, sensitive thresholds)
# Author: Albert Antwi | Date: 2026-01-03

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import optimize
import requests

TWO_PI = 2*np.pi
LAT, LON = 25.774, -80.194  # Miami, US 
TIMEZONE = "auto"


# ---------------- utils ----------------
def wrap_angle(x):
    return np.mod(x, TWO_PI)

def circ_diff(a, b):
    d = np.abs(a - b)
    return np.minimum(d, TWO_PI - d)

def angle_to_days(angle):
    return (angle / TWO_PI) * 365.0

# Safe SE fallback to avoid degenerate (deterministic) simulations
def safe_se(x, default=0.08):
    try:
        xv = float(x)
        if not np.isfinite(xv) or xv <= 0:
            return float(default)
        return xv
    except Exception:
        return float(default)

# ---------------- SS-GvM core ----------------
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
        if res.fun < best['fun']:
            best = res
    return best.x, -best.fun

# ---------------- Numerical Hessian & SE ----------------
def numerical_hessian(func, x, eps_rel=1e-4, eps_abs=1e-6):
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    f0 = func(x)
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

# ---------------- Data ingestion (Open-Meteo) ----------------
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
    df["doy"] = df["date"].dt.dayofyear
    thr = np.quantile(df["tmax"].dropna(), q)
    hot = df[df["tmax"] >= thr]
    phi = TWO_PI * (hot["doy"].values - 1) / 365.0
    return phi, df

# ---------------- Simulators ----------------

def simulate_angles(mu_base, mu_mon, drift_factor, sigma_per_day, horizon_days, step_days, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    delta_obs = circ_diff(mu_mon, mu_base)
    drift = drift_factor * (delta_obs / 365.0)  # radians/day
    n_steps = int(np.ceil(horizon_days/step_days))
    grid_days = np.arange(1, n_steps+1) * step_days
    paths = np.empty((n_paths, n_steps), dtype=float)
    mu = np.full(n_paths, mu_mon)
    for t in range(n_steps):
        dt = step_days
        noise = rng.normal(0.0, sigma_per_day*np.sqrt(dt), size=n_paths)
        mu = wrap_angle(mu + drift*dt + noise)
        paths[:, t] = angle_to_days(circ_diff(mu, mu_base))
    return grid_days, paths

def simulate_k_log_ou(k_base, k_init, se_k_base, alpha=0.05, horizon_days=365, step_days=7, n_paths=5000, seed=42):
    rng = np.random.default_rng(seed)
    x_base = np.log(max(k_base, 1e-6))
    x = np.full(n_paths, np.log(max(k_init, 1e-6)))
    sigma_log = (safe_se(se_k_base) / max(k_base, 1e-6)) * 0.6
    n_steps = int(np.ceil(horizon_days/step_days))
    grid_days = np.arange(1, n_steps+1) * step_days
    paths = np.empty((n_paths, n_steps), dtype=float)
    for t in range(n_steps):
        dt = step_days
        noise = rng.normal(0.0, sigma_log*np.sqrt(dt), size=n_paths)
        x = x + alpha*(x_base - x)*dt + noise
        k = np.exp(x)
        paths[:, t] = k
    return grid_days, paths

def simulate_eta_ou(eta_base, eta_init, se_eta_base, alpha=0.05, horizon_days=365, step_days=7, n_paths=5000, seed=42):
    rng = np.random.default_rng(seed)
    clip = lambda e: np.clip(e, -0.999, 0.999)
    y_base = np.arctanh(clip(eta_base))
    y = np.full(n_paths, np.arctanh(clip(eta_init)))
    sigma_y = safe_se(se_eta_base, default=0.10) * 0.8  # slightly higher default to help movement
    n_steps = int(np.ceil(horizon_days/step_days))
    grid_days = np.arange(1, n_steps+1) * step_days
    paths = np.empty((n_paths, n_steps), dtype=float)
    for t in range(n_steps):
        dt = step_days
        noise = rng.normal(0.0, sigma_y*np.sqrt(dt), size=n_paths)
        y = y + alpha*(y_base - y)*dt + noise
        eta = np.tanh(y)
        paths[:, t] = eta
    return grid_days, paths

# ---------------- UI ----------------
st.title("Dashboard Application for forecasting monthly risk probabilities of phase shift in extreme temperature in arid & semi-arid regions")

st.sidebar.header("Configuration")
baseline_start = st.sidebar.date_input("Baseline start", date(2009,1,1))
baseline_end   = st.sidebar.date_input("Baseline end",   date(2023,12,31))
monitor_start  = st.sidebar.date_input("Monitor start",  date(2024,1,1))
monitor_end    = st.sidebar.date_input("Monitor end",    date.today())
quantile_q     = st.sidebar.slider("Hot-day threshold quantile", 0.80, 0.99, 0.90)
n_starts       = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

# Alert thresholds (sensitive defaults to help curves move)
st.sidebar.header("Alert thresholds")
theta_mu1_days = st.sidebar.number_input("Phase shift threshold mu1 (days)", value=10.0, min_value=0.0)
theta_mu2_days = st.sidebar.number_input("Phase shift threshold mu2 (days)", value=7.0, min_value=0.0)
theta_eta      = st.sidebar.number_input("Skew change threshold (eta)", value=0.05, min_value=0.0)
theta_nu_days  = st.sidebar.number_input("Skew orientation threshold nu (days)", value=10.0, min_value=0.0)
cl_factor      = st.sidebar.selectbox("Control limit width (SE multiples)", [1.0, 1.5, 2.0, 2.5], index=2)

# Simulation settings
st.sidebar.header("Simulation settings")
horizon_days   = st.sidebar.slider("Horizon (days)", 180, 365, 365, step=15)
step_days      = st.sidebar.selectbox("Time step (days)", [1, 3, 7, 14, 30], index=2)
n_paths        = st.sidebar.slider("Monte Carlo paths", 1000, 20000, 5000, step=1000)
trend_mu       = st.sidebar.slider("Trend continuation (mu1)", 0.0, 1.0, 0.6)
trend_mu2      = st.sidebar.slider("Trend continuation (mu2)", 0.0, 1.0, 0.7)
alpha_k        = st.sidebar.slider("Mean reversion (k1,k2) OU alpha", 0.00, 0.50, 0.05)
alpha_eta      = st.sidebar.slider("Mean reversion (eta) OU alpha", 0.00, 0.50, 0.05)

labels = ["mu1","mu2","k1","k2","eta","nu"]

try:
    # Ingest
    phi_base, df_base = fetch_hotday_phases(str(baseline_start), str(baseline_end), q=quantile_q)
    phi_mon,  df_mon  = fetch_hotday_phases(str(monitor_start),  str(monitor_end),  q=quantile_q)
    st.write(f"Baseline hot-day samples: {len(phi_base)} | Monitoring hot-day samples: {len(phi_mon)}")

    # Fit
    params_base, ll_base = fit_ssgvm_mle_all_starts(phi_base, n_starts=n_starts)
    params_mon,  ll_mon  = fit_ssgvm_mle_all_starts(phi_mon,  n_starts=n_starts)

    # Baseline SEs
    base_negll = lambda p: -np.sum(logpdf_ssgvm(phi_base, p))
    se_base, cov_base = se_from_hessian(base_negll, params_base)

    base_df = pd.DataFrame({"Param": labels, "Baseline": params_base, "SE": se_base})
    mon_df  = pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    st.markdown("**Baseline parameters (±SE):**")
    st.dataframe(base_df)
    st.markdown("**Monitoring parameters:**")
    st.dataframe(mon_df)

    # ---------------- Alerts ----------------
    mu1_base, mu1_mon = params_base[0], params_mon[0]
    mu2_base, mu2_mon = params_base[1], params_mon[1]
    k1_base, k2_base  = params_base[2], params_base[3]
    k1_mon,  k2_mon   = params_mon[2], params_mon[3]
    eta_base, eta_mon = params_base[4], params_mon[4]
    nu_base,  nu_mon  = params_base[5], params_mon[5]

    delta_mu1_days = angle_to_days(circ_diff(mu1_mon, mu1_base))
    delta_mu2_days = angle_to_days(circ_diff(mu2_mon, mu2_base))
    delta_eta      = abs(eta_mon - eta_base)
    delta_nu_days  = angle_to_days(circ_diff(nu_mon, nu_base))

    # Control limits for k1,k2
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
        for a in alerts:
            st.error(a)
    else:
        st.success("No alerts triggered under current thresholds.")

    # ---------------- Density visualization ----------------
    st.subheader("Baseline vs Monitoring density")
    days = np.arange(1, 366)
    theta_grid = 2*np.pi*days/365.0

    def ssgvm_density(params, theta):
        Z = normalizer_ssgvm(*params)
        return np.exp(
            np.clip(params[2]*np.cos(theta-params[0]) + params[3]*np.cos(2*(theta-params[1])), -700, 700)
        ) * np.maximum(1.0 + params[4]*np.sin(theta-params[5]), 1e-12) / Z

    dens_base = ssgvm_density(params_base, theta_grid)
    dens_mon  = ssgvm_density(params_mon,  theta_grid)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(days, dens_base, label="Baseline", color="steelblue", lw=2)
    ax.plot(days, dens_mon,  label="Monitoring", color="crimson",  lw=2)
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
    ax.set_title("SS-GvM density: baseline vs monitoring")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ---------------- Simulation ----------------
    st.subheader("Multi-parameter simulation (next year)")
    st.markdown("Stochastic projections for mu1, mu2, k1, k2, eta using phase drift + OU dynamics.")

    # Volatility fallbacks (sensitive defaults)
    sigma_mu1_day = safe_se(se_base[0], default=0.08) * 0.5
    sigma_mu2_day = safe_se(se_base[1], default=0.08) * 0.5

    grid_days, paths_mu1 = simulate_angles(mu1_base, mu1_mon, trend_mu,  sigma_mu1_day, horizon_days, step_days, n_paths, seed=42)
    _,         paths_mu2 = simulate_angles(mu2_base, mu2_mon, trend_mu2, sigma_mu2_day, horizon_days, step_days, n_paths, seed=43)

    grid_days_k1, paths_k1 = simulate_k_log_ou(k1_base, k1_mon, se_base[2], alpha=alpha_k,   horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=44)
    grid_days_k2, paths_k2 = simulate_k_log_ou(k2_base, k2_mon, se_base[3], alpha=alpha_k,   horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=45)
    grid_days_eta, paths_eta = simulate_eta_ou(eta_base, eta_mon, se_base[4], alpha=alpha_eta, horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=46)

    # Probabilities across the grid
    prob_mu1 = (paths_mu1 > theta_mu1_days).mean(axis=0)
    prob_mu2 = (paths_mu2 > theta_mu2_days).mean(axis=0)
    prob_k1  = ((paths_k1 > k1_upper) | (paths_k1 < k1_lower)).mean(axis=0)
    prob_k2  = ((paths_k2 > k2_upper) | (paths_k2 < k2_lower)).mean(axis=0)
    prob_eta = (np.abs(paths_eta - eta_base) > theta_eta).mean(axis=0)

    # Monthly summary (approximate: month m ~ day 30.4*m)
    month_days = (np.arange(1,13) * 30.4).astype(int)
    summary_rows = []
    for m_idx, d in enumerate(month_days, start=1):
        i_mu = min(np.searchsorted(grid_days, d), len(grid_days)-1)
        i_k1 = min(np.searchsorted(grid_days_k1, d), len(grid_days_k1)-1)
        i_k2 = min(np.searchsorted(grid_days_k2, d), len(grid_days_k2)-1)
        i_et = min(np.searchsorted(grid_days_eta, d), len(grid_days_eta)-1)
        summary_rows.append({
            "MonthAhead": m_idx,
            f"P(Delta mu1 > {theta_mu1_days:.1f}d)": prob_mu1[i_mu],
            f"P(Delta mu2 > {theta_mu2_days:.1f}d)": prob_mu2[i_mu],
            "P(k1 outside CL)": prob_k1[i_k1],
            "P(k2 outside CL)": prob_k2[i_k2],
            f"P(|Delta eta| > {theta_eta:.2f})": prob_eta[i_et],
        })
    monthly_summary = pd.DataFrame(summary_rows)

    st.subheader("Monthly forward risk (probabilities)")
    st.dataframe(monthly_summary)

    # Probability curves
    fig_prob, axpr = plt.subplots(figsize=(10,5))
    axpr.plot(grid_days,    prob_mu1, label=f"P(Delta mu1 > {theta_mu1_days:.1f}d)", color="navy")
    axpr.plot(grid_days,    prob_mu2, label=f"P(Delta mu2 > {theta_mu2_days:.1f}d)", color="purple")
    axpr.plot(grid_days_k1, prob_k1,  label="P(k1 outside CL)", color="teal")
    axpr.plot(grid_days_k2, prob_k2,  label="P(k2 outside CL)", color="brown")
    axpr.plot(grid_days_eta,prob_eta, label=f"P(|Delta eta| > {theta_eta:.2f})", color="darkgreen")
    axpr.set_ylim(0,1); axpr.set_xlabel("Days ahead"); axpr.set_ylabel("Probability"); axpr.grid(alpha=0.3); axpr.legend(ncol=2)
    st.pyplot(fig_prob)

    # Downloads
    horizons = [90,180,270,360]
    rows = []
    for H in horizons:
        idx_mu = min(np.searchsorted(grid_days, H), len(grid_days)-1)
        rows.append({
            "Horizon_days": H,
            "mu1_P5":   np.percentile(paths_mu1[:, idx_mu], 5),
            "mu1_Med":  np.percentile(paths_mu1[:, idx_mu], 50),
            "mu1_P95":  np.percentile(paths_mu1[:, idx_mu], 95),
            "mu2_P5":   np.percentile(paths_mu2[:, idx_mu], 5),
            "mu2_Med":  np.percentile(paths_mu2[:, idx_mu], 50),
            "mu2_P95":  np.percentile(paths_mu2[:, idx_mu], 95),
            f"P(Delta mu1 > {theta_mu1_days:.1f}d)": prob_mu1[idx_mu],
            f"P(Delta mu2 > {theta_mu2_days:.1f}d)": prob_mu2[idx_mu],
        })
    quarterly = pd.DataFrame(rows)

    csv_monthly  = monthly_summary.to_csv(index=False).encode("utf-8")
    csv_quarterly= quarterly.to_csv(index=False).encode("utf-8")
    st.download_button("Download monthly probabilities (CSV)", data=csv_monthly,
                       file_name="monthly_forward_risk_probabilities.csv", mime="text/csv")
    st.download_button("Download quarterly percentiles & probs (CSV)", data=csv_quarterly,
                       file_name="quarterly_simulation_summary.csv", mime="text/csv")

    report = pd.concat([
        pd.DataFrame({"Param": labels, "Baseline": params_base, "Baseline_SE": se_base}),
        pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    ], axis=1)
    csv_report = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download parameter report (CSV)", data=csv_report,
                       file_name="kimberley_ssgvm_parameter_monitoring.csv", mime="text/csv")

except Exception as e:
    st.error(f"Data, fitting, or simulation failed: {e}")
    st.info("Tip: Ensure sufficient hot-day samples, adjust quantile or windows, and verify network access.")




# -----------------------------
# Hot Days Benchmarking (add-on)
# Historical percentile calc + Auto-suggest + WMO preset + Visual charts
# -----------------------------
import io as _io
import zipfile as _zipfile

st.markdown("---")
st.header("Hot Days Benchmarking — Add-on")

@st.cache_data(show_spinner=False)
def hotdays_geocode(name: str, count: int = 5, language: str = 'en'):
    if not name:
        return []
    url = ('https://geocoding-api.open-meteo.com/v1/search'
           f'?name={requests.utils.quote(name)}&count={count}&language={language}&format=json')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get('results', []) or []

@st.cache_data(show_spinner=False)
def hotdays_fetch_daily(lat: float, lon: float, start: str, end: str, tz: str = TIMEZONE):
    base = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start,
        'end_date': end,
        'daily': ['temperature_2m_max'],
        'timezone': tz
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    daily = js.get('daily', {})
    if not daily:
        return pd.DataFrame()
    df = pd.DataFrame({'date': daily.get('time', []), 'tmax': daily.get('temperature_2m_max', [])})
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# Leap-day removal

def _hotdays_fix_leap(df: pd.DataFrame):
    s = pd.to_datetime(df['date'])
    return df[~((s.dt.month == 2) & (s.dt.day == 29))].copy()

# DOY percentile threshold

def hotdays_compute_doy_percentile(df: pd.DataFrame, baseline_start: date, baseline_end: date,
                                   pctl: float = 90.0, window: int = 15) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['doy','thresh'])
    s = pd.to_datetime(df['date']).dt.date
    base = df[(s >= baseline_start) & (s <= baseline_end)].copy()
    if base.empty:
        return pd.DataFrame(columns=['doy','thresh'])
    base = _hotdays_fix_leap(base)
    base['doy'] = pd.to_datetime(base['date']).dt.dayofyear
    recs = []
    for doy in range(1, 366):
        win = [((d-1) % 365) + 1 for d in list(range(doy-window, doy)) + list(range(doy, doy+window+1))]
        vals = base.loc[base['doy'].isin(win), 'tmax'].values
        thr = float(np.nanpercentile(vals, pctl)) if len(vals) else np.nan
        recs.append({'doy': doy, 'thresh': thr})
    return pd.DataFrame(recs)

# Label series

def hotdays_label(df_actual: pd.DataFrame, thresh_doy: pd.DataFrame):
    if df_actual.empty or thresh_doy.empty:
        return df_actual.assign(thresh=np.nan, hot_day=np.nan)
    tmp = df_actual.copy()
    tmp['doy'] = pd.to_datetime(tmp['date']).dt.dayofyear
    tmp = tmp.merge(thresh_doy, on='doy', how='left')
    tmp['hot_day'] = tmp['tmax'] >= tmp['thresh']
    return tmp

# Heatwaves

def hotdays_heatwaves(binary_series: pd.Series, min_len: int = 3):
    runs = []
    in_run = False
    start = None
    for i, v in enumerate(binary_series.fillna(False).tolist()):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            if i - start >= min_len:
                runs.append((start, i-1))
            in_run = False
    if in_run and (len(binary_series) - start >= min_len):
        runs.append((start, len(binary_series)-1))
    return runs

# Sidebar controls
st.sidebar.header("Hot-day location")
place = st.sidebar.text_input("Place name (auto-suggest)", value="Miami")
autosuggest = st.sidebar.button("🔎 Auto-suggest")
if 'hot_lat' not in st.session_state:
    st.session_state.hot_lat = LAT
if 'hot_lon' not in st.session_state:
    st.session_state.hot_lon = LON
colA, colB = st.sidebar.columns(2)
hot_lat = colA.number_input("Latitude", value=float(st.session_state.hot_lat), format="%.6f")
hot_lon = colB.number_input("Longitude", value=float(st.session_state.hot_lon), format="%.6f")
if autosuggest and place:
    with st.spinner('Finding places...'):
        res = hotdays_geocode(place)
    if res:
        best = res[0]
        st.session_state.hot_lat = best.get('latitude', hot_lat)
        st.session_state.hot_lon = best.get('longitude', hot_lon)
        hot_lat, hot_lon = st.session_state.hot_lat, st.session_state.hot_lon
        st.sidebar.success(f"Selected: {best.get('name')}, {best.get('country_code')} ({hot_lat:.3f}, {hot_lon:.3f})")
    else:
        st.sidebar.warning("No suggestions found.")

st.sidebar.header("Hot-day percentile & baseline")
percentile = st.sidebar.slider("Percentile threshold (≥)", 50, 99, int(round(quantile_q*100)))
win = st.sidebar.slider("Smoothing window (± days)", 5, 30, 15, 1)
min_run = st.sidebar.slider("Heatwave min length (days)", 2, 7, 3, 1)
if st.sidebar.button("🌍 WMO preset (user pref)"):
    percentile = 90
    baseline_start = date(2015,1,1)
    baseline_end   = date(2023,12,31)
    min_run = 3
    st.sidebar.info("Applied: ≥90th percentile, baseline 2015–2025, heatwave length ≥3 days")

st.subheader("Daily Tmax vs historical percentile threshold")
run_hot = st.button("⬇️ Fetch & Analyze Hot Days")
if run_hot:
    with st.spinner("Fetching daily Tmax for baseline & monitoring..."):
        df_base_daily = hotdays_fetch_daily(hot_lat, hot_lon, str(baseline_start), str(baseline_end))
        df_mon_daily  = hotdays_fetch_daily(hot_lat, hot_lon, str(monitor_start), str(monitor_end))
    if df_base_daily.empty or df_mon_daily.empty:
        st.warning("No daily data returned for the given periods/location.")
    else:
        thr_doy = hotdays_compute_doy_percentile(df_base_daily, baseline_start, baseline_end,
                                                 pctl=float(percentile), window=int(win))
        labeled = hotdays_label(df_mon_daily, thr_doy)
        runs = hotdays_heatwaves(labeled['hot_day'], min_len=int(min_run))
        spans = []
        for s, e in runs:
            spans.append({'start': labeled.iloc[s]['date'],
                          'end':   labeled.iloc[e]['date'],
                          'length': int((pd.to_datetime(labeled.iloc[e]['date']) - pd.to_datetime(labeled.iloc[s]['date'])).days) + 1})
        total_days = len(labeled)
        hot_days = int(labeled['hot_day'].sum())
        share = 100.0 * hot_days / total_days if total_days else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Monitoring days", f"{total_days}")
        c2.metric("Hot days (≥ threshold)", f"{hot_days}", f"{share:.1f}%")
        c3.metric(f"Heatwaves (≥{min_run} days)", f"{len(spans)}")
        fig_ts, ax_ts = plt.subplots(figsize=(12,4))
        ts_dates = pd.to_datetime(labeled['date'])
        ax_ts.plot(ts_dates, labeled['tmax'], label='Tmax (°C)', color='#1f77b4', lw=1.2)
        ax_ts.plot(ts_dates, labeled['thresh'], label=f"{percentile}th percentile threshold", color='#d62728', lw=1.2)
        for sp in spans:
            ax_ts.axvspan(pd.to_datetime(sp['start']), pd.to_datetime(sp['end']), color='orange', alpha=0.2)
        ax_ts.set_ylabel('°C'); ax_ts.set_xlabel('Date'); ax_ts.legend(); ax_ts.grid(alpha=0.3)
        st.pyplot(fig_ts, use_container_width=True)
        fig_hist, ax_h = plt.subplots(figsize=(6,4))
        ax_h.hist(labeled['tmax'], bins=40, color='#1f77b4', alpha=0.7, edgecolor='white')
        med_thr = float(np.nanmedian(labeled['thresh']))
        ax_h.axvline(med_thr, color='#d62728', ls='--', lw=2, label=f"Median threshold ≈ {med_thr:.1f}°C")
        ax_h.set_xlabel('Tmax (°C)'); ax_h.set_ylabel('Frequency'); ax_h.legend(); ax_h.grid(alpha=0.3)
        st.pyplot(fig_hist, use_container_width=True)
        st.subheader("Detected heatwaves (monitoring period)")
        if spans:
            st.dataframe(pd.DataFrame(spans))
        else:
            st.info("No heatwaves meeting the criteria were detected.")
        st.subheader("Downloads")
        csv_lbl = labeled.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download labeled monitoring data (CSV)", data=csv_lbl,
                           file_name="monitoring_labeled_hotdays.csv", mime="text/csv")
        b1, b2 = _io.BytesIO(), _io.BytesIO()
        fig_ts.savefig(b1, format='png', dpi=200, bbox_inches='tight'); b1.seek(0)
        fig_hist.savefig(b2, format='png', dpi=200, bbox_inches='tight'); b2.seek(0)
        zip_buf = _io.BytesIO()
        with _zipfile.ZipFile(zip_buf, 'w', _zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('timeseries_threshold.png', b1.read())
            zf.writestr('distribution_threshold.png', b2.read())
        zip_buf.seek(0)
        st.download_button(label="Download charts (ZIP)", data=zip_buf.getvalue(),
                           file_name='hotdays_benchmark_charts.zip', mime='application/zip')

st.caption("Tip: Use the 🌍 WMO preset to quickly align with ≥90th percentile and 3‑day heatwave duration.")




# -----------------------------
# Automatic Threshold Suggestion (data-driven)
# -----------------------------
st.markdown("---")
st.subheader("Automatic threshold suggestion")
st.caption("Data-driven thresholds based on baseline circular variability and parameter SEs.")

# Helper: circular resultant length and circular std (Fisher, 1995)
def _circular_stats(angles_rad: np.ndarray):
    if angles_rad.size == 0:
        return np.nan, np.nan
    C = np.nanmean(np.cos(angles_rad))
    S = np.nanmean(np.sin(angles_rad))
    R = np.sqrt(C*C + S*S)
    # Circular standard deviation (radians)
    s = np.sqrt(np.maximum(0.0, -2.0*np.log(np.clip(R, 1e-12, 1.0))))
    return R, s

sens = st.radio("Sensitivity level", ["Early warning", "Balanced", "Trend detection"], index=0, horizontal=True)
mult = {"Early warning": 2.0, "Balanced": 3.0, "Trend detection": 4.0}[sens]

# Compute variability from baseline hot-day phases if available; otherwise from baseline daily DOY
try:
    # Prefer phi_base from SS-GvM ingestion if present
    phi_source = None
    if 'phi_base' in globals() and isinstance(phi_base, np.ndarray) and phi_base.size>0:
        phi_source = phi_base
    else:
        # Fallback: build phases from baseline daily Tmax
        df_base_daily = hotdays_fetch_daily(st.session_state.get('hot_lat', LAT), st.session_state.get('hot_lon', LON), str(baseline_start), str(baseline_end))
        if not df_base_daily.empty:
            doy = pd.to_datetime(df_base_daily['date']).dt.dayofyear.values
            phi = 2*np.pi*(doy-1)/365.0
            phi_source = phi
    if phi_source is None or len(phi_source)==0:
        st.warning("Insufficient baseline phase data to compute suggestions.")
    else:
        R, s_rad = _circular_stats(np.asarray(phi_source))
        s_days = (s_rad/(2*np.pi))*365.0
        # Recommended thresholds
        rec_mu1 = float(mult*s_days)
        rec_mu2 = float(mult*s_days)
        # Skew & skew orientation from baseline SEs if available
        rec_eta = None; rec_nu_days = None
        try:
            if 'se_base' in globals() and isinstance(se_base, (list, np.ndarray)) and len(se_base)>=6:
                # eta SE ~ scale of variability; use multiplier
                rec_eta = float(mult*float(se_base[4]))
                # nu SE is in radians -> convert to days
                nu_se = float(se_base[5])
                rec_nu_days = float(mult*((nu_se/(2*np.pi))*365.0))
        except Exception:
            pass

        cols = st.columns(4)
        cols[0].metric("Suggested θ_mu1 (days)", f"{rec_mu1:.1f}")
        cols[1].metric("Suggested θ_mu2 (days)", f"{rec_mu2:.1f}")
        if rec_eta is not None:
            cols[2].metric("Suggested θ_eta", f"{rec_eta:.3f}")
        if rec_nu_days is not None:
            cols[3].metric("Suggested θ_nu (days)", f"{rec_nu_days:.1f}")

        # Optional: apply to session_state for convenience
        if st.button("Apply suggested thresholds"):
            st.session_state['theta_mu1_days'] = rec_mu1
            st.session_state['theta_mu2_days'] = rec_mu2
            if rec_eta is not None:
                st.session_state['theta_eta'] = rec_eta
            if rec_nu_days is not None:
                st.session_state['theta_nu_days'] = rec_nu_days
            st.success("Suggested thresholds stored in session state. Adjust original sidebar inputs to match if needed.")
except Exception as _e:
    st.info(f"Threshold suggestion skipped: {_e}")


# -------------------------------------------------------------
# Monthly forward hot-day risk (≤ 3 months)
# Hybrid SS-GvM + ML (if available) + Monte Carlo aggregated to months
# -------------------------------------------------------------
st.markdown("---")
st.header("Monthly forward hot‑day risk (Hybrid SS‑GvM + ML + Monte Carlo)")

# Sidebar controls (separate from other sections)
st.sidebar.header("Monthly forward risk (≤ 3 months)")
month_horizon = st.sidebar.slider("Months ahead (1–3)", 1, 3, 2, step=1)
month_nsims    = st.sidebar.slider("Monte Carlo simulations (monthly)", 500, 20000, 5000, step=500)
month_blend_w  = st.sidebar.slider("Blend weight: SS‑GvM vs ML/climatology (monthly)", 0.0, 1.0, 0.5)
month_model    = st.sidebar.selectbox("ML model (monthly)", ["GradientBoosting", "RandomForest"]) 

# Use monitoring SS-GvM parameters if available; fall back to baseline
try:
    params_for_hot_m = params_mon
except Exception:
    try:
        params_for_hot_m = params_base
    except Exception:
        params_for_hot_m = None

# Build SS-GvM probabilities for all DOYs
try:
    if params_for_hot_m is not None:
        days_all_m = np.arange(1, 366)
        theta_all_m = 2*np.pi*days_all_m/365.0
        Z_m = normalizer_ssgvm(*params_for_hot_m)
        dens_m = np.exp(np.clip(params_for_hot_m[2]*np.cos(theta_all_m-params_for_hot_m[0]) 
                                + params_for_hot_m[3]*np.cos(2*(theta_all_m-params_for_hot_m[1])), -700, 700)) \
                 * np.maximum(1.0 + params_for_hot_m[4]*np.sin(theta_all_m - params_for_hot_m[5]), 1e-12) / Z_m
        hotprob_ssgvm_all_m = dens_m / dens_m.sum()
    else:
        hotprob_ssgvm_all_m = np.full(365, 0.01)
except Exception:
    hotprob_ssgvm_all_m = np.full(365, 0.01)

# Prepare monitoring daily data for ML/climatology
try:
    df_mon_daily_m = None
    if 'df_mon' in globals() and isinstance(df_mon, pd.DataFrame) and not df_mon.empty:
        df_mon_daily_m = df_mon.copy()
    else:
        # Use helper from add-on
        df_mon_daily_m = hotdays_fetch_daily(st.session_state.get('hot_lat', LAT),
                                             st.session_state.get('hot_lon', LON),
                                             str(monitor_start), str(monitor_end))
    if df_mon_daily_m is not None and not df_mon_daily_m.empty:
        if 'doy' not in df_mon_daily_m.columns:
            df_mon_daily_m['doy'] = pd.to_datetime(df_mon_daily_m['date']).dt.dayofyear
        thr_q_m = float(np.nanquantile(df_mon_daily_m['tmax'].dropna(), quantile_q))
        df_mon_daily_m['hot'] = (df_mon_daily_m['tmax'] >= thr_q_m).astype(int)
        df_ml_m = df_mon_daily_m.dropna(subset=['tmax']).copy()
        df_ml_m['sin_doy'] = np.sin(2*np.pi*df_ml_m['doy']/365.0)
        df_ml_m['cos_doy'] = np.cos(2*np.pi*df_ml_m['doy']/365.0)
        df_ml_m['tmax_roll3'] = df_ml_m['tmax'].rolling(3, min_periods=1).mean()
        df_ml_m['tmax_roll7'] = df_ml_m['tmax'].rolling(7, min_periods=1).mean()
        df_ml_m['tmax_diff1'] = df_ml_m['tmax'].diff().fillna(0.0)
        doy_idx_m = (df_ml_m['doy'].values - 1) % 365
        df_ml_m['hotprob_ssgvm'] = hotprob_ssgvm_all_m[doy_idx_m]
        features_m = ['sin_doy','cos_doy','tmax','tmax_roll3','tmax_roll7','tmax_diff1','hotprob_ssgvm']
        Xm = df_ml_m[features_m].values
        ym = df_ml_m['hot'].values.astype(int)
    else:
        Xm, ym, df_ml_m = None, None, None
except Exception:
    Xm, ym, df_ml_m = None, None, None

# Try scikit-learn; graceful fallback
model_m = None
sklearn_ok_m = False
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    sklearn_ok_m = True
except Exception:
    sklearn_ok_m = False

if sklearn_ok_m and df_ml_m is not None and Xm is not None and len(df_ml_m) >= 30 and np.unique(ym).size > 1:
    if month_model == "GradientBoosting":
        model_m = GradientBoostingClassifier(random_state=42)
    else:
        model_m = RandomForestClassifier(n_estimators=300, random_state=42)
    try:
        model_m.fit(Xm, ym)
    except Exception:
        model_m = None

# Simulation horizon (approx 30 days per month)
month_days = int(month_horizon * 30)
try:
    doy0_m = int(pd.Timestamp.today(tz=TIMEZONE).dayofyear)
except Exception:
    doy0_m = int(pd.Timestamp.today().dayofyear)
future_doys_m = np.array([((doy0_m + i - 1) % 365) + 1 for i in range(1, month_days + 1)])
ssgvm_probs_m = hotprob_ssgvm_all_m[future_doys_m - 1]

# Climatology by DOY
if df_mon_daily_m is not None and not df_mon_daily_m.empty:
    clim_m = df_mon_daily_m.groupby('doy')['tmax'].agg(['mean','std']).reindex(np.arange(1,366)).fillna(method='ffill').fillna(method='bfill')
    mu_clim_m = clim_m['mean'].values
    sd_clim_m = np.clip(clim_m['std'].values, 0.5, None)
else:
    mu_clim_m = np.full(365, 30.0)
    sd_clim_m = np.full(365, 2.0)

# Monte Carlo daily paths (for month_days)
sim_probs_m = np.zeros((month_nsims, month_days))
sim_hot_m   = np.zeros((month_nsims, month_days), dtype=bool)

for s in range(month_nsims):
    temps_m = np.random.normal(mu_clim_m[future_doys_m - 1], sd_clim_m[future_doys_m - 1])
    sin_d_m = np.sin(2*np.pi*future_doys_m/365.0)
    cos_d_m = np.cos(2*np.pi*future_doys_m/365.0)
    t_roll3_m = pd.Series(temps_m).rolling(3, min_periods=1).mean().values
    t_roll7_m = pd.Series(temps_m).rolling(7, min_periods=1).mean().values
    t_diff1_m = np.diff(np.concatenate([[temps_m[0]], temps_m]))
    Xm_h = np.column_stack([sin_d_m, cos_d_m, temps_m, t_roll3_m, t_roll7_m, t_diff1_m, ssgvm_probs_m])
    if model_m is not None:
        try:
            ml_probs_m = model_m.predict_proba(Xm_h)[:,1]
        except Exception:
            ml_probs_m = 1/(1+np.exp(-(0.08*(temps_m-30.0) + 2.0*(ssgvm_probs_m-0.05))))
    else:
        hot_freq_m = (df_mon_daily_m.groupby('doy')['hot'].mean().reindex(np.arange(1,366)).fillna(df_mon_daily_m['hot'].mean()).values
                      if df_mon_daily_m is not None and 'hot' in df_mon_daily_m.columns else np.full(365, 0.05))
        ml_probs_m = hot_freq_m[future_doys_m - 1]
    hybrid_p_m = np.clip(month_blend_w*ssgvm_probs_m + (1.0 - month_blend_w)*ml_probs_m, 0.0, 1.0)
    sim_probs_m[s] = hybrid_p_m
    sim_hot_m[s]   = np.random.rand(month_days) < hybrid_p_m

# Aggregate to months (approx 30 days per month)
months = list(range(1, month_horizon+1))
month_bins = [(30*(m-1), min(30*m, month_days)) for m in months]
rows = []
for m, (a,b) in zip(months, month_bins):
    counts = sim_hot_m[:, a:b].sum(axis=1)  # hot-day count in month m
    p_at_least_one = float((counts >= 1).mean())
    rows.append({
        'MonthAhead': m,
        'DaysInMonthApprox': b-a,
        'HotDays_Mean': float(counts.mean()),
        'HotDays_p05': float(np.percentile(counts, 5)),
        'HotDays_p50': float(np.percentile(counts, 50)),
        'HotDays_p95': float(np.percentile(counts, 95)),
        'Pr(≥1 hot day)': round(p_at_least_one, 4),
    })
monthly_hot_summary = pd.DataFrame(rows)

st.subheader("Monthly forward hot‑day risk (summary)")
st.dataframe(monthly_hot_summary)

# Simple plot: probability of ≥1 hot day per month
fig_m, ax_m = plt.subplots(figsize=(7,3.5))
ax_m.bar(monthly_hot_summary['MonthAhead'], monthly_hot_summary['Pr(≥1 hot day)'], color='teal', alpha=0.75)
ax_m.set_xlabel('Months ahead'); ax_m.set_ylabel('Pr(≥1 hot day)')
ax_m.set_ylim(0,1); ax_m.grid(alpha=0.3, axis='y')
st.pyplot(fig_m)

# Downloads
csv_monthly_hot = monthly_hot_summary.to_csv(index=False).encode('utf-8')
st.download_button("Download monthly hot‑day risk (CSV)", data=csv_monthly_hot,
                   file_name="monthly_forward_hotday_risk.csv", mime="text/csv")
