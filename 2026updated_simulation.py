
# -*- coding: utf-8 -*-
# Updated Simulation: SS-GvM Monitoring + Multi-Parameter Simulation — Kimberley
# Author: M365 Copilot | Date: 2026-01-03

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import optimize
import requests

TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719
TIMEZONE = "Africa/Johannesburg"

def wrap_angle(x):
    return np.mod(x, TWO_PI)

def circ_diff(a, b):
    d = np.abs(a - b)
    return np.minimum(d, TWO_PI - d)

def angle_to_days(angle):
    return (angle / TWO_PI) * 365.0

def normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu, n_grid=4096):
    grid = np.linspace(0, TWO_PI, n_grid, endpoint=False)
    expo = (k1*np.cos(grid-mu1) + k2*np.cos(2*(grid-mu2)) + np.log(np.maximum(1+eta*np.sin(grid-nu), 1e-12)))
    m = np.max(expo)
    return (TWO_PI/n_grid)*np.exp(m)*np.sum(np.exp(expo-m))

def logpdf_ssgvm(x, params):
    mu1, mu2, k1, k2, eta, nu = params
    Z = normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu)
    xw = wrap_angle(x)
    return (k1*np.cos(xw-mu1) + k2*np.cos(2*(xw-mu2)) + np.log(np.maximum(1+eta*np.sin(xw-nu), 1e-12)) - np.log(Z))

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

# ------- stochastic simulators -------

def simulate_angles(mu_base, mu_mon, drift_factor, sigma_per_day, horizon_days, step_days, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    delta_obs = circ_diff(mu_mon, mu_base)
    drift = drift_factor * (delta_obs / 365.0)
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
    sigma_log = (se_k_base / max(k_base, 1e-6)) * 0.6
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
    sigma_y = max(1e-6, se_eta_base) * 0.8
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

# --------------- UI ---------------
st.title("Updated Simulation — Phase (mu1,mu2), Concentration (k1,k2), Skew (eta)")

st.sidebar.header("Configuration")
baseline_start = st.sidebar.date_input("Baseline start", date(2019,1,1))
baseline_end   = st.sidebar.date_input("Baseline end",   date(2023,12,31))
monitor_start  = st.sidebar.date_input("Monitor start",  date(2024,1,1))
monitor_end    = st.sidebar.date_input("Monitor end",    date.today())
quantile_q     = st.sidebar.slider("Hot-day threshold quantile", 0.80, 0.99, 0.90)
n_starts       = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

# thresholds
theta_mu1_days = st.sidebar.number_input("Phase shift threshold mu1 (days)", value=10.0, min_value=0.0)
theta_mu2_days = st.sidebar.number_input("Phase shift threshold mu2 (days)", value=10.0, min_value=0.0)
theta_eta      = st.sidebar.number_input("Skew change threshold (eta)", value=0.10, min_value=0.0)
cl_factor      = st.sidebar.selectbox("Control limit width (SE multiples)", [1.0, 1.5, 2.0, 2.5], index=2)

st.sidebar.header("Simulation settings")
horizon_days   = st.sidebar.slider("Horizon (days)", 180, 365, 365, step=15)
step_days      = st.sidebar.selectbox("Time step (days)", [1, 3, 7, 14, 30], index=2)
n_paths        = st.sidebar.slider("Monte Carlo paths", 1000, 20000, 5000, step=1000)
trend_mu       = st.sidebar.slider("Trend continuation (phases)", 0.0, 1.0, 0.5)
alpha_k        = st.sidebar.slider("Mean reversion (k1,k2) OU alpha", 0.00, 0.50, 0.05)
alpha_eta      = st.sidebar.slider("Mean reversion (eta) OU alpha", 0.00, 0.50, 0.05)

labels = ["mu1","mu2","k1","k2","eta","nu"]

try:
    phi_base, df_base = fetch_hotday_phases(str(baseline_start), str(baseline_end), q=quantile_q)
    phi_mon,  df_mon  = fetch_hotday_phases(str(monitor_start),  str(monitor_end),  q=quantile_q)
    st.write(f"Baseline hot-day samples: {len(phi_base)} | Monitoring hot-day samples: {len(phi_mon)}")

    params_base, ll_base = fit_ssgvm_mle_all_starts(phi_base, n_starts=n_starts)
    params_mon,  ll_mon  = fit_ssgvm_mle_all_starts(phi_mon,  n_starts=n_starts)

    base_negll = lambda p: -np.sum(logpdf_ssgvm(phi_base, p))
    se_base, cov_base = se_from_hessian(base_negll, params_base)

    base_df = pd.DataFrame({"Param": labels, "Baseline": params_base, "SE": se_base})
    mon_df  = pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    st.markdown("**Baseline parameters (±SE):**")
    st.dataframe(base_df)
    st.markdown("**Monitoring parameters:**")
    st.dataframe(mon_df)

    # control limits for k1,k2
    k1_base, k2_base = params_base[2], params_base[3]
    k1_se, k2_se     = se_base[2], se_base[3]
    k1_upper = k1_base + cl_factor * k1_se
    k1_lower = max(0.0, k1_base - cl_factor * k1_se)
    k2_upper = k2_base + cl_factor * k2_se
    k2_lower = max(0.0, k2_base - cl_factor * k2_se)

    # phases simulation
    mu1_base, mu1_mon = params_base[0], params_mon[0]
    mu2_base, mu2_mon = params_base[1], params_mon[1]
    sigma_mu1_day = max(1e-6, se_base[0]) * 0.5
    sigma_mu2_day = max(1e-6, se_base[1]) * 0.5
    grid_days, paths_mu1 = simulate_angles(mu1_base, mu1_mon, trend_mu, sigma_mu1_day, horizon_days, step_days, n_paths, seed=42)
    _,         paths_mu2 = simulate_angles(mu2_base, mu2_mon, trend_mu, sigma_mu2_day, horizon_days, step_days, n_paths, seed=43)

    # k1,k2 simulation
    grid_days_k1, paths_k1 = simulate_k_log_ou(k1_base, params_mon[2], k1_se, alpha=alpha_k, horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=44)
    grid_days_k2, paths_k2 = simulate_k_log_ou(k2_base, params_mon[3], k2_se, alpha=alpha_k, horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=45)

    # eta simulation
    eta_base, eta_mon, se_eta = params_base[4], params_mon[4], se_base[4]
    grid_days_eta, paths_eta = simulate_eta_ou(eta_base, eta_mon, se_eta, alpha=alpha_eta, horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, seed=46)

    # probabilities
    prob_mu1 = (paths_mu1 > theta_mu1_days).mean(axis=0)
    prob_mu2 = (paths_mu2 > theta_mu2_days).mean(axis=0)
    prob_k1 = ((paths_k1 > k1_upper) | (paths_k1 < k1_lower)).mean(axis=0)
    prob_k2 = ((paths_k2 > k2_upper) | (paths_k2 < k2_lower)).mean(axis=0)
    prob_eta = (np.abs(paths_eta - eta_base) > theta_eta).mean(axis=0)

    horizons = [90, 180, 270, 360]
    rows = []
    for H in horizons:
        idx = min(np.searchsorted(grid_days, H), len(grid_days)-1)
        rows.append({
            "Horizon_days": grid_days[idx],
            "mu1_P5":   np.percentile(paths_mu1[:, idx], 5),
            "mu1_Med":  np.percentile(paths_mu1[:, idx], 50),
            "mu1_P95":  np.percentile(paths_mu1[:, idx], 95),
            "mu2_P5":   np.percentile(paths_mu2[:, idx], 5),
            "mu2_Med":  np.percentile(paths_mu2[:, idx], 50),
            "mu2_P95":  np.percentile(paths_mu2[:, idx], 95),
            "k1_P5":    np.percentile(paths_k1[:, idx], 5),
            "k1_Med":   np.percentile(paths_k1[:, idx], 50),
            "k1_P95":   np.percentile(paths_k1[:, idx], 95),
            "k2_P5":    np.percentile(paths_k2[:, idx], 5),
            "k2_Med":   np.percentile(paths_k2[:, idx], 50),
            "k2_P95":   np.percentile(paths_k2[:, idx], 95),
            "eta_P5":   np.percentile(paths_eta[:, idx], 5),
            "eta_Med":  np.percentile(paths_eta[:, idx], 50),
            "eta_P95":  np.percentile(paths_eta[:, idx], 95),
            f"P(Delta mu1 > {theta_mu1_days:.1f}d)": prob_mu1[idx],
            f"P(Delta mu2 > {theta_mu2_days:.1f}d)": prob_mu2[idx],
            f"P(k1 outside [{k1_lower:.2f},{k1_upper:.2f}])": prob_k1[idx],
            f"P(k2 outside [{k2_lower:.2f},{k2_upper:.2f}])": prob_k2[idx],
            f"P(|Delta eta| > {theta_eta:.2f})": prob_eta[idx],
        })
    summary = pd.DataFrame(rows)
    st.subheader("Simulation summary (percentiles & probabilities)")
    st.dataframe(summary)

    # phase envelopes
    p5_mu1 = np.percentile(paths_mu1, 5, axis=0)
    p50_mu1 = np.percentile(paths_mu1, 50, axis=0)
    p95_mu1 = np.percentile(paths_mu1, 95, axis=0)
    p5_mu2 = np.percentile(paths_mu2, 5, axis=0)
    p50_mu2 = np.percentile(paths_mu2, 50, axis=0)
    p95_mu2 = np.percentile(paths_mu2, 95, axis=0)
    fig_phase, axp = plt.subplots(figsize=(10,4))
    axp.plot(grid_days, p50_mu1, label="mu1 median (days)", color="navy")
    axp.fill_between(grid_days, p5_mu1, p95_mu1, color="skyblue", alpha=0.35, label="mu1 5–95% band")
    axp.plot(grid_days, p50_mu2, label="mu2 median (days)", color="purple")
    axp.fill_between(grid_days, p5_mu2, p95_mu2, color="plum", alpha=0.35, label="mu2 5–95% band")
    axp.axhline(theta_mu1_days, color="crimson", ls="--", label=f"theta mu1 = {theta_mu1_days:.1f}d")
    axp.axhline(theta_mu2_days, color="orange", ls="--", label=f"theta mu2 = {theta_mu2_days:.1f}d")
    axp.set_xlabel("Days ahead"); axp.set_ylabel("Delta phase (days)"); axp.grid(alpha=0.3); axp.legend()
    st.pyplot(fig_phase)

    # k envelopes
    p5_k1 = np.percentile(paths_k1, 5, axis=0)
    p50_k1 = np.percentile(paths_k1, 50, axis=0)
    p95_k1 = np.percentile(paths_k1, 95, axis=0)
    p5_k2 = np.percentile(paths_k2, 5, axis=0)
    p50_k2 = np.percentile(paths_k2, 50, axis=0)
    p95_k2 = np.percentile(paths_k2, 95, axis=0)
    fig_k, axk = plt.subplots(figsize=(10,4))
    axk.plot(grid_days_k1, p50_k1, label="k1 median", color="teal")
    axk.fill_between(grid_days_k1, p5_k1, p95_k1, color="lightseagreen", alpha=0.35, label="k1 5–95% band")
    axk.plot(grid_days_k2, p50_k2, label="k2 median", color="brown")
    axk.fill_between(grid_days_k2, p5_k2, p95_k2, color="burlywood", alpha=0.35, label="k2 5–95% band")
    axk.axhline(k1_lower, color="gray", ls=":", label="k1 lower CL")
    axk.axhline(k1_upper, color="gray", ls=":", label="k1 upper CL")
    axk.axhline(k2_lower, color="black", ls=":", label="k2 lower CL")
    axk.axhline(k2_upper, color="black", ls=":", label="k2 upper CL")
    axk.set_xlabel("Days ahead"); axk.set_ylabel("Concentration (k)"); axk.grid(alpha=0.3); axk.legend(ncol=2)
    st.pyplot(fig_k)

    # eta envelope
    p5_eta = np.percentile(paths_eta, 5, axis=0)
    p50_eta = np.percentile(paths_eta, 50, axis=0)
    p95_eta = np.percentile(paths_eta, 95, axis=0)
    fig_eta, axe = plt.subplots(figsize=(10,3))
    axe.plot(grid_days_eta, p50_eta, label="eta median", color="darkgreen")
    axe.fill_between(grid_days_eta, p5_eta, p95_eta, color="palegreen", alpha=0.35, label="eta 5–95% band")
    axe.axhline(eta_base + theta_eta, color="crimson", ls="--", label="+theta_eta")
    axe.axhline(eta_base - theta_eta, color="crimson", ls="--", label="-theta_eta")
    axe.set_xlabel("Days ahead"); axe.set_ylabel("eta (skew)"); axe.grid(alpha=0.3); axe.legend()
    st.pyplot(fig_eta)

    # probability curves plot
    fig_prob, axpr = plt.subplots(figsize=(10,5))
    axpr.plot(grid_days, prob_mu1, label=f"P(Delta mu1 > {theta_mu1_days:.1f}d)", color="navy")
    axpr.plot(grid_days, prob_mu2, label=f"P(Delta mu2 > {theta_mu2_days:.1f}d)", color="purple")
    axpr.plot(grid_days_k1, prob_k1, label="P(k1 outside CL)", color="teal")
    axpr.plot(grid_days_k2, prob_k2, label="P(k2 outside CL)", color="brown")
    axpr.plot(grid_days_eta, prob_eta, label=f"P(|Delta eta| > {theta_eta:.2f})", color="darkgreen")
    axpr.set_ylim(0,1); axpr.set_xlabel("Days ahead"); axpr.set_ylabel("Probability"); axpr.grid(alpha=0.3); axpr.legend(ncol=2)
    st.pyplot(fig_prob)

    # download summary
    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download multi-parameter simulation summary (CSV)", data=csv_bytes,
                       file_name="kimberley_multiparameter_simulation_summary.csv", mime="text/csv")

except Exception as e:
    st.error(f"Estimation or simulation failed: {e}")
    st.info("Tip: Ensure sufficient hot-day samples, adjust quantile or windows, and verify network access.")
