
# -*- coding: utf-8 -*-
# SS-GvM Monitoring + Phase-Shift Simulation — Kimberley
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
        init = np.array([np.random.rand()*TWO_PI, np.random.rand()*TWO_PI, np.random.gamma(2.0,1.0), np.random.gamma(2.0,1.0), np.tanh(np.random.randn()), np.random.rand()*TWO_PI])
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
    params = {"latitude": LAT, "longitude": LON, "start_date": start_date, "end_date": end_date, "daily": ["temperature_2m_max"], "timezone": TIMEZONE}
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

def simulate_phase_shift(mu1_base, mu1_mon, k1_mon, se_mu1_base, horizon_days=365, step_days=7, n_paths=5000, trend_factor=0.5, seed=42):
    rng = np.random.default_rng(seed)
    delta_obs_rad = circ_diff(mu1_mon, mu1_base)
    drift_rad_per_day = trend_factor * (delta_obs_rad / 365.0)
    base_sigma = se_mu1_base
    sigma_rad_per_day = base_sigma * (1.0 / (1.0 + max(1e-6, k1_mon))) * 0.5
    n_steps = int(np.ceil(horizon_days/step_days))
    grid_days = np.arange(1, n_steps+1) * step_days
    paths = np.empty((n_paths, n_steps), dtype=float)
    mu = np.full(n_paths, mu1_mon)
    for t in range(n_steps):
        dt = step_days
        noise = rng.normal(0.0, sigma_rad_per_day*np.sqrt(dt), size=n_paths)
        mu = wrap_angle(mu + drift_rad_per_day*dt + noise)
        paths[:, t] = angle_to_days(circ_diff(mu, mu1_base))
    horizons = [90, 180, 270, 360]
    rows = []
    for H in horizons:
        idx = min(np.searchsorted(grid_days, H), n_steps-1)
        vals = paths[:, idx]
        p5, p50, p95 = np.percentile(vals, [5,50,95])
        rows.append({"Horizon_days": grid_days[idx], "P5": p5, "Median": p50, "P95": p95})
    summary = pd.DataFrame(rows)
    return grid_days, paths, summary

st.title("SS-GvM Monitoring + Phase-Shift Simulation — Kimberley")

st.sidebar.header("Configuration")
baseline_start = st.sidebar.date_input("Baseline start", date(2019,1,1))
baseline_end   = st.sidebar.date_input("Baseline end",   date(2023,12,31))
monitor_start  = st.sidebar.date_input("Monitor start",  date(2024,1,1))
monitor_end    = st.sidebar.date_input("Monitor end",    date.today())
quantile_q     = st.sidebar.slider("Hot-day threshold quantile", 0.80, 0.99, 0.90)
n_starts       = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

# Alert thresholds
theta_mu_days  = st.sidebar.number_input("Phase shift threshold (days)", value=10.0, min_value=0.0)
theta_eta      = st.sidebar.number_input("Skew change threshold", value=0.10, min_value=0.0)
cl_factor      = st.sidebar.selectbox("Control limit width (SE multiples)", [1.0, 1.5, 2.0, 2.5], index=2)

st.subheader("Data ingestion and model fitting")
labels = ["mu1","mu2","k1","k2","eta","nu"]

try:
    phi_base, df_base = fetch_hotday_phases(str(baseline_start), str(baseline_end), q=quantile_q)
    phi_mon,  df_mon  = fetch_hotday_phases(str(monitor_start),  str(monitor_end),  q=quantile_q)
    st.write(f"Baseline hot-day samples: {len(phi_base)} | Monitoring hot-day samples: {len(phi_mon)}")

    params_base, ll_base = fit_ssgvm_mle_all_starts(phi_base, n_starts=n_starts)
    params_mon,  ll_mon  = fit_ssgvm_mle_all_starts(phi_mon,  n_starts=n_starts)

    base_negll = lambda p: -np.sum(logpdf_ssgvm(phi_base, p))
    se_base, cov_base = se_from_hessian(base_negll, params_base)

    base_df = pd.DataFrame({"Param": labels, "Estimate": params_base, "SE": se_base})
    mon_df  = pd.DataFrame({"Param": labels, "Estimate": params_mon})
    st.markdown("**Baseline parameters (±SE):**")
    st.dataframe(base_df)
    st.markdown("**Monitoring parameters:**")
    st.dataframe(mon_df)

    # Alerts
    delta_mu1_rad  = circ_diff(params_mon[0], params_base[0])
    delta_mu1_days = angle_to_days(delta_mu1_rad)

    k1_base, k2_base = params_base[2], params_base[3]
    k1_se,   k2_se   = se_base[2], se_base[3]
    k1_upper = k1_base + cl_factor * k1_se
    k1_lower = max(0.0, k1_base - cl_factor * k1_se)
    k2_upper = k2_base + cl_factor * k2_se
    k2_lower = max(0.0, k2_base - cl_factor * k2_se)

    k1_mon, k2_mon = params_mon[2], params_mon[3]
    k1_flag = (k1_mon > k1_upper) or (k1_mon < k1_lower)
    k2_flag = (k2_mon > k2_upper) or (k2_mon < k2_lower)
    delta_eta = abs(params_mon[4] - params_base[4])

    alerts = []
    if delta_mu1_days > theta_mu_days:
        alerts.append(f"Phase shift alert: Delta mu1 = {delta_mu1_days:.1f} days > theta_mu = {theta_mu_days:.1f}")
    if k1_flag or k2_flag:
        msg = "Concentration change alert: "
        if k1_flag:
            msg += f"k1={k1_mon:.2f} outside [{k1_lower:.2f}, {k1_upper:.2f}] "
        if k2_flag:
            msg += f"k2={k2_mon:.2f} outside [{k2_lower:.2f}, {k2_upper:.2f}] "
        alerts.append(msg.strip())
    if delta_eta > theta_eta:
        alerts.append(f"Skew change alert: Delta eta = {delta_eta:.3f} > theta_eta = {theta_eta:.3f}")

    st.subheader("Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("No alerts triggered under current thresholds.")

    # Density visualization
    st.subheader("Baseline vs Monitoring density")
    days = np.arange(1, 366)
    theta = 2*np.pi*days/365.0
    def ssgvm_density(params, theta):
        Z = normalizer_ssgvm(*params)
        return np.exp(np.clip(params[2]*np.cos(theta-params[0]) + params[3]*np.cos(2*(theta-params[1])), -700, 700)) * np.maximum(1.0 + params[4]*np.sin(theta-params[5]), 1e-12) / Z
    dens_base = ssgvm_density(params_base, theta)
    dens_mon  = ssgvm_density(params_mon,  theta)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(days, dens_base, label="Baseline", color="steelblue", lw=2)
    ax.plot(days, dens_mon,  label="Monitoring", color="crimson",  lw=2)
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
    ax.set_title("SS-GvM density: baseline vs monitoring")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Change summary + report
    change_df = pd.DataFrame({
        "Metric": ["Delta mu1 (days)", "k1 change", "k2 change", "Delta eta"],
        "Value":  [delta_mu1_days, k1_mon - k1_base, k2_mon - k2_base, delta_eta],
        "Control/Threshold": [f"theta_mu = {theta_mu_days:.1f}", f"[{k1_lower:.2f}, {k1_upper:.2f}]", f"[{k2_lower:.2f}, {k2_upper:.2f}]", f"theta_eta = {theta_eta:.3f}"],
        "Alert": ["Yes" if delta_mu1_days > theta_mu_days else "No", "Yes" if k1_flag else "No", "Yes" if k2_flag else "No", "Yes" if delta_eta > theta_eta else "No"]
    })
    st.subheader("Change summary")
    st.dataframe(change_df)

    report = pd.concat([
        pd.DataFrame({"Param": labels, "Baseline": params_base, "Baseline_SE": se_base}),
        pd.DataFrame({"Param": labels, "Monitoring": params_mon})
    ], axis=1)
    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download parameter report (CSV)", data=csv_bytes, file_name="kimberley_ssgvm_parameter_monitoring.csv", mime="text/csv")

    # ---------------- Simulation (auto-fit + manual override) -----------------
    st.subheader("Phase-shift simulation (Delta mu1) over next year")
    st.markdown("Uses fitted mu1 (baseline vs monitoring), k1 (monitoring), and SE(mu1 baseline). Alternatively, provide manual overrides.")

    st.sidebar.header("Simulation settings")
    horizon_days = st.sidebar.slider("Simulation horizon (days)", 180, 365, 365, step=15)
    step_days    = st.sidebar.selectbox("Time step (days)", [1, 3, 7, 14, 30], index=2)
    n_paths      = st.sidebar.slider("Monte Carlo paths", 1000, 20000, 5000, step=1000)
    trend_factor = st.sidebar.slider("Trend continuation factor", 0.0, 1.0, 0.5)
    use_manual   = st.sidebar.checkbox("Manual overrides for simulation", value=False)

    if use_manual:
        mu1_base_in = st.sidebar.number_input("mu1 baseline (radians)", min_value=0.0, max_value=float(TWO_PI), value=float(params_base[0]))
        mu1_mon_in  = st.sidebar.number_input("mu1 monitoring (radians)", min_value=0.0, max_value=float(TWO_PI), value=float(params_mon[0]))
        k1_mon_in   = st.sidebar.number_input("k1 monitoring", min_value=0.0, value=float(params_mon[2]))
        se_mu1_in   = st.sidebar.number_input("SE(mu1 baseline, radians)", min_value=0.0, value=float(se_base[0]) if np.isfinite(se_base[0]) else 0.05)
        mu1_base_sim, mu1_mon_sim, k1_mon_sim, se_mu1_sim = mu1_base_in, mu1_mon_in, k1_mon_in, se_mu1_in
    else:
        mu1_base_sim, mu1_mon_sim, k1_mon_sim = float(params_base[0]), float(params_mon[0]), float(params_mon[2])
        se_mu1_sim = float(se_base[0]) if np.isfinite(se_base[0]) else 0.05

    if st.button("Run simulation"):
        grid_days, paths, summary = simulate_phase_shift(mu1_base_sim, mu1_mon_sim, k1_mon_sim, se_mu1_sim, horizon_days=horizon_days, step_days=step_days, n_paths=n_paths, trend_factor=trend_factor, seed=42)
        st.dataframe(summary)
        p5, p50, p95 = np.percentile(paths, [5,50,95], axis=0)
        fig_s, ax_s = plt.subplots(figsize=(10,4))
        ax_s.plot(grid_days, p50, label="Median", color="navy")
        ax_s.fill_between(grid_days, p5, p95, color="skyblue", alpha=0.4, label="5–95% band")
        ax_s.axhline(theta_mu_days, color="crimson", ls="--", label=f"Alert threshold = {theta_mu_days:.1f}d")
        ax_s.set_xlabel("Days ahead"); ax_s.set_ylabel("Delta mu1 (days)"); ax_s.set_title("Projected phase shift (next year)"); ax_s.legend(); ax_s.grid(alpha=0.3)
        st.pyplot(fig_s)
        probs = (paths > theta_mu_days).mean(axis=0)
        fig_p, ax_p = plt.subplots(figsize=(10,3))
        ax_p.plot(grid_days, probs, color="darkgreen")
        ax_p.set_ylim(0,1); ax_p.set_xlabel("Days ahead"); ax_p.set_ylabel("Probability"); ax_p.set_title(f"P(Delta mu1 > {theta_mu_days:.1f} days)"); ax_p.grid(alpha=0.3)
        st.pyplot(fig_p)
        sim_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download simulation summary (CSV)", data=sim_csv, file_name="kimberley_phase_shift_simulation_summary.csv", mime="text/csv")

except Exception as e:
    st.error(f"Data, fitting, or simulation failed: {e}")
    st.info("Tip: Adjust date windows to periods with sufficient hot-day samples and ensure network connectivity.")
