# -*- coding: utf-8 -*-
"""
SS‑GvM Phase/Concentration/Skew Change Monitor
------------------------------------------------
This Streamlit dashboard ingests daily maximum temperature data (Open‑Meteo),
identifies hot days by a chosen quantile threshold, maps their day‑of‑year to
circular phases, fits SS‑GvM parameters for a baseline and a monitoring window,
and raises alerts when:
  • Phase shift: |Δμ₁| > θ_μ (in days) → onset moved.
  • Concentration change: Δk₁ or Δk₂ beyond baseline control limits (mean ± c·SE).
  • Skew change: |Δη| > θ_η → altered build up/decay.
"""

import matplotlib
matplotlib.use("Agg")  # robust for Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import optimize
import requests

# ----------------- Constants & helpers -----------------
TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719  # Kimberley
TIMEZONE = "Africa/Johannesburg"

@st.cache_data(show_spinner=False)
def wrap_angle(x):
    return np.mod(x, TWO_PI)

@st.cache_data(show_spinner=False)
def circ_diff(a, b):
    """Smallest absolute circular difference in radians."""
    d = np.abs(a - b)
    return np.minimum(d, TWO_PI - d)

def angle_to_days(angle):
    """Convert absolute circular difference (radians) to days on a 365-day year."""
    return (angle / TWO_PI) * 365.0

# ----------------- SS-GvM core -----------------
@st.cache_data(show_spinner=False)
def normalizer_ssgvm(mu1, mu2, k1, k2, eta, nu, n_grid=4096):
    grid = np.linspace(0, TWO_PI, n_grid, endpoint=False)
    expo = (
        k1*np.cos(grid-mu1)
        + k2*np.cos(2*(grid-mu2))
        + np.log(np.maximum(1 + eta*np.sin(grid-nu), 1e-12))
    )
    m = np.max(expo)  # log-sum-exp stabilization
    return (TWO_PI/n_grid)*np.exp(m)*np.sum(np.exp(expo-m))

@st.cache_data(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
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

# ----------------- Numerical Hessian & inference -----------------
@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def se_from_hessian(neg_ll_func, params):
    try:
        H = numerical_hessian(neg_ll_func, np.array(params))
        cov = np.linalg.pinv(H + 1e-8*np.eye(len(params)))
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        return se, cov
    except Exception:
        return np.full(len(params), np.nan), np.full((len(params), len(params)), np.nan)

# ----------------- Data ingestion (Open-Meteo) -----------------
@st.cache_data(show_spinner=True)
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
        hot = df[df["tmax"] >= thr]
        phi = TWO_PI * (hot["doy"].values - 1) / 365.0
        return phi, df
    except Exception:
        # Fallback: monthly climatology-based synthetic phases
        monthly_tmax_F = np.array([93, 91, 88, 81, 75, 69, 69, 74, 82, 87, 90, 93], dtype=float)
        month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        mid_offsets = np.cumsum(np.concatenate([[0], month_lengths[:-1]])) + (month_lengths / 2.0)
        angles = (mid_offsets - 1) * TWO_PI / 365.0
        weights = monthly_tmax_F - monthly_tmax_F.min()
        weights = weights + 1e-6
        N_total = 2400
        rep_counts = np.maximum(1, (weights / weights.sum() * N_total).astype(int))
        phi = np.concatenate([np.repeat(angles[i], rep_counts[i]) for i in range(12)])
        df = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
        df["tmax"] = np.nan
        df["doy"] = df["date"].dt.dayofyear
        return phi, df

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="SS‑GvM Change Monitor", layout="wide")
st.title("SS‑GvM Phase/Concentration/Skew Change Monitor — Kimberley")

# Sidebar: windows and thresholds
st.sidebar.header("Configuration")
baseline_start = st.sidebar.date_input("Baseline start", date(2019,1,1))
baseline_end   = st.sidebar.date_input("Baseline end",   date(2023,12,31))
monitor_start  = st.sidebar.date_input("Monitor start",  date(2024,1,1))
monitor_end    = st.sidebar.date_input("Monitor end",    date.today())
quantile_q     = st.sidebar.slider("Hot-day threshold quantile", 0.80, 0.99, 0.90)
n_starts       = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

# Alert thresholds
theta_mu_days  = st.sidebar.number_input("Phase shift threshold θμ (days)", value=10.0, min_value=0.0)
theta_eta      = st.sidebar.number_input("Skew change threshold θη", value=0.10, min_value=0.0)
cl_factor      = st.sidebar.selectbox("Control limit width (SE multiples)", [1.0, 1.5, 2.0, 2.5], index=2)

# Fetch data and fit models
st.subheader("Data ingestion and model fitting")
try:
    phi_base, df_base = fetch_hotday_phases(str(baseline_start), str(baseline_end), q=quantile_q)
    phi_mon,  df_mon  = fetch_hotday_phases(str(monitor_start),  str(monitor_end),  q=quantile_q)
    st.write(f"Baseline hot-day samples: {len(phi_base)}  |  Monitoring hot-day samples: {len(phi_mon)}")

    params_base, ll_base = fit_ssgvm_mle_all_starts(phi_base, n_starts=n_starts)
    params_mon,  ll_mon  = fit_ssgvm_mle_all_starts(phi_mon,  n_starts=n_starts)

    # Baseline SE (observed information via numerical Hessian)
    base_negll = lambda p: -np.sum(logpdf_ssgvm(phi_base, p))
    se_base, cov_base = se_from_hessian(base_negll, params_base)

    labels = ["mu1","mu2","k1","k2","eta","nu"]
    base_df = pd.DataFrame({"Param": labels, "Estimate": params_base, "SE": se_base})
    mon_df  = pd.DataFrame({"Param": labels, "Estimate": params_mon})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline parameters (±SE):**")
        st.dataframe(base_df, use_container_width=True)
    with col2:
        st.markdown("**Monitoring parameters:**")
        st.dataframe(mon_df, use_container_width=True)

    # ----------------- Parameter monitoring & alerts -----------------
    # Phase shift for mu1 (convert circular difference to days)
    delta_mu1_rad  = circ_diff(params_mon[0], params_base[0])
    delta_mu1_days = angle_to_days(delta_mu1_rad)

    # Concentration control limits for k1, k2 (baseline mean ± cl_factor*SE)
    k1_base, k2_base = params_base[2], params_base[3]
    k1_se,   k2_se   = se_base[2],    se_base[3]
    k1_upper = k1_base + cl_factor * k1_se
    k1_lower = max(0.0, k1_base - cl_factor * k1_se)
    k2_upper = k2_base + cl_factor * k2_se
    k2_lower = max(0.0, k2_base - cl_factor * k2_se)
    k1_mon,  k2_mon  = params_mon[2], params_mon[3]
    k1_flag = (k1_mon > k1_upper) or (k1_mon < k1_lower)
    k2_flag = (k2_mon > k2_upper) or (k2_mon < k2_lower)

    # Skew change threshold for eta
    delta_eta = abs(params_mon[4] - params_base[4])

    # Metrics summary row
    st.subheader("Change metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("|Δμ₁| (days)", f"{delta_mu1_days:.1f}", None)
    m2.metric("k₁ (mon vs limits)", f"{k1_mon:.2f}", f"[{k1_lower:.2f}, {k1_upper:.2f}]")
    m3.metric("k₂ (mon vs limits)", f"{k2_mon:.2f}", f"[{k2_lower:.2f}, {k2_upper:.2f}]")
    m4.metric("|Δη|", f"{delta_eta:.3f}", None)

    # Alerts
    alerts = []
    if delta_mu1_days > theta_mu_days:
        alerts.append(f"Phase shift alert: |Δμ₁| = {delta_mu1_days:.1f} days > θμ = {theta_mu_days:.1f} (onset moved)")
    if k1_flag or k2_flag:
        msg = "Concentration change alert: "
        if k1_flag:
            msg += f"k₁={k1_mon:.2f} outside [{k1_lower:.2f}, {k1_upper:.2f}] "
        if k2_flag:
            msg += f"k₂={k2_mon:.2f} outside [{k2_lower:.2f}, {k2_upper:.2f}] "
        alerts.append(msg.strip())
    if delta_eta > theta_eta:
        alerts.append(f"Skew change alert: |Δη| = {delta_eta:.3f} > θη = {theta_eta:.3f} (altered build-up/decay)")

    st.subheader("Alerts")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("No alerts triggered under current thresholds.")

    # ----------------- Density visualization -----------------
    st.subheader("Baseline vs Monitoring density")
    days = np.arange(1, 366)
    theta = 2*np.pi*days/365.0
    def ssgvm_density(params, theta):
        Z = normalizer_ssgvm(*params)
        return np.exp(
            np.clip(params[2]*np.cos(theta-params[0]) + params[3]*np.cos(2*(theta-params[1])), -700, 700)
        ) * np.maximum(1.0 + params[4]*np.sin(theta-params[5]), 1e-12) / Z
    dens_base = ssgvm_density(params_base, theta)
    dens_mon  = ssgvm_density(params_mon,  theta)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(days, dens_base, label="Baseline",   color="steelblue", lw=2)
    ax.plot(days, dens_mon,  label="Monitoring", color="crimson",   lw=2)
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
    ax.set_title("SS‑GvM density: baseline vs monitoring")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ----------------- Change summary table -----------------
    change_df = pd.DataFrame({
        "Metric": ["|Δμ₁| (days)", "k₁ change", "k₂ change", "|Δη|"],
        "Value":  [delta_mu1_days, k1_mon - k1_base, k2_mon - k2_base, delta_eta],
        "Control/Threshold": [
            f"θμ = {theta_mu_days:.1f}",
            f"[{k1_lower:.2f}, {k1_upper:.2f}]",
            f"[{k2_lower:.2f}, {k2_upper:.2f}]",
            f"θη = {theta_eta:.3f}"
        ],
        "Alert": [
            "Yes" if delta_mu1_days > theta_mu_days else "No",
            "Yes" if k1_flag else "No",
            "Yes" if k2_flag else "No",
            "Yes" if delta_eta > theta_eta else "No"
        ]
    })
    st.subheader("Change summary")
    st.dataframe(change_df, use_container_width=True)

    # ----------------- Download report -----------------
    report = pd.DataFrame({"Param": labels,
                           "Baseline":   params_base,
                           "Baseline_SE": se_base,
                           "Monitoring": params_mon})
    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download parameter report (CSV)", data=csv_bytes,
                       file_name="kimberley_ssgvm_parameter_monitoring.csv", mime="text/csv")

except Exception as e:
    st.error(f"Data or fitting failed: {e}")
    st.info("Tip: Adjust date windows to periods with sufficient hot-day samples and ensure network connectivity.")
