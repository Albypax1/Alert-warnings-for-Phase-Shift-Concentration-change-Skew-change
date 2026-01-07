# !pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy import optimize
import requests

# -------------------- Constants & helpers --------------------
TWO_PI = 2*np.pi
LAT, LON = -28.7419, 24.7719  # Kimberley
TIMEZONE = "Africa/Johannesburg"

def wrap_angle(x):
    return np.mod(x, TWO_PI)

def exp_safe(expo):
    return np.exp(np.clip(expo, -700, 700))

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

# -------------------- Data ingestion (Open-Meteo) --------------------

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

# -------------------- Streamlit UI --------------------
st.title("Kimberley SS‑GvM monitoring with stochastic treated water level simulation")

st.sidebar.header("Model configuration")
start_date = st.sidebar.date_input("Data start", date(2019,1,1))
end_date = st.sidebar.date_input("Data end", date.today())
hot_q = st.sidebar.slider("Hot-day quantile (threshold)", 0.80, 0.99, 0.90)
n_starts = st.sidebar.slider("MLE multi-starts", 10, 60, 30, step=5)

st.sidebar.header("System parameters")
current_level_m3 = st.sidebar.number_input("Current tank level (m³)", value=8000.0, min_value=0.0)
buffer_level_m3  = st.sidebar.number_input("Operational buffer level (m³)", value=6000.0, min_value=0.0)
tank_volume_m3   = st.sidebar.number_input("Tank nominal volume (m³)", value=10000.0, min_value=1.0)
production_m3pd  = st.sidebar.number_input("Plant production (m³/day)", value=9500.0, min_value=0.0)
alpha_base       = st.sidebar.number_input("Demand base α (m³/day)", value=8000.0, min_value=0.0)
beta_hot         = st.sidebar.number_input("Demand sensitivity β (m³/day per hot day)", value=4000.0, min_value=0.0)
gamma_evap       = st.sidebar.number_input("Evap coefficient γ (m³/day per hot day)", value=300.0, min_value=0.0)

st.sidebar.header("Simulation configuration")
horizon_days = st.sidebar.slider("Simulation horizon (days)", 3, 30, 7)
n_sims       = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
seed         = st.sidebar.number_input("Random seed (optional)", value=0, min_value=0, step=1)
sigma_demand = st.sidebar.number_input("Demand noise σ_d (m³/day)", value=600.0, min_value=0.0)
sigma_evap   = st.sidebar.number_input("Evap noise σ_e (m³/day)", value=80.0, min_value=0.0)
sigma_prod   = st.sidebar.number_input("Production noise σ_p (m³/day)", value=400.0, min_value=0.0)
use_hot_indicator = st.sidebar.checkbox("Sample hot days Bernoulli(p)", value=True,
                                        help="If checked, daily 'hot' is sampled ~ Bernoulli(HotProb). If unchecked, uses probability as continuous intensity.")

if seed:
    np.random.seed(int(seed))

# -------------------- Fit SS-GvM and build daily hot probabilities --------------------
st.subheader("Data ingestion and SS‑GvM fit")
phi_samples, df_temp = fetch_hotday_phases(str(start_date), str(end_date), q=hot_q)
st.write(f"Hot-day samples: {len(phi_samples)}")
params, ll = fit_ssgvm_mle_all_starts(phi_samples, n_starts=n_starts)
mu1, mu2, k1, k2, eta, nu = params

days = np.arange(1, 366)
theta = 2*np.pi*days/365.0
Z = normalizer_ssgvm(*params)
density = np.exp(np.clip(k1*np.cos(theta-mu1) + k2*np.cos(2*(theta-mu2)), -700, 700)) \
          * np.maximum(1.0 + eta*np.sin(theta - nu), 1e-12) / Z
hotprob = density / density.sum()  # per-day probability (sum over 365 = 1)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(days, density, color="crimson", lw=2, label="SS‑GvM density")
ax.set_xlabel("Day of Year"); ax.set_ylabel("Probability density")
ax.set_title("SS‑GvM daily density (Kimberley)")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# -------------------- Monte Carlo simulation of treated water levels --------------------
st.subheader("Stochastic treated water level simulation")
doy = doy_today()
idxs = np.clip(np.arange(doy - 1, doy - 1 + horizon_days), 0, 365 - 1)
probs_window = hotprob[idxs]

levels = np.zeros((n_sims, horizon_days))
demand_mat = np.zeros((n_sims, horizon_days))
evap_mat   = np.zeros((n_sims, horizon_days))

for s in range(n_sims):
    level = current_level_m3
    for d, idx in enumerate(idxs):
        p_hot = probs_window[d]
        if use_hot_indicator:
            hot = np.random.rand() < p_hot
            # demand/evap jump on hot day
            demand = alpha_base + (beta_hot if hot else 0.0) + np.random.normal(0.0, sigma_demand)
            evap   = (gamma_evap if hot else 0.0) + np.random.normal(0.0, sigma_evap)
        else:
            # intensity model: expected uplift scaled by probability
            demand = alpha_base + beta_hot*p_hot + np.random.normal(0.0, sigma_demand)
            evap   = gamma_evap*p_hot + np.random.normal(0.0, sigma_evap)
        demand = max(0.0, demand)
        evap   = max(0.0, evap)
        prod   = max(0.0, np.random.normal(production_m3pd, sigma_prod))
        level  = np.clip(level + (prod - demand - evap), 0.0, tank_volume_m3)
        levels[s, d] = level
        demand_mat[s, d] = demand
        evap_mat[s, d] = evap

# Percentiles for fan chart
q_lo, q_q1, q_med, q_q3, q_hi = np.percentile(levels, [5, 25, 50, 75, 95], axis=0)

fig2, ax2 = plt.subplots(figsize=(10,4))
x = np.arange(1, horizon_days+1)
ax2.fill_between(x, q_lo, q_hi, color='steelblue', alpha=0.20, label='5–95% interval')
ax2.fill_between(x, q_q1, q_q3, color='steelblue', alpha=0.35, label='25–75% (IQR)')
ax2.plot(x, q_med, color='navy', lw=2, label='Median')
ax2.axhline(buffer_level_m3, color='red', ls='--', label='Buffer level')
ax2.set_xlabel("Days ahead"); ax2.set_ylabel("Level (m³)")
ax2.set_title("Simulated treated water level (fan chart)")
ax2.legend(); ax2.grid(alpha=0.3)
st.pyplot(fig2)

# Table of daily summaries
calendar_doy = np.array([(doy + i) if (doy + i) <= 365 else (doy + i - 365) for i in range(horizon_days)])
df_daily = pd.DataFrame({
    "DayAhead": x,
    "CalendarDOY": calendar_doy,
    "HotProb": np.round(probs_window, 5),
    "Level_p05": np.round(q_lo, 1),
    "Level_p25": np.round(q_q1, 1),
    "Level_p50": np.round(q_med, 1),
    "Level_p75": np.round(q_q3, 1),
    "Level_p95": np.round(q_hi, 1),
})
st.dataframe(df_daily)

# -------------------- Alerts and recommended actions (probabilistic) --------------------
st.subheader("Alerts")
min_levels = levels.min(axis=1)
prob_breach_horizon = float(np.mean(min_levels <= buffer_level_m3))
prob_breach_48h = float(np.mean((levels[:, :min(2,horizon_days)] <= buffer_level_m3).any(axis=1)))
current_breach = current_level_m3 <= buffer_level_m3

# Heat pressure index retained from original: sum over next 7 days of HotProb
window7 = np.arange(doy, min(doy+7, 365+1))
heat_pressure = hotprob[window7 - 1].sum()

high_density_thresh = st.sidebar.slider("High-density threshold (7-day sum)", 0.30, 1.20, 0.60, step=0.05)
high_density = heat_pressure >= high_density_thresh

# Tiering logic (probabilistic)
tier = "Normal"
reasons = []
if current_breach or prob_breach_48h >= 0.50:
    tier = "Warning"
    reasons.append(f"Pr(breach ≤48h) = {prob_breach_48h:.0%} or current level ≤ buffer")
elif prob_breach_horizon >= 0.30 and high_density:
    tier = "Watch"
    reasons.append(f"Pr(breach within {horizon_days}d) = {prob_breach_horizon:.0%} with high heat pressure")
elif prob_breach_horizon >= 0.15:
    tier = "Advisory"
    reasons.append(f"Pr(breach within {horizon_days}d) = {prob_breach_horizon:.0%}")

st.write(f"Tier: {tier}")
if reasons:
    for r in reasons:
        st.error(r)
else:
    st.success("No immediate risks under current thresholds.")

st.subheader("Recommended actions")
if tier == "Warning":
    st.markdown("- **Production:** Increase output immediately; ensure CT compliance while ramping.\n- **Network:** Prioritize critical zones; schedule inter-reservoir transfers; throttle non-critical branches.\n- **Monitoring:** Hourly tank level and filter effluent; intensify leak hunt during night flow.\n- **Chemicals:** Prepare high-dose coagulation and PAC if water quality stress is concurrent.")
elif tier == "Watch":
    st.markdown("- **Production:** Pre-emptive output increase; stage staff and deliveries.\n- **Network:** Plan transfers to protect tank; adjust booster schedules to off-peak.\n- **Monitoring:** Tighten turbidity alarms; shorten filter runs.\n- **Comms:** Ready demand management messaging for peak hours.")
elif tier == "Advisory":
    st.markdown("- **Production:** Verify ramp capability; run jar tests to confirm dose curves.\n- **Network:** Check pump readiness; validate valve and transfer paths.\n- **Monitoring:** Calibrate instruments; track daily level vs simulation median.")
else:
    st.markdown("- **Maintain normal operations.** Continue daily monitoring.")

# -------------------- Downloads --------------------
st.subheader("Downloads")
param_df = pd.DataFrame({"Param": ["mu1","mu2","k1","k2","eta","nu"], "Estimate": params})
csv_params = param_df.to_csv(index=False).encode("utf-8")
st.download_button("Download SS‑GvM parameters (CSV)", data=csv_params, file_name="kimberley_ssgvm_params.csv", mime="text/csv")

# Ensemble daily summary
csv_daily = df_daily.to_csv(index=False).encode("utf-8")
st.download_button("Download simulation daily summary (CSV)", data=csv_daily, file_name="treated_level_sim_summary.csv", mime="text/csv")

# A sample of raw paths (first 50 sims)
raw_paths = pd.DataFrame(levels[:min(50, n_sims)], columns=[f"Day{d+1}" for d in range(horizon_days)])
csv_paths = raw_paths.to_csv(index=False).encode("utf-8")
st.download_button("Download raw simulation paths (CSV)", data=csv_paths, file_name="treated_level_sim_paths.csv", mime="text/csv")
