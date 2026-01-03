#GHS‑GvM Phase shift, Concentration and Skew Change Monitor — Kimberley 

Overview 

This dashboard monitors changes in the seasonal timing and distribution of hot days using the Generalized Harmonic Skew von Mises (GHS-vM) model. It compares two-time windows — Baseline and Monitoring — to detect shifts in key parameters that describe: 

Phase (μ₁): Timing of the primary mode (onset of hot-day season). 

Concentration (k₁, k₂): Strength of clustering around modal directions. 

Skewness (η): Asymmetry in build-up and decay of hot-day probability. 

The dashboard raises alerts when these changes exceed user-defined thresholds. 

 

Key Features 

Data ingestion: Retrieves daily maximum temperature data from Open‑Meteo for Kimberley. 

Hot-day identification: Uses a percentile threshold (default 90th percentile) to classify hot days. 

Circular transformation: Converts day-of-year to an angle $\phi = 2\pi \times (\text{DOY}-1)/365$ to capture annual cyclicity. 

Model fitting: Fits SS‑GvM parameters for Baseline and Monitoring periods using multi-start L‑BFGS‑B maximum likelihood estimation. 

Alerts: 

Phase Shift: |Δμ₁| > θμ (days) → onset moved. 

Concentration Change: k₁ or k₂ outside baseline control limits (mean ± c·SE). 

Skew Change: |Δη| > θη → altered build-up/decay. 

Visualizations: 

Parameter tables for Baseline and Monitoring. 

Metrics summary and alert indicators. 

Density plot comparing Baseline vs Monitoring SS‑GvM distributions. 

Downloadable report: Export parameter estimates and alerts as CSV. 

 

Why It Matters 

Monitoring these changes provides early warnings for shifts in seasonal heat patterns, enabling proactive planning for: 

Water resource management (anticipating demand spikes). 

Infrastructure stress (heat-related load changes). 

Climate adaptation strategies (tracking long-term shifts). 

 

Installation 

1     # Clone the repository 

2     git clone https://github.com/<your-username>/kimberley-ssgvm-dashboard.git 

3     cd kimberley-ssgvm-dashboard 

4      

5     # Create and activate a virtual environment 

6     python -m venv .venv 

7     source .venv/bin/activate   # Windows: .venv\Scripts\activate 

8      

9     # Install dependencies 

10     pip install -r requirements.txt 

11      

 

Usage 

Run the dashboard locally: 

1     streamlit run app_alerts.py 

2      

Sidebar Configuration 

Baseline & Monitoring windows: Select date ranges for comparison. 

Hot-day threshold: Choose percentile (e.g., 0.90). 

MLE multi-starts: Number of random initializations for robust fitting. 

Alert thresholds: 

θμ (days): Phase shift threshold. 

θη: Skew change threshold. 

Control limit width: SE multiples for concentration parameters. 

 

Deployment 

To deploy on Streamlit Cloud: 

Push app_alerts.py and requirements.txt to your GitHub repo. 

Go to https://share\.streamlit\.io → New app. 

Select your repo and branch. 

Set Main file path to app_alerts.py. 

Click Deploy. 

 

Alert Logic Summary 

Phase Shift Alert: Triggered when |Δμ₁| (converted to days) exceeds θμ. 

Concentration Alert: Triggered when k₁ or k₂ falls outside baseline control limits. 

Skew Alert: Triggered when |Δη| exceeds θη. 

 

Outputs 

Alerts section: Displays triggered warnings. 

Density plot: Visual comparison of Baseline vs Monitoring distributions. 

Change summary table: Shows metric values, thresholds, and alert status. 

Download button: Exports parameter report as CSV. 

 

Requirements 

Python 3.9+ 

streamlit 

pandas 

numpy 

matplotlib 

scipy 

requests 

 

License 

MIT License . 
