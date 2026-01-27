“Hot‑Day Monitoring & Early‑Warning System”
(Simple guide for everyday use – non‑technical)


Methodology behind the APP
##############################################################################################

The app uses a circular distribution to model the annual timing of hot days and then compares the baseline and monitoring parameters via maximum‑likelihood estimation with Hessian‑based standard errors, and then projects future behaviour using drift‑based circular simulations for phase parameters and mean‑reverting Ornstein–Uhlenbeck (OU) processes for concentration and skew parameters through Monte‑Carlo forecasting, while additionally incorporating a machine‑learning model(Gradient Boosting or Random Forest) trained on recent daily temperature features and the circular distribution seasonal probabilities to enhance short‑horizon prediction of monthly hot‑day risk estimates
####################################################################################################

🟩 1. What this tool does
This tool helps you monitor unusual heat patterns and gives early warnings when hot days begin changing compared to normal. It also shows the risk of hot days in the coming months.
 
####################################################################################################
🟩 2. The three things you check
✔️ A. Alerts
This section tells you if recent heat behaviour has shifted in terms of 
(a) Hot days coming earlier or later
(b) Heat becoming more intense or clustered and
(c) The shape/timing of the hot‑day season changing
####################################################################################################
🟩 GREEN = Normal: Everything is behaving as expected.
🟥 RED = Unusual Change: Pay attention. Something has shifted.

✔️ B. Baseline vs Monitoring Curve
This chart compares:
(a) Blue line = what was normal in past years
(b) Red line = what is happening now

If the red line moves away from the blue line, heat patterns are shifting.


✔️ C. Future Risk
This section shows the likelihood of 
(a) Heat patterns continuing to shift
(b) Hot‑day intensity increasing
(c) Unusual heat emerging in coming months
The section also provides a monthly risk table showing how many hot days may occur.

🟩 3. The 5 controls you use most
1️⃣ Baseline period
Choose years representing “normal” (recommended: 2015–2023).

2️⃣ Monitoring period
Choose the recent period you want to analyse (e.g., past 12 months).

3️⃣ Hot‑day threshold: 90% defines what counts as a “hot day”.
Keep at 90% for best general use.
 
4️⃣ Alert thresholds (leave as default). It control how sensitive the system is.
The default values balance early warning and stability.
 
5️⃣ Monthly risk forecast: it shows the expected hot‑day risks from 1 to 3 months ahead.
Recommended settings: 2 months, 5000 simulations, blend = 0.6.

🟥 4. What to do when a RED alert appears
🔥 Phase shift alert: Hot days are arriving earlier/later than normal.
➡️ Monitor closely and notify your supervisor or take action if you are the decision maker.

🔥 Concentration alert:Heat is becoming more intense or tightly packed.
➡️ Prepare for operational impacts.

🔥 Skew or orientation alert: The shape of the heat season is changing.
➡️ Check the monthly risk forecast.

🔥 High future probability (> 50%): The coming month may be significantly hotter.
➡️ Flag for follow‑up or further investigation.

🟩 5. Heatwave detection (optional): Use this section when reporting events:

Enter location (e.g., California)
Keep threshold at 90%
Heatwave length = 3 days

The tool will show detected heatwaves with the dates and duration.
 
🟩 6. Recommended default settings

Baseline:                    2015–2023
Monitoring:                  Last 12 months or as desired
Hot‑day threshold:           90%
Alert thresholds:            keep default settings
Forecast horizon:            365 days
Trend continuation settings: μ1 = 0.6, μ2 = 0.7
Mean‑reversion:              0.05
Monthly risk:                2 months
Blend weight:                0.6
Heatwave length:             Set at 3 days

🟩 7. Daily 30‑second routine
Open the app
Look at Alerts

 If the allert is 🟩GREEN → move on
 If the allert is 🟥 RED → review shift

Check the baseline vs monitoring chart
Check monthly hot‑day risk

🟥 8. When to escalate
Report to your supervisor when:

(a) More than one alert turns 🟥 red
(b) Hot‑day risk > 50% for next month
(c) A new heatwave occurs
(d) Heat timing or intensity changes sharply


Hot Days Benchmarking — Add-on
Click on the ⬇️ Fetch & Analyse Hot Days button to obtain Daily Tmax vs historical percentile threshold and Detected heatwaves within the monitoring period.


Send all your questions to albert.aantwi@outlook.com
