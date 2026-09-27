# Can We Predict Traffic 15 Minutes Ahead?

**Dataset:** Zhao, Gkountouna & Pfoser (2019) — 36 sensors on two highways in the Northern Virginia / Washington D.C. region, 15-minute intervals, 1,261 training and 840 test timesteps (45,396 / 30,240 sensor-timestep rows once flattened). Findings below describe this dataset/network only — they are not claimed to generalize to other roads or cities.

## Feature layout (decoded from the data, not just the provided description)

| Columns | Meaning | Width |
|---|---|---|
| 0–9 | lagged traffic, T-150 → T (col 9 = most recent) | 10 |
| 10–14 | weekday one-hot | 5 |
| 15–38 | hour-of-day one-hot | 24 |
| 39 | number of lanes (int 1–5) | 1 |
| 40–43 | road direction one-hot | 4 |
| 44–47 | road identity one-hot | 4 |

The provided docx description says weekday=7 and road name=1 feature; the actual one-hot blocks in the data are width 5 and width 4 respectively. Treat the numbers above as ground truth for modeling.

## Master results (all models, all information levels)

See `master_comparison_results.csv` for the full table. Summary:

| Level | Best model | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline: persistence (predict T+15 = T) | — | 0.0372 | 0.0545 | 0.929 |
| Baseline: moving avg of 10 lags | — | 0.0762 | 0.1082 | 0.720 |
| Baseline: sensor × hour-of-day mean | — | 0.0417 | 0.0598 | 0.914 |
| Level 1: 10 lags only | XGBoost | 0.0300 | 0.0437 | 0.954 |
| Level 2: + weekday/hour | XGBoost | 0.0261 | 0.0385 | 0.964 |
| Level 3: + lanes/direction/road | XGBoost | 0.0243 | 0.0360 | 0.969 |
| Level 4: + neighboring-sensor lags | XGBoost | 0.0243 | 0.0360 | 0.969 |
| Level 5a: feedforward NN (Level-4 features) | NN | **0.0242** | 0.0363 | 0.968 |
| Level 5b: LSTM/GRU on raw lag sequence | GRU | 0.0301 | 0.0445 | 0.953 |

## What actually moved the needle

1. **Every real model beats the naive persistence baseline**, but not by a huge margin (persistence MAE 0.0372 → best model 0.0242, a ~35% reduction). Traffic 15 minutes out is genuinely close to traffic right now — this is a "hard to beat a strong baseline by a lot" problem, which is itself a legitimate finding.
2. **Calendar features (weekday/hour) were the single biggest lever**: MAE dropped from 0.0300 → 0.0261 (~13% relative reduction) going from Level 1 to Level 2 — bigger than any other single addition.
3. **Road/lane identity (Level 3) added a real but smaller gain**: 0.0261 → 0.0243 (~7% reduction).
4. **Neighboring-sensor information (Level 4) added essentially nothing** on top of Level 3 (0.02435 → 0.02427, within noise). At a 15-minute horizon, a sensor's own recent history plus calendar/road context already captures almost everything the neighbors would tell you — the spatial signal likely matters more at longer horizons (30–60+ min), where congestion has time to propagate along the highway.
5. **Deep learning did not beat gradient boosting.** A plain feedforward NN on the same tabular features essentially tied XGBoost (0.0242 vs 0.0243 — not a meaningfully different result given dataset size). The LSTM/GRU sequence models, given only the raw 10-step lag sequence (no calendar/road features), did clearly *worse* than the tabular models (~0.030) — sequence modeling alone doesn't substitute for the engineered calendar/road context here, and with only ~45k rows there isn't enough data for the recurrent models to earn back that gap.
6. **Error is lowest overnight** (MAE ≈ 0.008–0.016 between 21:00–01:00, when traffic is flat) **and highest midday/early-afternoon** (MAE ≈ 0.030–0.033, roughly 11:00–16:00) — not during the sharpest morning/evening rush-hour transitions as might be expected. This suggests the hardest-to-predict traffic here is moderate, somewhat volatile midday flow rather than the highly regular rush-hour surge.

## Bottom line

> On this NoVA/DC 36-sensor network, historical traffic plus calendar information explains almost all of the predictable variation in 15-minute-ahead traffic volume (R² ≈ 0.97). Road/lane identity adds a modest further gain, neighboring-sensor information adds essentially none at this short horizon, and deep sequence models (LSTM/GRU) underperform simpler tabular gradient-boosted trees once given the same feature richness.

## Caveats

- Dataset is modest (2,101 total instances / 45k+30k sensor-rows once flattened) — not large enough to lean heavily on deep learning; the neural results here should be read as "no clear win," not "deep learning doesn't work for traffic."
- No explicit timestamps are provided (only weekday + hour), so a "same time yesterday" baseline was deliberately not attempted — including it would have required guessing at date alignment.
- All models were evaluated on the dataset's own provided train/test split (temporal, not shuffled), consistent with a forecasting setup.
