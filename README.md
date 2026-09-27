# Can We Predict Traffic 15 Minutes Ahead?

A model-comparison / forecasting project on a 36-sensor highway traffic dataset from the Northern Virginia / Washington D.C. region. The goal isn't just "train a model" — it's to quantify **how much each type of information (history, calendar, road, neighboring sensors) actually contributes** to 15-minute-ahead traffic forecasts, and to check whether more complex models (deep learning, sequence models) are worth it on a dataset this size.

## Dataset

Zhao, Gkountouna & Pfoser (2019), *ACM Trans. Spatial Algorithms Syst.* 5(3). 36 sensors on two highways, traffic volume sampled every 15 minutes, 1,261 training and 840 test timesteps.

Each timestep's input is a 36×48 matrix (36 sensors × 48 features). The 48 columns were decoded empirically (by checking which contiguous column ranges sum to 1 across every row of the training set) rather than trusted blindly from the dataset's own docs, since the docs undercounted a couple of the feature blocks:

| Columns | Feature | Width |
|---|---|---|
| 0–9 | lagged traffic volume, T-150 → T (col 9 = most recent) | 10 |
| 10–14 | weekday (one-hot) | 5 |
| 15–38 | hour of day (one-hot) | 24 |
| 39 | number of lanes | 1 |
| 40–43 | road direction (one-hot) | 4 |
| 44–47 | road identity (one-hot) | 4 |

`tra_adj_mat` (36×36 binary) gives the spatial connectivity between sensors — used for the Level 4 neighbor-sensor features.

## Approach

Built up in progressively richer information levels, always compared against dumb baselines, so no result is reported without knowing what it beat:

1. **Baselines** — persistence (predict T+15 = T), moving average of the 10 lags, per-sensor per-hour historical mean.
2. **Level 1** — Linear/Ridge, Random Forest, XGBoost on the 10 lagged values only.
3. **Level 2** — + weekday/hour.
4. **Level 3** — + lanes/direction/road identity.
5. **Level 4** — + adjacency-weighted average of neighboring sensors' own lagged history (via `tra_adj_mat`).
6. **Level 5** — a feedforward neural net on the Level-4 tabular features, and an LSTM/GRU treating the 10 lags (own + neighbor) as an explicit sequence.

Train/test split is the one provided with the dataset (temporal, not shuffled) — appropriate for a forecasting task. A "same time yesterday" baseline was deliberately **not** built, since the data gives weekday + hour but no explicit date, and guessing the alignment risked a subtle leak.

## Results

| Level | Best model | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline: persistence | — | 0.0372 | 0.0545 | 0.929 |
| Baseline: moving avg | — | 0.0762 | 0.1082 | 0.720 |
| Baseline: sensor × hour mean | — | 0.0417 | 0.0598 | 0.914 |
| Level 1: 10 lags only | XGBoost | 0.0300 | 0.0437 | 0.954 |
| Level 2: + weekday/hour | XGBoost | 0.0261 | 0.0385 | 0.964 |
| Level 3: + lanes/direction/road | XGBoost | 0.0243 | 0.0360 | 0.969 |
| Level 4: + neighboring sensors | XGBoost | 0.0243 | 0.0360 | 0.969 |
| Level 5a: feedforward NN | NN | **0.0242** | 0.0363 | 0.968 |
| Level 5b: LSTM/GRU (sequence only) | GRU | 0.0301 | 0.0445 | 0.953 |

Full table: `master_comparison_results.csv`. Full narrative writeup: `summary_report.md`.

**Headline findings:**
- Calendar features (Level 2) gave the single largest error reduction of any addition.
- Neighboring-sensor information (Level 4) added essentially nothing once own-history + calendar + road were already in the model — spatial propagation likely matters more at longer forecast horizons than 15 minutes.
- Deep learning did not beat gradient-boosted trees; a sequence-only LSTM/GRU (no calendar/road features) underperformed the tabular models, most likely because the dataset (~45K flattened rows) isn't large enough for recurrence to earn back what it gives up by dropping the engineered features.
- Error is lowest overnight and highest in the early afternoon, not at rush-hour peaks.

## Repo contents

```
traffic_pipeline.py          # baselines + Level 1-3 ablation
level4_neighbors.py          # Level 4: adjacency-weighted neighbor features
level5_neural.py             # Level 5: feedforward NN + LSTM/GRU
consolidate_final.py         # merges all results, regenerates final plots
summary_report.md            # full narrative writeup with caveats
master_comparison_results.csv
baseline_results.csv / level_ablation_results.csv / level4_neighbor_results.csv / level5_neural_results.csv
actual_vs_predicted.png / error_by_hour.png / mae_by_information_level.png
```

Run in order: `traffic_pipeline.py` → `level4_neighbors.py` → `level5_neural.py` → `consolidate_final.py`. Each script reads `traffic_dataset.mat` directly and writes its results/intermediate arrays for the next script to pick up.

## Caveats

- Only 2,101 total instances — treat this as a careful model-comparison project, not a deep-learning showcase.
- Findings describe this NoVA/DC 36-sensor network only; they aren't claimed to generalize to other roads or cities.
- Requires `scipy`, `pandas`, `scikit-learn`, `xgboost`, `torch`.

## Citation

Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019. Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints. *ACM Trans. Spatial Algorithms Syst.* 5, 3, Article 19.
