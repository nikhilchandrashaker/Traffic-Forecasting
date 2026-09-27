"""
Traffic Flow Prediction — Baselines + Level 1-3 Feature Ablation
Dataset: Zhao, Gkountouna & Pfoser (2019), 36 sensors, NoVA/DC highways.

Feature column layout in the 36x48 per-timestep matrices (confirmed empirically
by checking which contiguous column blocks sum to 1 across every row of the
full training set):
  cols  0-9   (10) : lagged traffic values, T-150 ... T (col 9 = most recent, T)
  cols 10-14  (5)  : weekday one-hot
  cols 15-38  (24) : hour-of-day one-hot
  col  39     (1)  : number of lanes (integer 1-5)
  cols 40-43  (4)  : road direction one-hot
  cols 44-47  (4)  : road identity one-hot
Target (tra_Y_*) is traffic volume at T+15, already normalized to [0,1].

We do NOT use a "same time yesterday" baseline: the arrays give no explicit
date/timestamp, only weekday + hour, so "yesterday" can't be reconstructed
without risking a wrong alignment.
"""
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RNG = 42

# ---------- 1. Load ----------
d = sio.loadmat('/mnt/user-data/uploads/traffic_dataset.mat')

def densify(X_cell):
    n = X_cell.shape[1]
    return np.stack([X_cell[0, i].toarray() for i in range(n)])  # (T, 36, 48)

X_tr = densify(d['tra_X_tr'])          # (1261, 36, 48)
X_te = densify(d['tra_X_te'])          # (840, 36, 48)
Y_tr = d['tra_Y_tr'].T                 # (1261, 36)
Y_te = d['tra_Y_te'].T                 # (840, 36)

T_tr, S, F = X_tr.shape
T_te = X_te.shape[0]

# Flatten to (T*36, 48) / (T*36,) — one row per (timestep, sensor).
# Sensor axis is inner so hour/weekday repeat per timestep block of 36 rows.
Xtr_flat = X_tr.reshape(-1, F)
Xte_flat = X_te.reshape(-1, F)
ytr_flat = Y_tr.reshape(-1)
yte_flat = Y_te.reshape(-1)

hour_tr = X_tr[:, :, 15:39].argmax(axis=2).reshape(-1)   # 0-23, per row
hour_te = X_te[:, :, 15:39].argmax(axis=2).reshape(-1)

# ---------- 2. Baselines (no fitting needed) ----------
def evaluate(name, y_true, y_pred, rows):
    rows.append({
        'model': name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'R2': r2_score(y_true, y_pred),
    })

results = []

# Persistence: predict T+15 = current value (col 9, most recent lag)
evaluate('Baseline: Persistence (T)', yte_flat, Xte_flat[:, 9], results)

# Moving average of the 10 lags
evaluate('Baseline: Moving avg (10 lags)', yte_flat, Xte_flat[:, 0:10].mean(axis=1), results)

# Per-sensor, per-hour mean traffic from the TRAINING set, applied to test
train_df = pd.DataFrame({
    'sensor': np.tile(np.arange(S), T_tr),
    'hour': hour_tr,
    'y': ytr_flat,
})
hour_sensor_mean = train_df.groupby(['sensor', 'hour'])['y'].mean()
sensor_te = np.tile(np.arange(S), T_te)
seasonal_pred = np.array([
    hour_sensor_mean.get((s, h), ytr_flat.mean())
    for s, h in zip(sensor_te, hour_te)
])
evaluate('Baseline: Sensor x hour-of-day mean', yte_flat, seasonal_pred, results)

# ---------- 3. Level 1-3 ablation with real models ----------
feature_levels = {
    'Level 1: 10 lags only': list(range(0, 10)),
    'Level 2: + weekday/hour': list(range(0, 10)) + list(range(10, 39)),
    'Level 3: + lanes/direction/road': list(range(0, 48)),
}

model_builders = {
    'Linear/Ridge': lambda: Ridge(alpha=1.0),
    'Random Forest': lambda: RandomForestRegressor(
        n_estimators=300, max_depth=12, n_jobs=-1, random_state=RNG),
}
if HAS_XGB:
    model_builders['XGBoost'] = lambda: XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1)

ablation_rows = []
best_model_preds = {}  # keyed by level name -> (y_true, y_pred) for the best model, for plotting

for level_name, cols in feature_levels.items():
    Xtr_l = Xtr_flat[:, cols]
    Xte_l = Xte_flat[:, cols]
    level_scores = []
    for model_name, build in model_builders.items():
        model = build()
        model.fit(Xtr_l, ytr_flat)
        pred = model.predict(Xte_l)
        row = {
            'level': level_name,
            'model': model_name,
            'MAE': mean_absolute_error(yte_flat, pred),
            'RMSE': mean_squared_error(yte_flat, pred) ** 0.5,
            'R2': r2_score(yte_flat, pred),
        }
        ablation_rows.append(row)
        level_scores.append((row['MAE'], model_name, pred))
    level_scores.sort(key=lambda r: r[0])
    best_model_preds[level_name] = (level_scores[0][1], level_scores[0][2])

ablation_df = pd.DataFrame(ablation_rows)
baseline_df = pd.DataFrame(results)

# ---------- 4. Save outputs ----------
baseline_df.to_csv('/mnt/user-data/outputs/baseline_results.csv', index=False)
ablation_df.to_csv('/mnt/user-data/outputs/level_ablation_results.csv', index=False)

# Best model overall (lowest MAE among all ablation rows)
best_row = ablation_df.loc[ablation_df['MAE'].idxmin()]
print("=== Baselines ===")
print(baseline_df.to_string(index=False))
print("\n=== Level 1-3 Ablation ===")
print(ablation_df.to_string(index=False))
print(f"\nBest overall: {best_row['model']} @ {best_row['level']} "
      f"(MAE={best_row['MAE']:.4f}, RMSE={best_row['RMSE']:.4f}, R2={best_row['R2']:.4f})")

np.save('/home/claude/yte_flat.npy', yte_flat)
np.save('/home/claude/hour_te.npy', hour_te)
import pickle
with open('/home/claude/best_model_preds.pkl', 'wb') as f:
    pickle.dump(best_model_preds, f)
