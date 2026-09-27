import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

baseline_df = pd.read_csv('/mnt/user-data/outputs/baseline_results.csv')
baseline_df.insert(0, 'level', 'Baseline')

l123_df = pd.read_csv('/mnt/user-data/outputs/level_ablation_results.csv')
l4_df = pd.read_csv('/mnt/user-data/outputs/level4_neighbor_results.csv')
l5_df = pd.read_csv('/mnt/user-data/outputs/level5_neural_results.csv')

master = pd.concat([baseline_df, l123_df, l4_df, l5_df], ignore_index=True)
master = master.sort_values('MAE').reset_index(drop=True)
master.to_csv('/mnt/user-data/outputs/master_comparison_results.csv', index=False)
print(master.to_string(index=False))

best = master.iloc[0]
print(f"\nOverall best: {best['model']} @ {best['level']}  MAE={best['MAE']:.4f}  RMSE={best['RMSE']:.4f}  R2={best['R2']:.4f}")

# ---------- Regenerate diagnostic plots using the overall best model's predictions ----------
S = 36
yte_flat = np.load('/home/claude/yte_flat.npy')
hour_te = np.load('/home/claude/hour_te.npy')
T_te = len(yte_flat) // S

with open('/home/claude/level5_preds.pkl', 'rb') as f:
    level5_preds = pickle.load(f)
with open('/home/claude/level4_preds.pkl', 'rb') as f:
    level4_preds = pickle.load(f)
with open('/home/claude/best_model_preds.pkl', 'rb') as f:
    level123_preds = pickle.load(f)  # dict: level_name -> (model_name, pred)

# map best row to a prediction array
if best['level'].startswith('Level 5'):
    pred_flat = level5_preds[best['model']]
elif best['level'].startswith('Level 4'):
    pred_flat = level4_preds[best['model']]
elif best['level'].startswith('Level'):
    _, pred_flat = level123_preds[best['level']]
else:
    raise ValueError('unexpected best level for plotting')

y_mat = yte_flat.reshape(T_te, S)
p_mat = pred_flat.reshape(T_te, S)

sensor_idx = 0
window = 200
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(y_mat[:window, sensor_idx], label='Actual', color='#1f77b4', linewidth=1.6)
ax.plot(p_mat[:window, sensor_idx], label=f"Predicted ({best['model']}, {best['level']})",
        color='#d62728', linewidth=1.3, linestyle='--')
ax.set_title(f'Actual vs Predicted Traffic — Sensor {sensor_idx}, first {window} test intervals (best model)')
ax.set_xlabel('Test time step (15-min intervals)')
ax.set_ylabel('Normalized traffic volume')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/actual_vs_predicted.png', dpi=150)
plt.close(fig)

abs_err = np.abs(y_mat - p_mat).reshape(-1)
mae_by_hour = np.array([abs_err[hour_te == h].mean() for h in range(24)])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(24), mae_by_hour, color='#4c72b0')
ax.set_title(f"Prediction Error (MAE) by Hour of Day — {best['model']}, {best['level']}")
ax.set_xlabel('Hour of day')
ax.set_ylabel('MAE (normalized traffic units)')
ax.set_xticks(range(24))
ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/error_by_hour.png', dpi=150)
plt.close(fig)

# ---------- MAE-vs-information-level summary chart ----------
level_order = [
    'Baseline',
    'Level 1: 10 lags only',
    'Level 2: + weekday/hour',
    'Level 3: + lanes/direction/road',
    'Level 4: + neighbor sensors',
    'Level 5: Neural Net (tabular, Level-4 features)',
    'Level 5: Sequence model (own + neighbor lag sequence)',
]
best_per_level = master.loc[master.groupby('level')['MAE'].idxmin()]
best_per_level = best_per_level.set_index('level').reindex(level_order).dropna()

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(range(len(best_per_level)), best_per_level['MAE'], marker='o', color='#2ca02c', linewidth=2)
for i, (lvl, row) in enumerate(best_per_level.iterrows()):
    ax.annotate(f"{row['model']}\n{row['MAE']:.4f}", (i, row['MAE']),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
ax.set_xticks(range(len(best_per_level)))
xlabels = ['Baseline', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5a\n(tabular NN)', 'Level 5b\n(sequence)']
ax.set_xticklabels(xlabels, rotation=0, ha='center')
ax.set_ylabel('Test MAE (best model at that information level)')
ax.set_title('How Much Does Each Additional Piece of Information Reduce Forecasting Error?')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/mae_by_information_level.png', dpi=150)
plt.close(fig)

print('\nSaved: master_comparison_results.csv, actual_vs_predicted.png, error_by_hour.png, mae_by_information_level.png')
