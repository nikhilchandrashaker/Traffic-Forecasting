"""
Level 4 — add neighboring-sensor information via tra_adj_mat.

For each (timestep, sensor), we add the adjacency-weighted average of each
neighboring sensor's own 10 lagged traffic values (cols 0-9), computed at the
SAME timestep t (so it's information legitimately available at prediction
time, not a future leak). Sensors with no neighbors get a zero vector.

We also add neighbor_count as a feature (how connected a sensor is), since
that's cheap and may matter (well-connected sensors may be more/less
predictable).
"""
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

RNG = 42
d = sio.loadmat('/mnt/user-data/uploads/traffic_dataset.mat')

def densify(X_cell):
    n = X_cell.shape[1]
    return np.stack([X_cell[0, i].toarray() for i in range(n)])

X_tr = densify(d['tra_X_tr'])   # (1261, 36, 48)
X_te = densify(d['tra_X_te'])   # (840, 36, 48)
Y_tr = d['tra_Y_tr'].T
Y_te = d['tra_Y_te'].T
adj = d['tra_adj_mat'].astype(float)   # (36, 36)
S = adj.shape[0]

neighbor_counts = adj.sum(axis=1)  # (36,)
# row-normalize adjacency so neighbor features are an average, not a sum
adj_norm = np.divide(adj, neighbor_counts[:, None], out=np.zeros_like(adj),
                      where=neighbor_counts[:, None] != 0)

def add_neighbor_features(X):
    """X: (T, 36, 48) -> neighbor_lags (T, 36, 10), neighbor_count (T, 36, 1) broadcast"""
    lags = X[:, :, 0:10]                              # (T, 36, 10)
    # for each timestep, neighbor_avg[s,:] = adj_norm[s,:] @ lags[t]
    neighbor_avg = np.einsum('sn,tnl->tsl', adj_norm, lags)   # (T, 36, 10)
    nc = np.broadcast_to(neighbor_counts, (X.shape[0], S))[:, :, None]  # (T,36,1)
    return neighbor_avg, nc

nbr_lags_tr, nbr_cnt_tr = add_neighbor_features(X_tr)
nbr_lags_te, nbr_cnt_te = add_neighbor_features(X_te)

Xtr_flat_base = X_tr.reshape(-1, 48)
Xte_flat_base = X_te.reshape(-1, 48)
Xtr_l4 = np.concatenate([Xtr_flat_base, nbr_lags_tr.reshape(-1, 10), nbr_cnt_tr.reshape(-1, 1)], axis=1)
Xte_l4 = np.concatenate([Xte_flat_base, nbr_lags_te.reshape(-1, 10), nbr_cnt_te.reshape(-1, 1)], axis=1)

ytr_flat = Y_tr.reshape(-1)
yte_flat = Y_te.reshape(-1)

model_builders = {
    'Linear/Ridge': lambda: Ridge(alpha=1.0),
    'Random Forest': lambda: RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=RNG),
    'XGBoost': lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1),
}

rows = []
level4_preds = {}
for name, build in model_builders.items():
    m = build()
    m.fit(Xtr_l4, ytr_flat)
    pred = m.predict(Xte_l4)
    rows.append({
        'level': 'Level 4: + neighbor sensors',
        'model': name,
        'MAE': mean_absolute_error(yte_flat, pred),
        'RMSE': mean_squared_error(yte_flat, pred) ** 0.5,
        'R2': r2_score(yte_flat, pred),
    })
    level4_preds[name] = pred

level4_df = pd.DataFrame(rows)
print(level4_df.to_string(index=False))

level4_df.to_csv('/mnt/user-data/outputs/level4_neighbor_results.csv', index=False)
np.save('/home/claude/Xtr_l4.npy', Xtr_l4)
np.save('/home/claude/Xte_l4.npy', Xte_l4)
with open('/home/claude/level4_preds.pkl', 'wb') as f:
    pickle.dump(level4_preds, f)
