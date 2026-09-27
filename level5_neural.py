"""
Level 5 — neural models.
  5a) Feedforward NN on the same tabular features as Level 4 (tests whether a
      deeper/nonlinear model beats tree ensembles on this feature set).
  5b) LSTM and GRU treating the 10 lagged values as an explicit length-10
      sequence per (timestep, sensor) — optionally with the neighbor-average
      lag sequence as a second channel — to test whether sequence modeling
      captures anything the tabular models miss.

Given only ~45k train rows (many correlated across sensors/time), this is
kept intentionally small (few layers, modest hidden size, early stopping on
a held-out slice of the training period) rather than a large deep model.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Xtr_l4 = np.load('/home/claude/Xtr_l4.npy')   # (T_tr*36, 59)
Xte_l4 = np.load('/home/claude/Xte_l4.npy')
yte_flat = np.load('/home/claude/yte_flat.npy')

import scipy.io as sio
d = sio.loadmat('/mnt/user-data/uploads/traffic_dataset.mat')
Y_tr = d['tra_Y_tr'].T
ytr_flat = Y_tr.reshape(-1)

S = 36
T_tr = Xtr_l4.shape[0] // S

# time-based split: last 10% of training TIMESTEPS held out for validation
# (keep whole sensor-blocks of 36 together so a timestep isn't split)
n_val_t = max(1, int(T_tr * 0.1))
n_train_t = T_tr - n_val_t
train_idx = np.arange(n_train_t * S)
val_idx = np.arange(n_train_t * S, T_tr * S)

# ---------- normalize tabular features (fit on train slice only) ----------
mu, sigma = Xtr_l4[train_idx].mean(axis=0), Xtr_l4[train_idx].std(axis=0) + 1e-8
Xtr_norm = (Xtr_l4 - mu) / sigma
Xte_norm = (Xte_l4 - mu) / sigma

def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

# ================= 5a: Feedforward NN on tabular Level-4 features =================
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_model(model, Xtr, ytr, Xval, yval, epochs=60, lr=1e-3, batch_size=1024, patience=8):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr)
    Xval_t, yval_t = to_t(Xval), to_t(yval)
    n = Xtr_t.shape[0]
    best_val, best_state, bad_epochs = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t)
            val_mae = loss_fn(val_pred, yval_t).item()
        if val_mae < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val

mlp = MLP(Xtr_norm.shape[1]).to(device)
mlp, best_val_mae = train_model(mlp, Xtr_norm[train_idx], ytr_flat[train_idx],
                                 Xtr_norm[val_idx], ytr_flat[val_idx])
mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(to_t(Xte_norm)).cpu().numpy()

# ================= 5b: LSTM / GRU on the 10-lag sequence =================
# sequence channel 1: own 10 lags (T-150..T); channel 2: neighbor-avg 10 lags
own_lags_tr = Xtr_l4[:, 0:10]
nbr_lags_tr = Xtr_l4[:, 48:58]
own_lags_te = Xte_l4[:, 0:10]
nbr_lags_te = Xte_l4[:, 48:58]

seq_tr = np.stack([own_lags_tr, nbr_lags_tr], axis=-1)   # (N, 10, 2)
seq_te = np.stack([own_lags_te, nbr_lags_te], axis=-1)

seq_mu = seq_tr[train_idx].mean(axis=(0, 1), keepdims=True)
seq_sigma = seq_tr[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
seq_tr_n = (seq_tr - seq_mu) / seq_sigma
seq_te_n = (seq_te - seq_mu) / seq_sigma

class SeqModel(nn.Module):
    def __init__(self, cell='lstm', input_size=2, hidden=32):
        super().__init__()
        rnn_cls = nn.LSTM if cell == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

def train_seq_model(model, Xtr, ytr, Xval, yval, epochs=40, lr=2e-3, batch_size=1024, patience=6):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr)
    Xval_t, yval_t = to_t(Xval), to_t(yval)
    n = Xtr_t.shape[0]
    best_val, best_state, bad_epochs = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mae = loss_fn(model(Xval_t), yval_t).item()
        if val_mae < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val

results_rows = []

mae = mean_absolute_error(yte_flat, mlp_pred)
rmse = mean_squared_error(yte_flat, mlp_pred) ** 0.5
r2 = r2_score(yte_flat, mlp_pred)
results_rows.append({'level': 'Level 5: Neural Net (tabular, Level-4 features)', 'model': 'Feedforward NN',
                      'MAE': mae, 'RMSE': rmse, 'R2': r2})
print(f'Feedforward NN: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')

seq_preds = {}
for cell in ['lstm', 'gru']:
    model = SeqModel(cell=cell).to(device)
    model, best_val = train_seq_model(model, seq_tr_n[train_idx], ytr_flat[train_idx],
                                       seq_tr_n[val_idx], ytr_flat[val_idx])
    model.eval()
    with torch.no_grad():
        pred = model(to_t(seq_te_n)).cpu().numpy()
    seq_preds[cell] = pred
    mae = mean_absolute_error(yte_flat, pred)
    rmse = mean_squared_error(yte_flat, pred) ** 0.5
    r2 = r2_score(yte_flat, pred)
    results_rows.append({'level': 'Level 5: Sequence model (own + neighbor lag sequence)',
                          'model': cell.upper(), 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    print(f'{cell.upper()}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}')

level5_df = pd.DataFrame(results_rows)
level5_df.to_csv('/mnt/user-data/outputs/level5_neural_results.csv', index=False)
print(level5_df.to_string(index=False))

with open('/home/claude/level5_preds.pkl', 'wb') as f:
    pickle.dump({'Feedforward NN': mlp_pred, **{k.upper(): v for k, v in seq_preds.items()}}, f)
