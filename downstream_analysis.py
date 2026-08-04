# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:09:31 2026

@author: TS
"""

"""
03b_downstream_extra.py (修正版)
补充下游分析：梯度敏感度、重构误差诊断、材料嵌入向量、循环稳定性指纹。
修复梯度导致的重构误差报错。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ================== 路径与配置 ==================
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis\downstream')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POINTS = 200

# ================== 模型定义（与训练时完全一致） ==================
class FiLM(torch.nn.Module):
    def __init__(self, condition_dim=9, feature_dim=64):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(condition_dim, 16), torch.nn.ReLU(), torch.nn.Linear(16, feature_dim*2))
    def forward(self, condition):
        out = self.fc(condition)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)

class ResBlock1D(torch.nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = torch.nn.LayerNorm(channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
    def forward(self, x, film_params=None):
        residual = x
        x = self.conv1(x)
        if film_params is not None:
            gamma, beta = film_params
            x = gamma * x + beta
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x + residual

class TransMICRONet(torch.nn.Module):
    def __init__(self, n_points=200, n_outputs=3, d_model=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.embedding = torch.nn.Linear(1, d_model)
        self.material_embed = torch.nn.Embedding(3, 8)
        self.film = FiLM(condition_dim=8+1, feature_dim=d_model)
        self.conv_block1 = ResBlock1D(d_model, dropout=dropout)
        self.conv_block2 = ResBlock1D(d_model, dropout=dropout)
        self.attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = torch.nn.LayerNorm(d_model)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.gate_recon = torch.nn.Linear(d_model, d_model)
        self.gate_pred = torch.nn.Linear(d_model, d_model)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, n_points)
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, n_outputs)
        )

    def forward(self, x, material_onehot, conc, return_attn=False, return_latent=False, return_recon=True):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        mat_idx = torch.argmax(material_onehot, dim=-1) if material_onehot.dim() > 1 else material_onehot
        mat_emb = self.material_embed(mat_idx)
        condition = torch.cat([mat_emb, conc], dim=-1)
        gamma, beta = self.film(condition)
        x = self.conv_block1(x, (gamma, beta))
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.attn_norm(x + attn_out)
        x = x.permute(0, 2, 1)
        shared = self.global_pool(x).squeeze(-1)
        gate_r = torch.sigmoid(self.gate_recon(shared))
        gate_p = torch.sigmoid(self.gate_pred(shared))
        recon = self.decoder(shared * gate_r)
        pred_log = self.predictor(shared * gate_p)

        out = (recon, pred_log)
        if return_attn:
            out += (attn_weights,)
        if return_latent:
            out += (shared,)
        return out

# ================== 数据准备 ==================
X_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_curves.npy'))
y_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_labels.npy'))
X_test_raw = np.load(os.path.join(DATA_DIR, 'test_curves.npy'))
y_test_raw = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
info_df = pd.read_csv(os.path.join(DATA_DIR, 'train_val_info.csv'))
strain_grid = np.load(os.path.join(DATA_DIR, 'strain_grid.npy'))

material_map = {'OHA-GEL': 0, 'Alginate': 1, 'ADA-GEL': 2}
train_material_ids = info_df['material'].map(material_map).values
train_conc1 = info_df['conc1'].values.astype(np.float32)
test_material_ids = np.full(len(X_test_raw), 2, dtype=np.int64)
test_conc = np.full(len(X_test_raw), 0.6, dtype=np.float32)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

def transform_labels(y): return np.log1p(y)
def inverse_transform_labels(y_log): return np.expm1(y_log)

y_train_log = transform_labels(y_train_raw)
global_scaler_y = StandardScaler()
global_scaler_y.fit(y_train_log)

# ================== 加载模型 ==================
model = TransMICRONet().to(DEVICE)
state = torch.load(os.path.join(MODEL_DIR, 'final_model.pth'), map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ================== 1. 梯度敏感度（对屈服点） ==================
print("Computing gradient sensitivity...")
sensitivity = []
labels = []
for i in range(len(X_train)):
    X_t = torch.tensor(X_train[i:i+1], dtype=torch.float32, requires_grad=True).to(DEVICE)
    mat_t = torch.tensor([train_material_ids[i]], dtype=torch.long).to(DEVICE)
    conc_t = torch.tensor([[train_conc1[i]]], dtype=torch.float32).to(DEVICE)
    mat_onehot = F.one_hot(mat_t, num_classes=3).float()
    recon, pred_log = model(X_t, mat_onehot, conc_t)
    pred_yield = pred_log[:, 1]  # log标准化空间
    grad = torch.autograd.grad(pred_yield, X_t, retain_graph=False)[0]
    sensitivity.append(grad.detach().cpu().numpy().flatten())
    labels.append(train_material_ids[i])
    # 清理梯度，避免影响下一次
    model.zero_grad()
    if X_t.grad is not None:
        X_t.grad.zero_()

sensitivity = np.array(sensitivity)
labels = np.array(labels)
np.savez(os.path.join(OUTPUT_DIR, 'gradient_sensitivity.npz'), sensitivity=sensitivity, labels=labels)

# ================== 2. 重构误差诊断 ==================
print("Computing reconstruction error...")
recon_errors = []
materials = []
concs = []
with torch.no_grad():
    for i in range(len(X_train)):
        X_t = torch.tensor(X_train[i:i+1], dtype=torch.float32).to(DEVICE)
        mat_t = torch.tensor([train_material_ids[i]], dtype=torch.long).to(DEVICE)
        conc_t = torch.tensor([[train_conc1[i]]], dtype=torch.float32).to(DEVICE)
        mat_onehot = F.one_hot(mat_t, num_classes=3).float()
        recon, _ = model(X_t, mat_onehot, conc_t)
        error = (recon.cpu().numpy() - X_train[i]).flatten()
        recon_errors.append(error)
        materials.append(train_material_ids[i])
        concs.append(train_conc1[i])

recon_errors = np.array(recon_errors)
materials = np.array(materials)
concs = np.array(concs)
np.savez(os.path.join(OUTPUT_DIR, 'recon_error_diagnosis.npz'), 
         errors=recon_errors, materials=materials, concs=concs)

# ================== 3. 材料嵌入向量 ==================
print("Extracting material embeddings...")
embeddings = model.material_embed.weight.data.cpu().numpy()  # (3, 8)
np.save(os.path.join(OUTPUT_DIR, 'material_embeddings.npy'), embeddings)

# ================== 4. 循环稳定性指纹 ==================
print("Extracting cycle stability fingerprints...")
def extract_cycle(filename):
    if 'c1' in filename: return 1
    if 'c2' in filename: return 2
    if 'c3' in filename: return 3
    return None

info_df['cycle'] = info_df['source_file'].apply(extract_cycle)

all_preds = []
with torch.no_grad():
    for i in range(len(X_train)):
        X_t = torch.tensor(X_train[i:i+1], dtype=torch.float32).to(DEVICE)
        mat_t = torch.tensor([train_material_ids[i]], dtype=torch.long).to(DEVICE)
        conc_t = torch.tensor([[train_conc1[i]]], dtype=torch.float32).to(DEVICE)
        mat_onehot = F.one_hot(mat_t, num_classes=3).float()
        _, pred_log = model(X_t, mat_onehot, conc_t)
        pred_log_np = global_scaler_y.inverse_transform(pred_log.cpu().numpy())
        preds = inverse_transform_labels(pred_log_np)
        all_preds.append(preds[0])

all_preds = np.array(all_preds)  # (18, 3)
cycle_info = info_df['cycle'].values

results_df = info_df[['material', 'conc1', 'conc2', 'cycle', 'source_file']].copy()
for j, name in enumerate(['peak_stress', 'yield_point', 'toughness']):
    results_df[f'pred_{name}'] = all_preds[:, j]
    results_df[f'true_{name}'] = y_train_raw[:, j]

results_df.to_csv(os.path.join(OUTPUT_DIR, 'cycle_stability.csv'), index=False)

print("All extra downstream data saved to", OUTPUT_DIR)