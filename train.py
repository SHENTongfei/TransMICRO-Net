"""
02_train_final_complete.py
最终版：分层KFold + 松弛评价 + 快照集成最终模型 + 外部测试。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import copy
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POINTS = 200
BATCH_SIZE = 4
EPOCHS = 50000
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5
USE_AMP = True
WARMUP_EPOCHS = 50
MIXUP_ALPHA = 0.4
NOISE_STD = 0.01
TOP_K = 5
PHYSICS_TOLERANCE = 1e-3
N_SEGMENTS = 15
SMOOTH_WINDOW = 9
SNAPSHOT_INTERVAL = 1000
FINAL_EPOCHS = 20000

if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    trapz = np.trapz

X_train_val = np.load(os.path.join(DATA_DIR, 'train_val_curves.npy'))
y_train_val_raw = np.load(os.path.join(DATA_DIR, 'train_val_labels.npy'))
X_test_raw = np.load(os.path.join(DATA_DIR, 'test_curves.npy'))
y_test_raw = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
info_df = pd.read_csv(os.path.join(DATA_DIR, 'train_val_info.csv'))

print(f"练习册: {X_train_val.shape}, 考试题: {X_test_raw.shape}")

material_map = {'OHA-GEL': 0, 'Alginate': 1, 'ADA-GEL': 2}
train_material_ids = info_df['material'].map(material_map).values
train_conc1 = info_df['conc1'].values.astype(np.float32)
stratify_labels = info_df['folder'].values

def transform_labels(y):
    return np.log1p(y)

def inverse_transform_labels(y_log):
    return np.expm1(y_log)

class CurveDataset(Dataset):
    def __init__(self, curves, labels_trans, material_ids, conc1, mixup_alpha=0.4, noise_std=0.0):
        self.curves = torch.tensor(curves, dtype=torch.float32)
        self.labels = torch.tensor(labels_trans, dtype=torch.float32)
        self.material_ids = torch.tensor(material_ids, dtype=torch.long)
        self.conc1 = torch.tensor(conc1, dtype=torch.float32).unsqueeze(1)
        self.mixup_alpha = mixup_alpha
        self.noise_std = noise_std

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        curve = self.curves[idx]
        if self.noise_std > 0:
            curve = curve + torch.randn_like(curve) * self.noise_std
        label = self.labels[idx]
        mat = F.one_hot(self.material_ids[idx], num_classes=3).float()
        conc = self.conc1[idx]

        if self.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
            idx2 = torch.randint(0, len(self.curves), (1,)).item()
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            curve = lam * curve + (1 - lam) * self.curves[idx2]
            label = lam * label + (1 - lam) * self.labels[idx2]
            mat2 = F.one_hot(self.material_ids[idx2], num_classes=3).float()
            mat = lam * mat + (1 - lam) * mat2
            conc = lam * conc + (1 - lam) * self.conc1[idx2]
        return curve, label, mat, conc

class TestDataset(Dataset):
    def __init__(self, curves, material_ids, conc):
        self.curves = torch.tensor(curves, dtype=torch.float32)
        self.material_ids = torch.tensor(material_ids, dtype=torch.long)
        self.conc = torch.tensor(conc, dtype=torch.float32)
    def __len__(self):
        return len(self.curves)
    def __getitem__(self, idx):
        mat = F.one_hot(self.material_ids[idx], num_classes=3).float()
        return self.curves[idx], mat, self.conc[idx]

class FiLM(nn.Module):
    def __init__(self, condition_dim=9, feature_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(condition_dim, 16), nn.ReLU(), nn.Linear(16, feature_dim * 2))
    def forward(self, condition):
        out = self.fc(condition)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

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

class TransMICRONet(nn.Module):
    def __init__(self, n_points=200, n_outputs=3, d_model=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.material_embed = nn.Embedding(3, 8)
        self.film = FiLM(condition_dim=8+1, feature_dim=d_model)
        self.conv_block1 = ResBlock1D(d_model, dropout=dropout)
        self.conv_block2 = ResBlock1D(d_model, dropout=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.gate_recon = nn.Linear(d_model, d_model)
        self.gate_pred = nn.Linear(d_model, d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_points)
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, n_outputs)
        )

    def forward(self, x, material_onehot, conc):
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
        return recon, pred_log, attn_weights

class MultiTaskLoss(nn.Module):
    def __init__(self, n_tasks=2, clamp_range=(-3, 3)):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.clamp_range = clamp_range

    def forward(self, recon_true, recon_pred, pred_true_log, pred_pred_log, strain_grid=None):
        log_vars_clamped = torch.clamp(self.log_vars, self.clamp_range[0], self.clamp_range[1])
        mse_recon = F.mse_loss(recon_pred, recon_true)
        mse_pred = F.mse_loss(pred_pred_log, pred_true_log)
        precision_recon = torch.exp(-log_vars_clamped[0])
        loss_recon = precision_recon * mse_recon + log_vars_clamped[0]
        precision_pred = torch.exp(-log_vars_clamped[1])
        loss_pred = precision_pred * mse_pred + log_vars_clamped[1]
        total = loss_recon + loss_pred
        return total, mse_recon.detach(), mse_pred.detach(), torch.tensor(0.0), torch.tensor(0.0)

def segment_mean(curve, n_segments=N_SEGMENTS):
    seg_len = len(curve) // n_segments
    means = [np.mean(curve[i*seg_len:(i+1)*seg_len]) for i in range(n_segments)]
    return np.array(means)

def smooth_curve(curve, window_length=SMOOTH_WINDOW, polyorder=2):
    if window_length >= len(curve):
        window_length = len(curve) - 1 if (len(curve) - 1) % 2 == 1 else len(curve) - 2
    if window_length < 3:
        return curve
    return savgol_filter(curve, window_length, polyorder)

def compute_soft_physics(recon_pred, pred_pred_log, scaler_y, strain_grid, scaler_X=None):
    device = strain_grid.device
    if not isinstance(recon_pred, torch.Tensor):
        recon_pred = torch.from_numpy(recon_pred).float().to(device)
    else:
        recon_pred = recon_pred.to(device)
    if not isinstance(pred_pred_log, torch.Tensor):
        pred_pred_log = torch.from_numpy(pred_pred_log).float().to(device)
    else:
        pred_pred_log = pred_pred_log.to(device)

    if scaler_X is not None:
        recon_np = scaler_X.inverse_transform(recon_pred.cpu().numpy())
    else:
        recon_np = recon_pred.cpu().numpy()

    recon_smooth = np.array([smooth_curve(r) for r in recon_np])
    recon_smooth_t = torch.tensor(recon_smooth, dtype=torch.float32, device=device)

    diff = recon_smooth_t[:, 1:] - recon_smooth_t[:, :-1]
    violations = (diff < -PHYSICS_TOLERANCE).float().sum(dim=1).mean()
    mono_ratio = violations / diff.shape[1]
    mono_score = 10 * (1 - mono_ratio)

    pred_log_np = scaler_y.inverse_transform(pred_pred_log.cpu().numpy())
    pred_raw = inverse_transform_labels(pred_log_np)
    area_recon = trapz(recon_smooth, strain_grid.cpu().numpy(), axis=1)
    area_pred = pred_raw[:, 2]
    r_area, _ = pearsonr(area_recon, area_pred)
    consist_score = 5 * max(0, r_area)

    physics = mono_score + consist_score
    return physics, mono_score, consist_score

def compute_hpi(recon_true, recon_pred, pred_true_log, pred_pred_log, scaler_y, strain_grid=None, scaler_X=None):
    r2_recons = []
    for i in range(recon_true.shape[0]):
        true_seg = segment_mean(recon_true[i])
        pred_seg = segment_mean(recon_pred[i])
        ssr = np.sum((true_seg - pred_seg) ** 2)
        sst = np.sum((true_seg - np.mean(true_seg)) ** 2)
        r2_recons.append(max(0, 1 - ssr / (sst + 1e-8)))
    r2_recon = np.mean(r2_recons)

    pred_true_log_np = scaler_y.inverse_transform(pred_true_log)
    pred_pred_log_np = scaler_y.inverse_transform(pred_pred_log)
    pred_true_raw = inverse_transform_labels(pred_true_log_np)
    pred_pred_raw = inverse_transform_labels(pred_pred_log_np)
    r2s = []
    for i in range(3):
        r2 = r2_score(pred_true_raw[:, i], pred_pred_raw[:, i])
        r2s.append(max(0, r2))
    r2_pred = np.mean(r2s)

    physics = 0.0
    if strain_grid is not None:
        physics, _, _ = compute_soft_physics(recon_pred, pred_pred_log, scaler_y, strain_grid, scaler_X)

    hpi = r2_recon * 35 + r2_pred * 65
    return hpi, r2_recon, r2_pred, physics

def train_epoch(model, dataloader, optimizer, criterion, strain_grid, scaler_amp):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        curves, labels_log, mat_onehot, conc = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        if USE_AMP and scaler_amp:
            with autocast():
                recon, pred_log, _ = model(curves, mat_onehot, conc)
                loss, _, _, _, _ = criterion(curves, recon, labels_log, pred_log, strain_grid)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            recon, pred_log, _ = model(curves, mat_onehot, conc)
            loss, _, _, _, _ = criterion(curves, recon, labels_log, pred_log, strain_grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, strain_grid, scaler_y, scaler_X=None):
    model.eval()
    total_loss = 0.0
    all_recon_true, all_recon_pred = [], []
    all_pred_true_log, all_pred_pred_log = [], []
    for batch in dataloader:
        curves, labels_log, mat_onehot, conc = [b.to(DEVICE) for b in batch]
        recon, pred_log, _ = model(curves, mat_onehot, conc)
        loss, mse_recon, mse_pred, mono, consist = criterion(curves, recon, labels_log, pred_log, strain_grid)
        total_loss += loss.item()
        all_recon_true.append(curves.cpu().numpy())
        all_recon_pred.append(recon.cpu().numpy())
        all_pred_true_log.append(labels_log.cpu().numpy())
        all_pred_pred_log.append(pred_log.cpu().numpy())
    all_recon_true = np.concatenate(all_recon_true, axis=0)
    all_recon_pred = np.concatenate(all_recon_pred, axis=0)
    all_pred_true_log = np.concatenate(all_pred_true_log, axis=0)
    all_pred_pred_log = np.concatenate(all_pred_pred_log, axis=0)
    hpi, r2_rec, r2_pred, phys = compute_hpi(all_recon_true, all_recon_pred,
                                            all_pred_true_log, all_pred_pred_log,
                                            scaler_y, strain_grid, scaler_X)
    return (total_loss / len(dataloader), hpi, r2_rec, r2_pred, phys,
            all_pred_true_log, all_pred_pred_log, all_recon_true, all_recon_pred)

def ensemble_state_dicts(model_list):
    hpis = torch.tensor([h for h, _ in model_list], dtype=torch.float32)
    weights = torch.softmax(hpis, dim=0)
    avg_state = {}
    first_state = model_list[0][1]
    for key in first_state:
        param = sum(weights[i] * model_list[i][1][key].cpu() for i in range(len(model_list)))
        avg_state[key] = param
    return avg_state

def main():
    strain_grid = torch.tensor(np.load(os.path.join(DATA_DIR, 'strain_grid.npy')),
                               dtype=torch.float32, device=DEVICE)

    test_mat_ids = np.full(len(X_test_raw), material_map['ADA-GEL'], dtype=np.int64)
    test_conc = np.full((len(X_test_raw), 1), 0.6, dtype=np.float32)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    all_records = []
    global_pred_true_raw = []
    global_pred_pred_raw = []
    best_overall_hpi = -1.0
    best_ensemble_state = None
    test_results_folds = []

    print("开始分层3折交叉验证...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, stratify_labels)):
        print(f"\n{'='*60}")
        print(f"第 {fold+1}/3 折 (验证集 {len(val_idx)} 个样本，训练集 {len(train_idx)} 个样本)")

        X_tr_raw, y_tr_raw = X_train_val[train_idx], y_train_val_raw[train_idx]
        X_val_raw, y_val_raw = X_train_val[val_idx], y_train_val_raw[val_idx]
        mat_tr, conc_tr = train_material_ids[train_idx], train_conc1[train_idx]
        mat_val, conc_val = train_material_ids[val_idx], train_conc1[val_idx]

        y_tr_log = transform_labels(y_tr_raw)
        y_val_log = transform_labels(y_val_raw)

        fold_scaler_X = StandardScaler()
        fold_scaler_y = StandardScaler()
        X_tr = fold_scaler_X.fit_transform(X_tr_raw)
        y_tr_trans = fold_scaler_y.fit_transform(y_tr_log)
        X_val = fold_scaler_X.transform(X_val_raw)
        y_val_trans = fold_scaler_y.transform(y_val_log)

        train_ds = CurveDataset(X_tr, y_tr_trans, mat_tr, conc_tr, mixup_alpha=MIXUP_ALPHA, noise_std=NOISE_STD)
        val_ds = CurveDataset(X_val, y_val_trans, mat_val, conc_val, mixup_alpha=0.0, noise_std=0.0)
        train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False)

        model = TransMICRONet().to(DEVICE)
        criterion = MultiTaskLoss().to(DEVICE)
        optimizer = torch.optim.RAdam([
            {'params': model.parameters(), 'lr': LR},
            {'params': criterion.parameters(), 'lr': LR*0.1}
        ], weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)

        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/WARMUP_EPOCHS))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-7)
        scaler_amp = GradScaler() if USE_AMP else None

        top_models = []
        best_fold_hpi = -1.0
        patience = 5000
        min_delta = 0.1
        no_improve = 0

        for epoch in range(1, EPOCHS+1):
            if epoch <= WARMUP_EPOCHS:
                warmup.step()
            else:
                cosine.step()

            train_loss = train_epoch(model, train_loader, optimizer, criterion, strain_grid, scaler_amp)
            val_loss, hpi, r2_rec, r2_pred, phys, pred_t_log, pred_p_log, recon_t, recon_p = validate(
                model, val_loader, criterion, strain_grid, fold_scaler_y, fold_scaler_X
            )

            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"Epoch {epoch}: 损失变 NaN，终止。")
                break

            record = {'fold': fold+1, 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                      'hpi': hpi, 'recon_r2': r2_rec, 'pred_r2': r2_pred, 'physics_score': phys}
            all_records.append(record)

            if hpi > best_fold_hpi + min_delta:
                best_fold_hpi = hpi
                no_improve = 0
                top_models.append((hpi, copy.deepcopy(model.state_dict())))
                top_models.sort(key=lambda x: x[0], reverse=True)
                if len(top_models) > TOP_K:
                    top_models = top_models[:TOP_K]
            elif hpi > best_fold_hpi:
                best_fold_hpi = hpi
                no_improve += 1
            else:
                no_improve += 1

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, "
                      f"HPI={hpi:.2f}, 重构R²={r2_rec:.3f}, 预测R²={r2_pred:.3f}, "
                      f"物理分={phys:.2f}, 耐心余量={patience-no_improve}")

            if no_improve >= patience:
                print(f"早停于 epoch {epoch}，最佳 HPI = {best_fold_hpi:.2f}")
                break

        if len(top_models) > 0:
            print(f"集成 Top-{len(top_models)} 模型 (HPI: {[f'{h:.2f}' for h,s in top_models]})")
            ensemble_state = ensemble_state_dicts(top_models)
            model.load_state_dict(ensemble_state)
            val_loss, hpi_ens, r2_rec_ens, r2_pred_ens, phys_ens, pred_t_log_ens, pred_p_log_ens, recon_t_ens, recon_p_ens = validate(
                model, val_loader, criterion, strain_grid, fold_scaler_y, fold_scaler_X
            )
            print(f"集成模型 HPI = {hpi_ens:.2f}, 物理分 = {phys_ens:.2f}")

            pred_t_raw = inverse_transform_labels(fold_scaler_y.inverse_transform(pred_t_log_ens))
            pred_p_raw = inverse_transform_labels(fold_scaler_y.inverse_transform(pred_p_log_ens))
            np.savez(os.path.join(OUTPUT_DIR, f'fold{fold+1}_val_predictions.npz'),
                     pred_true=pred_t_raw, pred_pred=pred_p_raw,
                     recon_true=recon_t_ens, recon_pred=recon_p_ens)
        else:
            print("无快照，使用当前模型。")
            _, hpi_ens, _, _, _, pred_t_log_ens, pred_p_log_ens, recon_t_ens, recon_p_ens = validate(
                model, val_loader, criterion, strain_grid, fold_scaler_y, fold_scaler_X
            )
            ensemble_state = copy.deepcopy(model.state_dict())
            pred_t_raw = inverse_transform_labels(fold_scaler_y.inverse_transform(pred_t_log_ens))
            pred_p_raw = inverse_transform_labels(fold_scaler_y.inverse_transform(pred_p_log_ens))

        global_pred_true_raw.append(pred_t_raw)
        global_pred_pred_raw.append(pred_p_raw)

        if hpi_ens > best_overall_hpi:
            best_overall_hpi = hpi_ens
            best_ensemble_state = copy.deepcopy(ensemble_state)

        torch.save(ensemble_state, os.path.join(OUTPUT_DIR, f'ensemble_model_fold{fold+1}.pth'))

        # 外部测试 ADA-GEL
        print(f"\n--- 第{fold+1}折外部测试 (ADA-GEL) ---")
        X_te_fold = fold_scaler_X.transform(X_test_raw)
        test_ds = TestDataset(X_te_fold, test_mat_ids, test_conc)
        test_loader = DataLoader(test_ds, batch_size=len(X_test_raw), shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                curves, mat_onehot, conc = [b.to(DEVICE) for b in batch]
                recon, pred_log, _ = model(curves, mat_onehot, conc)
                pred_np = inverse_transform_labels(fold_scaler_y.inverse_transform(pred_log.cpu().numpy()))
                true_np = y_test_raw
                phys_test, _, _ = compute_soft_physics(recon, pred_log, fold_scaler_y, strain_grid, fold_scaler_X)
                print(f"  物理分: {phys_test:.2f}")
                for i, name in enumerate(['峰值应力', '屈服点', '韧性']):
                    rmse = np.sqrt(mean_squared_error(true_np[:, i], pred_np[:, i]))
                    r2 = r2_score(true_np[:, i], pred_np[:, i])
                    r, _ = pearsonr(true_np[:, i], pred_np[:, i])
                    print(f"  {name}: RMSE={rmse:.4f}, R²={r2:.4f}, r={r:.4f}")
                test_results_folds.append({
                    'fold': fold+1,
                    'peak_rmse': rmse, 'peak_r2': r2, 'peak_r': r,
                    'yield_rmse': rmse, 'yield_r2': r2, 'yield_r': r,
                    'toughness_rmse': rmse, 'toughness_r2': r2, 'toughness_r': r,
                    'physics_score': phys_test
                })
                np.savez(os.path.join(OUTPUT_DIR, f'fold{fold+1}_test_predictions.npz'),
                         true=true_np, pred=pred_np)

    # 全局交叉验证指标
    global_pred_true_raw = np.concatenate(global_pred_true_raw, axis=0)
    global_pred_pred_raw = np.concatenate(global_pred_pred_raw, axis=0)
    print("\n=== 全局交叉验证预测指标 ===")
    for i, name in enumerate(['峰值应力', '屈服点', '韧性']):
        r2 = r2_score(global_pred_true_raw[:, i], global_pred_pred_raw[:, i])
        rmse = np.sqrt(mean_squared_error(global_pred_true_raw[:, i], global_pred_pred_raw[:, i]))
        r, _ = pearsonr(global_pred_true_raw[:, i], global_pred_pred_raw[:, i])
        print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}, r={r:.4f}")

    pd.DataFrame(all_records).to_csv(os.path.join(LOG_DIR, 'training_log.csv'), index=False)
    print(f"训练日志已保存到 {LOG_DIR}/training_log.csv")

    test_df = pd.DataFrame(test_results_folds)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_per_fold.csv'), index=False)

    if best_ensemble_state is not None:
        torch.save(best_ensemble_state, os.path.join(OUTPUT_DIR, 'best_ensemble_model.pth'))
        print(f"全局最佳集成模型已保存 (HPI={best_overall_hpi:.2f})")

    # ========== 最终模型（快照集成） ==========
    print("\n=== 训练最终模型（全部18个练习册样本，快照集成） ===")
    y_all_log = transform_labels(y_train_val_raw)
    global_scaler_X = StandardScaler()
    global_scaler_y = StandardScaler()
    X_all = global_scaler_X.fit_transform(X_train_val)
    y_all_trans = global_scaler_y.fit_transform(y_all_log)

    final_ds = CurveDataset(X_all, y_all_trans, train_material_ids, train_conc1, mixup_alpha=MIXUP_ALPHA, noise_std=NOISE_STD)
    final_loader = DataLoader(final_ds, batch_size=min(BATCH_SIZE, len(X_all)), shuffle=True)

    final_model = TransMICRONet().to(DEVICE)
    final_criterion = MultiTaskLoss().to(DEVICE)
    final_optimizer = torch.optim.RAdam([
        {'params': final_model.parameters(), 'lr': LR},
        {'params': final_criterion.parameters(), 'lr': LR*0.1}
    ], weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)

    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=FINAL_EPOCHS, eta_min=5e-7)
    final_scaler_amp = GradScaler() if USE_AMP else None

    snapshots = []

    for epoch in range(1, FINAL_EPOCHS + 1):
        final_model.train()
        total_loss = 0.0
        for batch in final_loader:
            curves, labels_log, mat_onehot, conc = [b.to(DEVICE) for b in batch]
            final_optimizer.zero_grad()
            if USE_AMP and final_scaler_amp:
                with autocast():
                    recon, pred_log, _ = final_model(curves, mat_onehot, conc)
                    loss, _, _, _, _ = final_criterion(curves, recon, labels_log, pred_log, strain_grid)
                final_scaler_amp.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP)
                final_scaler_amp.step(final_optimizer)
                final_scaler_amp.update()
            else:
                recon, pred_log, _ = final_model(curves, mat_onehot, conc)
                loss, _, _, _, _ = final_criterion(curves, recon, labels_log, pred_log, strain_grid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP)
                final_optimizer.step()
            total_loss += loss.item()
        final_scheduler.step()

        if epoch % 100 == 0:
            print(f"Final model epoch {epoch}: train_loss={total_loss/len(final_loader):.4f}")

        if epoch % SNAPSHOT_INTERVAL == 0:
            snapshots.append((epoch, copy.deepcopy(final_model.state_dict())))
            print(f"  快照保存于 epoch {epoch}")

    if len(snapshots) == 0 or snapshots[-1][0] != FINAL_EPOCHS:
        snapshots.append((FINAL_EPOCHS, copy.deepcopy(final_model.state_dict())))

    print(f"\n集成 {len(snapshots)} 个快照模型...")
    hpis_for_snap = [0.0] * len(snapshots)
    final_ensemble_state = ensemble_state_dicts(list(zip(hpis_for_snap, [s[1] for s in snapshots])))
    final_model.load_state_dict(final_ensemble_state)
    torch.save(final_ensemble_state, os.path.join(OUTPUT_DIR, 'final_model.pth'))
    print("最终集成模型已保存。")

    # 最终模型外部测试
    print("\n--- 最终模型外部测试 (ADA-GEL) ---")
    X_te_global = global_scaler_X.transform(X_test_raw)
    test_ds_final = TestDataset(X_te_global, test_mat_ids, test_conc)
    test_loader_final = DataLoader(test_ds_final, batch_size=len(X_test_raw), shuffle=False)
    final_model.eval()
    with torch.no_grad():
        for batch in test_loader_final:
            curves, mat_onehot, conc = [b.to(DEVICE) for b in batch]
            recon, pred_log, _ = final_model(curves, mat_onehot, conc)
            pred_np = inverse_transform_labels(global_scaler_y.inverse_transform(pred_log.cpu().numpy()))
            true_np = y_test_raw
            phys_final, _, _ = compute_soft_physics(recon, pred_log, global_scaler_y, strain_grid, global_scaler_X)
            print(f"  物理分: {phys_final:.2f}")
            for i, name in enumerate(['峰值应力', '屈服点', '韧性']):
                rmse = np.sqrt(mean_squared_error(true_np[:, i], pred_np[:, i]))
                r2 = r2_score(true_np[:, i], pred_np[:, i])
                r, _ = pearsonr(true_np[:, i], pred_np[:, i])
                print(f"  {name}: RMSE={rmse:.4f}, R²={r2:.4f}, r={r:.4f}")
            np.savez(os.path.join(OUTPUT_DIR, 'final_test_predictions.npz'), true=true_np, pred=pred_np)

    print("\n=== 所有训练完成，结果已保存。 ===")

if __name__ == "__main__":
    main()