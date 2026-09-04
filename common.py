import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 2000
EMD_VALID_MIN_MS = 30
EMD_VALID_MAX_MS = 100


def normalize_and_smooth_curve(curve, smooth_sigma=1.0):
    """Normalize a 1D curve to [0, 1] and apply optional Gaussian smoothing."""
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return curve
    span = curve.max() - curve.min()
    if span <= 1e-9:
        norm = np.zeros_like(curve)
    else:
        norm = (curve - curve.min()) / (span + 1e-9)
    if smooth_sigma and curve.size > 1:
        return gaussian_filter1d(norm, sigma=smooth_sigma)
    return norm


def detect_st_sri_peak_ms(lags_ms, synergy_curve, constrained=False, smooth_sigma=1.0,
                          boundary_exclusion_ms=5.0):
    """
    Shared ST-SRI peak extractor used across E3/E17/E18.

    unconstrained mode excludes the sub-5 ms boundary region to avoid edge artifacts.
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    smooth_curve = normalize_and_smooth_curve(synergy_curve, smooth_sigma=smooth_sigma)
    if lags_ms.size == 0 or smooth_curve.size == 0:
        return None

    time_axis = np.abs(lags_ms)
    if constrained:
        valid_mask = (time_axis >= EMD_VALID_MIN_MS) & (time_axis <= EMD_VALID_MAX_MS)
    else:
        valid_mask = time_axis >= boundary_exclusion_ms

    if not np.any(valid_mask):
        return None

    peak_idx = int(np.argmax(smooth_curve[valid_mask]))
    return float(time_axis[valid_mask][peak_idx])


def select_predicted_class_shap(model, samples, shap_values):
    """
    Collapse SHAP outputs to predicted-class temporal attributions, matching the
    target semantics used by the other baselines.
    Returns an array shaped (N, T).
    """
    with torch.no_grad():
        pred_targets = model(samples).argmax(dim=1).detach().cpu().numpy()

    if isinstance(shap_values, list):
        per_sample = []
        for sample_idx, target_idx in enumerate(pred_targets):
            class_attr = np.asarray(shap_values[target_idx])[sample_idx]
            per_sample.append(np.abs(class_attr).sum(axis=-1))
        return np.stack(per_sample, axis=0)

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 4 and shap_arr.shape[0] == len(pred_targets):
        # Newer SHAP variants may return (N, T, C, num_classes)
        per_sample = []
        for sample_idx, target_idx in enumerate(pred_targets):
            per_sample.append(np.abs(shap_arr[sample_idx, :, :, target_idx]).sum(axis=-1))
        return np.stack(per_sample, axis=0)

    if shap_arr.ndim == 4 and shap_arr.shape[0] != len(pred_targets):
        # Fallback for (num_classes, N, T, C)
        per_sample = []
        for sample_idx, target_idx in enumerate(pred_targets):
            per_sample.append(np.abs(shap_arr[target_idx, sample_idx]).sum(axis=-1))
        return np.stack(per_sample, axis=0)

    if shap_arr.ndim == 3:
        return np.abs(shap_arr).sum(axis=-1)

    raise ValueError(f"Unsupported SHAP output shape: {shap_arr.shape}")


class NinaProDataset(Dataset):
    def __init__(self, root_dir, subject_id, window_ms=300, target_fs=2000, step_ms=50,
                 anticipation_ms=0):
        self.root = root_dir
        self.fs = target_fs
        self.window_len = int(window_ms * self.fs / 1000)
        self.stride = int(step_ms * self.fs / 1000)
        self.anticipation_steps = int(anticipation_ms * self.fs / 1000)
        d_path = os.path.join(root_dir, f"S{subject_id}_data.npy")
        l_path = os.path.join(root_dir, f"S{subject_id}_label.npy")
        raw_data = np.load(d_path)
        raw_labels = np.load(l_path)
        self.data = (raw_data - np.mean(raw_data, axis=0)) / (np.std(raw_data, axis=0) + 1e-6)
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(raw_labels).long()
        # 有提前量时，末尾需要预留 anticipation_steps 个样本作为标签参考
        self.num_samples = (len(self.data) - self.window_len - self.anticipation_steps) // self.stride + 1

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_len
        if self.anticipation_steps == 0:
            # 原始行为：取窗口内众数标签
            label = torch.mode(self.labels[start:end]).values
        else:
            # 提前预测：取窗口结束后 anticipation_steps 处的标签
            label_pos = end - 1 + self.anticipation_steps
            label = self.labels[label_pos]
        return self.data[start:end, :], label


class LSTMModel(nn.Module):
    # ... (这部分保持不变)
    def __init__(self, input_size=12, hidden_size=256, num_layers=3, num_classes=18, dropout=0.3, pooling="last"):
        super(LSTMModel, self).__init__()
        if pooling not in ("last", "mean", "attention"):
            raise ValueError(f"unknown LSTM pooling: {pooling}")
        self.pooling = pooling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        if pooling == "attention":
            self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        if self.pooling == "mean":
            out = out.mean(dim=1)
        elif self.pooling == "attention":
            weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (B, T)
            out = torch.bmm(out.transpose(1, 2), weights.unsqueeze(-1)).squeeze(-1)  # (B, H)
        else:
            out = out[:, -1, :]
        return self.fc(out)


# ================= 2b. TCN 模型 =================

class _CausalConv1d(nn.Module):
    """因果卷积：只看过去，不看未来。"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class _TCNBlock(nn.Module):
    """TCN 残差块：2 层因果卷积 + 残差连接。"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.downsample(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network.
    输入: (B, T, C)  输出: (B, num_classes)
    3 层 TCN，指数递增 dilation，取最后时间步做分类。
    """
    def __init__(self, input_size=12, hidden_size=256, num_layers=3,
                 num_classes=18, kernel_size=7, dropout=0.3):
        super().__init__()
        channels = [hidden_size] * num_layers
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        out = self.network(x.transpose(1, 2))
        # 取最后时间步
        return self.fc(out[:, :, -1])


# ================= 2c. Transformer 模型 =================

class _PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class _ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet1DModel(nn.Module):
    """Simple ResNet1D for time-series classification.

    Input: (B, T, C) -> Output: (B, num_classes)
    """

    def __init__(self, input_size=12, hidden_size=64, num_blocks=3, num_classes=18, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            _ResBlock1D(hidden_size, dropout=dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.fc(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder for time-series classification.
    输入: (B, T, C)  输出: (B, num_classes)
    先用线性层把 C 维投影到 d_model，再过 Transformer Encoder，
    取最后时间步（或 CLS token）做分类。
    """
    def __init__(self, input_size=12, d_model=128, nhead=8, num_layers=4,
                 num_classes=18, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, max_len=2000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x)         # (B, T, d_model)
        x = self.pos_encoder(x)        # (B, T, d_model)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        return self.fc(x[:, -1, :])    # 取最后时间步


# ================= 模型注册表 =================

MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'tcn': TCNModel,
    'transformer': TransformerModel,
    'resnet1d': ResNet1DModel,
}


def build_model(arch='lstm', input_size=12, num_classes=18, **kwargs):
    """统一的模型构建接口。"""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](input_size=input_size, num_classes=num_classes, **kwargs)


# ================= 3. ST-SRI 解释器 (Interpreter) =================
class ST_SRI_Interpreter:
    def __init__(self, model, background_data, device=None, predict_proba=None):
        self.device = DEVICE if device is None else torch.device(device)
        self.model = model.to(self.device).eval()
        self.predict_proba = predict_proba
        self.baseline = torch.mean(background_data, dim=0).to(self.device)
        self.T, self.C = self.baseline.shape

    def get_score_batch(self, x_batch, target_cls=None):
        """
        专用批量打分函数
        target_cls: 如果为 None，自动选择预测概率最高的类
        """
        with torch.no_grad():
            if self.predict_proba is not None:
                probs = self.predict_proba(x_batch)
            else:
                logits = self.model(x_batch)
                probs = torch.softmax(logits, dim=1)

            # 如果没有指定目标类，就取第一个样本预测最高的类
            if target_cls is None:
                target_cls = torch.argmax(probs[0]).item()

            # 返回该类别的概率 (B, )
            return probs[:, target_cls].cpu().numpy()

    def scan_fast(
        self,
        x,
        max_lag_ms=150,
        stride=1,
        block_size=2,
        current_endpoint=None,
        target_cls=None,
    ):
        """
        极速版扫描：利用 GPU 的并行能力，一次算完所有 Lag
        """
        # 1. 预计算所有参数
        if x.ndim != 2 or tuple(x.shape) != (self.T, self.C):
            raise ValueError(f"x must have shape {(self.T, self.C)}, got {tuple(x.shape)}")
        if stride < 1:
            raise ValueError("stride must be at least one sample")
        if block_size < 1:
            raise ValueError("block_size must be at least one sample")

        max_lag_points = int(max_lag_ms * (FS / 1000))
        curr_t = self.T - 1 if current_endpoint is None else int(current_endpoint)
        if curr_t < block_size - 1 or curr_t >= self.T:
            raise ValueError(
                f"current_endpoint={curr_t} cannot support a {block_size}-sample current block"
            )
        max_lag_points = min(max_lag_points, curr_t - block_size + 1)

        lags = list(range(stride, max_lag_points + 1, stride))
        if not lags: return [], [], []

        N_lags = len(lags)

        # 2. 构造超级 Batch
        x_base = x.unsqueeze(0).repeat(N_lags, 1, 1)

        x_lag = x_base.clone()
        x_curr = x_base.clone()
        x_none = x_base.clone()

        # 构建 Mask (注意：这里正确使用了 block_size)
        for i, tau in enumerate(lags):
            # 遮挡区间计算
            # 我们希望遮挡 [t - block_size + 1, t] 这一段，包含 t 本身
            t_end = curr_t + 1
            t_start = t_end - block_size

            # 边界保护
            t_start = max(0, t_start)

            # 遮挡 t (curr)
            x_lag[i, t_start:t_end, :] = self.baseline[t_start:t_end, :]
            x_none[i, t_start:t_end, :] = self.baseline[t_start:t_end, :]

            # 遮挡 t-tau (lag)
            t_prev_end = curr_t - tau + 1
            t_prev_start = t_prev_end - block_size

            x_curr[i, t_prev_start:t_prev_end, :] = self.baseline[t_prev_start:t_prev_end, :]
            x_none[i, t_prev_start:t_prev_end, :] = self.baseline[t_prev_start:t_prev_end, :]

        # 3. 确定目标类别 (Target Class)
        # 我们基于原始输入确定模型想预测什么，确保所有 Batch 关注同一个类
        if target_cls is None:
            with torch.no_grad():
                if self.predict_proba is not None:
                    orig_probs = self.predict_proba(x.unsqueeze(0))
                else:
                    orig_logits = self.model(x.unsqueeze(0))
                    orig_probs = torch.softmax(orig_logits, dim=1)
                target_cls = torch.argmax(orig_probs[0]).item()
        elif target_cls < 0:
            raise ValueError("target_cls must be nonnegative")

        # 4. 批量推理 (传入 target_cls，解决报错！)
        f_both = self.get_score_batch(x.unsqueeze(0), target_cls)[0]

        s_lag = self.get_score_batch(x_lag, target_cls)
        s_curr = self.get_score_batch(x_curr, target_cls)
        s_none = self.get_score_batch(x_none, target_cls)

        # 5. SII 计算
        interactions = f_both - s_lag - s_curr + s_none

        synergy = np.maximum(interactions, 0)
        redundancy = np.minimum(interactions, 0)

        lags_ms = [l * (1000 / FS) for l in lags]

        return lags_ms, synergy, redundancy


# ================= 无泄漏数据划分工具 =================

def blocked_time_split(dataset, train_ratio=0.8, gap_ratio=1.0, window_len=None, stride=None):
    """
    基于连续时间块的无泄漏划分，避免重叠窗口被分到不同集合。
    
    原则：
    - 训练集取前 train_ratio 比例的连续时间
    - 测试/验证集取后 (1-train_ratio) 比例的连续时间
    - 在两者之间预留 gap，gap 大小为 window_len - stride，确保没有共享原始采样点
    
    参数:
        dataset: NinaProDataset 对象，需包含 .stride 和 .window_len 属性
        train_ratio: 训练集比例
        gap_ratio: gap 倍数，默认 1.0 即预留 (window_len - stride)
        window_len: 如果 dataset 没有该属性，手动传入
        stride: 如果 dataset 没有该属性，手动传入
    
    返回:
        (train_indices, val_indices): 索引列表，可用于 Subset
    """
    total_samples = len(dataset)
    
    # 获取窗口参数
    wl = window_len if window_len is not None else getattr(dataset, 'window_len', 600)
    st = stride if stride is not None else getattr(dataset, 'stride', 100)
    
    # gap 样本数（窗口级）：需要预留一个 window_len - stride 长度的间隔，约等于 1 个窗口
    gap_samples = int((wl - st) / st * gap_ratio)
    gap_samples = max(1, gap_samples)
    
    train_end = int(total_samples * train_ratio) - gap_samples // 2
    val_start = int(total_samples * train_ratio) + (gap_samples - gap_samples // 2)
    
    if val_start >= total_samples:
        # 样本太少，退化成无 gap 划分
        train_end = int(total_samples * train_ratio)
        val_start = train_end
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(val_start, total_samples))

    # 防御性检查：确保 train/val 之间存在正向 gap（原始采样点不重叠）
    if train_indices and val_indices:
        gap_raw = val_start * st - (train_end - 1) * st - wl
        assert gap_raw >= 0, (
            f"blocked_time_split leak: gap_raw={gap_raw} < 0 "
            f"(train_end={train_end}, val_start={val_start}, wl={wl}, st={st})"
        )

    return train_indices, val_indices


def create_blocked_split(dataset, train_ratio=0.8, gap_ratio=1.0, window_len=None, stride=None):
    """
    创建 blocked split 的 Subset 对象，直接替代 random_split。

    返回:
        (train_dataset, val_dataset): Subset 对象
    """
    train_idx, val_idx = blocked_time_split(dataset, train_ratio, gap_ratio, window_len, stride)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_val_subset(subject_id, data_root="./data", window_ms=300, step_ms=50, train_ratio=0.8):
    """返回单个受试者的 val-only Subset，供所有评估实验使用。"""
    ds = NinaProDataset(data_root, subject_id, window_ms=window_ms, target_fs=FS, step_ms=step_ms)
    _, val_ds = create_blocked_split(ds, train_ratio=train_ratio)
    return ds, val_ds


def get_val_background(val_ds, n_bg=20, seed=42):
    """从 val_ds 中提取背景数据（优先 rest 样本），确保 background 不来自训练集。"""
    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    all_x, all_y = next(iter(loader))
    rest_mask = (all_y == 0)
    if rest_mask.sum() >= n_bg:
        bg_data = all_x[rest_mask][:200]
    else:
        # 如果 val 中 rest 样本不够，取 val 前 n_bg 个
        bg_data = all_x[:n_bg]
    return bg_data.to(DEVICE)


def collect_onset_windows(val_ds, target_count=None, rest_label=0):
    """
    从 val split 中提取 rest -> active 的 onset 窗口，确保所有相关实验共享同一采样协议。

    Returns:
        (windows, labels): 两个列表，元素分别为单个窗口张量和对应标签
    """
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    onset_windows = []
    onset_labels = []
    prev_label = rest_label

    for x, y in loader:
        curr_label = int(y.item())
        is_onset = (prev_label == rest_label) and (curr_label != rest_label)
        prev_label = curr_label

        if is_onset:
            onset_windows.append(x[0].clone())
            onset_labels.append(curr_label)
            if target_count is not None and len(onset_windows) >= target_count:
                break

    return onset_windows, onset_labels


def describe_blocked_split(dataset, train_ratio=0.8, gap_ratio=1.0, window_len=None, stride=None):
    """
    返回 blocked split 的索引与原始采样点范围，便于打印和验证无泄漏。
    """
    train_idx, val_idx = blocked_time_split(dataset, train_ratio, gap_ratio, window_len, stride)
    wl = window_len if window_len is not None else getattr(dataset, 'window_len', 600)
    st = stride if stride is not None else getattr(dataset, 'stride', 100)

    def to_span(indices):
        if not indices:
            return None
        start_window = indices[0]
        end_window = indices[-1]
        raw_start = start_window * st
        raw_end = end_window * st + wl - 1
        return {
            "window_range": [int(start_window), int(end_window)],
            "raw_range": [int(raw_start), int(raw_end)],
        }

    return {
        "train": to_span(train_idx),
        "val": to_span(val_idx),
        "gap_windows": int(max(0, (val_idx[0] - train_idx[-1] - 1) if train_idx and val_idx else 0)),
        "gap_raw_samples": int(max(0, (val_idx[0] * st) - (train_idx[-1] * st + wl)) if train_idx and val_idx else 0),
    }


def multi_subject_blocked_split(subject_datasets, train_ratio=0.8):
    """
    多受试者数据集的无泄漏划分：对每个受试者独立做 blocked split 然后合并。
    
    参数:
        subject_datasets: 每个元素是单个受试者的 Dataset 对象
    
    返回:
        (train_dataset, val_dataset): 合并后的 Subset 列表合并
    """
    from torch.utils.data import ConcatDataset
    train_parts = []
    val_parts = []
    for ds in subject_datasets:
        tr, val = create_blocked_split(ds, train_ratio)
        train_parts.append(tr)
        val_parts.append(val)
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


# ================= 统计工具函数 (从 experiments_improved 提取) =================
def calculate_cohens_d(group1, group2):
    """
    计算 Cohen's d 效应量

    Args:
        group1, group2: 两组数据（列表或数组）

    Returns:
        float: Cohen's d 值
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def interpret_cohens_d(d):
    """
    解释 Cohen's d 效应量大小

    Args:
        d: Cohen's d 值

    Returns:
        str: 效应量解释
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data, n_bootstrap=10000, ci=95, statistic=np.mean):
    """
    计算 Bootstrap 置信区间

    Args:
        data: 原始数据
        n_bootstrap: Bootstrap 采样次数
        ci: 置信水平（百分比）
        statistic: 统计量函数（默认均值）

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, 100 - (100 - ci) / 2)

    return lower, upper
