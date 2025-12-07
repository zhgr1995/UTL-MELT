#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UMuLT, original name: UTL-MELT; Some symbols in the text and code are inconsistent, please check and identify them carefully
"""

import os, math, random, warnings, cv2, torchaudio, torch
import numpy as np, pandas as pd
from typing import Dict, List
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import pickle
import torch.nn as nn
import torch.nn.functional as F
import argparse, csv, datetime
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchmetrics.classification import CalibrationError
from sklearn.metrics import recall_score

# ---------- 1 配置 ----------
CONFIG: Dict = {
    "num_classes":        7,#情感分类标签数目
    "batch_size":         36,#最大就64吧
    "epochs_stage1":      12, # Adapter 预训练轮次
    "epochs_stage2":      24,# 全量训练轮次
    "warmup_epochs":      6,
    "base_lr":            2e-5,#太大：训练易发散，指标抖动；太小：收敛慢，容易陷入鞍点。
    "weight_decay":       5e-5,#过大：模型容量受限，欠拟合（特别是尾部类学习不足）；过小或 =0：易过拟合头部类，尾部样本噪声影响更大；
    "video_frames":       4,#视频帧数4或8；
    "tau_unc":            0.08,#小于 τ 的不确定性 u 直接赋权 1；大于 τ 的 u 会按 max（） 线性衰减。
    "kappa":              0.85,#越大 → u>τ 时权重迅速衰减，极端不确定性专家几乎被忽略；越小 → 对高不确定性专家更宽容，保留多模态信息。
    "eps":                1e-6,
    "beta_cb":            0.9997,#β 越接近 1 → 对小样本（尾部）惩罚越大，提升尾部损失权重；随 β→0 → 权重趋于 1，退化为普通交叉熵。
    "ldam_C":             1,#C 越大 → 对尾部类别 margin 增大，模型对尾部更“宽容”；C 越小 → margin 减弱，更接近无边界的 CLF。
    "ema_mom":            0.995,#m 越大 → 历史统计更平滑，对短期波动不敏感；m 越小 → 更快响应当前 batch 失衡。
    "alpha_fair":         0.65,#α 越大 → 对高历史 loss 类别（通常是尾部）加大惩罚；α 越小 → 各类别权重趋于一致。越低越好
    "lambda_cons":        0.80,#λ_cons 越大 → 强化模态间一致性，有助于尾部样本。
    "sampler_replacement": True,# 样本采样有放回，提升尾部样本曝光频率
    "async_av_k":         1,#=1：文本、音、视同步更新；>1：每 k 步更新一次音/视，文本更新更频繁。
    "audio_len":          100,   # 固定MFCC序列长度
    "prefix_temp":        5, #用于冲突感知前缀加权
    "freeze_bert_layers": 3,        # 可设 0 表示完全不冻结
    "enable_ema_fair": True,  #消融开关，下同
    "enable_uncertainty_gate": True,
    "enable_cross_modal_align": True,
    "use_ldam_cb": True,
    "enable_text": True,
    "enable_audio": True,
    "enable_video": True,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_cons", type=float, default=CONFIG["lambda_cons"])
    p.add_argument("--alpha_fair",  type=float, default=CONFIG["alpha_fair"])
    p.add_argument("--tau_unc", type=float, default=CONFIG["tau_unc"])
    return p.parse_args()
args = parse_args()
CONFIG["lambda_cons"] = args.lambda_cons
CONFIG["alpha_fair"]  = args.alpha_fair
CONFIG["tau_unc"]  = args.tau_unc

# ---------- 2 路径与标签 ，自行修改路径----------
PATHS = {
    'train_csv' : 'train_subset.csv',
    'train_audio': 'MELD.Raw/audio/train_splits',
    'train_video': 'MELD.Raw/video/train_splits',
    'dev_csv'  : 'dev_subset.csv',
    'dev_audio': 'MELD.Raw/audio/dev_splits',
    'dev_video': 'MELD.Raw/video/dev_splits',
    'test_csv' : 'test_subset.csv',
    'test_audio': 'MELD.Raw/audio/test_splits',
    'test_video': 'MELD.Raw/video/test_splits',
}


LABEL_MAP = {
    'neutral':0, 'joy':1, 'sadness':2, 'anger':3,
    'surprise':4, 'fear':5, 'disgust':6
}
GROUP = {
    "head":[0,1],        # neutral, joy
    "medium":[3,4],      # anger, surprise
    "tail":[2,5,6],      # sadness, fear, disgust
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 3. 数据集定义 ----------
class MELDDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_root, video_root, tokenizer, max_len:int=128):
        self.df = pd.read_csv(csv_path)
        self.ar = audio_root
        self.vr = video_root
        self.tok = tokenizer
        self.max_len = max_len
        self.mfcc_mean  = 0.0
        self.mfcc_std   = 1.0
        self.frame_mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
        self.frame_std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
        self.mfcc = torchaudio.transforms.MFCC(16000, n_mfcc=40)
        self.img_tfms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.num_f     = CONFIG["video_frames"]
        self.audio_len = CONFIG["audio_len"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y = LABEL_MAP[r.Emotion]
        tok = self.tok(r.Utterance, truncation=True, padding="max_length", 
                       max_length=self.max_len, return_tensors="pt")
        text = {k:v.squeeze(0) for k,v in tok.items()}
        # 音频 MFCC
        af = os.path.join(self.ar, f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.wav")
        if os.path.exists(af):
            wav, sr = torchaudio.load(af)
            wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr,16000)(wav)
            mfcc = self.mfcc(wav).squeeze(0)
        else:
            mfcc = torch.randn(40, self.audio_len)*self.mfcc_std + self.mfcc_mean
        T = mfcc.shape[1]
        if T >= self.audio_len:
            mfcc = mfcc[:,:self.audio_len]
        else:
            pad = torch.zeros(40, self.audio_len - T)
            mfcc = torch.cat([mfcc, pad], dim=1)
        # 视频帧提取
        vf = os.path.join(self.vr, f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}.mp4")
        frames = []
        if os.path.exists(vf):
            cap = cv2.VideoCapture(vf)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(self.num_f):
                fid = min(int(total * i / self.num_f), total-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, fr = cap.read()
                if ret:
                    frames.append(self.img_tfms(Image.fromarray(fr[...,::-1])))
            cap.release()
        # 补帧
        if len(frames) < self.num_f:
            fake = torch.randn(self.num_f, 3, 224, 224)*self.frame_std + self.frame_mean
            frames = frames + [fake[i] for i in range(self.num_f-len(frames))]
        video = torch.stack(frames[:self.num_f], 0)
        return text, mfcc, video, torch.tensor(y)

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_ch=40, out_dim=256):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_ch, 64, 7, 2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(512*block.expansion, out_dim)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out).squeeze(-1)
        return self.fc(out)

# ---------- 4. 分支与模块 ----------
class TextBranch(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        n_freeze = CONFIG.get("freeze_bert_layers", 0)
        if n_freeze > 0:
            for p in self.bert.encoder.layer[:n_freeze]:
                p.requires_grad_(False)
        self.proj = nn.Linear(self.bert.config.hidden_size, d)
    def forward(self, x):
        return self.proj(self.bert(**x).pooler_output)

class AudioBranch(nn.Module):
    def __init__(self, d=256): super().__init__(); self.net = ResNet1D(BasicBlock1D, 
                                                                       [2,2,2,2], in_ch=40, out_dim=d)
    def forward(self, a): return self.net(a)

class VisualBranch(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        res = models.resnet50(weights=weights); res.fc = nn.Identity()
        self.backbone = res; self.fc = nn.Linear(2048, d)
    def forward(self, v):
        B,F,C,H,W = v.shape
        z = self.backbone(v.view(B*F,C,H,W)).view(B,F,-1).mean(1)
        return self.fc(z)

class ResidualAdapter(nn.Module):
    def __init__(self, hid, bn=64): super().__init__(); self.down=nn.Linear(hid,bn); self.up=nn.Linear(bn,hid)
    def forward(self, x): return x + self.up(F.relu(self.down(x)))

class CrossAttn(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(2 * d_model, d_model)  # KV输入=2*d_model
        self.v = nn.Linear(2 * d_model, d_model)  # 适配拼接维度
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out  = nn.Linear(d_model, d_model)
    
    def forward(self, q_in, kv_in):
        """
        Args:
            q_in: [B, d_model] 查询模态
            kv_in: [B, 2*d_model] 拼接的键值模态
        """
        q = self.q(q_in).unsqueeze(1)      # [B, 1, d_model]
        k = self.k(kv_in).unsqueeze(1)     # [B, 1, d_model]
        v = self.v(kv_in).unsqueeze(1)     # [B, 1, d_model]
        y, _ = self.attn(q, k, v, need_weights=False)
        return self.out(y.squeeze(1))      # [B, d_model]

class EvidenceHead(nn.Module):
    def __init__(self, d, num_cls): super().__init__(); self.fc=nn.Linear(d,num_cls)
    def forward(self, x): return F.relu(self.fc(x))

# ---------- 5. UTL-MELT ----------
class UTL_MELT(nn.Module):
    def __init__(self, d=256, num_cls=7):
        super().__init__()
        self.CONFIG = CONFIG
        self.txt = TextBranch(d)
        self.aud = AudioBranch(d)
        self.vis = VisualBranch(d)
        if CONFIG["enable_cross_modal_align"]:
            self.align_t = CrossAttn(d)
            self.align_a = CrossAttn(d)
            self.align_v = CrossAttn(d)
        else:
            self.align_t = None
            self.align_a = None
            self.align_v = None
        self.adp_t = ResidualAdapter(d)
        self.adp_a = ResidualAdapter(d)
        self.adp_v = ResidualAdapter(d)
        self.eh_t = EvidenceHead(d, num_cls)
        self.eh_a = EvidenceHead(d, num_cls)
        self.eh_v = EvidenceHead(d, num_cls)

    def forward(self, text, audio, video, return_intermediates=False):
        # ========== Step 1: 特征提取 ==========
        ft0 = self.txt(text)
        fa0 = self.aud(audio)
        fv0 = self.vis(video)

        # ========== Step 2: 模态消融开关 ==========
        if not self.CONFIG["enable_text"]:
            ft0 = torch.zeros_like(ft0)
        if not self.CONFIG["enable_audio"]:
            fa0 = torch.zeros_like(fa0)
        if not self.CONFIG["enable_video"]:
            fv0 = torch.zeros_like(fv0)

        # ========== Step 3: 跨模态对齐（论文Eq. 2-4） ==========
        if self.CONFIG["enable_cross_modal_align"]:
            # Text查询Audio+Visual的拼接
            ft1 = self.align_t(ft0, torch.cat([fa0, fv0], dim=-1))
            # Audio查询Text+Visual的拼接
            fa1 = self.align_a(fa0, torch.cat([ft0, fv0], dim=-1))
            # Visual查询Text+Audio的拼接
            fv1 = self.align_v(fv0, torch.cat([ft0, fa0], dim=-1))
        else:
            ft1, fa1, fv1 = ft0, fa0, fv0

        # ========== Step 4: 适配器（论文Eq. 1+5） ==========
        ft = self.adp_t(ft1)
        fa = self.adp_a(fa1)
        fv = self.adp_v(fv1)

        # ========== Step 5: 生成Evidence（论文Eq. 6） ==========
        et = self.eh_t(ft)  # [B, K]
        ea = self.eh_a(fa)
        ev = self.eh_v(fv)
        E = torch.stack((et, ea, ev), dim=1)  # [B, 3, K]

        # ========== Step 6: 证据融合（论文Eq. 6-14） ==========
        K = self.CONFIG['num_classes']
        eps = self.CONFIG['eps']
        
        # 6.1 计算Dirichlet参数和不确定性（论文Eq. 6）
        alpha_t = et + 1.0  # [B, K]
        alpha_a = ea + 1.0
        alpha_v = ev + 1.0
        
        S_t = alpha_t.sum(dim=1, keepdim=True)  # [B, 1]
        S_a = alpha_a.sum(dim=1, keepdim=True)
        S_v = alpha_v.sum(dim=1, keepdim=True)
        
        u_t = K / S_t  # [B, 1] 不确定性
        u_a = K / S_a
        u_v = K / S_v
        
        # 6.2 计算belief mass（论文Eq. 7）
        bm_t = et / S_t  # [B, K]
        bm_a = ea / S_a
        bm_v = ev / S_v
        
        # 6.3 归一化belief用于冲突度计算（论文Eq. 8）
        eps_norm = 1e-6
        belief_sum_t = bm_t.sum(dim=1, keepdim=True)  # [B, 1]，应≈1-u_m
        belief_sum_a = bm_a.sum(dim=1, keepdim=True)
        belief_sum_v = bm_v.sum(dim=1, keepdim=True)
        
        b_tilde_t = torch.where(
            belief_sum_t >= eps_norm,
            bm_t / (belief_sum_t + eps_norm),
            torch.ones_like(bm_t) / K
        )  # [B, K]
        b_tilde_a = torch.where(
            belief_sum_a >= eps_norm,
            bm_a / (belief_sum_a + eps_norm),
            torch.ones_like(bm_a) / K
        )
        b_tilde_v = torch.where(
            belief_sum_v >= eps_norm,
            bm_v / (belief_sum_v + eps_norm),
            torch.ones_like(bm_v) / K
        )
        
        # 6.4 计算冲突度（论文Eq. 9）
        C_a = 1.0 - (b_tilde_a * b_tilde_t).sum(dim=1)  # [B] Audio vs Text
        C_v = 1.0 - (b_tilde_v * b_tilde_a).sum(dim=1)  # [B] Visual vs Audio
        C_a = torch.clamp(C_a, 0.0, 1.0)
        C_v = torch.clamp(C_v, 0.0, 1.0)
        
        # 6.5 计算prefix权重（论文Eq. 10）
        u_t_scalar = u_t.squeeze(1)  # [B]
        u_a_scalar = u_a.squeeze(1)
        u_v_scalar = u_v.squeeze(1)
        
        w_t = 1.0 - u_t_scalar                      # Text: 1 - u^(1)
        w_a = (1.0 - u_a_scalar) * (1.0 - C_a)      # Audio: (1-u^(2))(1-C^(2))
        w_v = (1.0 - u_v_scalar) * (1.0 - C_v)      # Visual: (1-u^(3))(1-C^(3))
        
        w_pref = torch.stack([w_t, w_a, w_v], dim=1)  # [B, 3]
        
        # 6.6 Temperature-controlled softmax（论文Eq. 11）
        a_tilde = F.softmax(w_pref / self.CONFIG['prefix_temp'], dim=1)  # [B, 3]
        
        # 6.7 不确定性门控（论文Eq. 12）
        u_stack = torch.cat([u_t, u_a, u_v], dim=1)  # [B, 3]
        
        if self.CONFIG["enable_uncertainty_gate"]:
            tau = self.CONFIG['tau_unc']
            kappa = self.CONFIG['kappa']
            
            reliable = (u_stack <= tau).float()
            w_unc = reliable + (1 - reliable) * torch.clamp(
                1.0 - kappa * (u_stack - tau),
                min=0.01
            )  # [B, 3]
        else:
            w_unc = torch.ones_like(u_stack) / 3.0
        
        # 6.8 混合融合（论文Eq. 13-14）
        w_hat = a_tilde * w_unc  # [B, 3]
        wexp = w_hat * torch.exp(-u_stack)  # [B, 3]
        alpha = wexp / (wexp.sum(dim=1, keepdim=True) + eps)  # [B, 3] 最终权重
        
        e_final = (alpha.unsqueeze(-1) * E).sum(dim=1)  # [B, K]
        
        # ========== Step 7: 返回 ==========
        if return_intermediates:
            return e_final, (et, ea, ev), u_stack
        else:
            return e_final, (et, ea, ev)
            
# ---------- 6. 复合损失 ----------
class CompositeLossUTL(nn.Module):
    def __init__(self, freq: List[int], num_cls: int = 7):
        super().__init__()
        freq = torch.tensor(freq, dtype=torch.float32)
        # LDAM margin
        self.register_buffer('margin', CONFIG['ldam_C'] / torch.sqrt(torch.sqrt(freq)))
        self.register_buffer('freq', freq)
        # EMA fairness buffer
        self.register_buffer('ema', torch.zeros(num_cls))

    def forward(self, logits, y, evidences, return_base=False):
        B, K = logits.shape
        # ===== LDAM-CB  =====
        if CONFIG.get('use_ldam_cb', True):
            # LDAM margin adjustment
            margin_vector = self.margin[y]
            margin_mat = torch.zeros_like(logits)
            margin_mat[torch.arange(B), y] = margin_vector
            adj_logits = logits - margin_mat
            # Class-balanced weighting
            gamma = (1 - CONFIG['beta_cb']) / (1 - CONFIG['beta_cb'] ** self.freq[y])
            base = F.cross_entropy(adj_logits, y, reduction='none') * gamma
        else:
            # Standard cross entropy
            base = F.cross_entropy(logits, y, reduction='none')

        # ===== EMA-Fairness 权重（只读，不在这里更新 EMA） =====
        if CONFIG.get('enable_ema_fair', True):
            lam = F.softmax(CONFIG['alpha_fair'] * (self.ema - self.ema.mean()), dim=0)
        else:
            lam = torch.ones_like(base)

        # 一致性损失（由 lambda_cons 控制）
        et, ea, ev = evidences
        cons_loss = CONFIG['lambda_cons'] * (
            F.mse_loss(et, ea, reduction='mean') +
            F.mse_loss(et, ev, reduction='mean') +
            F.mse_loss(ea, ev, reduction='mean')
        )

        # 总loss = 权重 * 基loss + 一致性
        loss = (lam[y] * base).mean() + cons_loss
        if return_base:
            return loss, base.detach()
        return loss

    @torch.no_grad()
    def epoch_update_ema(self, class_mean_loss: torch.Tensor):
        self.ema = self.ema * CONFIG['ema_mom'] + (1 - CONFIG['ema_mom']) * class_mean_loss
# ---------- 7. 训练与评估 ----------
ece_meter = CalibrationError(task="multiclass", num_classes=CONFIG["num_classes"],n_bins=10, norm="l1").to(DEVICE)
@torch.no_grad()
def evaluate(model, loader, save_prefix=None):
    model.eval()
    logits_list, y_true_list = [], []
    with torch.no_grad():
        for text, audio, video, y in loader:
            # 和 train_epoch 保持一致的解包方式
            text  = {k: v.to(DEVICE) for k, v in text.items()}
            audio = audio.to(DEVICE)
            video = video.to(DEVICE)
            y     = y.to(DEVICE)
            # forward 返回 (logits, evidences)
            logits, _ = model(text, audio, video)
            logits_list.append(logits)
            y_true_list.append(y)
    logits = torch.cat(logits_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    probs = torch.softmax(logits, dim=1)
    y_pred = probs.argmax(dim=1)
    # 基本指标
    res = {
        "acc": accuracy_score(y_true.cpu(), y_pred.cpu()),
        "f1":  f1_score(y_true.cpu(), y_pred.cpu(), average="macro"),
        "gmean": geometric_mean(y_true.cpu(), y_pred.cpu()),
    }

    # ECE & AUC
    res["ece"] = ece_meter(probs, y_true).item()
    res["auc"] = roc_auc_score(
        y_true.cpu().numpy(),
        probs.cpu().numpy(),
        multi_class="ovo"
    )

    # 各组指标
    for name, cls_ids in GROUP.items():
        mask = torch.isin(y_true, torch.tensor(cls_ids, device=y_true.device))
        if mask.any():
            y_t, y_p = y_true[mask].cpu(), y_pred[mask].cpu()
            res[f"{name}_acc"]   = accuracy_score(y_t, y_p)
            res[f"{name}_f1"]    = f1_score(y_t, y_p, average="macro")
            res[f"{name}_gmean"] = geometric_mean(y_t, y_p)
    # 保存到 CSV
    if save_prefix:
        # 全局指标
        df_global = pd.DataFrame([{
            'acc': res['acc'],
            'f1': res['f1'],
            'gmean': res['gmean'],
            'ece': res['ece'],
            'auc': res['auc']
        }])
        df_global.to_csv(f"{save_prefix}_global_metrics.csv", index=False)

        # 组级指标
        rows = []
        for grp in GROUP:
            rows.append({
                'group': grp,
                'acc': res.get(f"{grp}_acc", float('nan')),
                'f1': res.get(f"{grp}_f1", float('nan')),
                'gmean': res.get(f"{grp}_gmean", float('nan'))
            })
        pd.DataFrame(rows).to_csv(f"{save_prefix}_group_metrics.csv", index=False)

    return res
    
def geometric_mean(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    # 避免负值或零导致 nan
    recalls = np.clip(recalls, a_min=1e-6, a_max=None)
    return float(np.prod(recalls) ** (1.0 / len(recalls)))

MANUAL_TAIL = {2, 5, 6}  # sadness, disgust, fear
def build_tail_loader(dataset):
    labels = np.array(dataset.df.Emotion.map(LABEL_MAP))
    idx = np.where(np.isin(labels, list(MANUAL_TAIL)))[0]
    subset = Subset(dataset, idx.tolist())
    freq = np.bincount(labels, minlength=CONFIG["num_classes"])
    tail_labels = labels[idx]
    weights = 1.0 / freq[tail_labels]
    sampler=WeightedRandomSampler(
            weights.tolist(),num_samples=len(weights), replacement=True
        )
    loader = DataLoader(
        subset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=0, 
        pin_memory=False
        )
    print(f"Tail subset → {len(subset)} samples, classes {sorted(MANUAL_TAIL)}")
    return loader 
    
def build_sampler(df: pd.DataFrame):
    freq=df.Emotion.value_counts().reindex(LABEL_MAP.keys()).astype(int)
    inv=1./freq
    w=torch.tensor([inv[e] for e in df.Emotion],dtype=torch.float)
    return WeightedRandomSampler(w,len(w),replacement=CONFIG['sampler_replacement']),freq.tolist()

def train_epoch(model, loader, crit, opt, scaler, stage: int):
    model.train()
    total_batches=len(loader)
    pbar=tqdm(loader,total=total_batches,desc=f"Stage-{stage} Train",unit="batch",dynamic_ncols=True)
       # 每个 epoch 累计的类损失
    K = CONFIG["num_classes"]
    epoch_cls_loss_sum  = torch.zeros(K, device=DEVICE)
    epoch_cls_loss_cnt  = torch.zeros(K, device=DEVICE)
    for step,(text,audio,video,y) in enumerate(pbar):
        text={k:v.to(DEVICE) for k,v in text.items()}
        audio=audio.to(DEVICE); video=video.to(DEVICE); y=y.to(DEVICE)
        if stage==2 and CONFIG['async_av_k']>1:
            for p in list(model.aud.parameters())+list(model.vis.parameters()):
                p.requires_grad_((step%CONFIG['async_av_k'])==0)
        with torch.cuda.amp.autocast():
            logits,evid=model(text,audio,video)
            loss, base_vec = crit(logits, y, evid, return_base=True)
                # 累计每类平均 base loss
        with torch.no_grad():
            for c in range(K):
                mask = (y == c)
                if mask.any():
                    epoch_cls_loss_sum[c] += base_vec[mask].mean()
                    epoch_cls_loss_cnt[c] += 1
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        pbar.set_postfix({"loss":f"{loss.item():.4f}"})
    pbar.close()
        # 计算 epoch 内各类平均损失
    with torch.no_grad():
        # 防止除0
        epoch_cls_loss_cnt = torch.clamp(epoch_cls_loss_cnt, min=1)
        epoch_cls_mean = epoch_cls_loss_sum / epoch_cls_loss_cnt

    return epoch_cls_mean
# ---------- 8. 主程序 ----------
def main():
    warnings.filterwarnings('ignore')
    tok=BertTokenizer.from_pretrained("bert-base-uncased")
    # ---- Dataset & Loader ----
    train_ds = MELDDataset(PATHS['train_csv'], PATHS['train_audio'], PATHS['train_video'], tok)
    dev_ds   = MELDDataset(PATHS['dev_csv'],   PATHS['dev_audio'],   PATHS['dev_video'],   tok)
    test_ds  = MELDDataset(PATHS['test_csv'],  PATHS['test_audio'],  PATHS['test_video'],  tok)

    sampler, freq = build_sampler(train_ds.df)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                              num_workers=0, pin_memory=False)
    dev_loader   = DataLoader(dev_ds,  batch_size=CONFIG['batch_size'], shuffle=False,
                              num_workers=0, pin_memory=False)
    tail_loader  = build_tail_loader(train_ds)   

    model=UTL_MELT().to(DEVICE)
    crit=CompositeLossUTL(freq).to(DEVICE)
    # 优化器与 scheduler 
    bert_params=list(model.txt.bert.parameters())
    bert_ids=set(map(id,bert_params))
    other = [(n, p) for n,p in model.named_parameters() if p.requires_grad and id(p) not in bert_ids]
    decay,nod=[],[]
    for n,p in other:
        if n.endswith('bias') or 'LayerNorm' in n: nod.append(p)
        else: decay.append(p)
    opt=torch.optim.AdamW([
        {'params':decay,'weight_decay':CONFIG['weight_decay']},
        {'params':bert_params,'weight_decay':1e-6},
    ],lr=CONFIG['base_lr'],betas=(0.9,0.98))

    total_steps=(CONFIG['epochs_stage1']+CONFIG['epochs_stage2'])*len(train_loader)
    warm_steps=CONFIG['warmup_epochs']*len(train_loader)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: (s+1)/warm_steps 
            if s<warm_steps else 0.5*(1+math.cos(math.pi*(s-warm_steps)/(total_steps-warm_steps))))
    scaler=torch.cuda.amp.GradScaler()
    # ---------- Stage‑1 : Tail Adapter ----------
    for p in model.parameters():
        p.requires_grad_(False)
    for n,p in model.named_parameters():
        if n.startswith(('adp_','eh_')):
            p.requires_grad_(True)
    # 细节冻结与解冻

    print("\n== Stage‑1 (tail adapter) ==")
    for _ in range(CONFIG['epochs_stage1']):
        train_epoch(model, tail_loader, crit, opt, scaler, stage=1)   # <‑‑ 使用 tail_loader
        sched.step()

    # ---------- Stage‑2 : Full Fine‑tune ----------
    for p in model.parameters():
        p.requires_grad_(True)
    best=0.0; history=[]
    print("\n== Stage-2 (full fine-tune) ==")
    for ep in range(1,CONFIG['epochs_stage2']+1):
        cls_mean = train_epoch(model,train_loader,crit,opt,scaler,stage=2)
        if CONFIG.get('enable_ema_fair', True):
            crit.epoch_update_ema(cls_mean)
        sched.step()
        m=evaluate(model,dev_loader)
        history.append({'epoch':ep,**m})
        print(f"[E{ep:02d}] ACC={m['acc']:.3f} F1={m['f1']:.3f} H/M/T {m['head_f1']:.3f}/{m['medium_f1']:.3f}/{m['tail_f1']:.3f}")
        if m['f1']>best:
            best=m['f1']; torch.save(model.state_dict(),'utl_melt_best.pth')
    pd.DataFrame(history).to_csv('dev_history.csv',index=False)
    print('Best Dev Macro-F1 =',best)
    # Test 评估
    test_ds=MELDDataset(PATHS['test_csv'],PATHS['test_audio'],PATHS['test_video'],tok)
    test_loader=DataLoader(test_ds,batch_size=CONFIG['batch_size'],shuffle=False,num_workers=4)
    tm=evaluate(model,test_loader,save_prefix='test')
    print("\n== TEST ==")
    print(f"GLOBAL ACC {tm['acc']:.3f} F1 {tm['f1']:.3f} G-mean {tm['gmean']:.3f}")
    print("全局、组级指标已保存。")
    # === 写入超参敏感性结果 ===
    csv_path = "sensitivity_results.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["lambda_cons", "alpha_fair", "acc", "f1", "gmean"])
        w.writerow([CONFIG["lambda_cons"], CONFIG["alpha_fair"],
                    tm["acc"], tm["f1"], tm["gmean"]])
    # === 写入 τ_unc 敏感性结果 ===
    tau_csv = "tau_sensitivity.csv"
    first = not os.path.exists(tau_csv)
    with open(tau_csv, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["tau_unc", "acc"])
        w.writerow([CONFIG["tau_unc"], tm["acc"]])

def collect_and_export_vis_data(model, tokenizer):
    ds = MELDDataset(PATHS['test_csv'], PATHS['test_audio'], PATHS['test_video'], tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    model.eval()

    # --------- 初始化累加器 ----------
    sums_ef = {tag: {m: torch.zeros(CONFIG["num_classes"])       # 7 维
                     for m in ("text","audio","visual","fused")}
               for tag in ("head","tail")}
    sums_u  = {tag: {m: 0.0 for m in ("text","audio","visual")}
               for tag in ("head","tail")}
    counts  = {"head":0, "tail":0}

    with torch.no_grad():
        for text,audio,video,y in loader:
            lbl = int(y.item())
            tag = "head" if lbl in GROUP["head"] else "tail" if lbl in GROUP["tail"] else None
            if tag is None:
                continue

            # forward
            text  = {k:v.to(DEVICE) for k,v in text.items()}
            audio = audio.to(DEVICE); video=video.to(DEVICE)
            e_final, (et,ea,ev), us = model(text,audio,video,return_intermediates=True)
            u_t, u_a, u_v = us.squeeze(0).cpu().tolist()

            # 累加
            sums_ef[tag]["text"]   += et.squeeze(0).cpu()
            sums_ef[tag]["audio"]  += ea.squeeze(0).cpu()
            sums_ef[tag]["visual"] += ev.squeeze(0).cpu()
            sums_ef[tag]["fused"]  += e_final.squeeze(0).cpu()
            sums_u[tag]["text"]   += u_t
            sums_u[tag]["audio"]  += u_a
            sums_u[tag]["visual"] += u_v
            counts[tag] += 1

    # --------- 求平均 ----------
    for tag in ("head","tail"):
        for m in sums_ef[tag]:
            sums_ef[tag][m] /= counts[tag]
        for m in sums_u[tag]:
            sums_u[tag][m]  /= counts[tag]

    # --------- 写 CSV ----------
    #  evidence
    ev_rows = []
    for tag in ("tail","head"):
        for m, vec in sums_ef[tag].items():        # vec.shape = [7]
            for cls_idx, val in enumerate(vec.tolist()):
                ev_rows.append({"sample": tag, "modality": m,
                                "class_idx": cls_idx, "evidence": val})
    pd.DataFrame(ev_rows).to_csv("vis_evidence.csv", index=False)

    # uncertainty
    unc_rows = []
    for tag in ("tail","head"):
        for m, val in sums_u[tag].items():
            unc_rows.append({"sample": tag, "modality": m, "uncertainty": val})
    pd.DataFrame(unc_rows).to_csv("vis_uncertainty.csv", index=False)

    print("vis_evidence.csv 与 vis_uncertainty.csv 已生成（均值统计）")


if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("Device:", DEVICE)
    main()

    # **训练+测试跑完后导出
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = UTL_MELT().to(DEVICE)
    model.load_state_dict(torch.load('utl_melt_best.pth', map_location=DEVICE))
    collect_and_export_vis_data(model, tok)
