# UMuLT (aka UTL-MELT)
#Quantifying Epistemic Uncertainty in Multimodal Long-Tailed Classification: A Belief Entropy-Based Evidential Fusion Framewor#


> This repository implements the UMuLT framework (original code name: UTL-MELT) proposed in the paper.
> It targets multimodal long-tailed distribution learning by combining evidence-theoretic uncertainty modeling, conflict-aware prefix weighting, EMA-based fairness reweighting, and a two-stage curriculum training, achieving robust text–audio–visual classification.

---

## 🔑 Highlights

* **Unified tri-modal modeling**: Text (BERT), audio (MFCC + 1D ResNet18), and visual (ImageNet ResNet50) feature extraction.
* **Cross-modal alignment**: Single-pass Cross-Attention (t→a, a→v, v→t) + lightweight Residual Adapter.
* **Evidence-theoretic fusion**: Dempster–Shafer theory to construct evidence vectors, estimate modality uncertainty $u_m$, and belief masses $b_{k,m}$.
* **Conflict-aware prefix weighting**: Sequential modality introduction with conflict-dependent trust weighting.
* **Double normalization & gating**: Temperature Softmax, threshold gating, and $\exp(-u_m)$ to prevent any modality from dominating.
* **CB-LDAM + EMA fairness reweighting**: Balances class-frequency discrepancy and difficulty, amplifying tail-class loss adaptively.
* **Two-stage training strategy**: Stage‑1 trains only adapters and heads with tail data; Stage‑2 fully fine-tunes all parameters.
* **Visualization export**: Generates `vis_evidence.csv` and `vis_uncertainty.csv` to analyze modality contributions.

---

## 📦 Environment & Dependencies

* Python ≥ 3.8
* PyTorch ≥ 1.12 (CUDA recommended)
* torchvision, torchaudio, transformers
* scikit-learn, torchmetrics, pandas, numpy, tqdm, opencv-python

```bash
conda create -n umult python=3.8 -y
conda activate umult
pip install -r requirements.txt
```

`requirements.txt` example:

```text
torch>=1.12
torchvision>=0.13
torchaudio>=0.12
transformers>=4.30
scikit-learn
torchmetrics
pandas
numpy
tqdm
opencv-python
```

---

## 📂 Dataset Preparation

The implementation uses **MELD** (7-class emotion, text-audio-visual).
Please download and unpack MELD, ensuring the following paths or modify them in `PATHS`:

```python
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
```

CSV files should contain: `Dialogue_ID, Utterance_ID, Emotion, Utterance`.

---

## 🚀 Training & Evaluation

### 1. Run main script

```bash
python utl_melt.py \
  --lambda_cons 0.8 \
  --alpha_fair 0.65 \
  --tau_unc 0.08
```

### 2. Two-stage training

* **Stage‑1**: Train only Adapters (`adp_*`) + Evidence Heads (`eh_*`) using the tail subset loader.
* **Stage‑2**: Unfreeze all parameters, train on the full dataset, and update EMA fairness weights every epoch.

### 3. Key switches & hyperparameters

Modify directly in the `CONFIG` dictionary.

### 4. Validation / Testing

* Development history is saved in `dev_history.csv`.
* The best model is saved as `utl_melt_best.pth`.
* Test evaluation generates:

  * `test_global_metrics.csv`
  * `test_group_metrics.csv` (Head/Medium/Tail)

### 5. Visualization export

`collect_and_export_vis_data()` generates:

* `vis_evidence.csv`: mean evidence per modality and fused output.
* `vis_uncertainty.csv`: mean uncertainty per modality.

---

## 🧠 Theory-to-Code Mapping

| Paper Component               | Code Location                                 | Description                                            |
| ----------------------------- | --------------------------------------------- | ------------------------------------------------------ |
| Evidence vector $\mathbf e_m$ | `EvidenceHead.forward()` & `UTL_MELT.forward` | ReLU ensures non-negativity                            |
| Uncertainty $u_m = K/S_m$     | `UTL_MELT.forward`, “uncertainty computation” | `a0_* = e.sum + K`, `u_stack = K/a0_*`                 |
| Belief mass $b_{k,m}$         | Same as above                                 | `bm_* = e / a0_*`                                      |
| Conflict $C_m$                | “conflict computation”                        | Outer-product minus diagonal (i≠j)                     |
| Prefix weighting              | “prefix weight (Eq.8-10)”                     | `w_a = u_t; w_v = w_a*u_a/(1-C_v+eps)`                 |
| Double normalization & gating | `a_tilde / w_unc / exp(-u)`                   | Matches Eq.(11) & Eq.(13)                              |
| CB-LDAM                       | `CompositeLossUTL.forward()`                  | True-class margin subtraction + class-balanced `gamma` |
| EMA fairness reweighting      | `epoch_update_ema()` & `forward()`            | EMA updated per epoch, softmax yields `lam`            |
| Cross-modal consistency loss  | `cons_loss`                                   | Sum of 3 MSE terms, weighted by `lambda_cons`          |
| Two-stage training            | `main()` + `train_epoch()`                    | Freeze/unfreeze + tail\_loader usage                   |

---

## 📜 Citation & Acknowledgements

If you use this code or method, please cite:

```bibtex
@article{your2025umult,
  title   = {Quantifying Epistemic Uncertainty in Multimodal Long-Tailed Classification: A Belief Entropy-Based Evidential Fusion Framework},
  author  = {Guorui Zhu},
  journal = {Entropy},
  year    = {2026}
}
```

* MELD dataset copyright belongs to the original authors.
* Implementation partly refers to LDAM and Class-Balanced Loss.

---

## 📄 License

Choose a suitable license (MIT / Apache-2.0 / GPL-3.0) and add a `LICENSE` file.

---

## ❓ FAQ

**Q1: Can I train on a single modality?**
A: Yes, set `enable_audio=False`, etc.

**Q2: Why `async_av_k` exists?**
A: It’s an optional engineering trick for balancing update frequency. Not used in the paper (default `1`).

**Q3: Why update EMA fairness weights epoch-wise?**
A: To match the paper: class mean losses are aggregated every epoch before updating EMA.

---

## ℹ️ About Baseline Code & Ablation Studies

This repository currently **only releases the UMuLT core implementation**. Baselines and explicit ablation scripts are not provided due to:

1. **Licensing & dependency constraints**: Many baseline methods rely on third-party repositories; directly merging them risks license violations and heavy dependencies.
2. **Fairness & reproducibility**: The paper provides uniform dataset splits, metrics, and hyperparameter ranges. Re-implementing baselines independently ensures objective comparisons.
3. **Project scheduling**: The focus was on releasing the core algorithm first. Baseline adaptation scripts will be released after the paper acceptance for consistency.

### 🔬 Ablation Studies

Ablation can be easily conducted by toggling `CONFIG` flags:

* **Module-level**:

  * `enable_cross_modal_align` – disable Cross-Attention
  * `enable_uncertainty_gate` – disable uncertainty gating
  * `enable_ema_fair` – disable EMA fairness reweighting
  * `use_ldam_cb` – disable CB-LDAM, reverting to standard CE

* **Modality-level**:

  * `enable_text`, `enable_audio`, `enable_video` – control each modality.

Example:

```bash
# Disable cross-modal alignment and uncertainty gating
python utl_melt.py --lambda_cons 0.8 --alpha_fair 0.65 --tau_unc 0.08
# and in the script:
# CONFIG["enable_cross_modal_align"] = False
# CONFIG["enable_uncertainty_gate"] = False
```

### 📅 Future Work

* Baseline adaptation scripts will be released after paper acceptance.
* Automated batch ablation scripts are planned.

---

# UMuLT (aka UTL-MELT)

**Quantifying Epistemic Uncertainty in Multimodal Long-Tailed Classification: A Belief Entropy-Based Evidential Fusion Framewor**

> 本仓库实现了论文中提出的 UMuLT 框架（原始代码名为 UTL-MELT）。该方法针对多模态长尾分布数据，结合证据理论的不确定性建模、冲突感知的前缀加权、EMA 公平重加权以及两阶段课程训练，实现鲁棒的文本-音频-视觉三模态分类。

---

## 🔑 主要特性 / Highlights

* **三模态统一建模**：文本(BERT)、音频(MFCC+1D ResNet18)、视觉(ImageNet ResNet50) 特征抽取。
* **跨模态对齐**：单轮循环 Cross-Attention（t→a、a→v、v→t）+ 轻量级 Residual Adapter。
* **证据理论融合**：依据 Dempster–Shafer 证据理论构建证据向量、计算模态不确定性 $u_m$ 与信念质量 $b_{k,m}$。
* **冲突感知前缀权重**：按固定顺序逐模态引入，利用冲突度调节后续模态的信任权重。
* **双重归一化 + 门控**：温度 Softmax、阈值门控、$\exp(-u_m)$ 精细调权，避免单模态主导。
* **CB-LDAM + EMA 公平重加权**：兼顾类别样本数差异与实时难度，动态放大尾部类别损失。
* **两阶段训练策略**：Stage-1 冻结主干，仅训练 Adapter/Head 并只看尾部数据；Stage-2 全量解冻联合优化。
* **可视化导出**：训练后可导出 `vis_evidence.csv`、`vis_uncertainty.csv` 用于分析各模态贡献。

---

## 📦 环境与依赖

* Python ≥ 3.8
* PyTorch ≥ 1.12 (支持 CUDA)
* torchvision, torchaudio, transformers
* scikit-learn, torchmetrics, pandas, numpy, tqdm, opencv-python

```bash
conda create -n umult python=3.8 -y
conda activate umult
pip install -r requirements.txt
```

> 一个 `requirements.txt`，如不足以支持正常运行，请自行下载，注意使用cuda驱动而不是cpu驱动模块：

```text
torch>=1.12
torchvision>=0.13
torchaudio>=0.12
transformers>=4.30
scikit-learn
torchmetrics
pandas
numpy
tqdm
opencv-python
```

---

## 📂 数据准备

本实现默认使用 **MELD** 数据集（文本、音频、视频三模态，7 类情感）。

请下载并解压数据集后，保证以下路径与文件命名匹配（或在 `PATHS` 中自行修改）：

```python
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
```

同样，你需要保证 `csv` 文件包含 `Dialogue_ID, Utterance_ID, Emotion, Utterance` 等字段。

---

## 🚀 训练与评估

### 1. 运行主训练脚本

```bash
python utl_melt.py \
  --lambda_cons 0.8 \
  --alpha_fair 0.65 \
  --tau_unc 0.08
```

### 2. 两阶段训练流程

* **Stage‑1**：仅训练 Adapter(`adp_*`) + Evidence Head(`eh_*`) 模块，并使用尾部数据加载器 `tail_loader`。
* **Stage‑2**：解冻全网参数，使用全体训练集继续训练，并在每个 epoch 末利用 `epoch_update_ema()` 更新 EMA 权重。

### 3. 开关与超参（CONFIG）

请直接查看 `CONFIG` 字典。



### 4. 验证 / 测试

训练过程中会在 Dev 集上评估并保存 `dev_history.csv`；最佳模型自动保存为 `utl_melt_best.pth`。最终测试集评估生成：

* `test_global_metrics.csv`（全局指标）
* `test_group_metrics.csv`（Head/Medium/Tail 分组指标）

### 5. 可视化导出

脚本末尾的 `collect_and_export_vis_data()` 会生成：

* `vis_evidence.csv`：各模态/融合证据均值
* `vis_uncertainty.csv`：各模态不确定性均值

你可以据此绘制条形图或热力图，分析不同模态对不同类别的贡献。

---

## 🧠 理论到代码的映射

| 论文模块               | 代码对应位置                                          | 说明                                     |
| ------------------ | ----------------------------------------------- | -------------------------------------- |
| 证据向量 $\mathbf e_m$ | `EvidenceHead.forward()` & UTL\_MELT.forward    | ReLU 保证非负证据                            |
| 不确定性 $u_m=K/S_m$   | UTL\_MELT.forward，第  “计算不确定性” 处                 | `a0_* = e.sum + K`，`u_stack = K/a0_*`  |
| 信念质量 $b_{k,m}$     | 同上                                              | `bm_* = e / a0_*`                      |
| 冲突度 $C_m$          | UTL\_MELT.forward，“冲突度” 处                       | 通过外积减去对角线实现 i≠j 求和                     |
| 前缀权重递推             | UTL\_MELT.forward，“前缀权重(Eq.8-10)”               | `w_a = u_t; w_v = w_a*u_a/(1-C_v+eps)` |
| 双重归一化 & 门控         | UTL\_MELT.forward，“a\_tilde / w\_unc / exp(-u)” | 与论文 Eq.(11) & Eq.(13) 对应               |
| CB-LDAM            | `CompositeLossUTL.forward()`                    | 只对真类减 margin；类平衡权重 `gamma`             |
| EMA 公平重加权          | `epoch_update_ema()` + `forward()`              | 按 epoch 更新 EMA，softmax 得到 `lam`        |
| 跨模态一致性损失           | `cons_loss`                                     | 三对 MSE 求和，权重 `lambda_cons`             |
| 两阶段训练              | `main()` + `train_epoch()`                      | 冻结/解冻 + `tail_loader` 使用               |

---

## 📜 引用 & 致谢

如果你使用了本代码或方法，请引用我们的论文（请根据最终发表的期刊/会议填写 BibTeX）：

```bibtex
@article{your2025umult,
  title   = {Quantifying Epistemic Uncertainty in Multimodal Long-Tailed Classification: A Belief Entropy-Based Evidential Fusion Framewor},
  author  = {Guorui Zhu},
  journal = {Entropy},
  year    = {2026}
}
```

* 本仓库部分实现参考了 LDAM, Class-Balanced Loss 等经典方法。
* MELD 数据集版权归原作者所有。

---

## 📄 许可证

请根据你的实际情况选择合适的开源协议（MIT / Apache-2.0 / GPL-3.0 等）并在仓库根目录添加 `LICENSE` 文件。

---

## ❓ FAQ

**Q1: 可以只用单模态训练吗？**
A: 可以，设置 `enable_audio=False` 等即可做消融实验。

**Q2: 为什么有 `async_av_k`？**
A: 这是一个可选的工程技巧，用于降低计算量或平衡模态更新频率。论文未使用，默认设为 1 即可。

**Q3: EMA 公平权重为什么是按 epoch 更新？**
A: 为了与论文定义一致，我们在每个 epoch 汇总类平均损失后再更新 EMA；forward 中只读取，不更新。

---

关于未提供对比方法代码与消融实验脚本的说明

本仓库当前仅开放 UMuLT 主干实现，未附带具体对比方法（Baseline）的完整代码与显式的消融实验脚本，原因如下：

版权与依赖限制：多数对比方法来自他人项目，直接合并可能触犯许可证或引入大量依赖，维护成本高。我们建议读者按论文描述自行实现或直接使用原作者的公开代码。

公平性与可复现性：论文中统一给出了数据划分、指标与超参范围。让读者在同一框架下自行复现 Baseline，更能检验方法的泛化与客观性。

项目节奏与代码整理：我们优先开源核心算法实现。对比方法适配代码将在论文录用后再行清理与发布，以保证接口统一、文档完整。

如何进行消融实验？

主代码中已提供模块/模态级的开关，所有消融均可通过修改 CONFIG 实现：

模块级消融：

enable_cross_modal_align：关闭跨模态对齐（Cross-Attn）

enable_uncertainty_gate：关闭不确定性门控

enable_ema_fair：关闭 EMA 公平重加权

use_ldam_cb：关闭 CB-LDAM，退化为普通 CE

模态级消融：

enable_text, enable_audio, enable_video 分别控制文本/音频/视觉分支。

示例：

# 关闭跨模态对齐和不确定性门控
python utl_melt.py --lambda_cons 0.8 --alpha_fair 0.65 --tau_unc 0.08
# 并在脚本中手动设置：
# CONFIG["enable_cross_modal_align"] = False
# CONFIG["enable_uncertainty_gate"] = False

后续计划

对比方法适配代码：视稿件录用情况开放或补充。

批处理消融脚本：计划提供一份自动化脚本，快速跑完常见消融组合。

简言之：核心代码已完整开源；对比方法请自行实现或等待论文录用后的后续开源；消融实验可依赖现有注释与开关轻松完成。

