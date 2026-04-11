非常抱歉之前的格式或排版依然给你带来了困扰。

为了确保你能**一次性、零误差地提取所有内容**，以下是完全纯净的整份 README 源码。请直接点击下方代码块右上角的\*\*“复制”\*\*按钮，粘贴到你的 `.md` 文件中即可：

````markdown
# diffuser-acceleration

<p align="center">
  <strong>Kinematics-aware diffusion acceleration for mobile manipulation</strong>
</p>

<p align="center">
  <a href="https://github.com/Haotian020527/diffuser-acceleration">
    <img src="https://img.shields.io/github/stars/Haotian020527/diffuser-acceleration?style=social" alt="GitHub Stars">
  </a>
  <a href="https://m2diffuser.github.io/assets/paper/M2Diffuser.pdf">
    <img src="https://img.shields.io/badge/Paper-T--PAMI%202025-red?style=flat-square" alt="Paper">
  </a>
  <a href="https://arxiv.org/pdf/2410.11402">
    <img src="https://img.shields.io/badge/arXiv-2410.11402-b31b1b?style=flat-square" alt="arXiv">
  </a>
  <a href="https://m2diffuser.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-0a66c2?style=flat-square" alt="Project Page">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.8-3776AB?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.6-76B900?style=flat-square" alt="CUDA">
  <img src="https://img.shields.io/badge/License-Pending-lightgrey?style=flat-square" alt="License">
</p>

![teaser](./assests/teaser.jpg)

## 📖 项目简介 (Introduction)

`diffuser-acceleration` 致力于解决一个现实痛点：扩散模型（Diffusion Models）在轨迹生成任务上表现优异，但 Dense Denoiser 的推理与训练计算成本过高。

本仓库以核心模块 `moe-cokin` 为基础，将 CoKin 的双空间扩散模型重构为**带有运动学约束的稀疏专家系统（Sparse MoE）**。在最大程度保留轨迹质量与物理一致性的前提下，显著削减无效计算开销。

> **Note:** 当前公开分支主要聚焦于 MecKinova 的 `goal-reach` 任务。`pick` 与 `place` 任务的预处理及历史资源保留在仓库中，但核心脚本和配置均以 `goal-reach` 任务为入口。

---

## ✨ 核心特性 (Features)

* 🚀 **`moe-cokin` 旗舰加速模块：** 在 Joint 和 Pose 双分支上执行 Paired Sparse Routing，将稠密 FFN 替换为按层激活的 MoE Expert Bank。
* 🤖 **CoKin 双空间扩散：** 同步建模 10-DoF 关节轨迹与 7D 末端位姿轨迹，并通过可微正向运动学（Differentiable FK）实现一致性耦合。
* 🛠️ **研究友好的解耦配置：** 基于 Hydra 构建，将 `diffuser`、`pose_model`、`joint_model`、`task`、`optimizer` 与 `planner` 彻底解耦，支持极速实验切换。
* 🌍 **3D 场景条件建模：** 集成 PointTransformer / PointNet 场景编码器，支持点云场景 Token 注入去噪器。
* 📊 **完备的基线对比：** 包含 M2Diffuser、MPiNets、MPiFormer 等主流模型实现，便于进行精度、速度与复杂度的全面 Benchmark 对比。
* 🏗️ **清晰的工程架构：** 提供 PyTorch Lightning 训练框架、脚本化启动、数据模块封装，以及评估与后处理模块的清晰分离。

---

## 🔥 深度解析：moe-cokin (Deep Dive)

### 什么是 `moe-cokin`？

`moe-cokin` 是构建在 CoKin (`ConsistencyCoupledKinematicsDiffuser`) 架构之上的稀疏化增强版。有别于传统的单分支 Diffusion Policy，CoKin 是一个**双分支系统**：
1.  **Joint 分支**：预测机器人关节轨迹。
2.  **Pose 分支**：预测末端执行器位姿轨迹。
*(二者通过正向运动学 Forward Kinematics 保持严格的几何一致性。)*

`moe-cokin` 的核心思想并非盲目增加模型参数，而是**将最昂贵、最可替换的 FFN 层转化为混合专家系统（MoE）**。每一层仅激活最适配的 Expert 组合，而非让所有 Token 均经过全量 Dense 通路。

**当前主线链路：**
```text
scripts/model-m2diffuser/goal-reach/train.sh (mode=moe_cokin)
  -> configs/diffuser/cokin_moe.yaml
  -> models/m2diffuser/cokin_moe.py :: CoKinMoEDiffuser
  -> models/model/moe_unet.py :: MoEUNetModel + PairedRouter
````

### 运作机制 (How it works)

`moe-cokin` 并非简单地套用 MoE，而是设计了创新的 **Joint-Pose Paired Routing** 机制。模型不再为两个分支独立挑选 Expert，而是为每一层联合选择一个 `(joint_expert, pose_expert)` 对，确保稀疏路径严格尊重双空间的物理耦合关系。

**1. 路由上下文 (Routing Context):**
路由决策融合了四维度信息：

$$r^l = \text{MLP}([\phi(t), z_{\text{scene}}, \text{Pool}(h_{\text{joint}}^l), \text{Pool}(h_{\text{pose}}^l)])$$

其中 $\phi(t)$ 为时间步嵌入，$z_{\text{scene}}$ 为场景潜在特征，$h_{\text{joint}}^l$ 与 $h_{\text{pose}}^l$ 为当前层的隐藏特征。

**2. 专家对评分 (Expert-Pair Scoring):**
PairedRouter 为每一层构建得分矩阵：

$$S^l_{ij} = u_i(r^l) + v_j(r^l) + B_{ij}$$

训练阶段采用 Gumbel-Softmax (Straight-through) 进行近似离散采样，推理阶段采用确定性的 Top-1 / Argmax 路由。

**3. 损失函数约束 (Loss formulation):**
不仅优化去噪误差，更强调路由选择与运动学结构对齐，并引入负载均衡防止专家坍塌（Expert Collapse）：

$$\mathcal{L} = \lambda_{\text{pose}} \mathcal{L}_{\text{diff\_pose}} + \lambda_{\text{joint}} \mathcal{L}_{\text{diff\_joint}} + \lambda_{\text{fk}} \mathcal{L}_{\text{fk\_route}} + \lambda_{\text{lb}} \mathcal{L}_{\text{lb}}$$

### 性能对比与优势 (Advantages)

`moe-cokin` 的核心收益来源于 **Active Path Sparsification（激活路径稀疏化）**。每次前向传播实际参与计算的子网络显著减小。

| 维度 | Dense CoKin | `moe-cokin` |
| --- | --- | --- |
| **网络结构** | 双分支 Dense UNet | 双分支 MoE UNet |
| **FFN 计算** | 全量稠密计算 | 每层仅激活选定的 Expert Pair |
| **路由决策** | 静态无路由 | 基于 Timestep + Scene Latent + Hidden States 动态路由 |
| **空间协同** | 依赖 FK 一致性 | FK 一致性 + **Paired Routing** |
| **负载均衡** | N/A | $\mathcal{L}_{\text{lb}}$ 正则项防止坍塌 |

*(注：真实计算加速收益取决于 Batch Size、Kernel 实现效率及专家调度开销，具体 Benchmark 数据见后续发布。)*

-----

## 🚀 快速开始 (Quick Start)

### 1\. 环境配置 (Installation)

推荐环境：`Linux` + `NVIDIA GPU` | `Python 3.8` | `PyTorch 1.13.1` | `CUDA 11.6`

```bash
git clone [https://github.com/Haotian020527/diffuser-acceleration.git](https://github.com/Haotian020527/diffuser-acceleration.git)
cd diffuser-acceleration

# 一键安装依赖 (One-shot setup)
bash setup_env.sh
```

> `setup_env.sh` 将自动处理 `PointNet2 ops`, `PyTorch3D`, `Kaolin` 等复杂的 3D 依赖。

### 2\. 数据与资源获取 (Data & Assets)

如需复现实验，请下载以下 M2Diffuser 公开资源，并修改配置中的对应路径：

  * [Dataset (Zip)](https://huggingface.co/datasets/M2Diffuser/mec_kinova_mobile_manipulation/blob/main/dataset.zip)
  * [Checkpoints (Zip)](https://huggingface.co/datasets/M2Diffuser/mec_kinova_mobile_manipulation/blob/main/checkpoints.zip)
  * [URDF & USD Assets](https://www.google.com/search?q=https://huggingface.co/datasets/M2Diffuser/mec_kinova_mobile_manipulation/tree/main) (含 Robot / Scene 的物理与渲染资产)

**需修改路径的地方：**

1.  `configs/task/mk_m2diffuser_goal_reach.yaml` 中的 `data_dir`。
2.  `utils/path.py` 中的机器人及场景资源路径。

### 3\. 模型训练 (Training)

```bash
# Baseline: DDPM
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1 ddpm

# Baseline: Dense CoKin
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1 cokin

# 推荐运行: moe-cokin
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1 moe_cokin

# 从最新 Checkpoint 恢复训练 (Resume training)
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1 moe_cokin latest
```

> **注意：** 当前提供的脚本默认采用**单卡训练**。由于稀疏路由在 DDP 下易触发 `find_unused_parameters` 相关问题，单卡方案是当前最稳定的公开入口。

### 4\. 轻量自检与推理 (Smoke Test & Inference)

**环境连通性测试：**

```bash
python scripts/test_cokin_smoke.py
```

该脚本用于快速验证 CoKin 双分支架构与 FK 一致性计算链路是否跑通。

**推理评估：**
若需对 `moe-cokin` 执行推理，可通过 Hydra 覆写默认配置：

```bash
python inference_m2diffuser.py \
    diffuser=cokin_moe \
    model@pose_model=cokin_moe_pose_mk \
    model@joint_model=cokin_moe_joint_mk
```

-----

## 📂 目录结构 (Repository Structure)

```text
diffuser-acceleration/
├── configs/              # Hydra 配置目录 (diffuser, model, task)
├── datamodule/           # 数据集封装与加载器
├── models/
│   ├── m2diffuser/       # 扩散模型核心 (ddpm, cokin, cokin_moe)
│   ├── model/            # UNet 与 MoE 网络组件 (moe_unet)
│   ├── mpiformer/        # MPiFormer 基线实现
│   └── mpinets/          # MPiNets 基线实现
├── preprocessing/        # 数据预处理脚本
├── postprocessing/       # 结果聚合与可视化
├── eval/                 # 评价指标与轨迹平滑性工具
├── env/                  # 机器人与场景交互环境代码
├── scripts/              # 训练、推理与测试的 Shell 脚本
└── third_party/          # 第三方依赖封装
```

-----

## 📚 引用与协议 (Citation & License)

### Citation

如果您在研究中使用了本仓库的代码，请引用 **M2Diffuser**：

```bibtex
@article{yan2025m2diffuser,
  title={M2Diffuser: Diffusion-based Trajectory Optimization for Mobile Manipulation in 3D Scenes},
  author={Yan, Sixu and Zhang, Zeyu and Han, Muzhi and Wang, Zaijin and Xie, Qi and Li, Zhitian and Li, Zhehan and Liu, Hangxin and Wang, Xinggang and Zhu, Song-Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

### License

*Pending License Clarification.* (项目代码基于开源协议，但在正式 Release 前具体许可正在确认中。)

```
```