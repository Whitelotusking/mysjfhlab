# 项目文档

## 1. 项目简介

本项目是 `BlendRL` 在 Atari `Riverraid` 环境上的强化学习实验分支，核心目标是验证“符号策略 + 神经策略 + World Model 仲裁器”的训练与评估流程。

项目当前主要解决两个问题：

1. 在 `Riverraid` 中训练 baseline / clean world model / world-model-assisted policy。
2. 当符号策略和神经策略在动作选择上存在冲突时，使用 learned world model 对候选动作做短视 rollout，并选择预测回报更高的动作。

当前 Riverraid 主线采用与 L40 机器一致的版本：奖励函数为 `sign(original_reward)`，不是 shaped reward。此前 yingnan 主仓存在 reward 版本混杂问题，已经通过同步 L40 版本修正；受污染的旧实验已移动到 quarantine。

## 2. 主要功能

项目当前已有的 Riverraid 核心功能如下：

1. `Riverraid` 多进程环境后端  
   使用 `MultiprocessRiverraidEnv` 支持并行环境采样，当前训练配置为：

   ```text
   --num-envs 128
   --num-steps 128
   --mp-num-workers 32
   ```

2. BlendRL policy 训练  
   主训练入口为：

   ```text
   train_blenderl.py
   ```

   当前使用的策略结构包括：

   ```text
   algorithm = blender
   blender_mode = logic
   blend_function = softmax
   actor_mode = hybrid
   reasoner = nsfr
   ```

3. Riverraid 符号规则系统  
   Riverraid 规则文件位于：

   ```text
   in/envs/riverraid/logic/default/
   ```

   当前保留原始 4 元谓词规则，例如：

   ```prolog
   neural_agent(X):-close_by_enemy(P,S,H,J).
   logic_agent(X):-nothing_around(X).
   logic_agent(X):-close_by_fuel(P,F).
   logic_agent(X):-visible_bridge(B).
   ```

4. World Model 仲裁器  
   核心实现位于：

   ```text
   blendrl/agents/world_model_arbiter.py
   blendrl/agents/blender_agent.py
   ```

   当前接入方式为 frozen clean world model：

   ```text
   --use-world-model-arbiter
   --world-model-freeze
   --world-model-planning-mode always_plan
   --world-model-candidate-action-mode sample_triplet_vs_blend
   --world-model-horizon 5
   ```

5. Clean World Model checkpoint 复用  
   当前使用的 clean WM checkpoint 为：

   ```text
   /home/yingnan/riverraid_wm_from_l40_20260707/out/02_world_model_training/riverraid_gpu1_parallel/clean/runs/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__1_gpu1_parallel_online_wm_clean_seed1/checkpoints/step_20000000.pth
   ```

6. 训练输出管理  
   当前正在运行的 seed0 + frozen WM 实验输出到：

   ```text
   /home/yingnan/??/czx/blendRL/out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
   ```

## 3. 技术栈

本项目不是传统 Web 项目，没有前端、后端服务和数据库。

| 类型 | 当前情况 |
|---|---|
| 前端 | 无传统前端；训练监控主要依赖日志和 TensorBoard |
| 后端 | 无 Web 后端；核心为 Python 训练脚本 |
| 数据库 | 未发现数据库依赖 |
| 主要语言 | Python |
| 深度学习框架 | PyTorch |
| 强化学习环境 | Atari / ALE / OCAtari，具体版本需确认 |
| 并行环境 | Python multiprocessing / forkserver，多进程 Atari 环境 |
| 符号推理 | NSFR / Nudge 相关模块 |
| 可视化与监控 | TensorBoard event logs，命令行日志 |
| 部署方式 | Linux 服务器上通过 `.venv/bin/python`、bash 脚本、GPU CUDA 运行 |
| 第三方在线服务 | 当前训练使用 `--no-track`，未启用 WandB；是否还有其他服务需确认 |

当前使用的 Python 环境路径：

```text
/home/yingnan/??/czx/blendRL/.venv/bin/python
```

当前 GPU 运行环境：

```text
yingnan: 2 × NVIDIA GeForce RTX 4090
```

L40 机器上的参考仓库路径：

```text
/data/czx/blendRL
```

## 4. 项目结构

Riverraid 相关的主要目录和文件如下。

```text
/home/yingnan/??/czx/blendRL
```

当前 yingnan 主仓。已经同步为 L40 Riverraid 版本。主要用于后续 Riverraid 训练与评估。

```text
train_blenderl.py
```

主训练入口。当前包含 Riverraid fast multiprocessing backend 接入逻辑：

```text
MultiprocessRiverraidEnv
```

```text
train_blenderl_clean_isolated.py
```

clean world model 训练入口。用于训练 policy 不受 WM 干预、但在线训练并保存 clean WM 的实验。当前是否继续用于新 clean WM seed0/seed2 需确认。

```text
alt_runtime/riverraid_mp_env.py
```

Riverraid 多进程环境实现。用于加速 128 环境并行采样。

```text
in/envs/riverraid/blenderl_reward.py
```

Riverraid reward 函数。当前已同步为 L40 版本：

```python
import numpy as np

def reward_function(self) -> float:
    return float(np.sign(self.org_reward))
```

```text
in/envs/riverraid/env.py
in/envs/riverraid/env_vectorized.py
```

Riverraid 环境封装和向量化环境逻辑。

```text
in/envs/riverraid/valuation.py
```

Riverraid 符号谓词的 Python valuation 函数，例如 `close_by_fuel`、`close_by_enemy`、`visible_bridge` 等。

```text
in/envs/riverraid/logic/default/
```

Riverraid 规则目录，核心文件包括：

```text
clauses.txt
blender_clauses.txt
neural_preds.txt
preds.txt
consts.txt
bk.txt
```

```text
blendrl/agents/world_model_arbiter.py
```

World Model 仲裁器定义。核心类：

```text
WorldModelArbiterConfig
VisualSymbolicWorldModelArbiter
```

主要功能包括：

```text
predict_transition
rollout_return
compute_losses
```

```text
blendrl/agents/blender_agent.py
```

BlendRL agent 主逻辑。与 Riverraid+WM 速度最相关的逻辑包括：

```text
_compute_world_model_source
_rollout_policy_action
_rollout_blended_value
```

```text
/home/yingnan/riverraid_wm_from_l40_20260707
```

从 L40 拷贝来的 Riverraid isolated 目录。当前主要保留 clean WM seed1 checkpoint 和历史参考脚本。

```text
/home/yingnan/riverraid_mismatch_quarantine_20260709_1750
```

旧的混杂版本实验留档目录。里面保存了此前 reward 版本不一致导致不适合作为最终结论的实验记录。

```text
/home/yingnan/??/czx/blendRL/.riverraid_sync_backup_20260709_1750_l40_sync
```

同步 L40 Riverraid 代码前，对 yingnan 主仓旧 Riverraid 相关文件做的备份。

## 5. 运行方式

### 5.1 环境要求

已确认的运行环境：

```text
OS: Linux
GPU: NVIDIA GeForce RTX 4090
Python: 使用项目内 .venv
CUDA: 需可用
```

依赖安装方式需确认。当前可用环境为：

```bash
cd /home/yingnan/??/czx/blendRL
./.venv/bin/python -V
```

### 5.2 当前 seed0 + WM 训练启动方式

当前训练脚本：

```text
/home/yingnan/??/czx/blendRL/launch_riverraid_purewm_bd0_d05_seed0_l40sync_20260709.sh
```

已使用以下方式在 GPU0 上启动：

```bash
cd /home/yingnan/??/czx/blendRL
nohup bash launch_riverraid_purewm_bd0_d05_seed0_l40sync_20260709.sh > /tmp/riverraid_purewm_seed0_l40sync_gpu0.out 2>&1 &
```

当前训练的核心命令由脚本生成并保存到：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/commands/seed0_gpu0_l40sync_command.txt
```

### 5.3 当前训练日志

当前日志路径：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/launch_logs/seed0_gpu0_l40sync_20260709_180946.log
```

查看日志：

```bash
tail -n 120 out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/launch_logs/seed0_gpu0_l40sync_20260709_180946.log
```

### 5.4 当前训练输出

输出目录：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
```

checkpoint 目录：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/runs/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__0_purewm_bd0_d05_frozen_cleanwm_seed0_20m_l40sync/checkpoints
```

TensorBoard 目录：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/tensorboard/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__0_purewm_bd0_d05_frozen_cleanwm_seed0_20m_l40sync
```

启动 TensorBoard 的端口和访问方式需确认。可参考命令：

```bash
tensorboard --logdir out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709/tensorboard --port 6006
```

是否能从本地浏览器访问该端口需确认。

## 6. 配置说明

### 6.1 当前训练环境变量

当前 seed0 + WM 训练脚本中使用：

```bash
CUDA_VISIBLE_DEVICES=0
BLENDRL_OUT_PATH=/home/yingnan/??/czx/blendRL/out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=1
```

含义：

```text
CUDA_VISIBLE_DEVICES=0
```

指定使用 yingnan 的物理 GPU0。

```text
BLENDRL_OUT_PATH
```

指定本次训练输出根目录，避免写入默认 `out/01_blendrl_training`。

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

缓解 PyTorch CUDA 显存碎片问题。此前 GPU1 强行运行 clean WM 时出现过 OOM，此项可降低碎片风险，但不能解决显存不足。

```text
OMP_NUM_THREADS=1
```

限制 OpenMP 线程数，避免多进程环境中 CPU 线程过度竞争。

### 6.2 当前训练核心参数

```text
env_name = riverraid
seed = 0
total_timesteps = 20005000
num_envs = 128
num_steps = 128
mp_num_workers = 32
save_steps = 5000000
```

优化器和 PPO/BlendRL 相关参数：

```text
learning_rate = 0.00025
logic_learning_rate = 0.00025
blender_learning_rate = 0.00025
gamma = 0.99
gae_lambda = 0.95
num_minibatches = 4
update_epochs = 10
clip_coef = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
```

World Model 接入参数：

```text
use_world_model_arbiter = true
world_model_checkpoint = /home/yingnan/riverraid_wm_from_l40_20260707/out/02_world_model_training/riverraid_gpu1_parallel/clean/runs/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__1_gpu1_parallel_online_wm_clean_seed1/checkpoints/step_20000000.pth
world_model_freeze = true
world_model_planning_mode = always_plan
world_model_candidate_action_mode = sample_triplet_vs_blend
world_model_delta = 0.5
world_model_branch_delta = 0.0
world_model_horizon = 5
world_model_warmup_steps = 0
world_model_reward_head_type = scalar
world_model_update_epochs = 2
world_model_learning_rate = 0.0001
world_model_state_loss_coef = 1.0
world_model_neural_state_loss_coef = 1.0
world_model_reward_loss_coef = 1.0
world_model_done_loss_coef = 0.25
```

### 6.3 数据库、端口和第三方服务

数据库：

```text
未发现数据库配置。
```

端口：

```text
训练本身不占用固定 Web 端口。
TensorBoard 端口需手动指定，常用 6006，但实际端口需确认。
```

第三方服务：

```text
--no-track
```

当前训练关闭在线 tracking。是否需要 WandB 或其他服务需确认。

## 7. 核心模块说明

### 7.1 Riverraid 多进程环境

文件：

```text
alt_runtime/riverraid_mp_env.py
```

作用：

1. 封装 OCAtari/ALE Riverraid 环境。
2. 将多个环境分配到多进程 worker 中。
3. 将 `logic_state`、`neural_state`、`reward`、`done` 等返回给训练主进程。
4. 支持 `num_envs=128` 和 `mp_num_workers=32` 的并行采样。

当前 `train_blenderl.py` 已确认包含：

```text
MultiprocessRiverraidEnv
```

并且 Riverraid 会走 fast multiprocessing backend。

### 7.2 Riverraid reward

文件：

```text
in/envs/riverraid/blenderl_reward.py
```

当前 reward：

```python
return float(np.sign(self.org_reward))
```

这是当前版本一致性的关键。此前 yingnan 主仓曾使用 shaped reward，导致与 L40 baseline / clean WM 不一致；现在已修正为 L40 sign reward。

### 7.3 Riverraid 逻辑规则

目录：

```text
in/envs/riverraid/logic/default/
```

主要规则：

```prolog
noop(X):-.
right_to_fuel(X):-close_by_fuel(P,F),left_of_fuel(P,F).
left_to_fuel(X):-close_by_fuel(P,F),right_of_fuel(P,F).
fire_at_bridge(X):-visible_bridge(B),same_level_bridge(P,B).
right_to_bridge(X):-visible_bridge(B),left_of_bridge(P,B).
left_to_bridge(X):-visible_bridge(B),right_of_bridge(P,B).
```

Blender 选择神经/逻辑分支的规则：

```prolog
neural_agent(X):-close_by_enemy(P,S,H,J).
logic_agent(X):-nothing_around(X).
logic_agent(X):-close_by_fuel(P,F).
logic_agent(X):-visible_bridge(B).
```

已知性能问题：

```prolog
close_by_enemy(P,S,H,J)
```

这是 4 元谓词，会增加 NSFR 推理开销。在 WM rollout 中被反复调用，是 Riverraid+WM SPS 明显低于 Seaquest+WM 的主要原因之一。当前决定暂不修改，继续使用原规则。

### 7.4 BlendRL Agent

文件：

```text
blendrl/agents/blender_agent.py
```

关键职责：

1. 计算神经策略动作分布。
2. 计算符号逻辑策略动作分布。
3. 通过 blender weights 合成最终动作分布。
4. 在启用 WM 时调用 `_compute_world_model_source` 决定是否用 WM 仲裁。
5. 在 WM rollout 中调用 `_rollout_policy_action` 和 `_rollout_blended_value`。

速度瓶颈主要在 `_rollout_policy_action` 内部反复调用 Riverraid 的逻辑推理和 blender weights。

### 7.5 World Model Arbiter

文件：

```text
blendrl/agents/world_model_arbiter.py
```

核心类：

```text
VisualSymbolicWorldModelArbiter
```

核心方法：

```text
predict_transition
rollout_return
compute_losses
```

当前运行中，WM 是 frozen 的：

```text
world_model_freeze = true
```

因此当前训练不会更新 WM 参数，只用 clean WM 对动作候选做 rollout 评分。

### 7.6 Clean WM checkpoint

当前接入的是 clean seed1 20M checkpoint：

```text
/home/yingnan/riverraid_wm_from_l40_20260707/out/02_world_model_training/riverraid_gpu1_parallel/clean/runs/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__1_gpu1_parallel_online_wm_clean_seed1/checkpoints/step_20000000.pth
```

此 checkpoint 文件已确认存在，大小约 `30M`。

是否需要补 clean WM seed0 / seed2：需确认。此前尝试补 seed0/seed2 时曾因版本混杂和 GPU1 OOM 中断，旧记录已 quarantine。

## 8. 常见问题

### 8.1 Riverraid 结果和 L40 baseline 对不上

原因：

此前 yingnan 主仓使用 shaped reward，而 L40 使用：

```python
return float(np.sign(self.org_reward))
```

导致同名 Riverraid 实验实际不在同一奖励定义下运行。

解决：

1. 确认当前 reward 文件：

   ```bash
   cat /home/yingnan/??/czx/blendRL/in/envs/riverraid/blenderl_reward.py
   ```

2. 应看到：

   ```python
   return float(np.sign(self.org_reward))
   ```

3. 不要使用 quarantine 中的旧评测结果作为最终结论。

### 8.2 GPU OOM

现象：

此前强行在 GPU1 上补 clean WM seed0/seed2 时出现：

```text
torch.OutOfMemoryError: CUDA out of memory
```

原因：

GPU1 已有其他任务占用显存，再启动 clean WM 后显存不足。

解决：

1. 启动前检查 GPU：

   ```bash
   nvidia-smi
   ```

2. 确保目标 GPU 有足够空闲显存。
3. 保留：

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

4. 不要强行和已有大显存任务共用同一张 4090。

### 8.3 Riverraid+WM 速度明显慢于 Seaquest+WM

现象：

历史检查中观察到：

```text
Seaquest + WM ≈ 355 SPS
Riverraid + WM ≈ 118~133 SPS
Riverraid clean / no WM ≈ 565 SPS
```

当前新 run 约：

```text
SPS ≈ 133
```

主要原因：

1. Riverraid+WM 的 `trigger_rate` 接近 1。
2. `always_plan + sample_triplet_vs_blend + horizon=5` 会对多个候选动作反复 rollout。
3. Riverraid 的 4 元谓词：

   ```prolog
   close_by_enemy(P,S,H,J)
   ```

   使 NSFR 推理成本较高。

当前决策：

```text
暂不改规则，继续使用原规则。
```

### 8.4 旧实验误用

不应再直接使用以下目录作为最终结论来源：

```text
/home/yingnan/riverraid_mismatch_quarantine_20260709_1750
```

其中包含旧的混杂 reward 实验、失败的 seed0/seed2 clean WM 尝试和旧评测脚本。

如需追溯，可只作历史参考，不能直接作为最终对比结论。

### 8.5 Git 工作区不干净

当前仓库存在较多 modified / untracked 文件。Riverraid 关键文件已和 L40 对齐，但整体 git 状态不干净。

解决建议：

1. 单独为 Riverraid 当前版本建立 commit 或 tag。
2. 将 quarantine、backup、历史脚本和正式实验脚本分开管理。
3. 运行前记录关键文件 hash。

## 9. 当前状态

### 9.1 已完成

1. 已确认 L40 和 yingnan 曾存在 Riverraid 版本不一致问题。
2. 已将 yingnan 主仓 Riverraid 关键代码同步为 L40 版本。
3. 已确认当前 reward 为 `sign(original_reward)`。
4. 已确认 `MultiprocessRiverraidEnv` fast backend 在当前 `train_blenderl.py` 中可用。
5. 已将旧的混杂实验移动到：

   ```text
   /home/yingnan/riverraid_mismatch_quarantine_20260709_1750
   ```

6. 已备份同步前的旧代码到：

   ```text
   /home/yingnan/??/czx/blendRL/.riverraid_sync_backup_20260709_1750_l40_sync
   ```

7. 已在 GPU0 启动新的 seed0 + frozen clean WM 训练：

   ```text
   out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
   ```

### 9.2 当前正在运行

当前运行：

```text
Riverraid seed0 + frozen clean WM seed1 20M
GPU = 0
```

启动脚本：

```text
launch_riverraid_purewm_bd0_d05_seed0_l40sync_20260709.sh
```

截至最近一次核对，状态为：

```text
GPU0 memory ≈ 14526 MiB
GPU0 utilization ≈ 11%~18%
SPS ≈ 133
latest TensorBoard episodic_return step ≈ 140544
episodic_return last = 1220
episodic_return last10 mean ≈ 1558
episodic_return last50 mean ≈ 1492.6
max episodic_return ≈ 2660
world_model_arbiter_trigger_rate ≈ 0.9998
world_model_arbiter_active = 1.0
```

当前 checkpoint：

```text
step_0.pth
```

尚未到第一个 `5M` checkpoint。

### 9.3 已知问题

1. Riverraid+WM SPS 明显低，当前约 `133 SPS`。
2. `world_model_arbiter_trigger_rate` 接近 `1.0`，说明几乎每步都在触发 WM 仲裁。
3. Riverraid 原规则中的 4 元谓词导致 NSFR 推理成本高。
4. 当前只重新启动了 seed0；clean WM seed0/seed2 尚未补齐，是否需要补需确认。
5. 当前使用的 clean WM 是 seed1，接到 policy seed0 上；这是否作为最终主线配置需确认。
6. Git 工作区不干净，需要后续整理。
7. TensorBoard 访问端口和远程访问方式需确认。

## 10. 后续待办

1. 持续观察当前 run 到关键里程碑：

   ```text
   0.5M
   1M
   2M
   5M
   10M
   20M
   ```

2. 到 `5M` 后检查是否生成：

   ```text
   checkpoints/step_5000000.pth
   ```

3. 与 L40 baseline seed0 在同 step 下做对比，不能再使用 quarantine 中的旧评测结果。

4. 重新决定是否补 clean WM seed0 / seed2：

   ```text
   需确认
   ```

5. 若要补 clean WM seed0 / seed2，必须使用当前已同步的 L40 sign reward 版本，并确认日志中 reward 路径指向：

   ```text
   /home/yingnan/??/czx/blendRL/in/envs/riverraid/blenderl_reward.py
   ```

6. 若当前 seed0+WM 速度过慢影响实验周期，可考虑以下方向，但是否执行需确认：

   ```text
   world_model_planning_mode: always_plan -> triggered
   提高 world_model_delta / branch_delta
   降低 horizon
   优化 Riverraid 逻辑谓词
   将训练迁移到 L40 空闲 GPU
   ```

7. 整理正式实验目录，避免将 quarantine 结果与正式结果混用。

8. 为当前 Riverraid 同步版本建立固定记录：

   ```text
   git commit / tag / hash manifest
   ```

9. 补充一份最小复现实验脚本：

   ```text
   smoke test: 10k 或 100k steps
   verify reward path
   verify WM checkpoint load
   verify SPS
   ```

10. 明确最终论文/报告中采用哪一组 Riverraid 结果：

   ```text
   需确认
   ```
