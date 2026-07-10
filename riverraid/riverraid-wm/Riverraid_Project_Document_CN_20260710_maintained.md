# Riverraid 项目文档（维护版）

更新时间：2026-07-10  
维护依据：`Riverraid_Project_Document_CN_20260709.md`、当前 yingnan 机器实际 GPU 状态、训练日志、输出目录与本次会话中已确认的操作记录。  
维护原则：代码和实际配置优先于旧文档；不确定内容标注“需确认”；只保留对后续开发、运行、部署、排错有价值的信息。

---

## 1. 项目简介

本项目是 `BlendRL` 在 Atari `Riverraid` 环境上的强化学习实验分支，核心目标是验证以下流程：

1. Riverraid baseline policy 训练。
2. Riverraid clean World Model 训练。
3. 使用 frozen clean World Model 作为仲裁器辅助 policy 训练。

当前 Riverraid 主线采用与 L40 机器一致的版本：奖励函数为 `sign(original_reward)`，不是 shaped reward。此前 yingnan 主仓存在 reward 版本混杂问题，已经通过同步 L40 版本修正；受污染的旧实验已移动到 quarantine，不应作为最终结果来源。

当前主线已经从“world-model-assisted policy 训练”切换为“补训练 Riverraid clean World Model seed0 / seed2”。

---

## 2. 当前项目状态

### 2.1 已完成

1. 已停止 GPU0 上原来的 Riverraid 世界模型辅助 baseline 训练。
2. 原辅助训练主进程 `PID 2693553` 已不再出现在 `nvidia-smi` 中。
3. 原辅助训练输出目录保留为历史实验目录：

   ```text
   /home/yingnan/??/czx/blendRL/out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
   ```

4. 该历史 run 曾运行到 5M steps 之后，并生成过 `step_5000000.pth`。
5. 已启动 GPU0 clean WM 守护脚本。
6. 已在 GPU0 上启动 Riverraid clean WM seed0。
7. clean WM seed2 已排队，等待 seed0 完成后串行启动。
8. GPU1 上两个 Seaquest 训练进程未被修改。

### 2.2 当前正在运行

当前正在运行的是：

```text
Riverraid clean World Model seed0
GPU = 0
PID = 2729122
process = @HS_BlendeRL#first_epoch
```

当前输出目录：

```text
/home/yingnan/??/czx/blendRL/out/61_riverraid_clean_wm_seed0_gpu0_20260710
```

训练日志：

```text
/home/yingnan/??/czx/blendRL/out/61_riverraid_clean_wm_seed0_gpu0_20260710/launch_logs/train.log
```

日志已确认出现：

```text
Using riverraid multiprocessing backend with 32 worker processes
Changed reward function to /home/yingnan/??/czx/blendRL/in/envs/riverraid/blenderl_reward.py
```

这说明当前 clean WM seed0 使用的是 Riverraid multiprocessing backend，并且 reward 路径指向当前项目内的 Riverraid reward 文件。

### 2.3 当前排队任务

seed2 已由 GPU0 guard 排队：

```text
/home/yingnan/??/czx/blendRL/out/62_riverraid_clean_wm_seed2_gpu0_20260710
```

当前状态：

```text
seed2 尚未启动，等待 seed0 完成并生成最终 checkpoint 后串行启动。
```

### 2.4 当前 GPU 状态

截至最近一次检查：

```text
GPU0:
  运行 Riverraid clean WM seed0
  PID 2729122
  显存占用约 16.7GB

GPU1:
  仍有两个 Seaquest 训练进程
  PID 2703432
  PID 2713685
  总显存占用约 20GB
```

GPU1 当前不建议再启动 Riverraid clean WM 或其他大显存任务。

---

## 3. 主要功能

项目当前已有的 Riverraid 核心功能如下。

### 3.1 Riverraid 多进程环境后端

当前训练使用 `MultiprocessRiverraidEnv` 支持并行环境采样。当前主线配置为：

```text
--num-envs 128
--num-steps 128
--mp-num-workers 32
```

相关实现：

```text
alt_runtime/riverraid_mp_env.py
```

### 3.2 BlendRL policy 训练

主训练入口：

```text
train_blenderl.py
```

当前常用策略结构：

```text
algorithm = blender
blender_mode = logic
blend_function = softmax
actor_mode = hybrid
reasoner = nsfr
rules = default
```

### 3.3 Clean World Model 训练

clean WM 训练入口：

```text
train_blenderl_clean_isolated.py
```

当前已确认该入口用于 Riverraid clean WM seed0 / seed2 补训练。

clean WM 训练的目标不是让 WM 介入 policy 决策，而是在线训练并保存 clean world model checkpoint。当前配置使用：

```text
--use-world-model-arbiter
--no-world-model-freeze
--world-model-warmup-steps 999999999
--world-model-planning-mode triggered
--world-model-candidate-action-mode argmax_pair
```

由于 `world_model_warmup_steps = 999999999`，训练过程中 policy 基本不受 WM 仲裁影响，主要目的是得到 clean WM checkpoint。

### 3.4 World Model 仲裁器

核心实现：

```text
blendrl/agents/world_model_arbiter.py
blendrl/agents/blender_agent.py
```

历史辅助训练使用 frozen clean WM 作为仲裁器：

```text
--use-world-model-arbiter
--world-model-freeze
--world-model-planning-mode always_plan
--world-model-candidate-action-mode sample_triplet_vs_blend
--world-model-horizon 5
```

该历史辅助训练已停止，目录保留用于结果分析。

---

## 4. 技术栈

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

当前 Python 环境路径：

```text
/home/yingnan/??/czx/blendRL/.venv/bin/python
```

当前 yingnan GPU 环境：

```text
yingnan: 2 × NVIDIA GeForce RTX 4090
```

L40 参考仓库路径：

```text
/data/czx/blendRL
```

---

## 5. 项目结构

主仓路径：

```text
/home/yingnan/??/czx/blendRL
```

Riverraid 相关主要文件：

```text
train_blenderl.py
train_blenderl_clean_isolated.py
alt_runtime/riverraid_mp_env.py
in/envs/riverraid/blenderl_reward.py
in/envs/riverraid/env.py
in/envs/riverraid/env_vectorized.py
in/envs/riverraid/valuation.py
in/envs/riverraid/logic/default/
blendrl/agents/world_model_arbiter.py
blendrl/agents/blender_agent.py
```

当前 clean WM GPU0 守护脚本：

```text
scripts/riverraid_clean_wm_gpu0_guard_20260710.sh
```

当前 clean WM seed0 输出目录：

```text
out/61_riverraid_clean_wm_seed0_gpu0_20260710
```

当前 clean WM seed2 预留输出目录：

```text
out/62_riverraid_clean_wm_seed2_gpu0_20260710
```

历史辅助训练目录：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
```

旧混杂实验留档目录：

```text
/home/yingnan/riverraid_mismatch_quarantine_20260709_1750
```

同步 L40 前的备份目录：

```text
/home/yingnan/??/czx/blendRL/.riverraid_sync_backup_20260709_1750_l40_sync
```

从 L40 拷贝来的 Riverraid isolated 目录：

```text
/home/yingnan/riverraid_wm_from_l40_20260707
```

---

## 6. 运行方式

### 6.1 环境要求

已确认环境：

```text
OS: Linux
GPU: NVIDIA GeForce RTX 4090
Python: 使用项目内 .venv
CUDA: 需可用
```

检查 Python：

```bash
cd /home/yingnan/??/czx/blendRL
./.venv/bin/python -V
```

检查 GPU：

```bash
nvidia-smi
```

### 6.2 当前 clean WM seed0 / seed2 运行方式

当前主线通过 GPU0 guard 串行运行 clean WM seed0 和 seed2。

脚本：

```text
scripts/riverraid_clean_wm_gpu0_guard_20260710.sh
```

查看状态：

```bash
cd /home/yingnan/??/czx/blendRL
scripts/riverraid_clean_wm_gpu0_guard_20260710.sh status
```

查看 seed0 日志：

```bash
cd /home/yingnan/??/czx/blendRL
tail -n 120 out/61_riverraid_clean_wm_seed0_gpu0_20260710/launch_logs/train.log
```

查看 GPU：

```bash
nvidia-smi
```

### 6.3 clean WM guard 运行逻辑

当前 GPU0 guard 的设计逻辑：

```text
1. 检查 GPU0 空闲显存。
2. 当 GPU0 空闲显存达到阈值后启动 seed0。
3. seed0 完成并生成 step_20000000.pth 后，启动 seed2。
4. 不主动停止其他 GPU 进程。
5. 单个 seed 失败后停止 guard，避免反复重启污染目录。
```

当前 guard 进程：

```text
PID 2728425
command: bash scripts/riverraid_clean_wm_gpu0_guard_20260710.sh daemon
```

说明：当前已观察到 `out/riverraid_clean_wm_gpu0_guard_20260710/guard.pid` 存在，但 `guard.log` 是否稳定写入需确认。排错时优先查看 seed0 训练日志和 `nvidia-smi`。

### 6.4 历史辅助训练运行方式

历史辅助训练脚本：

```text
launch_riverraid_purewm_bd0_d05_seed0_l40sync_20260709.sh
```

历史输出目录：

```text
out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
```

该 run 已停止，不再作为当前正在运行任务。

历史 run 的关键配置：

```text
world_model_freeze = true
world_model_planning_mode = always_plan
world_model_candidate_action_mode = sample_triplet_vs_blend
world_model_delta = 0.5
world_model_horizon = 5
world_model_warmup_steps = 0
```

该 run 使用 clean WM seed1 20M checkpoint 作为 frozen world model：

```text
/home/yingnan/riverraid_wm_from_l40_20260707/out/02_world_model_training/riverraid_gpu1_parallel/clean/runs/riverraid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_128_steps_128_backend_mp_workers_32__1_gpu1_parallel_online_wm_clean_seed1/checkpoints/step_20000000.pth
```

---

## 7. 配置说明

### 7.1 当前 clean WM 训练核心参数

```text
env_name = riverraid
seed = 0 / 2
total_timesteps = 20005000
num_envs = 128
num_steps = 128
mp_num_workers = 32
save_steps = 5000000
algorithm = blender
blender_mode = logic
blend_function = softmax
actor_mode = hybrid
rules = default
reasoner = nsfr
track = false
```

World Model 训练参数：

```text
use_world_model_arbiter = true
world_model_freeze = false
world_model_warmup_steps = 999999999
world_model_planning_mode = triggered
world_model_candidate_action_mode = argmax_pair
world_model_delta = 0.0
world_model_branch_delta = 0.0
world_model_horizon = 5
world_model_update_epochs = 4
world_model_learning_rate = 0.0001
world_model_reward_head_type = scalar
world_model_state_loss_coef = 1.0
world_model_neural_state_loss_coef = 1.0
world_model_reward_loss_coef = 1.0
world_model_done_loss_coef = 0.25
```

### 7.2 当前环境变量

推荐保留：

```bash
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=1
```

不同 seed 的输出目录通过 `BLENDRL_OUT_PATH` 指定，例如：

```bash
BLENDRL_OUT_PATH=/home/yingnan/??/czx/blendRL/out/61_riverraid_clean_wm_seed0_gpu0_20260710
```

### 7.3 Reward 配置

当前 Riverraid reward 文件：

```text
in/envs/riverraid/blenderl_reward.py
```

当前 reward 定义：

```python
import numpy as np

def reward_function(self) -> float:
    return float(np.sign(self.org_reward))
```

这是当前版本一致性的关键。不要把 shaped reward 实验和当前 sign reward 实验混用。

### 7.4 数据库、端口和第三方服务

数据库：

```text
未发现数据库配置。
```

端口：

```text
训练本身不占用固定 Web 端口。
TensorBoard 端口需手动指定，常用 6006，但实际访问方式需确认。
```

第三方服务：

```text
当前训练使用 --no-track。
WandB 或其他在线 tracking 是否需要启用：需确认。
```

---

## 8. 核心模块说明

### 8.1 Riverraid 多进程环境

文件：

```text
alt_runtime/riverraid_mp_env.py
```

作用：

1. 封装 OCAtari/ALE Riverraid 环境。
2. 将多个环境分配到多进程 worker 中。
3. 将 `logic_state`、`neural_state`、`reward`、`done` 等返回给训练主进程。
4. 支持 `num_envs=128` 和 `mp_num_workers=32` 的并行采样。

### 8.2 Riverraid 逻辑规则

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

已知性能问题：`close_by_enemy(P,S,H,J)` 是 4 元谓词，会增加 NSFR 推理开销。在 WM rollout 中被反复调用时，会降低 Riverraid+WM SPS。当前决定暂不修改规则。

### 8.3 BlendRL Agent

文件：

```text
blendrl/agents/blender_agent.py
```

关键职责：

1. 计算神经策略动作分布。
2. 计算符号逻辑策略动作分布。
3. 通过 blender weights 合成最终动作分布。
4. 启用 WM 时调用 `_compute_world_model_source` 决定是否用 WM 仲裁。
5. WM rollout 中调用 `_rollout_policy_action` 和 `_rollout_blended_value`。

### 8.4 World Model Arbiter

文件：

```text
blendrl/agents/world_model_arbiter.py
```

核心类：

```text
WorldModelArbiterConfig
VisualSymbolicWorldModelArbiter
```

核心方法：

```text
predict_transition
rollout_return
compute_losses
```

---

## 9. 常见问题与排错

### 9.1 Riverraid 结果和 L40 baseline 对不上

原因：此前 yingnan 主仓曾使用 shaped reward，而 L40 使用 sign reward。两者不能直接对比。

检查：

```bash
cat /home/yingnan/??/czx/blendRL/in/envs/riverraid/blenderl_reward.py
```

应看到：

```python
return float(np.sign(self.org_reward))
```

处理：不要使用 quarantine 中的旧实验作为最终结论。

### 9.2 GPU OOM / 显存不足

当前经验：Riverraid clean WM 训练在 4090 上可能占用约 16GB 以上显存。

启动前必须检查：

```bash
nvidia-smi
```

建议：

```text
1. 不要在 GPU1 当前已有两个 Seaquest 进程时再启动 Riverraid clean WM。
2. 保留 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
3. 保留 OMP_NUM_THREADS=1。
4. 如果目标 GPU 空闲显存不足，不要强行启动训练。
```

### 9.3 Riverraid+WM 速度明显慢于 Seaquest+WM

已观察到的历史现象：

```text
Seaquest + WM ≈ 355 SPS
Riverraid + WM ≈ 118~133 SPS
Riverraid clean / no WM ≈ 565 SPS
```

主要原因：

1. Riverraid+WM 的 `trigger_rate` 接近 1。
2. `always_plan + sample_triplet_vs_blend + horizon=5` 会对多个候选动作反复 rollout。
3. Riverraid 的 4 元谓词 `close_by_enemy(P,S,H,J)` 使 NSFR 推理成本较高。

当前决策：暂不改规则，继续使用原规则。

### 9.4 旧实验误用

不应再直接使用以下目录作为最终结论来源：

```text
/home/yingnan/riverraid_mismatch_quarantine_20260709_1750
```

该目录只能作为历史追溯参考。

### 9.5 Git 工作区不干净

当前仓库存在较多 modified / untracked 文件。Riverraid 关键文件已和 L40 对齐，但整体 git 状态仍需整理。

建议：

```text
1. 为当前 Riverraid 版本建立 commit 或 tag。
2. 将 quarantine、backup、历史脚本和正式实验脚本分开管理。
3. 运行前记录关键文件 hash。
```

### 9.6 B.task_relay 工具使用规范

以后所有 `B.task_relay` 调用必须先预览再执行：

```text
1. 先调用 translate=true，只生成预览命令。
2. 检查预览命令是否符合目的。
3. 再调用 translate=false 执行确认后的命令。
4. 不跳过预览直接执行。
```

适用场景：

```text
nvidia-smi 检查
训练启动
训练停止
日志检查
进程状态检查
```

对于停止进程等敏感操作，直接写 `kill` 命令可能被平台安全检查拦截。更稳的方式是用自然语言描述“安全停止某个已确认训练任务”，让工具端先生成预览命令。

---

## 10. 文档维护结果

### 10.1 本次变更摘要

本次变更的核心是：停止原 GPU0 上的 Riverraid 世界模型辅助 baseline 训练，并将当前主线切换到 GPU0 上补训练 Riverraid clean World Model。

具体变化：

```text
1. out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709 从“当前运行”改为“历史实验”。
2. 新增 out/61_riverraid_clean_wm_seed0_gpu0_20260710 作为当前运行目录。
3. 新增 out/62_riverraid_clean_wm_seed2_gpu0_20260710 作为后续排队目录。
4. 新增 scripts/riverraid_clean_wm_gpu0_guard_20260710.sh 作为当前 clean WM 串行训练守护脚本。
5. 更新 GPU0/GPU1 当前使用情况。
6. 更新 B.task_relay 运维规范。
```

### 10.2 需要更新的文档

建议更新：

```text
Riverraid_Project_Document_CN_20260709.md
riverraid/riverraid-wm/README.md
riverraid/riverraid-wm/maintenance-result-20260710.md
docs-index.md
CHANGELOG.md
```

如果仓库中存在以下文件，也建议同步更新：

```text
docs/troubleshooting.md
docs/experiments.md
docs/gpu-runbook.md
```

### 10.3 过时内容

旧文档中过时内容：

```text
1. out/51 仍是当前正在运行实验 —— 已过时，该 run 已停止。
2. 当前 checkpoint 只有 step_0.pth —— 已过时，out/51 曾生成 step_5000000.pth。
3. train_blenderl_clean_isolated.py 是否继续用于 seed0/seed2 需确认 —— 已确认正在用于 clean WM seed0/seed2。
4. 是否需要补 clean WM seed0/seed2 需确认 —— 已决定补，seed0 已启动，seed2 已排队。
5. GPU1 clean WM guard 作为候选路径 —— 当前不建议，GPU1 被 Seaquest 占用较高。
6. nohup 启动 out/51 的说明 —— 仅保留为历史，不再作为当前主线启动方式。
```

### 10.4 缺失内容

旧文档缺少：

```text
1. GPU0 clean WM guard 脚本路径。
2. 当前 guard 进程 PID 2728425。
3. 当前 clean WM seed0 训练进程 PID 2729122。
4. out/61 seed0 输出目录和日志路径。
5. out/62 seed2 排队目录。
6. 原辅助训练 PID 2693553 已停止的状态。
7. B.task_relay 必须先 translate=true 预览的运维规则。
8. guard.log 当前未确认有效写入的问题。
```

### 10.5 矛盾内容

当前发现的矛盾：

```text
1. 文档说当前主线是 frozen WM 辅助训练，但实际主线已变为 clean WM seed0 训练。
2. 文档说 clean WM seed0/seed2 是否补齐需确认，但实际 seed0 已启动、seed2 已排队。
3. 文档说 out/51 尚未到 5M checkpoint，但实际已生成 step_5000000.pth 后停止。
4. 文档中 GPU1 clean WM OOM 是历史问题，但当前 GPU1 仍高占用，不应再放 Riverraid clean WM。
5. 文档中 guard 日志路径未明确；实际 `guard.log` 未确认有效存在，应优先看 seed0 train.log。
```

### 10.6 建议修改内容

建议将文档中的“当前正在运行”“运行方式”“常见问题”“后续待办”按本维护版对应章节替换。

最小替换重点：

```text
1. 将 out/51 改为历史辅助训练。
2. 将 out/61 改为当前 clean WM seed0 运行目录。
3. 将 out/62 改为后续 clean WM seed2 排队目录。
4. 增加 scripts/riverraid_clean_wm_gpu0_guard_20260710.sh。
5. 增加查看状态、查看日志、查看 GPU 的命令。
6. 增加 B.task_relay 运维规范。
```

### 10.7 当前项目状态

当前已完成：

```text
1. 原 GPU0 Riverraid 世界模型辅助 baseline 训练已停止。
2. GPU0 clean WM guard 已启动。
3. Riverraid clean WM seed0 已启动。
4. seed2 已排队。
5. GPU1 Seaquest 进程未被修改。
```

当前未完成：

```text
1. clean WM seed0 尚未完成。
2. clean WM seed2 尚未启动。
3. seed0 的 step_5000000.pth 尚未确认生成。
4. seed0 的 step_20000000.pth 尚未生成。
5. guard.log 是否稳定写入需确认。
6. GitHub 远端文档是否同步需确认。
```

下一步建议：

```text
1. 持续观察 out/61 是否推进。
2. 到 5M 后检查 step_5000000.pth。
3. seed0 完成后确认 guard 是否自动启动 seed2。
4. 修正或确认 guard.log 输出位置。
5. 将本维护版文档同步到仓库。
6. clean WM seed0/seed2 完成后，再更新 frozen WM 辅助训练使用的 checkpoint 路径。
```

---

## 11. 后续待办

1. 观察当前 clean WM seed0 是否正常推进。
2. 到以下里程碑检查日志、SPS 和 checkpoint：

   ```text
   0.5M
   1M
   2M
   5M
   10M
   20M
   ```

3. 到 5M 后确认是否生成：

   ```text
   checkpoints/step_5000000.pth
   ```

4. 到 20M 后确认是否生成：

   ```text
   checkpoints/step_20000000.pth
   ```

5. seed0 完成后确认 seed2 是否自动启动。
6. clean WM seed0/seed2 完成后，再决定是否重新启动新的 frozen WM 辅助训练。
7. 更新 GitHub 文档并提交 commit：需确认远端权限和同步方式。
8. 整理 guard 脚本日志输出，确保 `guard.log` 可用于排错。

---

## 12. 信息不足 / 需确认

以下内容仍需补充或确认：

```text
1. GPU0 guard 脚本最终完整内容。
2. out/61 的完整 commands 文件。
3. out/61 TensorBoard 当前 latest step / SPS / checkpoint 状态。
4. guard.log 是否应存在，以及为什么当前未确认有效写入。
5. GitHub 远端仓库当前文档是否已同步。
6. clean WM seed0/seed2 完成后是否作为后续 frozen WM 辅助训练的正式 checkpoint。
7. TensorBoard 端口和本地访问方式。
```
