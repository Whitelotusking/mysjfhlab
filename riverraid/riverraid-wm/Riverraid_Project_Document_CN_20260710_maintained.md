# Riverraid 世界模型项目文档（维护版）

更新时间：2026-07-12  
维护依据：`Riverraid_Project_Document_CN_20260709.md`、`blendrl_document_maintenance.md`、当前仓库既有维护版文档，以及已确认的实验记录。  
维护原则：代码、配置、checkpoint、运行日志和实际评测脚本优先于旧文档；不确定内容标注“需确认”；删除已经过时的临时运行状态。

---

## 1. 项目定位

本项目是 `BlendRL` 在 Atari `Riverraid` 环境上的世界模型实验分支，核心目标是验证：

1. Riverraid baseline / BlendRL policy 的训练与评测。
2. Riverraid clean World Model checkpoint 的复用。
3. 三个冻结世界模型对候选动作进行 rollout 评分。
4. 在 sampled Blend 与 sampled Neural 候选动作冲突时，用三世界模型的预测优势决定是否允许 Neural 覆盖 Blend。

当前 Riverraid 主线采用与 L40 机器一致的奖励版本：

```python
return float(np.sign(self.org_reward))
```

因此当前文档不再沿用旧的 shaped reward 结果，也不再把旧的 reward 混杂实验作为正式结论来源。

当前主线已经从“补训练 Riverraid clean World Model seed0 / seed2”更新为：

```text
Riverraid 三世界模型评测与 tau 扫描
```

旧文档中关于 “clean WM seed0 正在运行、seed2 排队、out/51 保留为历史辅助训练目录” 的状态已经过时，本版已删除。

---

## 2. 当前状态

### 2.1 已完成

1. 已从原项目创建独立开发副本：

   ```text
   原项目真实目录：/home/yingnan/??/czx/blendRL
   原项目快捷链接：/home/yingnan/blendRL_project_k4 -> /home/yingnan/??/czx/blendRL
   当前开发目录：/home/yingnan/blendRL_k4_dev
   ```

2. 当前开发目录拥有独立源代码和 Git 仓库，但共享原项目的环境与输出目录：

   ```text
   /home/yingnan/blendRL_k4_dev/.venv -> 原项目/.venv
   /home/yingnan/blendRL_k4_dev/out   -> 原项目/out
   ```

3. 三世界模型 rollout 已实现并测试：

   ```text
   rollout_action_mode = argmax | dist
   mc_rollouts = K
   common_random_numbers = CRN
   planning_seed_offset = 独立规划随机种子
   ```

4. 已实现并测试：

   ```text
   dist + K4 + CRN
   dist + K1 + CRN
   独立规划 RNG 保存和恢复
   K 轨迹批量展开
   模型间标准差与轨迹内标准差分离
   训练和评测共用相同 rollout 逻辑
   ```

5. Riverraid 三世界模型已确认使用三份独立 clean checkpoint：

   ```text
   seed0：yingnan 完整训练结果
   seed1：已从 L40 同步到 yingnan
   seed2：已导入 yingnan 的 WM-only checkpoint
   ```

   后续 Riverraid 运行不需要再访问 L40。

6. Riverraid 评测已新增：

   ```text
   --wm-checkpoint-paths
   ```

   支持直接加载完整 checkpoint 或 WM-only checkpoint。

7. Riverraid 评测已新增：

   ```text
   --full-game-episodes
   ```

   用于修复 `EpisodicLifeEnv` 将“掉一条命”误算为完整游戏的问题。

8. 已删除容易造成误解的不完整实验目录：

   ```text
   out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709
   ```

9. 已审计纯 PPO 入口：

   ```text
   train_blenderl.py --algorithm ppo
   ```

   该入口实际仍会创建 BlendRL，不是真正的纯 PPO。旧 `train_neuralppo.py` 也不能直接作为正式公平基线。

### 2.2 当前运行中

截至本次维护依据中的检查，当前 Riverraid 主要运行任务为：

```text
Riverraid tau = 0.0 / 0.1 / 0.2 / 0.3 四组 100 局完整游戏评测
GPU = 0
每组 32 个环境
每组 8 个 worker
总 worker 数控制为 32
```

注意：这类运行状态会继续变化，文档中不再记录临时 PID、瞬时显存和未完成任务的“当前步数”。需要实时确认时使用：

```bash
nvidia-smi
ps -ef | grep python
tail -f out/<实验>/launch_logs/train.log
```

### 2.3 阶段性 Riverraid tau 结果

以下结果来自未完成的动态评测，只能作为阶段观察，不能写入最终论文表格：

| tau | 已完成局数 | 平均原始游戏分 | 中位数 | Neural 动作占比 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 36 | 2644.7 | 2675 | 34.7% |
| 0.1 | 45 | 2499.8 | 2540 | 19.1% |
| 0.2 | 58 | 2424.5 | 2360 | 11.3% |
| 0.3 | 69 | 2332.3 | 2340 | 7.5% |
| baseline | 100 | 2225.9 | 2165 | 0% |

阶段性观察：

```text
低 tau 暂时表现更好，tau=0 当前阶段性均值最高。
最终结论必须等待每组 100 局完整游戏完成。
```

---

## 3. 项目目录与环境风险

### 3.1 服务器目录

```text
原项目：/home/yingnan/??/czx/blendRL
原项目快捷链接：/home/yingnan/blendRL_project_k4
当前开发目录：/home/yingnan/blendRL_k4_dev
```

开发目录复用原项目环境和输出：

```text
.venv -> 原项目/.venv
out   -> 原项目/out
```

因此需要注意：

1. 不要在开发目录随意升级或删除 Python 依赖，因为会影响原项目环境。
2. 不要删除不属于本次实验的 `out` 目录，因为会删除原项目真实结果。
3. 两个目录不能使用相同输出名并发写入。
4. 每次训练必须使用唯一的 `BLENDRL_OUT_PATH`。
5. 启动前必须检查输出目录不存在。

### 3.2 环境变量

| 变量 | 作用 |
| --- | --- |
| `BLENDRL_OUT_PATH` | 指定本次实验输出目录 |
| `CUDA_VISIBLE_DEVICES` | 指定物理 GPU |
| `PYTHONUNBUFFERED=1` | 立即刷新训练日志 |
| `OMP_NUM_THREADS` | 控制 CPU 线程，评测并行时建议显式设置 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 减少 CUDA 内存碎片，可选 |
| `PYTHONPATH=.` | 从项目根目录运行脚本时保证本地模块可导入 |

当前没有 Docker、数据库、HTTP API 或 Web 服务配置。

### 3.3 环境使用

现有服务器环境：

```bash
cd /home/yingnan/blendRL_k4_dev
source .venv/bin/activate
```

开发目录的 `.venv` 是原项目环境的符号链接。除非明确重建环境，否则不要执行依赖升级。

重建环境时可参考：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd nsfr && ../.venv/bin/pip install -e . && cd ..
cd nudge && ../.venv/bin/pip install -e . && cd ..
```

Atari ROM、ALE、OCAtari 和 HackAtari 的系统安装方法仍需补充验证。

---

## 4. Riverraid 世界模型机制

### 4.1 三世界模型动作仲裁

当前推荐的三世界模型规划流程：

1. 使用三个冻结的独立世界模型。
2. 比较 sampled Blend 与 sampled Neural 候选动作。
3. 候选动作相同时跳过世界模型。
4. 候选动作不同时执行五步 world model rollout。
5. 后续动作使用 Blend policy 的 `argmax`。
6. 每个世界模型产生一条确定性轨迹。
7. 对三个模型的回报计算均值和标准差。
8. 当标准化优势大于 `tau` 时允许 Neural 覆盖 Blend。

当前 Seaquest 消融结果表明：

```text
dist + K1 + CRN ≈ dist + K4 + CRN
两者均明显低于 argmax + K1
```

因此当前推荐默认配置仍是：

```text
argmax + K1
CRN 关闭
horizon = 5
```

`dist`、K4 和 CRN 已经实现，但目前作为消融实验保留，不作为默认生产配置。

### 4.2 三世界模型关键参数

```text
--use-world-model-arbiter
--world-model-checkpoint <WM0>
--world-model-ensemble-checkpoints <WM0>,<WM1>,<WM2>
--world-model-freeze
--world-model-ensemble-tau 0.2
--world-model-ensemble-beta 1
--world-model-horizon 5
--world-model-warmup-steps 0
```

推荐确定性 rollout：

```text
--world-model-rollout-action-mode argmax
--world-model-mc-rollouts 1
```

argmax 模式下不要传：

```text
--world-model-common-random-numbers
```

随机 rollout 消融：

```text
--world-model-rollout-action-mode dist
--world-model-mc-rollouts 1
--world-model-common-random-numbers
```

多轨迹消融：

```text
--world-model-rollout-action-mode dist
--world-model-mc-rollouts 4
--world-model-common-random-numbers
```

参数限制：

1. `argmax` 只允许 `K=1`。
2. CRN 只能与 `dist` 一起使用。
3. `mc_rollouts` 必须大于等于 1。

非法组合包括：

```text
argmax + K4
argmax + CRN
K = 0
```

### 4.3 世界模型回报计算

单条五步轨迹回报：

```text
G = 预测折扣奖励之和 + 存活概率加权的末端 Blend critic value
```

每个世界模型在 K 轨迹模式下先计算：

```text
per_model_score = K 条轨迹回报的平均
```

然后在三个世界模型之间计算：

```text
blend_mean
blend_std
neural_mean
neural_std
```

最终置信度：

```text
confidence =
(neural_mean - blend_mean) /
max(sqrt(neural_std^2 + blend_std^2), 1e-8)
```

当：

```text
confidence > tau
```

执行 Neural 候选，否则保留 Blend 候选。

注意：当前三世界模型 Neural-only 最终切换依据是 `confidence > tau`。`ensemble_beta` 主要保留在诊断或其他保守分数中，不是当前 Neural-only 最终覆盖的主阈值。

---

## 5. Riverraid 三世界模型评测

### 5.1 评测口径

Riverraid 必须使用完整游戏口径：

```text
--full-game-episodes
```

原因：Riverraid 使用 `EpisodicLifeEnv`。如果不开启完整游戏模式，掉一条命可能被错误计为一个完整 episode，导致评测分数无法与正式完整游戏分比较。

### 5.2 checkpoint 加载

直接加载三个世界模型 checkpoint：

```text
--wm-checkpoint-paths <WM0>,<WM1>,<WM2>
```

当提供 `--wm-checkpoint-paths` 时，评测脚本应优先使用这些 checkpoint，并清空默认 Seaquest run 路径，避免 manifest 记录错误来源。

正式三模型来源：

```text
seed0 clean 20M
seed1 clean 20M（已同步到 yingnan）
seed2 clean 20M WM-only checkpoint
```

不应使用两个 seed0 模型代替 seed1。

### 5.3 tau 扫描配置

正式 tau 评测参数：

```text
policy：seed0 20M
WM：clean seed0、seed1、seed2
episodes：100 个完整游戏
num_eval_envs：32
rollout：argmax
K：1
CRN：关闭
horizon：5
beta：1
Neural-only：开启
tau：0、0.1、0.2、0.3
```

单组启动方式：

```bash
cd /home/yingnan/blendRL_k4_dev

bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.0 000
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.1 010
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.2 020
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.3 030
```

注意：脚本包含固定输出目录。重复评测前必须修改为新的唯一输出目录。

---

## 6. 训练与恢复注意事项

### 6.1 随机 rollout 恢复

随机 rollout 使用独立规划 RNG：

```text
planning_seed = training_seed + planning_seed_offset
```

训练 checkpoint 旁会保存：

```text
ensemble_planning_rng_step_<step>.pt
```

恢复训练时必须同时恢复对应的规划 RNG sidecar。

恢复前检查：

```bash
ls out/<实验>/checkpoints/
ls out/<实验>/ensemble_planning_rng_step_*.pt
```

缺少 RNG sidecar 时，不应继续随机 rollout 正式实验。

### 6.2 输出目录检查

每次启动前检查：

```bash
test ! -e "$BLENDRL_OUT_PATH"
```

输出目录已存在时应停止，不要复用。

---

## 7. 常见错误

### 7.1 argmax 模式启动失败

错误原因：

```text
argmax rollout requires mc_rollouts=1
```

处理：

```text
--world-model-rollout-action-mode argmax
--world-model-mc-rollouts 1
```

并关闭 CRN。

### 7.2 CRN 参数报错

错误原因：

```text
common random numbers require dist rollout mode
```

CRN 只能与 `dist` 配合。

### 7.3 Riverraid 评测分数只有个位数或几十

原因：把掉一条命当成完整游戏。

处理：

```text
--full-game-episodes
```

### 7.4 checkpoint 加载失败

直接 checkpoint 模式必须提供兼容的三个世界模型：

```text
--wm-checkpoint-paths WM0,WM1,WM2
```

加载器会严格检查缺失键、多余键和参数形状。

### 7.5 SPS 过低

依次检查：

```bash
nvidia-smi
ps -ef | grep python
```

确认：

1. 是否有多个训练共享 GPU。
2. `mp_num_workers` 是否与正式配置一致。
3. 是否只运行了很短的未预热测试。
4. 是否开启了大量 trace。
5. 是否有多个评测同时占用 CPU。

### 7.6 纯 PPO 不能直接启动

不要使用：

```text
train_blenderl.py --algorithm ppo
```

它仍会创建 BlendRL 模型。旧 `train_neuralppo.py` 当前也不是可复现的正式入口。必须先实现干净的纯 CNN PPO 脚本。

---

## 8. 当前不支持的内容

当前项目不是 Web 服务。

```text
Docker：未配置
Docker Compose：未配置
数据库：无
HTTP API：无
开放端口：无固定要求
```

实际运行方式：

```text
Linux 服务器
Python .venv
NVIDIA CUDA
shell 脚本
systemd 或前台进程
TensorBoard 可选
```

如需 Docker 化，需要额外补充：

1. CUDA 基础镜像。
2. Atari ROM 挂载。
3. OCAtari / HackAtari 系统依赖。
4. checkpoint 和 out 目录挂载。
5. GPU device 配置。
6. 共享内存和多进程限制。

---

## 9. 纯 PPO 公平基线状态

当前项目尚无可直接复现的纯 PPO 正式入口。

公平的纯 PPO 必须使用：

```text
与 BlendRL 神经分支相同的 CNN
18 个 ALE 原始动作
20,004,864 实际环境步
128 个环境
128 步 rollout
32 个环境 worker
4096 minibatch
10 个 update epochs
learning_rate = 0.00025
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
```

必须移除：

```text
Logic 策略
Blender
World Model
Skill package
```

正式训练前还需要实现对应的 CNN PPO 训练与 100 局完整游戏评测入口。

---

## 10. 已删除或归档的旧内容

以下旧内容不再作为当前状态维护：

1. `clean WM seed0 正在运行`。
2. `clean WM seed2 排队等待 seed0 完成`。
3. `GPU0 clean WM guard 当前状态`。
4. 临时 PID、瞬时显存、瞬时 GPU 利用率。
5. `out/51_riverraid_purewm_bd0_d05_seed0_l40sync_20260709` 作为保留历史目录的描述；该目录已被删除以避免误用。
6. “Riverraid 三世界模型仍依赖 L40”的描述；当前所需 seed1 已同步到 yingnan。
7. “评测只能从 run 目录加载 WM”的描述；当前支持 `--wm-checkpoint-paths`。
8. “Riverraid 一条 done 就是一整局”的评测假设；当前必须使用 `--full-game-episodes`。
9. “dist、K4、CRN 尚未实现”的待办描述；这些功能已经实现并测试。
10. “K4 正式训练尚未启动”的描述；Seaquest 相关 K1/K4 消融已经运行，Riverraid 当前采用 argmax + K1 作为推荐评测配置。

---

## 11. 未完成事项

1. 等待 Riverraid 四组 tau 完成 100 局并生成最终汇总。
2. 完成后优先比较均值、中位数和方差，而不是只看最高分。
3. 与无世界模型的 Riverraid BlendRL 完整正式对照做同口径比较。
4. 完成 Seaquest K1/K4 最终 20M 结果和固定 100 局评测，用于解释为何 Riverraid 默认采用 argmax + K1。
5. 建立统一实验注册表，记录输出目录、seed、checkpoint、tau、rollout 模式和结论。
6. 新建干净的纯 CNN PPO 训练和评测入口。
7. 补充 ROM、CUDA、系统依赖和 TensorBoard 访问文档。
8. 明确 `/home/yingnan/blendRL_k4_dev` 是否长期作为独立研究分支，还是后续合并回原项目。

### 仍需补充的信息

为了完成最终文档，还需要：

1. Riverraid tau 扫描完成后的 `tau_sweep_summary.csv/json`。
2. Riverraid 无世界模型 baseline 的完整 100 局同口径结果。
3. Seaquest K1/K4 训练结束后的最终 checkpoint 和 100 局结果。
4. 当前服务器 CUDA 驱动、ROM 路径和系统依赖清单。
5. 未来纯 PPO 新入口的实际文件名和最终命令。
