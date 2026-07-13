# 文档维护结果

更新时间：2026-07-13  
维护对象：`Riverraid_Project_Document_CN_20260710_maintained(1).md`  
维护原则：

1. 当前项目代码、实际配置、checkpoint、运行日志和评测脚本优先于旧文档。
2. 官方仓库代码优先于论文描述；论文描述优先于二手总结。
3. 未直接检查当前项目代码或运行结果的内容标注“需确认”。
4. 临时 PID、瞬时 GPU 状态和未完成任务进度不作为长期文档内容。
5. 原始游戏分、训练奖励、简化环境回报和完整游戏回报必须分开记录。

---

## 1. 本次变更摘要

本次维护新增和修正了以下内容：

1. 保留 Riverraid 三世界模型动作仲裁、checkpoint 加载、完整游戏评测和 tau 扫描的已确认实现。
2. 删除“当前正在运行”“正在排队”等容易过时的瞬时状态，改为可重复执行的状态检查命令。
3. 增加 BlendRL 复现风险说明：
   - 公开 `requirements.txt` 未锁定 Gymnasium、Stable-Baselines3、OCAtari 等关键版本。
   - BlendRL 的 HackAtari 依赖指向未固定 commit 的 GitHub 默认分支。
   - 公开仓库默认训练参数与 BlendRL 原论文参数不一致。
   - `train_blenderl.py --algorithm ppo` 仍创建 BlendRL，不是纯 PPO。
   - 旧 `train_neuralppo.py` 不能直接视为公平、独立的纯 PPO 基线。
4. 增加 GRAIL 与 BlendRL 代码对比：
   - GRAIL 的 `train_blenderl.py` 与 BlendRL 当前公开版本核心代码相同。
   - GRAIL 新增可学习 valuation、Stage 1/Stage 2 模块加载与冻结机制。
   - GRAIL 修正 Seaquest 潜水员对象切片和水面坐标。
   - GRAIL 论文完整环境设置与公开脚本默认值、Seaquest 奖励代码之间存在不一致。
5. 增加 GRAIL、BlendRL 原论文和 H²RL 分数口径说明，避免把简化环境与完整环境直接比较。
6. 增加环境冻结、Git commit、ROM、checkpoint 哈希和实验注册表要求。
7. 增加后续开发优先级：先建立可信 PPO/BlendRL 基线，再评估世界模型仲裁增益。

---

## 2. 需要更新的文档

| 文档 | 状态 | 需要更新的内容 |
| --- | --- | --- |
| `Riverraid_Project_Document_CN_20260710_maintained(1).md` | 本次已更新 | 世界模型主线、复现问题、GRAIL/H²RL、依赖冻结、当前状态 |
| `README.md` | 需要更新 | 标准启动命令、输出目录、完整游戏评测、环境冻结要求 |
| `docs/architecture.md` | 建议新增或更新 | BlendRL、Neural、三世界模型仲裁的数据流和切换条件 |
| `docs/evaluation.md` | 建议新增 | 原始分数、训练奖励、完整游戏 episode、100 局评测口径 |
| `docs/reproducibility.md` | 建议新增 | Python/CUDA/Gymnasium/ALE/OCAtari/HackAtari 版本及 commit |
| `docs/troubleshooting.md` | 建议新增或更新 | `EpisodicLifeEnv`、checkpoint 不兼容、依赖漂移、低 SPS |
| `CHANGELOG.md` | 建议更新 | 三世界模型、CRN、K-rollout、完整游戏评测、直接 WM checkpoint 加载 |
| `requirements-lock.txt` 或 `environment.yml` | 必须新增 | 固定实际可运行环境，不再只依赖未锁版本的 `requirements.txt` |
| `experiments/registry.csv` | 建议新增 | 记录 seed、checkpoint、Git SHA、奖励、tau、评测结果和状态 |

当前未确认上述建议文档是否已经存在。若已有，应更新原文件，不要重复新建同类文档。

---

## 3. 过时内容

### 3.1 临时运行状态

旧文档中的以下内容会快速失效，不应继续作为“当前状态”：

- tau 四组评测正在运行。
- 某个 seed 正在训练或排队。
- 临时 PID、显存占用、GPU 利用率。
- 某个 worker 当前完成多少局。
- 当前仍依赖 L40 读取 checkpoint。

处理方式：

```bash
nvidia-smi
ps -ef | grep -E "python|train|eval"
tail -f out/<实验目录>/launch_logs/train.log
```

文档只记录启动方法、输出位置和最终完成结果。

### 3.2 阶段性 tau 结果被误当成最终结果

旧文档中的 tau 表只来自未完成评测：

| tau | 当时已完成局数 | 当时平均原始游戏分 |
| ---: | ---: | ---: |
| 0.0 | 36 | 2644.7 |
| 0.1 | 45 | 2499.8 |
| 0.2 | 58 | 2424.5 |
| 0.3 | 69 | 2332.3 |
| baseline | 100 | 2225.9 |

这些数据只能保留为“2026-07-12 阶段记录”，不能进入最终论文结果。四组是否已经完成 100 局：**需确认**。

### 3.3 直接使用 `requirements.txt` 可以复现作者环境

该说法不准确。

BlendRL 当前公开依赖主要写成：

```text
stable-baselines3
ocatari
hackatari @ git+https://github.com/k4ntz/HackAtari.git
gymnasium[atari, accept-rom-license]
torch==2.4.0
```

只有 `torch==2.4.0` 和少数包严格固定。其他包会根据安装时间、Python 版本和依赖解析结果变化；HackAtari 还会获取安装时默认分支代码。

GRAIL 的 `requirements.txt` 同样未锁定 Stable-Baselines3、OCAtari 和 Gymnasium。GRAIL Dockerfile使用仓库内 `third_party/hackatari`，但本地安装说明没有完整给出与论文实验一致的锁定环境。

### 3.4 `train_blenderl.py --algorithm ppo` 是纯 PPO

该说法不准确。当前审计结论是：

```text
train_blenderl.py --algorithm ppo
```

仍然创建 `BlenderActorCritic`，不能作为纯 CNN PPO。

旧 `train_neuralppo.py` 虽然只训练神经 actor-critic，但仍使用 BlendRL 环境构造路径，默认参数也与论文完整实验不完全一致。

### 3.5 GRAIL 表 1 的 983 与 BlendRL 原论文 4204 是同一实验

该说法不准确。

- GRAIL Table 1 的 Seaquest `BlendRL+Expert = 983±93` 属于 Stage 1 简化环境。
- GRAIL Figure 11 的 Seaquest hard-coded BlendRL `4706` 属于 Stage 2 完整环境。
- BlendRL 原论文 Seaquest BlendRL `4204±10` 属于原论文完整实验。

只有后两项在任务含义上较接近，但训练环境、代码版本和评测细节仍未完全对齐。

---

## 4. 缺失内容

### 4.1 当前项目代码与配置文件

本次只收到维护文档，没有收到当前项目源码和最终运行配置。以下结论仍需用当前仓库复核：

- 世界模型仲裁最终公式及 `ensemble_beta` 的实际用途。
- `--wm-checkpoint-paths` 的严格加载逻辑。
- `--full-game-episodes` 对终止、截断和生命损失的处理。
- tau 启动脚本中的实际环境数、worker 数和输出目录。
- 当前 Riverraid 奖励是否始终为 `sign(original_reward)`。
- 当前正式 baseline checkpoint 的训练参数和 Git SHA。

需要补充的文件：

```text
train_blenderl.py
实际评测入口
世界模型/ensemble/arbiter 实现文件
Riverraid env.py 与 env_vectorized.py
Riverraid reward 文件
scripts/run_riverraid_3wm_tau_parallel_one.sh
正式 run 的 config.yaml / manifest.json
pip freeze
Git commit SHA
最终 tau 汇总 CSV/JSON
```

### 4.2 可复现环境快照

当前缺少：

- Python 精确版本。
- CUDA driver、CUDA runtime、cuDNN 版本。
- `gymnasium`、`stable-baselines3`、`ale-py`、`ocatari`、`hackatari` 精确版本。
- OCAtari/HackAtari Git commit。
- Atari ROM 名称、来源和哈希。
- 当前项目 Git commit。
- checkpoint SHA256。

### 4.3 可信纯 PPO 基线

当前没有同时满足以下条件的正式 PPO 入口：

- 与 BlendRL 神经分支相同 CNN。
- 相同像素预处理和动作空间。
- 相同奖励和 episode 口径。
- 相同总环境步、rollout、minibatch、update epochs。
- 独立于 Logic、Blender、World Model。
- 100 局完整游戏固定评测。

### 4.4 GRAIL 完整实验配置

GRAIL 论文写明 Stage 2 使用：

```text
60M steps
32 parallel environments
128 rollout steps
10 PPO epochs
learning rate 2.5e-4 linear decay
gamma 0.99
GAE lambda 0.95
clip 0.1
action entropy 0.01
blend entropy 0.01
value coefficient 0.5
gradient clipping 0.5
```

但公开仓库没有提供 Figure 11 每个 seed 对应的完整命令、配置文件、checkpoint 和依赖锁文件。

### 4.5 H²RL 的 BlendRL 实现

截至 2026-07-13，未找到 H²RL 论文明确链接的官方公开代码仓库。无法确认其 BlendRL baseline 使用：

- 哪个 BlendRL commit。
- 原始或 GRAIL 修正版 Seaquest valuation。
- NSFR 或 NEUMANN。
- 哪个依赖环境。
- 哪个奖励函数。
- 单命 episode 或完整游戏。
- 是否修改 CNN、动作空间或训练入口。

因此 H²RL 的低分只能作为复现冲突证据，不能直接认定为官方 BlendRL 的真实性能。

---

## 5. 矛盾内容

### 5.1 三篇论文中的 Seaquest 分数不可直接混用

| 来源 | 环境/阶段 | PPO | BlendRL | 结论 |
| --- | --- | ---: | ---: | --- |
| BlendRL 原论文 | 原论文完整实验 | 837.3±46.7 | 4204±10 | 原作者报告 |
| GRAIL Table 1 | Stage 1 简化环境 | 453±93 | Expert 983±93 | 不可与完整环境直接比较 |
| GRAIL Figure 11 | Stage 2 完整环境 | 2216 | hard-coded 4706 | 与原论文较接近 |
| H²RL Table 13 | H²RL 统一设置 | 3247±881 | 117±62 | 实现未公开，原因需确认 |

当前项目中应只比较同一代码、同一环境、同一奖励和同一评测脚本下重新训练的结果。

### 5.2 BlendRL 原论文与当前公开脚本默认值

BlendRL 原论文 Table 4：

```text
num_envs = 512
num_steps = 128
total_timesteps = 20,000,000
learning_rate = 2.5e-4
gamma = 0.99
```

当前公开 `train_blenderl.py` 默认值：

```text
num_envs = 20
num_steps = 128
total_timesteps = 60,000,000
update_epochs = 10
```

公开脚本可以通过命令行覆盖部分参数，但默认运行不等于原论文复现。

### 5.3 GRAIL 论文 Stage 2 与公开代码

| 项目 | GRAIL 论文 | 公开代码/README | 状态 |
| --- | --- | --- | --- |
| 并行环境 | 32 | 脚本默认 20，README 示例 5 | 可覆盖，但默认不一致 |
| PPO epochs | 10 | `train_blenderl.py` 10 | 一致 |
| PPO baseline epochs | 论文 Stage 2 写 10 | `train_neuralppo.py` 默认 4 | 不一致 |
| Stage 2 valuation | 加载 Stage 1 valuation 并冻结 | README 调用旧 `train_blenderl.py` | README 路径不足 |
| Neural/Blender 初始化 | Stage 2 从头随机初始化 | `train_valuation.py` 配置路径支持重置 | 代码具备能力，缺正式配置 |
| Seaquest 奖励 | 主要目标 20，其他奖励 1 | 默认 reward 为救援 1、普通事件 0.5 | 明确不一致或缺配置 |
| step size | 论文写 1 | wrapper 使用 `MaxAndSkipEnv(skip=4)` | 含义需确认 |
| 完整运行命令 | 应包含完整 Stage 2 配置 | README 只给简化示例 | 缺失 |

### 5.4 GRAIL 与 BlendRL Seaquest 代码差异

GRAIL 并非完全原样使用旧 Seaquest 环境代码。

已确认差异：

```python
# BlendRL 当前公开 valuation
divers_vs = objs[:, -6:]

# GRAIL
divers_vs = objs[:, -7:-1]
```

```python
# BlendRL 当前公开奖励判断
player.y == 45

# GRAIL
player.y == 46
```

GRAIL 还注明 Seaquest 当前实际只有 42 个对象，但为了兼容旧 checkpoint 保持 `n_objects = 43`。

这说明 OCAtari/HackAtari 对象布局或坐标变化可能直接破坏旧符号谓词。当前项目若复现 Seaquest，必须记录依赖版本并为对象索引做单元测试。

### 5.5 依赖文件与复现目标

当前 `requirements.txt` 只能表达“可尝试安装”，不能表达“论文使用环境”。

需要同时维护：

```text
requirements.in        # 人工维护的直接依赖
requirements-lock.txt  # 完整固定版本
git-dependencies.txt   # OCAtari/HackAtari 等 commit
system-info.txt        # Python/CUDA/driver
rom-hashes.txt         # ROM 哈希
```

---

## 6. 建议修改内容

以下内容可直接作为新版项目主文档。

---

### Riverraid 世界模型辅助 BlendRL 项目文档

更新时间：2026-07-13

#### 1. 项目定位

本项目在 BlendRL 的 Atari Riverraid 分支上增加世界模型辅助动作仲裁。

核心流程：

1. BlendRL 产生 sampled Blend 候选动作。
2. 神经策略产生 sampled Neural 候选动作。
3. 两个动作相同则直接执行，不调用世界模型。
4. 两个动作不同时，三个冻结世界模型分别进行短期 rollout。
5. 比较两个候选动作的预测回报。
6. 只有 Neural 候选的标准化优势超过 `tau` 时，才允许 Neural 覆盖 Blend。

当前 Riverraid 正式奖励口径：

```python
return float(np.sign(self.org_reward))
```

旧 shaped reward、原始游戏分和训练 reward 不得混在同一结果表中。

#### 2. 已确认实现

当前维护文档确认已经实现：

```text
三世界模型 ensemble
rollout_action_mode = argmax | dist
mc_rollouts = K
common_random_numbers
独立 planning RNG
planning RNG checkpoint sidecar
K 轨迹批量展开
模型间标准差与轨迹内标准差分离
训练与评测共用 rollout 逻辑
--wm-checkpoint-paths
--full-game-episodes
```

正式 Riverraid 三个世界模型来源：

```text
seed0：yingnan 完整训练 checkpoint
seed1：从 L40 同步到 yingnan
seed2：yingnan WM-only checkpoint
```

具体 checkpoint 路径和 SHA256：**需确认并写入实验注册表**。

#### 3. 推荐世界模型配置

当前推荐的确定性配置：

```text
rollout_action_mode = argmax
mc_rollouts = 1
common_random_numbers = false
horizon = 5
world_model_freeze = true
```

原因：已有 Seaquest 消融记录显示 `dist + K1/K4 + CRN` 低于 `argmax + K1`。该结论是否已使用最终 20M checkpoint 和固定 100 局评测：**需确认**。

参数约束：

```text
argmax 只允许 K=1
CRN 只允许 dist
mc_rollouts >= 1
```

#### 4. 仲裁分数

当前维护文档记录的回报形式：

```text
G = 折扣预测奖励之和
  + 存活概率加权的末端 Blend critic value
```

K 轨迹模式：

```text
per_model_score = K 条轨迹回报均值
```

三个模型之间计算：

```text
blend_mean
blend_std
neural_mean
neural_std
```

标准化置信度：

```text
confidence =
(neural_mean - blend_mean) /
max(sqrt(neural_std^2 + blend_std^2), 1e-8)
```

最终动作：

```text
confidence > tau  -> Neural
otherwise         -> Blend
```

该公式及 `ensemble_beta` 是否仍与当前代码完全一致：**需确认**。

#### 5. 评测口径

Riverraid 正式评测必须使用：

```text
--full-game-episodes
```

原因：训练 wrapper 含 `EpisodicLifeEnv`。生命损失可能生成训练 episode 边界，但正式原始游戏分应累计到真实游戏结束。

正式结果至少记录：

```text
平均原始游戏分
标准差
中位数
四分位数
最小值/最大值
完整游戏局数
Neural 覆盖比例
候选动作冲突比例
世界模型调用比例
每个 tau 的 checkpoint 和 Git SHA
```

不得只比较最高分。

#### 6. tau 扫描

已有启动脚本记录：

```bash
cd /home/yingnan/blendRL_k4_dev

bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.0 000
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.1 010
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.2 020
bash scripts/run_riverraid_3wm_tau_parallel_one.sh 0.3 030
```

旧文档记录的目标配置：

```text
policy：Riverraid seed0 20M
WM：clean seed0、seed1、seed2
episodes：100 个完整游戏
num_eval_envs：32
rollout：argmax
K：1
CRN：关闭
horizon：5
beta：1
tau：0、0.1、0.2、0.3
```

脚本当前实际参数和四组最终完成状态：**需确认**。

#### 7. 目录与输出安全

当前开发目录记录：

```text
/home/yingnan/blendRL_k4_dev
```

共享资源记录：

```text
.venv -> 原项目 .venv
out   -> 原项目 out
```

风险：

1. 升级共享 `.venv` 会影响原项目。
2. 删除开发目录下的共享 `out` 会删除原始结果。
3. 并发实验不能使用相同输出目录。
4. 每个 run 必须设置唯一 `BLENDRL_OUT_PATH`。
5. 启动前检查目标目录不存在。

推荐：

```bash
test ! -e "$BLENDRL_OUT_PATH" || {
  echo "Output path already exists: $BLENDRL_OUT_PATH"
  exit 1
}
```

#### 8. 环境快照

在任何升级、重装或正式训练前执行：

```bash
cd /home/yingnan/blendRL_k4_dev
source .venv/bin/activate

mkdir -p reproducibility

python --version > reproducibility/python-version.txt
python -m pip freeze > reproducibility/pip-freeze.txt
nvidia-smi > reproducibility/nvidia-smi.txt

git rev-parse HEAD > reproducibility/project-git-sha.txt
git status --short > reproducibility/project-git-status.txt
```

记录关键包：

```bash
python - <<'PY' > reproducibility/key-packages.txt
from importlib.metadata import version, PackageNotFoundError

packages = [
    "torch",
    "numpy",
    "gymnasium",
    "stable-baselines3",
    "ale-py",
    "ocatari",
    "hackatari",
]

for name in packages:
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
```

Git 依赖若来自源码安装，还要记录：

```bash
git -C <OCATARI_DIR> rev-parse HEAD
git -C <HACKATARI_DIR> rev-parse HEAD
```

checkpoint 记录：

```bash
sha256sum <POLICY_CHECKPOINT> <WM0> <WM1> <WM2>
```

#### 9. BlendRL 复现注意事项

BlendRL 原论文报告的 Seaquest 分数：

```text
PPO：837.3 ± 46.7
BlendRL：4204 ± 10
```

原论文训练参数包括：

```text
num_envs = 512
num_steps = 128
total_timesteps = 20,000,000
learning_rate = 2.5e-4
gamma = 0.99
```

当前公开 `train_blenderl.py` 默认值并非上述配置：

```text
num_envs = 20
total_timesteps = 60,000,000
```

因此直接运行公开默认命令不能称为原论文复现。

当前项目已发现：

```text
train_blenderl.py --algorithm ppo
```

仍然构建 BlendRL。不得把它标记为纯 PPO。

#### 10. 纯 PPO 基线要求

正式纯 PPO 应满足：

```text
与 BlendRL 神经分支相同 CNN
18 个 ALE 原始动作
相同像素预处理
相同奖励函数
相同完整游戏评测
相同总实际环境步
相同 rollout 和 PPO 更新强度
无 Logic
无 Blender
无 World Model
无 Skill package
```

当前项目计划参数记录为：

```text
20,004,864 实际环境步
128 个环境
128 步 rollout
32 个环境 worker
4096 minibatch
10 update epochs
learning_rate = 0.00025
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.1
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
```

这些参数是否已经进入可运行的新 PPO 脚本：**需确认**。

#### 11. GRAIL 对本项目的影响

GRAIL 是 BlendRL 的后续工作，主要新增：

```text
LLM 生成概念 proxy
可学习 valuation network
concept alignment loss
Stage 1 概念学习
Stage 2 冻结 valuation 后训练神经策略和 Blender
```

可借鉴内容：

1. 用可学习谓词替代完全手写阈值。
2. 使用弱监督 proxy 解决稀疏奖励下的概念学习。
3. 将概念学习与完整策略训练分阶段处理。
4. 在本项目中可由世界模型仲裁器检查符号或神经候选动作。

不应直接继承的内容：

1. GRAIL 的公开完整实验命令和配置不完整。
2. 依赖仍未严格锁定。
3. Seaquest 论文奖励与默认代码不一致。
4. PPO baseline 默认 epochs 与论文描述不一致。
5. README 的 Stage 2 命令不能完整表达 valuation 加载、冻结和模块重置。

#### 12. GRAIL 与 BlendRL 的关键代码差异

核心 `train_blenderl.py` 当前基本相同，GRAIL 的主要变化在：

```text
valuation/models
train_valuation.py
ValuationConfig
checkpoint 层级加载与冻结
Seaquest 环境兼容修正
```

Seaquest 兼容修正：

```python
# 旧 BlendRL
divers_vs = objs[:, -6:]
surface_y = 45

# GRAIL
divers_vs = objs[:, -7:-1]
surface_y = 46
```

当前项目如果继续做 Seaquest 对照，应增加测试：

```text
对象数量和顺序
6 名潜水员对应的 tensor 切片
玩家到达水面时的 y 坐标
full_divers 是否真实触发
救援奖励是否只触发一次
```

#### 13. GRAIL 分数解释

GRAIL Stage 1 简化环境：

```text
Seaquest NeuralPPO：453 ± 93
Seaquest BlendRL+Expert：983 ± 93
```

GRAIL Stage 2 完整环境：

```text
Seaquest NeuralPPO：2216
Seaquest hard-coded BlendRL：4706
```

Stage 1 与 Stage 2 不能直接比较。

GRAIL Stage 2 hard-coded BlendRL 与原论文 `4204` 较接近，但公开代码仍不足以严格复现 `4706`。

#### 14. H²RL 分数解释

H²RL 在自己的统一环境中报告：

```text
Seaquest PPO：3247 ± 881
Seaquest BlendRL：117 ± 62
```

这与 BlendRL 原论文和 GRAIL Stage 2 严重冲突。

截至 2026-07-13，未找到 H²RL 官方公开实现，因此不能确认低分来自：

```text
BlendRL 方法本身
代码移植错误
Seaquest 索引/坐标版本问题
依赖版本
奖励函数
episode 口径
checkpoint 选择
训练配置
```

论文中只能将该结果表述为“后续工作中的复现冲突”。

#### 15. 常见错误

##### 15.1 Riverraid 只有个位数或几十分

可能原因：

```text
把生命损失当成完整游戏结束
```

处理：

```text
--full-game-episodes
```

##### 15.2 argmax rollout 参数错误

正确配置：

```text
--world-model-rollout-action-mode argmax
--world-model-mc-rollouts 1
```

不要开启 CRN。

##### 15.3 CRN 参数错误

CRN 只能用于：

```text
rollout_action_mode = dist
```

##### 15.4 checkpoint 不兼容

检查：

```text
模型结构
对象数量
动作数量
参数键
参数形状
训练代码 Git SHA
依赖版本
```

##### 15.5 重装后分数突然变化

优先检查：

```text
gymnasium
stable-baselines3
ale-py
ocatari
hackatari
ROM
wrapper 顺序
reward 文件
```

不要直接在共享 `.venv` 中升级。

##### 15.6 纯 PPO 分数异常

确认实际运行的类：

```text
ActorCritic
```

而不是：

```text
BlenderActorCritic
```

并检查 PPO `update_epochs` 是否与对照一致。

#### 16. 实验注册表

建议每个正式实验增加一行：

```csv
run_id,game,method,git_sha,env_hash,reward_fn,policy_ckpt,wm0,wm1,wm2,seed,total_steps,num_envs,num_steps,update_epochs,tau,horizon,rollout_mode,mc_rollouts,crn,full_game,episodes,mean,std,median,status,notes
```

`status` 只允许：

```text
planned
running
completed
failed
invalid
archived
```

#### 17. 当前不支持内容

当前项目不是 Web 服务：

```text
数据库：无
HTTP API：无
固定端口：无
Docker Compose：未配置
```

GRAIL 官方仓库有 Dockerfile，但本项目尚未确认已完成可复现 Docker 化。

若后续 Docker 化，需要：

```text
固定 CUDA 基础镜像 digest
锁定 Python 依赖
固定 OCAtari/HackAtari commit
ROM 挂载和哈希校验
checkpoint/out 挂载
GPU 和共享内存配置
多进程 worker 限制
```

#### 18. 参考来源

- BlendRL: A Framework for Merging Symbolic and Neural Policy Learning, ICLR 2025, arXiv:2410.11689
- GRAIL: Autonomous Concept Grounding for Neuro-Symbolic Reinforcement Learning, arXiv:2604.16871
- Boosting deep Reinforcement Learning using pretraining with Logical Options, arXiv:2603.06565
- `ml-research/blendrl`
- `ml-research/grail`

---

## 7. 当前项目状态

### 已完成

- Riverraid 三世界模型仲裁主流程。
- argmax/dist rollout。
- K-rollout 与 CRN。
- 独立 planning RNG 和恢复 sidecar。
- 直接加载三份 WM checkpoint。
- Riverraid 完整游戏评测模式。
- tau 扫描启动脚本。
- 纯 PPO 入口审计。
- BlendRL、GRAIL 公开代码初步对比。
- GRAIL、BlendRL、H²RL 分数口径梳理。

### 未完成或需确认

- tau 四组最终 100 局结果。
- 当前正式 baseline 的完整配置和 checkpoint SHA256。
- 世界模型仲裁公式与当前代码逐行复核。
- Seaquest K1/K4 最终固定评测。
- 干净纯 PPO 训练和评测入口。
- 当前环境完整 `pip freeze`。
- OCAtari/HackAtari commit。
- ROM 哈希。
- GRAIL Figure 11 对应的完整配置和 checkpoint。
- H²RL 官方实现是否后续公开。

### 已知问题

1. BlendRL 公开依赖未锁定，重装环境可能发生版本漂移。
2. 当前公开 BlendRL 默认参数不是原论文参数。
3. Seaquest 对象索引和水面坐标受依赖版本影响。
4. GRAIL 论文 Stage 2 与公开默认奖励/命令存在不一致。
5. H²RL 的 BlendRL 实现未公开，117 分原因无法定位。
6. 当前项目共享 `.venv` 和 `out`，误升级或误删除风险高。
7. 尚无可信的同口径纯 PPO 基线。

### 下一步建议

按优先级执行：

1. 导出当前环境、Git SHA、ROM 和 checkpoint 哈希。
2. 完成 tau 四组 100 局评测并生成不可变汇总。
3. 建立实验注册表，标记无效和非最终实验。
4. 实现干净纯 PPO，并与无世界模型 BlendRL 使用同一评测脚本。
5. 对 Seaquest 增加对象索引、坐标和救援触发单元测试。
6. 在复制的独立环境中做依赖版本 A/B，不修改共享 `.venv`。
7. 世界模型方法只与同代码、同奖励、同完整游戏口径的 baseline 比较。
8. GRAIL 可作为可学习谓词模块参考，不直接作为已复现基线。
