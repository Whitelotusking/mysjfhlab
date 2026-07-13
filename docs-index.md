# 文档索引

更新时间：2026-07-13

## riverraid

- 旧主文档路径：`riverraid/Riverraid_Project_Document_CN_20260709.md`
- 当前维护版路径：`riverraid/riverraid-wm/Riverraid_Project_Document_CN_20260710_maintained.md`
- 内容：BlendRL 在 Atari Riverraid 环境上的世界模型辅助动作仲裁维护文档，覆盖三世界模型、tau 扫描、完整游戏评测、复现风险、GRAIL/H²RL 分数口径、依赖冻结、纯 PPO 基线要求和实验注册表要求。
- 状态：2026-07-13 已根据 `Riverraid_WorldModel_BlendRL_Document_Maintained_20260713.md` 更新为最新版，删除临时运行状态，新增复现与基线风险说明。
- 后续维护：每次代码、实验配置、checkpoint、环境版本、依赖锁、评测结果或论文对照发生变化后更新该文档；动态评测结果必须标注检查时间，不能长期写成静态最终结论。

## 维护规则

1. 代码、配置文件和实际运行日志优先于旧文档。
2. 不确定的信息标注“需确认”。
3. 每次更新文档后，同步更新本索引。
4. 不把临时聊天内容直接写入正式文档，只保留对后续实验有价值的信息。
5. 原始游戏分、训练奖励、简化环境回报和完整游戏回报必须分开记录。
6. 正式实验必须记录 Git SHA、环境版本、ROM/依赖信息、checkpoint 哈希和评测口径。
