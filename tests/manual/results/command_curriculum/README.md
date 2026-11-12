# Curriculum 与 command 职责调整验收

初次验收：2026-11-10；task 职责复核：2026-11-11。范围仅为现有命令课程的依赖与调度，不包含 terrain 课程或联合采样实现。阶段计划已移至 `temp/阶段计划.md`。

## 实现边界

- 各 manager 提供 `get_curriculum_space_config(config)` 类方法，返回本模块的空间描述；未参与课程的 manager 返回空映射，不引入公共 manager 基类。
- CommandManager 解析命令范围与课程归属，提供连续维度描述和原约束过滤器；不再生成完整候选 cell。
- task 只收集并传递 manager 描述。CurriculumManager 在内部解析课程配置、调用 `curriculum/spaces.py` 合并维度并构造联合 cell；支持连续与数值离散维度，拒绝重复字段、未知课程、源配置冲突、错位过滤结果和过大联合空间。
- curriculum 不再导入 command 类或约束实现；保留 LPAC/FastLPAC 的奖励统计、学习进度及概率更新语义；每个实例只有一个空间，检查点使用单个 `space` 字段，不兼容旧 group 结构。
- 课程注册名统一为 `lpac_reward` / `fast_lpac_reward`，对应 `LpacReward` / `FastLpacReward`；实例名（如 `locomotion`）与算法类型分开；通用基类与缓冲分别为 `RewardCurriculum`、`CurriculumBuffer`，实现在 `curriculum/terms/reward.py`。旧课程注册名、类名和模块不保留别名，业务配置、命令显式引用及 Stage 指标路径同步采用新名称。
- command 仅通过 `get_command()` 读取分配，协议不再要求内部 buffers 或 resample 能力。
- task 分别调度课程 reset（结算/清理）、resample（分配）和 command reset（生成具体命令）。未结算回合禁止重新分配，读取不消耗课程随机数。
- 课程命令必须显式指定课程实例，支持多个同类算法实例；保留对称采样行为，统一使用 `Curriculum` / `SymmetricCurriculum`。
- 测试与播放冻结课程评价和概率，仍可从冻结概率采样；训练入口恢复更新。

尚未引入联合地形分配，也未新增跨物理 reset 的准备/提交事务。后续接入 terrain 时，再将联合分配准备移到物理 reset 前，并补齐相应失败重试语义。

## 最终检查

| 检查 | 结果 |
|---|---|
| quadruped_rl：core / mujoco | 1121 passed，11 skipped，47 deselected |
| quadruped_isaacsim：core / isaacsim | 1057 passed，11 skipped，111 deselected |
| MuJoCo Pyright | 0 errors，0 warnings |
| Isaac Pyright | 0 errors，0 warnings |
| 真实 OpenGL | 单项复验及最终 MuJoCo 全量回归通过 |
| git diff --check | 通过 |

新增测试覆盖统一 manager 接口、无需实例的 task 收集、多个提供方的联合空间、连续/离散维度组合、跨模块过滤与冲突检查，以及纯候选元数据输入、只读采样协议、独立命令共存、同算法多实例、局部回合结算与采样分离、活动回合覆盖拒绝和评估冻结。已有 LPAC/FastLPAC、Stage、RNG 和检查点恢复测试继续通过。

2026-11-11 职责复核修复两点：课程参数解析与空间构建归入 CurriculumManager，task 只传递原始描述；命令依赖通过构造参数接线，不再原地修改命令配置。新增回归验证 task 不解释课程配置、构建命令不修改输入，以及 BaseTask 不拥有 randomization/curriculum。任务的资源准备、manager 复用与生命周期调度、指标汇总、任务检查点包装仍由 task 负责。

## 历史对照与真实应用

先前职责调整的对照基线为 Git 提交 `b7561be54f72f4eabb1682bf86196fb74a9cafb9`，记录保存在下方历史 JSON 中。随后按要求取消旧设计兼容：移除了类属性回退、四个按算法命名的命令类和验收脚本中的旧实现加载分支。当前脚本只验证显式引用配置；旧配置不再自动兼容，旧源配置检查点也不做自动迁移。

历史对照使用当时的 ANYmal C 配置，候选 cell、相同种子下的分配、三个受控回合的课程状态及旧检查点加载结果一致。这些结论对应历史源码指纹，不代表当前仍提供旧设计兼容。LPAC 尚未评价的 cell 仍使用 NaN 占位，当前状态比较要求相同位置一致。

实际 MuJoCo + PPO 使用 4 环境、64 步 rollout、32 步回合、CPU 4 线程：

- 完成 2 次训练更新，信号与策略参数有限。
- 测试 4 回合，课程统计及概率保持不变。
- 从正式 runner 检查点恢复，课程状态一致。
- 续训完成 2 次更新，参数 L2 变化约 1.98748。
- 所有实例正常关闭，进程返回 0。

单空间结构的最新结果：[verification_single_space_20261111.json](verification_single_space_20261111.json)。`temp/single_space_ppo_20261111` 验证一个联合实例，`temp/two_instances_ppo_20261111` 使用临时 task 配置将平移、转向拆为两个同算法实例，均完成 2 次初始训练更新、4 回合评估和 2 次续训更新，状态恢复一致、信号有限、资源正常关闭。两实例的配置与复验入口保存在 `temp/verify_two_curriculum_instances.py`。

当前配置与接口已移除 group：manager 描述直接按课程实例组织，command 仅引用实例名，`reward_sources` 为实例自身的非空列表，指标为 `curriculum/<instance>/<metric>`。旧 group 配置明确拒绝；旧检查点不做迁移。回归覆盖多个同类型实例的独立空间、奖励统计和恢复，以及同实例多维联合采样；部分仅用于旧 group 配置结构的测试已替换为新结构校验。

此前名称调整结果：[verification_explicit_20261111.json](verification_explicit_20261111.json)；更早记录：[verification_20261111.json](verification_20261111.json)、[verification_20261110.json](verification_20261110.json)。历史结果对应当时的源码指纹。短训练不代表策略收敛，课程恢复不保证完整动力学轨迹逐步一致。

## Stage 指标接口解耦复核

StageCallback 仅消费 `info` 中显式指定的 `metric`、`revision_metric` 和 `validity_metric`，不识别课程路径、不推导控制字段，也不访问 task 或 curriculum manager。省略控制引用时保留普通指标采样；版本去重、回退清窗、有效性门控及未恢复环境时重建受控窗口的语义保留。非法控制输入不修改窗口与版本。检查点新增指标引用元数据，恢复时必须匹配，不兼容旧开关或缺少引用元数据的旧状态。

[verification_stage_metrics_20261111.json](verification_stage_metrics_20261111.json) 记录最新回归与源码指纹。普通指标和不同前缀的控制字段、无 task/manager 的采样、非法有效性值及引用不一致的恢复均有回归覆盖。真实 MuJoCo PPO 完成 2 次初始更新、4 回合评估及 2 次续训更新；正式检查点中的 Stage 状态在 runner 实际恢复回调后核对一致。复验入口为 `temp/verify_generic_stage_ppo.py`，产物位于 `temp/generic_stage_ppo_20261111`。

## 复跑

在项目根目录使用 `quadruped_rl`：

```powershell
python tests/manual/verify_command_curriculum.py --output-dir temp/command_curriculum_recheck
python -m pytest -m "core or mujoco" tests -q --basetemp temp/command_curriculum_tests -p no:cacheprovider
```

Isaac 兼容回归使用 `quadruped_isaacsim` 和 `core or isaacsim` marker；这里没有进行 Isaac 非平地应用验收。
