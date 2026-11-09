# 阶段 3：任务信号适配与首轮训练验收

最新结果见 [用户审核后最终检查](final_review_20261109.md)：MuJoCo/core 1044 项通过、Isaac/core 980 项通过，ANYmal 真实渲染和两次 PPO 短训通过；开发过程记录见 [阶段检查报告](review_20261109.md)。

当前地形示例已迁到现有 `configs/tasks/locomotion_anymal_c_phase.yaml`，配对 `configs/simulators/mujoco_anymal_c_rgb_array.yaml`；不再依赖已删除的 Go1 地形任务。两份手工脚本现使用 ANYmal C，以下 Go1 原始记录保留为历史验收证据。训练脚本在输出目录生成固定命令诊断任务，保留原 ANYmal 业务配置的 LPAC 命令课程；当前检查机身高度奖励，不再要求启用 Go1 抬脚奖励。

验收日期：2026-11-07（Asia/Shanghai）；2026-11-08 将验收参数移入脚本后，两次 PPO 短训复验通过。阶段目标已达成：高度相关观测、奖励、终止使用正确的局部地面参照；平地行为兼容；有限地形越界能结束并局部重置；真实 MuJoCo PPO 短训练正常推进。

## 最小接入方案

沿用原有 VectorEnv、simulator、runner 和 PPO 结构。SimulatorState 增加四项可选派生物理数据，由 simulator 在 get_state 时计算：

- `base_ground_height`：`[env]`，根位置下方的世界地面高度。
- `foot_ground_heights`：`[env, foot]`，ModelContext 足 geom 顺序的世界地面高度。
- `ground_valid`：`[env]`，机身和所有足端采样点均有有效地面高度。
- `terrain_in_bounds`：`[env]`，机身和所有足端均处于随机地形区域内；供可选 `TerrainBounds` 终止条目使用。

MuJoCo 使用 BaseSimulator 的 `_get_ground_state()` 将根位置与各足位置合并为一次批量查询，再与位置、速度、接触等数据一起构造完整 SimulatorState。范围来自 `get_ground_bounds()`（所选环境的世界 xy 下/上界 `[env,2]`），旧无限平面返回 None。Isaac 则由 runtime 的 `get_state()` 完整提供物理状态字段，外层只构造 SimulatorState，不再次查询或派生地面状态；当前仅支持无限平地，省略可选地面字段。

`BaseTaskLogic.build_task_context()` 只将传入 state 及任务数据组合为 TaskContext；不保存 simulator，不查询地面，randomization manager/term 也不参与每步信号采集。高度读取和奖励屏蔽函数在 `envs/tasks/utils/height.py` 中读取 `task_context.state`。奖励和观测即使分别构造 context，也共享同一份完整物理状态。

未配置有限地形时四项字段为 None，不执行采样；`as_dict()` 省略不可用字段，保持旧平地状态键集合兼容。`clone()` 和 `select()` 复制/选择可用 Tensor 并保留 None。地面派生 Tensor 每次状态采集独立生成，局部 reset 状态读取不会覆盖终止前完整状态中的地面数据；原有位置等复用缓冲仍遵守完整/局部读取通道分离的约定。

## 高度条目清点与处理

| 条目 | 阶段 3 行为 |
| --- | --- |
| observation `base_height` | 根位置 z 减去机身下方局部地面高度 |
| observation `foot_height` | 各足 geom 中心 z 减去各自下方地面高度；不额外改变足半径定义 |
| reward `base_height_l2`、`base_height_l2_xy_speed_weighted` | 目标高度与误差使用相同地面参照 |
| reward `foot_lift_height_diff_command_weighted_exp`、`foot_lift_height_diff_command_gated_l2` | 摆动脚高度使用各足下方地面，指令门控和支撑状态语义不变 |
| termination `base_height` | 根位置相对局部地面高度与 `min_height` 比较 |
| observation `base_position` | 保留世界坐标 |
| 机身平面相对的步态/相位奖励 | 保留机身平面参照，未替换为地面参照 |
| 世界垂直速度、重力投影、关节与接触信号 | 保留原定义，不借高度适配顺带改变语义 |

七个高度条目都支持 `height_reference: ground`（默认）或 `world`。观测和终止项直接配置该参数，奖励项放在 `params` 中。只有几何地形启用时才提供非零地面参照；需要绝对高度时可明确选择 `world`。

## 越界处理

MuJoCo 在随机地形四周增加每侧 1000 米的平坦碰撞体，内部凹陷不被覆盖，生成器边缘回到平面高度。机身/足端逐点查询，高度场外读取外围平面；`ground_valid` 与随机区域边界解耦。`TerrainBounds` 通过 `terrain_in_bounds` 决定是否终止，当前示例未启用，允许在外围继续运动；添加该项后越界终止；BaseTaskLogic 不再硬编码越界终止。

无定义行（如非有限坐标或超出外围覆盖）的地面相对高度观测使用零占位，高度奖励与高度阈值测试屏蔽。外围并非无限平面，超出覆盖的直接高度查询返回 NaN，不自动补充终止策略。普通 reset 不更换地形或推进分配代数。边缘衔接改变采样规则，分配状态版本为 5，拒绝旧版本 1/2/3/4。

原始短训记录曾启用 `base_height: {min_height: 0.1}`；当前业务示例保留原有 body_contact 终止，不再额外启用低高度策略。低高度终止能力由针对性测试验证，可按任务需要配置。平地逐接触点筛选与高度场按足汇总法向力的差异仍按既有约定保留。

## 自动验证

测试覆盖：

- 七类高度条目的平移不变性、显式 world 选项、零地面与旧平地输出逐位一致、非法参照拒绝及越界信号屏蔽。
- 真实线性坡面上，两个相同关节/朝向姿态位于 x=-1.2、+1.2 m，地面高度分别为 -0.24、+0.24 m，根世界 z 分别为 0.08、0.56 m。相对根高度均为 0.32 m（绝对容差 1e-6 m），各足高度观测及机身高度奖励一致，不误触发低高度终止。
- 环境子集 `[2,0]` 只执行一次 `[2,5,2]` 批量查询，后续观测共享结果，观测维度仍为 112。
- 根位置越界、根仍在块内但足端越界均只终止指定环境；外围查询仍有效，转移观测/奖励有限，局部 reset 后所有环境重新进入随机区域；关闭 TerrainBounds 后可继续运动。
- 在出生平台外 x=1.2 m 的 plane/slope/rough 表面制造真实 trunk 接触，非法接触奖励和 body_contact 终止均正确检测。
- 自审修复 IllegalContactL1 空接触输出使用总环境数的问题，改为当前批次大小；恢复训练及其他逻辑不受影响。
- 验收参数内置于手工脚本，组件组合测试检查生成的 8 环境/200 步配置，以及前进指令任务副本；算法、策略、仿真器和原任务均复用现有配置。无需额外的业务 smoke 配置或阶段转移参数。

完整回归：MuJoCo/公共 973 passed、11 skipped、44 deselected；Isaac/公共 938 passed、11 skipped、79 deselected。两套 Pyright 均为 0 errors / 0 warnings。Isaac 这里只做公共接口回归，没有启动实际 Isaac 非平地应用。

测试入口：`tests/test_terrain_signals.py`、`tests/test_mujoco_terrain_signals.py`。

## 真实 PPO 短训练

脚本直接使用现有 OnPolicyRunner、PPO 和 actor_critic；8 环境，rollout_length=128，seed=0，CPU，sim_dt=0.002、frame_skip=10，回合上限 200 控制步。

为避免“配置是 rough，但训练始终停在中央平坦平台”的假验收，脚本仅在训练开始时把非平地环境诊断性放到 x=1.2 m，按各足下方实际高度抬升根位置，并重新清理任务历史、构建初始观测。业务出生策略未改变；后续自动 reset 仍回到正常安全区。

| 项目 | 零指令站立 | 前进指令 0.3 m/s |
| --- | ---: | ---: |
| PPO 更新数 | 16 | 8 |
| 转移数 | 16,384 | 8,192 |
| 完成回合数 | 148 | 76 |
| 最短回合 | 10 步 | 33 步 |
| 平均回合长度 | 106.13 步 | 103.83 步 |
| 非平坦 slope 环境步样本 | 1,033 | 881 |
| 非平坦 rough 环境步样本 | 489 | 422 |
| 参数更新 L2 范数 | 6.6691 | 4.7731 |

两次均无观测、奖励、状态、优化指标或参数非有限值；全部环境均观察到四足接触；无出生后一步即终止；参数确实更新。前进指令下抬脚高度奖励范围为 [-20.5989,0]，确认指令门控打开后也经过实际训练验证。两次运行和清理正常结束，进程退出码为 0。

原始结果：[站立短训](training_verification.json)、[前进短训](moving/training_verification.json)。这是信号/训练流程验收，不是最终行走能力、收敛性、课程或正式地形恢复训练验收。

## 平地回归

使用原阶段 1 的 16 环境脚本重跑，八项行为比较全部完全一致，测量与进程关闭退出码 0。见 [比较结果](plane_comparison.json)、[原始测量](plane_regression.json)、[进程结果](plane_regression.process.json)。性能数值只作记录，不宣称本阶段已完成大规模性能预算。

## 复跑方式

从仓库根目录使用 quadruped_rl 解释器：

```powershell
python -m pytest tests/test_terrain_signals.py tests/test_mujoco_terrain_signals.py -q -p no:cacheprovider --basetemp temp/stage3_repeat
python tests/manual/verify_mujoco_terrain_training.py
python tests/manual/verify_mujoco_terrain_training.py --iterations 8 --command-speed 0.3 --output-dir tests/manual/results/stage3_mujoco/moving
```

短训练的环境数、回合上限、rollout 长度和日志回调均在 `tests/manual/verify_mujoco_terrain_training.py` 中定义，默认运行 16 次更新。运行时在 `--output-dir` 下保存 `environment.yaml`，前进指令模式另存 `moving_task.yaml`，供组件工厂读取和结果追溯；不再在业务 configs 目录维护独立 smoke 配置。平台外诊断放置也只存在于该脚本。

阶段 4 才实现 Isaac 静态地形及环境区域映射；阶段 5 才接入正式地形课程、检查点与评估恢复。阶段 3 没有修改已有 simulator、VectorEnv、runner 或 PPO 业务结构。


## 地面物理状态归属（2026-11-08）

三个地面字段已从 TaskContext 迁入 SimulatorState。simulator 负责物理范围、高度及有效性采集，task 负责参考系选择、奖励屏蔽及越界终止规则。已删除 BaseTaskLogic 的 simulator 引用和主动查询，也删除过渡性的 randomization 上下文接口。测试覆盖不同环境原点、乱序/空子集、可选字段 clone/select，以及自动重置后的终止前 ground_valid 保留。

本次实际 PPO 复验采用 0.3 m/s 指令、4 次更新（4,096 条转移），正常结束；记录位于 temp/ground_state_training_20261108。16 环境平地八项基线完全一致，进程退出码 0。

本次最终回归：MuJoCo/core 981 项通过、11 项跳过；Isaac/core 942 项通过、11 项跳过。两个解释器对应的 Pyright 检查均为 0 errors、0 warnings。

## 外围平地与可选越界终止（2026-11-08）

新增真实物理验证：外围四边、四角及远处的高度查询与碰撞射线一致；探针落地稳定，八个方向跨越接缝不坠落；外围碰撞体沿用地面的摩擦等接触参数，不覆盖内部负高度地形。保留添加外围前的场景统计尺度，避免大型碰撞体放大默认相机和数值尺度。真实 OpenGL 渲染通过。

任务集成覆盖启用 TerrainBounds 后仅重置越界环境、关闭后继续在外围运行、机身在内部而部分足端在外围时逐点采样，以及终止前 terrain_in_bounds 不被局部重置覆盖。公共测试覆盖无地形、乱序子集、空子集、非有限坐标和 clone/select。

最终回归：MuJoCo/core 1006 项通过、11 项跳过；Isaac/core 946 项通过、11 项跳过。两套 Pyright 均无错误或警告。实际 PPO 使用 0.3 m/s 指令、4 次更新（4096 条转移），验收通过，记录位于 temp/peripheral_training_20261108；本次未启动 Isaac 非平地应用。
