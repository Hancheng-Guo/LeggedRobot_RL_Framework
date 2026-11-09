# 阶段 3：用户审核后最终复核

日期：2026-11-09，Asia/Shanghai。

结论：阶段 3 的功能与验收目标已完成。本轮发现并修复两处明确问题，修正后的最终状态通过完整回归、真实 ANYmal 物理/渲染、站立与前进 PPO 短训、平地基线及两侧类型检查。没有扩展到 Isaac 随机地形、正式地形课程或恢复训练。

## 本轮发现及最小修正

| 问题 | 修正及证据 |
| --- | --- |
| 诊断任务的 YAML 导出默认按字母排序，改变地形类型的采样区间顺序和观测拼接顺序 | 使用 `sort_keys=False`；临时测试配置同步保留顺序。原回归测试补充与业务源配置的类型/观测顺序比较，修复前失败、修复后通过 |
| 地形模型初始化将整个 `hfield_data` 写成 0.5，覆盖模型原有的其他高度场资源 | 按新增高度场的地址和长度，仅初始化对应缓冲片段；新增带其他高度场碰撞资源的模型测试，修复前原数据全部被覆盖、修复后保持原值 |

业务代码的本轮修改仅涉及高度场初始化范围；另一项是验收脚本与测试配置的序列化。未更改配置语义、生成公式、种子规则、状态版本 5，也未改动用户暂存区。

## 验收结果

| 检查 | 最终结果 |
| --- | --- |
| MuJoCo/core 全部测试，包含真实 OpenGL 测试 | 1044 passed、11 skipped、47 deselected |
| Isaac/core 全部测试 | 980 passed、11 skipped、111 deselected |
| MuJoCo 与 Isaac Pyright | 均为 0 errors、0 warnings |
| ANYmal 三种地形、足端支撑、地面高度、局部 reset | 通过 |
| 实际渲染及变更后高度场上传 | 通过，平地/坡面/起伏与变更后图像已查看 |
| 普通 reset 地形复用、局部 reset 隔离 | 通过 |
| renderer、相机、OpenGL 上下文复用 | 通过 |
| ANYmal 固定零速度诊断短训 | 16 次 PPO 更新，16384 条转移，通过 |
| ANYmal 0.3 m/s 诊断短训 | 8 次 PPO 更新，8192 条转移，通过 |
| 16 环境平地回归 | 八项与阶段 1 完全一致，进程退出码 0 |
| `git diff --check` | 通过 |

两次短训均检测实际非平地样本、全足接触、有限观测/奖励/优化指标及参数更新；无出生后一步即终止，关闭流程正常返回。诊断任务只在输出目录中覆盖固定命令，现有 ANYmal 业务任务的 LPAC 课程不变。

平地兼容结论针对地面几何、重置及高度参照；用户明确授权的 xyz 足端运动惩罚及 MuJoCo 角速度参考系修正保留其行为变化，不宣称所有旧奖励逐位不变。

## 数据与复跑

[结构化结果及受检代码 SHA-256](final_verification_20261109.json) 记录最终测试结果、训练指标、渲染检查及源文件指纹。

原始数据位于仓库 `temp/stage3_audited_final_20261109`：

- `verified_standing/training_verification.json`
- `verified_moving/training_verification.json`
- `verified_render/verification.json` 及渲染 PNG
- `plane.json`、`plane.process.json`、`plane_comparison.json`

```powershell
# 使用 quadruped_rl 解释器
python -m pytest -m "core or mujoco" tests -q --basetemp temp/stage3_audited_repeat_mujoco -p no:cacheprovider
python tests/manual/verify_mujoco_terrain.py --render --output-dir temp/stage3_audited_repeat/render
python tests/manual/verify_mujoco_terrain_training.py --iterations 16 --output-dir temp/stage3_audited_repeat/standing
python tests/manual/verify_mujoco_terrain_training.py --iterations 8 --command-speed 0.3 --output-dir temp/stage3_audited_repeat/moving

# 使用 quadruped_isaacsim 解释器
python -m pytest -m "core or isaacsim" tests -q --basetemp temp/stage3_audited_repeat_isaac -p no:cacheprovider
```

## 范围与已接受边界

Isaac 本轮验证的是公共接口及 runtime 单元回归，不代表真实 Isaac 非平地应用已验收。PPO 短训验证训练链路与信号健康，不代表收敛或最终行走能力。外围每侧 1000 米仍有外边界，TerrainBounds 可选；`max_slope` 约束采样系数而非实际合成坡度；非法接触仍按接触条目计数。这些均为当前已确定的行为，不作为本轮新增错误。
