# 阶段 1：统一地形契约与平地基线

## 范围

本阶段交付公共配置、参考几何数据、分配生命周期和基线。两个后端仍只接受 `plane`，对 `slope`、`rough` 明确报错。参考几何生成器不创建碰撞；MuJoCo 和 Isaac 的非平地链路分别属于阶段 2、4，课程和训练检查点接线属于阶段 5。

## 公共语义（terrain-v1）

入口为 `envs/simulators/terrain.py`。所有长度为米，坡度为 rise/run，局部地形中心为 `(0,0)`。`patch_size` 是 x/y 全长，`resolution` 是最大网格间距；每轴使用 `ceil(length/resolution)` 个等长网格。高度行索引是 y，列索引是 x；碰撞和查询都必须沿每格左下到右上的对角线划分三角形。

`types` 是 plane/slope/rough 的不重复序列；`weights` 是对应的正采样权重，内部归一化。`difficulty_levels` 是 [0,1] 内的不重复等级值，默认 `[1]`；难度同时缩放 `max_height` 和 `max_slope`。`seed` 非空时覆盖运行种子。固定配置、种子、patch ID、难度索引产生相同的类型、坡度和高度数据，与后端和调用顺序无关。

- plane：所有顶点高度为 0。
- slope：沿 x 方向，坡度在 `[-max_slope*difficulty,+max_slope*difficulty)` 均匀采样；中央平台外连续延伸。
- rough：各网格顶点由标签哈希独立采样，高度在 `[-max_height*difficulty,+max_height*difficulty)`；`max_slope` 不约束 rough 的局部梯度。
- `spawn_size` 默认 `[0.5,0.5]`，是中央 z=0 平台的保证区域；向外多保留一格的零高度顶点，保证与出生矩形相交的三角形均平坦。`spawn_margin` 是边缘预留距离，出生矩形须严格容纳在预留后的区域内。后续后端必须另行验证机器人足迹能放入该区域；超出区域的随机出生点不受安全保证。

随机源使用版本化 BLAKE2b 标签派生 53 位 uniform。不会消费 Python、NumPy 或 Torch 的训练随机流。改变采样规则必须升级版本，不能沿用 v1 恢复旧状态。

## 分配、reset 和恢复契约

`TerrainAssignments` 是后端无关的参考分配器。建场景前创建一次，初始 generation=0、难度索引=0。逻辑 patch ID 为 `generation*num_envs+env_id`。局部 reset 只递增被选中环境的 generation，输入顺序不影响结果。难度由调用方在 reset 边界传入；本阶段不决定课程策略。难度更新和分配发生在仿真出生姿态设置之前，任务管理器 reset 之前。普通物理步进和状态读取不得改变分配。

静态地形后端可将逻辑分配映射到预建地形块，不要求回合中重建碰撞。该映射、世界原点和有效出生姿态在后端实现时也必须保存；此处的 reference patch ID 不是现成的 Isaac 区域索引。

`state_dict()` 保存 version、完整配置、有效种子、环境数、各环境 generation 和难度索引；状态支持 JSON 往返。恢复先校验全部字段再更新，配置、种子、环境数不同应报错，失败不能部分写入。已保存分配可复现当前网格及下次 reset 结果。分配器无需额外保存随机状态。阶段 5 接入 Runner 时，还须保存课程进度、预建区域映射及出生采样状态；现有 VectorEnv 检查点尚未保存这些数据。旧检查点仅在平地模式兼容；非平地不得悄悄重新抽样。

## 地面高度接口

`BaseSimulator.get_ground_heights(xy, env_ids=None)` 输入世界坐标 `[环境,采样点,2]`，输出同设备、同 dtype 的世界 z `[环境,采样点]`。子集按 env_ids 顺序对应行。机身位置可用单点查询，足端可一次查询全部足点。平地输出零；后续非平地后端必须先转到各 patch 局部坐标，并按上述碰撞三角形查询。patch 外行为和区域映射由后端接入时显式定义，不得返回伪造的零高度。

## 可复跑的平地验收

从仓库根目录，在对应解释器中运行：

```powershell
python tests/manual/record_plane_baseline.py mujoco --output tests/manual/results/stage1_mujoco_plane.json
python tests/manual/record_plane_baseline.py isaac_sim --output tests/manual/results/stage1_isaac_plane.json
```

固定 seed=0，16 环境，standing 任务，0.002s 物理步长，每控制步 10 子步；预热 10、测量 50 控制步。记录观测和状态形状、各足接触数/法向力、平地查询、完整 reset 和局部 reset。性能计时同步 CUDA，覆盖 VectorEnv.step（含物理、状态、任务及自动 reset），不含 PPO 和渲染。MuJoCo 配置保留 rgb_array 能力但测量期间不调用 render；Isaac 使用 headless 配置。

结果中的 `BASELINE MEASUREMENTS PASSED` 只代表业务验收；`BASELINE CLEANUP RETURNED` 及最终进程退出码另行判断清理。不得用强制退出掩盖 Isaac 清理异常。

脚本默认由父进程监督，超时 180 秒（可用 `--timeout` 调整），覆盖启动、测量和关闭。原始测量保存在指定 JSON；同名 `.process.json` 单独记录子进程退出码、是否超时、是否写出本轮测量。超时时父入口返回 124，不能算进程成功。

## 2026-11-04 实测

Windows Server 2025，AMD EPYC 9554，NVIDIA RTX 4080 16 GiB，驱动 581.57。使用 quadruped_rl / quadruped_isaacsim 解释器，CPU 与 GPU 设备不同；本表是各自回归参照，不是后端等价性能比较。原始数据见本目录两个 `stage1_*_plane.json`。

| 项目 | MuJoCo CPU | Isaac Sim CUDA |
| --- | ---: | ---: |
| 环境数 | 16 | 16 |
| 初始化 | 1.274 s | 27.349 s |
| 观测形状 | [16,112] | [16,112] |
| 环境步进 mean | 11.615 ms | 232.345 ms |
| 环境步进 p95 | 14.156 ms | 286.932 ms |
| 吞吐 | 1377.49 env steps/s | 68.86 env steps/s |
| 平地高度查询 | 全零 | 全零 |
| 局部 reset 其余环境状态 | 不变 | 不变 |
| reset 回到初始 qpos | 通过 | 修复默认姿态别名后通过 |

50 个测量步内，两后端的四只足在全部 16 个环境均有接触。末帧 env_0 的法向力分别为 MuJoCo `[29.20,29.73,32.81,33.28]` N、Isaac `[29.71,30.07,34.27,34.95]` N；原始数组保存在 JSON。这是各自平地站姿的回归记录，不要求两物理引擎逐位一致。

Isaac 的 `_capture_default_state()` 原先可能引用实时 Warp/PhysX 存储，物理步进会改变 reset 目标。本次改为克隆根位置、朝向和关节位置，并通过真实 reset 回归和单元测试确认。此修复保持配置定义的默认姿态语义。

额外运行原有 Isaac profiler（16 环境、预热 3、测量 20、无相机、Fabric 输出关闭），物理步进 mean=99.45 ms，状态读取 mean=8.66 ms，约 148 env steps/s。它不含任务管理器，不能与上述完整环境步进数值混用；数值也受机器负载影响。

已知限制：Isaac 业务测量完成后，现有关闭流程仍可能停在 `BASELINE CLEANUP BEGIN`，本次按超时监督记录为进程失败。该问题在原项目文档已有记录；阶段 1 的性能与行为数据有效，但不代表 Isaac 生命周期问题已解决。后续大规模性能预算和关闭流程修复需单独验证。

本轮监督结果：MuJoCo 子进程正常退出 0；Isaac 测量写出后关闭超过 90 秒总时限，监督入口终止子进程并返回 124（子进程结束码 1）。详见对应 `.process.json`。

## 检查结果

- MuJoCo：Pyright 0 错误，pytest 845 passed / 11 skipped / 43 deselected。
- Isaac：Pyright 0 错误，pytest 863 passed / 11 skipped / 25 deselected。
- 新增用例覆盖参考网格确定性、黄金值、种子覆盖、难度和出生区域校验、局部 reset 分配隔离、检查点 JSON 往返及后续采样、失败恢复原子性、全局 RNG 隔离，以及 Isaac 默认姿态存储独立性。
- Isaac 最终 pytest 显式使用工作区临时目录，避免 Windows 默认 pytest 临时目录访问权限导致的 fixture 建立失败；命令为 `python -m pytest -m 'core or isaacsim' tests -q -p no:cacheprovider --basetemp temp/stage1_pytest_isaac_final`。重复运行时换用新的临时目录名称。

阶段 1 的统一契约及平地基线已完成。非平地碰撞、训练信号适配、课程和正式检查点恢复仍按阶段 2–5 继续，不计入本阶段完成范围。
