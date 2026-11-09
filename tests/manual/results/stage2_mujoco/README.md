# 阶段 2：MuJoCo 地形链路验收

验收日期：2026-11-07（Asia/Shanghai，按当前代码复验）。本阶段的完成标准已满足：多个环境同时使用不同地形，机器人能在平地、坡面和起伏地形上形成正确地面接触，局部 reset 无串扰，默认平地回归通过。

## 实现

- `envs/simulators/mujoco_terrain.py` 将阶段 1 的 `TerrainPatch` 转为 MuJoCo 原生 heightfield。原 floor geom 在原位置替换，名称、ID、摩擦、材质和接触配置保留。各环境的模型和高度缓冲独立，reset 不重新编译碰撞模型。
- 使用固定高度包围范围，在 reset 边界更新高度缓冲。直接写入编译后的 `hfield_data`，避免 XML 自动归一化与行方向转换改变公共网格。MuJoCo 数据和公共网格均按 y 递增存储，沿左下到右上划分三角形。
- `get_ground_heights(xy, env_ids)` 对实际碰撞缓冲的 float32 高度值做同样的三角形插值，保留输入 Tensor 的设备和 dtype，支持机身、足端及环境子集查询。地形之外和非有限坐标明确报错。
- task 的 `RandomizationManager` 创建并拥有 `TerrainRandomizationTerm`，由该项完成地形分配、难度和出生采样。普通回合 reset 复用已有地形，不推进分配代数；仅首次使用或请求的难度变化时，为对应环境生成新地形并在成功提交后推进代数。后端重新绑定后重新应用缓存网格，不重新采样；普通步进不改变地形。可通过 `task.randomization_manager.terms["terrain"].patches` 查看当前分配（未首次 reset 的环境为 `None`）。`VectorEnv` 在其他任务状态 reset 之前调用 task 的物理随机化钩子；`simulator.reset()` 本身只恢复物理状态。
- 仿真器提供 reset 姿态中全部机器人碰撞形状的相对 AABB，管理器绑定后端时校验其是否适配 `spawn_size`。确定性出生采样仅在能够完整容纳机器人投影的范围内进行，并按高度场高度调整根位置，必要时抬升到至少 2 mm 的保守初始碰撞间隙。安全区校验、间隙和随机位置选择均不在后端实现中。
- 自审中发现高度场的多个三角形接触会分摊足部支撑力，逐点阈值可能误报腾空。新地形路径改为先汇总每只足与地面的法向力，再应用阈值；旧平地路径维持原来的接触规则，保证回归兼容。
- 可视窗口更新高度场时使用 viewer 锁和 `update_hfield()`；离屏渲染器保持上下文和相机，由 `simulator.playback_env_index` 指定当前展示环境（默认 0）；该环境的地形改变后标记待上传，下次绘制前使用 `mjr_uploadHField()` 更新高度场。该调用使用 MuJoCo 3.11 Renderer 的 `_gl_context` / `_mjr_context`（当前无公开上传方法），仅限后端内部。窗口渲染、离屏渲染、viewer 加锁和地形更新判断均使用同一播放索引，其他环境的地形变化不触发当前渲染器上传。

使用 heightfield 的原因及数据规范可参照 [MuJoCo 官方高度场说明](https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-hfield)。普通 MuJoCo mesh 的碰撞凸包不适合直接承载这些起伏网格。

## 运行

从仓库根目录，使用 `quadruped_rl` 解释器：

```powershell
python tests/manual/verify_mujoco_terrain.py --render
```

本历史记录运行 3 个独立 Go1 环境，`sim_dt=0.002`、`frame_skip=10`、3 个工作线程、固定地形 seed=0。第一次 reset 后三个环境分别是 plane/slope/rough，patch ID 为 3/4/5，初始难度值均为 0.25。当前手工脚本已改用 `configs/simulators/mujoco_anymal_c_rgb_array.yaml` 与 `configs/tasks/locomotion_anymal_c_phase.yaml`；旧 Go1 地形任务已删除。

先在安全出生区运行 300 个控制步，再将机器人移动到 x=1.2 m、y=0 的出生区外，按照各足下方高度放置并运行另 300 个控制步。维持 home 位置控制目标，不调用任务管理器或 PPO。验证所有状态有限、机身保持合理离地高度、四足在稳定阶段均出现接触，且最终四足均超过力阈值。随后执行一次普通完整 reset，核对地形对象、高度缓冲及分配代数保持不变，且不产生地形上传请求。最后仅重置示例播放环境 env_0 并请求最高难度，核对 env_1/2 的地形和姿态完全不变。渲染模式下若采样仍为 plane，则交替请求不同难度直至实际几何改变，避免把普通 reset 当作重新采样操作。

本次脚本退出码为 0，输出 `MUJOCO TERRAIN VERIFICATION PASSED`。原始结果见 [verification.json](verification.json)。出生区外最终足下高度（FR、FL、RR、RL，米）如下，确认坡面和起伏用例确实位于非平坦区域：

| 地形 | FR | FL | RR | RL |
| --- | ---: | ---: | ---: | ---: |
| plane | 0 | 0 | 0 | 0 |
| slope | 0.007845 | 0.007832 | 0.004116 | 0.004114 |
| rough | 0.004538 | -0.000370 | 0.001149 | -0.007005 |

## 自动验证与平地回归

- MuJoCo、Isaac 两套 Pyright：均为 0 errors / 0 warnings。
- MuJoCo / 公共 pytest：938 passed、11 skipped、44 deselected。
- Isaac / 公共 pytest：908 passed、11 skipped、74 deselected；这是测试回归，没有重新启动 Isaac Sim 应用。
- 地形后端及公共策略测试覆盖三类独立地形、600 个随机/边界高度点与 MuJoCo 原生射线逐点比对（绝对容差 1e-8 m）、出生碰撞包围盒、非平坦区域真实接触探针、串并行步进一致性、局部 reset 完整积分状态隔离、固定种子与 reset 顺序、难度更新、失败更新原子性、平移地面、非法地面及高度场上传。额外验证了后端 reset 不采样、应用网格不重置机器人、非法网格批量更新的原子校验、task 持有随机化状态与配置不变时保留状态、真实 VectorEnv 完整/回合结束 reset 经过 task 随机化，以及切换到普通 task 后清除旧高度场。补充覆盖普通 reset 不生成/应用地形、相同难度请求不推进代数、不同难度仅更新所选环境、后端重新绑定时应用缓存网格，以及非零播放索引下的渲染、地形上传和 viewer 锁。
- 恢复训练目录碰撞测试 `test_failed_archive_does_not_publish_or_switch_run` 固定 `app.application_entry.datetime.now()`，让首次失败与再次尝试始终使用相同目录时间，并检查 manifest 的 `run_time`。该测试不再依赖两次操作发生于真实时钟的同一秒；本轮两套完整回归均通过。仅修改测试，不改变恢复训练的业务行为。
- 增加依赖方向检查：simulator 代码不得导入 task 层；task 随机化代码不得导入 MuJoCo、Isaac 或具体后端实现，只可依赖公共物理契约。
- 使用原阶段 1 平地测量脚本重新运行 16 环境，正常关闭且退出码为 0。与阶段 1 的观测形状、状态形状、初始接触、50 步各足接触数、最终法向力、reset 隔离、reset 可重复性和零高度输出逐项比较，全部完全一致。见 [比较结果](plane_comparison.json)、[原始测量](plane_regression.json)及[进程状态](plane_regression.process.json)。耗时受机器负载影响，不要求逐位相同。

复跑测试时可指定新的工作区临时目录，避免 Windows 默认 pytest 临时目录权限问题：

```powershell
python -m pytest -m 'core or mujoco' tests -q -p no:cacheprovider --basetemp temp/stage2_validation_new
```

## 支持范围和后续边界

- 非平地模式要求明确指定一个世界坐标轴对齐、挂在 world body 下、启用碰撞的 plane 作为替换目标。不支持额外 plane、倾斜地面和非平凡父 frame；这些情况会在构建时失败。
- 新地形有有限边界，`patch_size` 外没有隐藏的备用平面。高度查询超出边界会报错；阶段 3 接入地形信号时需相应处理越界终止。
- 纯 `types: {plane: {}}`（以及不配置 terrain）保持原有无限平面和固定 reset 行为。混合分布中的 plane 是有限的平坦地形块，会参与分配和出生采样。
- `spawn_size` 必须适配机器人完整碰撞包围盒，Go1 示例使用 `[1,1]`。更大的机器人或不同 reset 姿态可能需要扩大区域。场景文件及原机器人资产未被修改，地形数据在内存中生成。
- 本轮实际渲染复验（2026-11-07T13:23:00Z，UTC）成功创建 WGL/OpenGL 上下文，完整 `python tests/manual/verify_mujoco_terrain.py --render` 退出码为 0。示例使用默认播放索引 0：脚本确认普通 reset 复用地形、不产生上传请求；难度更新后实际高度缓冲改变、reset 前后图像不同、renderer / camera / OpenGL context 对象均保持相同、上传标记已清除，且 env_1/2 地形与姿态保持不变。`verification.json` 中相关标记与 `passed` 均为 true；下方图片为本轮生成结果。非零播放索引另由参数化测试验证。之前 WGL 初始化失败的记录保留在 [渲染状态](render_validation.json) 的 `previous_attempt` 中，本轮未复现。
- 阶段 2 仅验收仿真链路。任务离地高度信号与短训练现已在 [阶段 3](../stage3_mujoco/README.md) 完成；Isaac 非平地仍属于阶段 4，课程、正式检查点和评估恢复仍属于阶段 5。

## 实际渲染图像

- [平地全景](terrain_0_plane.png)、[坡面全景](terrain_1_slope.png)、[起伏地形全景](terrain_2_rough.png)。
- [reset 前](before_reset.png)、[reset 后](after_reset.png)、[reset 后地形全景](terrain_after_reset.png)。

## 职责划分调整（2026-11-05）

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| `envs/simulators/terrain.py` | `TerrainGeometry`、`TerrainPatch`、`TerrainSpawnGeometry` 和 `TerrainBackend` 物理接口 | task 策略与采样 |
| task 的 `RandomizationManager` | 通过注册表创建并调度随机化项，在物理 reset 前后调度准备、应用和提交/回滚 | MuJoCo/Isaac 专有 API |
| `randomization/terms/terrain.py` 中的 `TerrainRandomizationTerm` | 地形配置、参考生成、类型和难度选择、分配代数、安全出生校验与采样 | MuJoCo/Isaac 专有 API |
| `VectorEnv` | 向 task 取得碰撞布局需求，在前后钩子之间执行唯一一次物理 reset，再清理任务状态 | 随机化策略和分配状态 |
| `MujocoSimulator` | 根据给定布局建场景，应用网格、重置物理状态、测量碰撞包围盒、执行给定根位置 | 随机采样、难度进度、分配状态 |
| `MujocoTerrain` | 高度场缓冲、网格校验和三角形高度查询 | 类型权重、种子、难度、出生规则 |

地形配置迁移到 **task YAML** 的 `randomization_manager_config.terms.terrain`。仿真器配置接口不再接受旧的 `terrain` 参数，由后端接口自然拒绝不支持的参数；VectorEnv 不保留地形迁移特判。直接做物理诊断的调用方可显式组合相同的 task 随机化类：

```python
randomization = RandomizationManager(num_envs, context, terms={"terrain": terrain})
simulator.config_update(..., **randomization.get_simulator_requirements(terms={"terrain": terrain}))
randomization.bind(simulator)  # 只校验与绑定，不生成或应用地形
try:
    randomization.before_simulator_reset()  # 复用地形，或按需生成并应用候选地形
    simulator.reset()                       # 环境层负责物理 reset
    randomization.after_simulator_reset()   # 设置出生位置，全部成功才提交
except Exception:
    randomization.abort_simulator_reset()
    raise
```

旧的 `simulator.terrain_patches` 改为 `task.randomization_manager.terms["terrain"].patches`；难度更新移到 `task.randomization_manager.terms["terrain"].set_difficulty(env_ids_tensor, difficulty_indices)`。task/随机化项不调用 simulator.reset，由 VectorEnv 在 task.before_simulator_reset 与 task.after_simulator_reset 之间执行物理 reset，然后单独清理 task 回合状态。公共代码及 task 随机化层只使用后端接口，没有导入任何 MuJoCo 或 Isaac 模块。空的 `randomization_manager_config: {}` 保留旧平地行为；配置不变时不会重建随机化状态。正式训练检查点接线仍属于阶段 5，此次没有引入课程策略。


## 本次自审修复（1–5、7）

1. 增加通用 `BaseRandomizationTerm` / 注册表；manager 不含 terrain 专用分支或属性。测试增加第二种 term，验证注册、配置、资源需求和失败回滚调度。
2. 空配置返回空需求，兼容不接收地形参数的旧后端。VectorEnv 只记录已应用资源键，复用同一后端时清除不再需要的资源；切换后端不携带旧资源清除参数。
3. 地形分配使用候选状态；网格、物理 reset 和出生设置全部成功后提交。失败恢复所选 MjData、高度缓冲及查询网格，分配代数、当前描述和待处理难度不被消耗。测试对完整与局部 reset 的三个失败阶段分别注入故障，逐项核对物理状态及重试结果。
4. bind 不生成或应用初始地形，首次 reset 才采样；保留既有首次 reset patch ID 3/4/5 的确定性行为。普通回合 reset 复用地形，仅难度变化才推进对应分配代数；重新绑定后端只重新应用缓存。
5. 后端 `TerrainPatch` 只包含 x / y / heights；任务 `TerrainAssignment` 持有元数据。完整碰撞 AABB 测量不再要求具名足，使用无足名球体模型验证出生过程。
7. 地形变化不销毁离屏 renderer；下次绘制上传高度场并保留相机。播放索引统一取 `simulator.playback_env_index`，不固定为 env_0。真实 WGL 渲染复验已通过，renderer / camera / OpenGL context 身份断言及画面更新均验证成功。

第 6 项接触语义差异按约定保留：平地按接触点筛选，高度场先按足汇总；两者现共用后续阈值判断和结果构造，既有数值语义不变。
