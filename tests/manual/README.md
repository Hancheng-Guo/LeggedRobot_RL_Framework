# 手动诊断脚本

这里的文件是按需运行的性能和故障排查工具，不是常规 pytest 用例。请从项目根目录运行，并先选择对应的 Python 环境。性能结果受硬件、系统负载和运行参数影响，不应当作固定的测试通过阈值。

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `profile_mujoco_state.py` | MuJoCo | 分析 `MujocoSimulator.get_state()` 及各状态字段的耗时占比。 |
| `benchmark_mujoco_step_threads.py` | MuJoCo | 比较不同环境数和工作线程数的仿真步进吞吐量。 |
| `isaac_shutdown_diagnostic.py` | Isaac Sim | 分阶段复现 Isaac Sim 启动、场景构建与关闭时的问题。 |
| `profile_isaac_sim_runtime.py` | Isaac Sim | 测量项目运行时的初始化、物理步进、状态读取与可选的相机渲染。 |

## MuJoCo 性能分析

```powershell
python tests/manual/profile_mujoco_state.py --num-envs 32 --repeats 200
python tests/manual/benchmark_mujoco_step_threads.py --num-envs 16,32,64 --workers 1,2,4,8
```

两个脚本都输出 CSV 格式的计时结果。前者可用 `--warmup` 调整预热次数；后者可用 `--frame-skip`、`--warmup-steps`、`--measured-steps` 和 `--repeats` 调整基准测试。运行 `python <脚本路径> --help` 可查看完整参数。

## Isaac Sim 关闭诊断

```powershell
python tests/manual/isaac_shutdown_diagnostic.py --mode baseline
python tests/manual/isaac_shutdown_diagnostic.py --mode world
python tests/manual/isaac_shutdown_diagnostic.py --mode model --num-envs 16 --steps 2
```

建议从 `baseline` 开始，再依次尝试 `bridge-before`、`bridge-after`、`world` 和场景相关模式，以定位触发问题的步骤。场景模式会使用默认的 Unitree Go1 USD 模型；可通过 `--model-path` 指定其他模型。脚本正常完成时输出 `DIAGNOSTIC COMPLETED SUCCESSFULLY`。运行 `--help` 可查看所有模式和参数。

## Isaac Sim 性能分析

在安装 Isaac Sim 的设备上，从项目根目录运行：

```powershell
python tests/manual/profile_isaac_sim_runtime.py --num-envs 16 --steps 5
python tests/manual/profile_isaac_sim_runtime.py --num-envs 64 --steps 5
```

脚本默认不创建相机。需要单独量化相机成本时，为相同环境数加上 `--with-camera`。每组输出 `step`（包含配置中的 10 个物理子步）、`get_state` 和可选的 `render` 耗时；请比较预热后的汇总数据。脚本不执行 PPO 更新，因此吞吐量仅代表仿真与状态读取。

验证 Fabric 输出的同步开销时，对相同环境数运行 `--disable-fabric-output`。此选项仅在无相机模式下可用，会在创建 Isaac Sim 应用后同时关闭 Fabric 的变换和速度发布；它不会修改正常训练配置。

若初始化停滞，查看最后一条 `Starting ...` 阶段日志。终端输出过长时可保存完整输出：

```powershell
python tests/manual/profile_isaac_sim_runtime.py --num-envs 16 --steps 5 *> isaac_profile.log
```

之后用 `Get-Content isaac_profile.log -Tail 80` 查看末尾，或直接提供日志文件。
