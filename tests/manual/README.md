# 手动诊断脚本

这里的文件是按需运行的性能和故障排查工具，不是常规 pytest 用例。请从项目根目录运行，并先选择对应的 Python 环境。性能结果受硬件、系统负载和运行参数影响，不应当作固定的测试通过阈值。

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `profile_mujoco_state.py` | MuJoCo | 分析 `MujocoSimulator.get_state()` 及各状态字段的耗时占比。 |
| `benchmark_mujoco_step_threads.py` | MuJoCo | 比较不同环境数和工作线程数的仿真步进吞吐量。 |
| `isaac_shutdown_diagnostic.py` | Isaac Sim | 分阶段复现 Isaac Sim 启动、场景构建与关闭时的问题。 |
| `verify_isaac_simulation_app_restart.py` | Isaac Sim | 验证同一 Python 进程能否连续启动和关闭 `SimulationApp`。 |
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

## Isaac Sim 无相机训练验证

下面的脚本使用独立的 128 环境配置，运行一个真实的 standing PPO 更新，分别统计物理步进、环境步进和 PPO 更新。模拟器配置关闭 Fabric 变换和速度输出，不创建相机：

```powershell
python tests/manual/train_isaac_without_camera.py *> isaac_headless_train.log
```

出现 `CAMERA-FREE TRAINING PASSED` 表示训练阶段已完成。结果会保存在独立的 `checkpoints/unitree_go1_isaac_cuda_headless_train_test_*` 目录；请同时检查终端日志和目录内的 `logs/training.log`。

使用该测试生成的 checkpoint 验证独立相机播放。`--train-time` 填输出目录末尾的时间戳：

```powershell
python tests/manual/play_isaac_headless_checkpoint.py --app-name unitree_go1_isaac_cuda_headless_train_test --train-time 2026-09-26_19-15-18 --render-mode rgb_array --num-steps 50 *> isaac_camera_playback.log
```

正式训练仍使用原有入口，选择 `unitree_go1_isaac_cuda_velocity`。`application.play()` 会在播放时检查环境的 render mode：无渲染模式时直接跳过。训练结束后运行上面的独立脚本播放 checkpoint；脚本会在新的 Isaac Sim 进程中调用原来的 `application.play()`。`--render-mode none` 跳过播放，`--render-mode rgb_array` 创建相机并生成视频。

## Isaac Sim 同进程重启验证

在安装 Isaac Sim 的设备上，从项目根目录运行：

```powershell
python tests/manual/verify_isaac_simulation_app_restart.py *> isaac_simulation_app_restart.log
```

脚本在同一个 Python 进程中连续两次启动、更新、关闭 `SimulationApp`，每步输出时间和阶段。出现 `ISAAC SIMULATIONAPP SAME-PROCESS RESTART PASSED` 表示最小生命周期测试通过；若卡住或崩溃，请保留完整日志和进程退出码。通过此测试仍不能证明 World、相机和项目环境能在同一进程内重建，后续需分别验证。
