# 手动诊断脚本

这里的文件是按需运行的性能和故障排查工具，不是常规 pytest 用例。请从项目根目录运行，并先选择对应的 Python 环境。性能结果受硬件、系统负载和运行参数影响，不应当作固定的测试通过阈值。

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `profile_mujoco_state.py` | MuJoCo | 分析 `MujocoSimulator.get_state()` 及各状态字段的耗时占比。 |
| `benchmark_mujoco_step_threads.py` | MuJoCo | 比较不同环境数和工作线程数的仿真步进吞吐量。 |
| `isaac_shutdown_diagnostic.py` | Isaac Sim | 分阶段复现 Isaac Sim 启动、场景构建与关闭时的问题。 |
| `verify_isaac_simulation_app_restart.py` | Isaac Sim | 验证同一 Python 进程能否连续启动和关闭 `SimulationApp`。 |
| `verify_isaac_camera_activation.py` | Isaac Sim | 验证同一个 `SimulationApp` 中，物理步进后再创建相机并读取图像。 |
| `verify_isaac_go1_camera_activation.py` | Isaac Sim | 真实 Go1 训练后，在同一个应用和环境中启用相机并调用 `ApplicationEntry.play()`。 |
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

下面的脚本使用 `tests/manual/configs/unitree_go1_isaac_cuda_headless_train_test.yaml` 中独立的 128 环境测试参数，运行一个真实的 standing PPO 更新，分别统计物理步进、环境步进和 PPO 更新。测试辅助入口会临时装载该配置，并将实际配置归档进 checkpoint；不会向正式任务目录增加任务。训练时关闭 Fabric 变换和速度输出，不创建相机：

```powershell
python tests/manual/train_isaac_without_camera.py *> isaac_headless_train.log
```

出现 `CAMERA-FREE TRAINING PASSED` 表示训练阶段已完成。结果会保存在独立的 `checkpoints/unitree_go1_isaac_cuda_headless_train_test_*` 目录；请同时检查终端日志和目录内的 `logs/training.log`。

使用该测试生成的 checkpoint 验证独立相机播放。`--train-time` 填输出目录末尾的时间戳：

```powershell
python tests/manual/play_isaac_headless_checkpoint.py --app-name unitree_go1_isaac_cuda_headless_train_test --train-time 2026-09-26_19-15-18 --render-mode rgb_array --num-steps 50 *> isaac_camera_playback.log
```

正式训练仍使用原有入口，选择 `unitree_go1_isaac_cuda_velocity`。Isaac 模拟器的 `train_render_mode` 默认是 `None`，训练时不创建相机并关闭 Fabric 输出；`render_mode` 仅在 `application.play()` 时生效。设置为 `rgb_array` 时，播放在原有 headless `SimulationApp` 中启用 Fabric 输出、创建相机并生成视频。设置为 `human` 时，若训练应用以 headless 启动，播放会明确报错，需另起非 headless 进程。上面的独立 checkpoint 脚本仍可用于单独验证播放。

## Isaac Sim 同进程重启验证

在安装 Isaac Sim 的设备上，从项目根目录运行：

```powershell
python tests/manual/verify_isaac_simulation_app_restart.py *> isaac_simulation_app_restart.log
```

脚本在同一个 Python 进程中连续两次启动、更新、关闭 `SimulationApp`，每步输出时间和阶段。出现 `ISAAC SIMULATIONAPP SAME-PROCESS RESTART PASSED` 表示最小生命周期测试通过；若卡住或崩溃，请保留完整日志和进程退出码。通过此测试仍不能证明 World、相机和项目环境能在同一进程内重建，后续需分别验证。

## Isaac Sim 运行中启用相机

此测试只启动一次 `SimulationApp`。它先在 headless、关闭 viewport 更新且不创建相机的 World 中步进，再启用 Fabric 变换和速度输出，在现有 World 中创建相机并读取图像：

```powershell
python tests/manual/verify_isaac_camera_activation.py *> isaac_camera_activation.log
```

出现 `ISAAC CAMERA ACTIVATION PASSED` 且 `CLEANUP: SimulationApp closed` 才表示相机切换与清理均完成。若卡住或报错，请提供完整日志。此测试不加载机器人或 PPO checkpoint；通过后仍需验证项目训练环境的相同切换。

## Go1 训练后同应用播放

下面的脚本使用 `unitree_go1_isaac_cuda_headless_train_test` 测试配置完成一次 PPO 训练，然后连续两次调用 `ApplicationEntry.play()`；每次播放都会在原来的 World 中启用相机并生成 GIF，结束后销毁相机、恢复 Fabric 设置，期间不关闭或重启 `SimulationApp`：

```powershell
python tests/manual/verify_isaac_go1_camera_activation.py *> isaac_go1_camera_activation.log
```

出现 `ISAAC GO1 SAME-APP CAMERA PLAYBACK PASSED` 表示训练、相机启用、播放、GIF 生成和应用关闭均完成。若失败，请提供完整日志及最后一条 `PHASE` 输出。

## 无相机训练的 viewport 性能对比

Go1 同应用播放验证通过后，可比较当前 headless 默认设置与关闭 viewport 更新的设置。以下两条命令均使用同一 128 环境配置、无相机、Fabric 输出关闭，并各运行一次 PPO 训练；唯一差别是第二条在创建 `SimulationApp` 时设置 `disable_viewport_updates=True`：

```powershell
python tests/manual/train_isaac_without_camera.py *> isaac_viewport_default.log
python tests/manual/train_isaac_without_camera.py --disable-viewport-updates *> isaac_viewport_disabled.log
```

两份日志都应出现 `CAMERA-FREE TRAINING PASSED`。比较 `TRAINING TIMINGS` 中 `IsaacSimRuntime.step`、`IsaacSimRuntime.get_state`、`VectorEnv.step`、`PPO.update` 的 `calls` 和 `mean`，以及整体训练耗时。尽量在相同设备负载下交替运行两组至少两轮；每次运行在独立进程中完成，以免 Isaac 状态相互影响。这个开关只作用于手动脚本，不修改正式运行时配置。
