# 手动诊断脚本

这里的文件是按需运行的性能和故障排查工具，不是常规 pytest 用例。请从项目根目录运行，并先选择对应的 Python 环境。性能结果受硬件、系统负载和运行参数影响，不应当作固定的测试通过阈值。

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `profile_mujoco_state.py` | MuJoCo | 分析 `MujocoSimulator.get_state()` 及各状态字段的耗时占比。 |
| `benchmark_mujoco_step_threads.py` | MuJoCo | 比较不同环境数和工作线程数的仿真步进吞吐量。 |
| `isaac_shutdown_diagnostic.py` | Isaac Sim | 分阶段复现 Isaac Sim 启动、场景构建与关闭时的问题。 |
| `verify_isaac_shutdown_exit.py` | Isaac Sim | 在独立进程中对比正常/快速关闭及 Python 错误的最终退出码。 |
| `verify_isaac_simulation_app_restart.py` | Isaac Sim | 验证同一 Python 进程能否连续启动和关闭 `SimulationApp`。 |
| `verify_isaac_camera_activation.py` | Isaac Sim | 验证同一个 `SimulationApp` 中，物理步进后再创建相机并读取图像。 |
| `verify_isaac_go1_camera_activation.py` | Isaac Sim | 真实 Go1 训练后，在同一个应用和环境中启用相机并调用 `ApplicationEntry.play()`。 |
| `verify_isaac_go1_shutdown_exit.py` | Isaac Sim | 验证完整 Go1 流程经逐扩展关闭后显式退出，并检查播放异常的退出码。 |
| `diagnose_isaac_go1_playback.py` | Isaac Sim | 为 Go1 训练、播放及 Isaac 关闭过程增加阶段标记与定时调用栈。 |
| `profile_isaac_sim_runtime.py` | Isaac Sim | 测量项目运行时的初始化、物理步进、状态读取与可选的相机渲染。 |
| `diagnose_isaac_runtime_shutdown.py` | Isaac Sim | 在运行时性能入口外记录 `IsaacSimRuntime.close()` 的前后标记，隔离 Runner/PPO。 |
| `diagnose_isaac_simulator_shutdown.py` | Isaac Sim | 单独构建并关闭 `IsaacSimSimulator`，隔离 `VectorEnv` 与 Runner。 |
| `diagnose_isaac_vector_env_shutdown.py` | Isaac Sim | 单独构建 `VectorEnv` 与 Isaac simulator，跳过任务、Runner 和 ApplicationEntry。 |

## MuJoCo 性能分析

```powershell
python tests/manual/profile_mujoco_state.py --num-envs 32 --repeats 200
python tests/manual/benchmark_mujoco_step_threads.py --num-envs 16,32,64 --workers 1,2,4,8
```

两个脚本都输出 CSV 格式的计时结果。前者可用 `--warmup` 调整预热次数；后者可用 `--frame-skip`、`--warmup-steps`、`--measured-steps` 和 `--repeats` 调整基准测试。运行 `python <脚本路径> --help` 可查看完整参数。

## Isaac Sim 关闭诊断

先用下面的退出码矩阵确认这台设备的关闭行为。每个场景运行两次，脚本自动创建 `temp/isaac_shutdown_exit_时间戳/`，分别保存标准输出、错误输出和 `summary.json`。父脚本自身返回 0 仅表示矩阵执行完毕，**不能代表所有子进程都成功**；应查看汇总中的 `exit_code`、`timed_out` 和最后一条 `MARK`。单个子进程最长运行 360 秒，可用 `--timeout` 调整。

```powershell
python -u tests/manual/verify_isaac_shutdown_exit.py
```

六种场景依次为：`graceful-success`（逐扩展关闭）、`graceful-explicit-exit-success` 和 `graceful-explicit-exit-error`（逐扩展关闭返回后分别以 0 和 23 显式退出）、`fast-success`（快速关闭）、`fast-error-default`（故意抛错后以默认退出码快速关闭）、`fast-error-preserved`（故意抛错后显式要求退出码 23）。`fast-error-default` 若最终为 0，说明快速关闭会掩盖 Python 错误；两个 `*-error` 显式退出场景应为 23。`fast-*` 场景可能直接终止进程，因此缺少 `CLOSE_RETURNED` 标记是预期行为。`graceful-success` 若有 `CLOSE_RETURNED` 和 `CHILD_FINISHED`，但退出码为 `-1073741819`（`0xC0000005`），则故障发生在 `close()` 返回后的原生/解释器退出阶段。显式退出场景仅用于定位问题，不能直接代替项目的通用 `close()` 语义。

需要单独重跑某个场景时：

```powershell
python -u tests/manual/verify_isaac_shutdown_exit.py --cases graceful-success --repeats 1
```

如需对照 Windows 故障模块，可在运行后查看最近的 Application Error 事件（事件 ID 1000），按 `summary.json` 中的 PID 和时间匹配：

```powershell
Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=(Get-Date).AddHours(-2)} | Select-Object -First 10 TimeCreated, Message
```

确认这些结果后，再决定是否修改正式关闭路径，并重新运行 Go1 训练及两轮播放。不要只依据快速关闭的退出码 0 判定业务成功，还须检查训练和播放完成标记。

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

定位运行时本身的关闭时点时，可运行不含 Runner/PPO 的 128 环境对照，并检查 `RUNTIME SHUTDOWN: close returned`、`script finished` 与最终退出码：

```powershell
python -u tests/manual/diagnose_isaac_runtime_shutdown.py --num-envs 128 --warmup 0 --steps 5 --disable-fabric-output 1> temp/isaac_runtime_shutdown_stdout.log 2> temp/isaac_runtime_shutdown_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_runtime_shutdown.py --import-application-entry --num-envs 128 --warmup 0 --steps 5 --disable-fabric-output 1> temp/isaac_runtime_with_app_import_stdout.log 2> temp/isaac_runtime_with_app_import_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_simulator_shutdown.py --num-envs 128 1> temp/isaac_simulator_shutdown_stdout.log 2> temp/isaac_simulator_shutdown_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_simulator_shutdown.py --num-envs 128 --preallocate-cuda-tensor 1> temp/isaac_simulator_preallocated_stdout.log 2> temp/isaac_simulator_preallocated_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_vector_env_shutdown.py 1> temp/isaac_vector_env_shutdown_stdout.log 2> temp/isaac_vector_env_shutdown_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_vector_env_shutdown.py --defer-cuda-buffer 1> temp/isaac_vector_env_deferred_stdout.log 2> temp/isaac_vector_env_deferred_stderr.log
$LASTEXITCODE
python -u tests/manual/diagnose_isaac_vector_env_shutdown.py --use-runtime-context-factory 1> temp/isaac_vector_env_runtime_context_stdout.log 2> temp/isaac_vector_env_runtime_context_stderr.log
$LASTEXITCODE
```

进一步比较 `RuntimeContext` 初始化路径时，可对同一脚本组合使用 `--use-runtime-context-factory`、`--indexed-runtime-device`、`--fixed-runtime-seed`，或在直接构造路径使用 `--probe-device`、`--initialize-seed`、`--threads-before-seed`、`--unindexed-cuda`。这些选项只用于隔离设备表示、种子和初始化顺序，不修改正式配置。记录 `close begin`、`close returned`、`script finished` 及退出码；即使 `close()` 返回，进程仍可能以 `0xC0000005` 退出。当前设备上，工厂路径加 `cuda:0`、固定种子 0 的两次相同运行均在 `close begin` 后访问冲突；直接构造路径即使执行设备检查、设种子和先设线程数，也能让 `close()` 返回，但退出阶段仍访问冲突。这说明关闭阶段位置受初始化路径影响，尚不能归因于单一配置项。

`--trace-close-steps` 会标记 `World.stop`、`World.clear`、`SimulationApp.close` 及 SimulationManager 的卸载步骤。当前设备上，`SimulationManager._shutdown()` 和原生接口释放都已返回，随后 Kit 卸载仍发生 `0xC0000005`；Windows 应用事件指向 `isaacsim.core.simulation_manager.plugin.dll` 的 `0x1c5fe`。`--skip-native-release` 仅用于隔离故障：它使 `SimulationApp.close()` 返回，但进程退出时仍访问冲突，不可作为正式清理策略。现阶段不要因看到 `close returned` 就判定关闭问题已修复。

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
New-Item -ItemType Directory -Force temp | Out-Null
python tests/manual/verify_isaac_go1_camera_activation.py *> temp/isaac_go1_camera_activation.log
$LASTEXITCODE
```

出现 `ISAAC GO1 SAME-APP CAMERA PLAYBACK PASSED` 表示训练、相机启用、播放和 GIF 生成已完成；还需确认退出码为 `0`。原生扩展可能在该标记打印后、进程退出前崩溃。若失败，请提供完整日志、退出码及最后一条 `PHASE` 输出。

若初始化或关闭长时间没有新日志，可运行带阶段标记的同一流程。它会在模型转换、场景克隆、`World.reset()` 等步骤前后打印 `DIAG`，并在超过两分钟时输出 Python 调用栈：

```powershell
python -u tests/manual/diagnose_isaac_go1_playback.py 1> temp/go1_diagnostic.log 2> temp/go1_diagnostic_error.log
$LASTEXITCODE
```

查看两份日志的末尾以定位最后完成的阶段。`Timeout (0:02:00)!` 是诊断调用栈的标记，并不单独表示脚本失败；仍须以退出码判断进程是否正常结束。

若最小关闭矩阵显示崩溃只发生在 `close()` 返回后的进程收尾阶段，可用下面的诊断入口验证完整业务流程。它沿用现有的 `fast_shutdown=False` 和 Go1 测试，待 `ApplicationEntry` 完成清理并返回后，显式结束这个独立进程。`--train-only` 用来区分训练清理与播放清理；`--inject-playback-error` 在训练后、首次播放前故意抛错。各次执行分别保存日志；请检查两轮 GIF 标记、`ISAAC GO1 SAME-APP CAMERA PLAYBACK PASSED`、`SHUTDOWN DIAG` 标记和退出码。如果崩溃发生在 `SimulationApp.close()` 内部，就不会出现显式退出标记，此时该办法不能解决问题。该入口仅用于诊断，不能代替通用 `close()` 方法。

```powershell
python -u tests/manual/verify_isaac_go1_shutdown_exit.py 1> temp/go1_graceful_exit_success.log 2> temp/go1_graceful_exit_success_error.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --train-only 1> temp/go1_graceful_exit_train_only.log 2> temp/go1_graceful_exit_train_only_error.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --train-only --without-tensorboard 1> temp/go1_graceful_exit_no_tensorboard.log 2> temp/go1_graceful_exit_no_tensorboard_error.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --train-only --algorithm-first 1> temp/go1_graceful_exit_algorithm_first.log 2> temp/go1_graceful_exit_algorithm_first_error.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --stop-before-runner-train 1> temp/go1_graceful_exit_runner_built.log 2> temp/go1_graceful_exit_runner_built_error.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --skip-runner-train 1> temp/go1_graceful_exit_runner_built_no_error.log 2> temp/go1_graceful_exit_runner_built_no_error_stderr.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --skip-algorithm-build 1> temp/go1_graceful_exit_no_algorithm.log 2> temp/go1_graceful_exit_no_algorithm_stderr.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --build-environment-only 1> temp/go1_graceful_exit_environment_only.log 2> temp/go1_graceful_exit_environment_only_stderr.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --build-simulator-only 1> temp/go1_graceful_exit_simulator_only.log 2> temp/go1_graceful_exit_simulator_only_stderr.log
$LASTEXITCODE
python -u tests/manual/verify_isaac_go1_shutdown_exit.py --inject-playback-error 1> temp/go1_graceful_exit_failure.log 2> temp/go1_graceful_exit_failure_error.log
$LASTEXITCODE
```

## 无相机训练的 viewport 性能对比

Go1 同应用播放验证通过后，可比较当前 headless 默认设置与关闭 viewport 更新的设置。以下两条命令均使用同一 128 环境配置、无相机、Fabric 输出关闭，并各运行一次 PPO 训练；唯一差别是第二条在创建 `SimulationApp` 时设置 `disable_viewport_updates=True`：

```powershell
python tests/manual/train_isaac_without_camera.py *> isaac_viewport_default.log
python tests/manual/train_isaac_without_camera.py --disable-viewport-updates *> isaac_viewport_disabled.log
```

两份日志都应出现 `CAMERA-FREE TRAINING PASSED`。比较 `TRAINING TIMINGS` 中 `IsaacSimRuntime.step`、`IsaacSimRuntime.get_state`、`VectorEnv.step`、`PPO.update` 的 `calls` 和 `mean`，以及整体训练耗时。尽量在相同设备负载下交替运行两组至少两轮；每次运行在独立进程中完成，以免 Isaac 状态相互影响。这个开关只作用于手动脚本，不修改正式运行时配置。
