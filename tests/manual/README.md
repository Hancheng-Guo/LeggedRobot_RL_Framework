# 手动诊断脚本

这里的文件是按需运行的性能和故障排查工具，不是常规 pytest 用例。请从项目根目录运行，并先选择对应的 Python 环境。性能结果受硬件、系统负载和运行参数影响，不应当作固定的测试通过阈值。

| 脚本 | 环境 | 用途 |
| --- | --- | --- |
| `profile_mujoco_state.py` | MuJoCo | 分析 `MujocoSimulator.get_state()` 及各状态字段的耗时占比。 |
| `benchmark_mujoco_step_threads.py` | MuJoCo | 比较不同环境数和工作线程数的仿真步进吞吐量。 |
| `isaac_shutdown_diagnostic.py` | Isaac Sim | 分阶段复现 Isaac Sim 启动、场景构建与关闭时的问题。 |

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
