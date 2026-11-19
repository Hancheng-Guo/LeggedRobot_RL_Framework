# LeggedRobot RL Framework

一个面向四足机器人运动控制的模块化强化学习框架。项目当前提供 PPO、可配置 Actor-Critic、向量化环境、MuJoCo/Isaac Sim 后端、多阶段训练、课程学习、检查点、TensorBoard 日志及视频导出。

本文档提供项目结构、安装运行、实验示例与常见问题。组件参数和任务配置见[开发者查阅手册](docs/guide.md)。

## 文档导航

- [项目结构](#1-项目结构) · [安装与运行](#2-安装与运行)
- [完整组合示例](#3-完整组合示例) · [校验与常见问题](#4-校验与常见问题) · [8. 扩展注册表](docs/8_extensions_zh.md)
- [开发者查阅手册](docs/guide.md)：实验配置、Runner、PPO、策略网络、环境、仿真器和任务 manager。

## 1. 项目结构

```text
.
├─ main.py                      # 当前运行入口
├─ tools/                       # 运行实验、播放检查点和检查工具
├─ app/                         # 应用生命周期、运行时上下文、多阶段训练
├─ docs/                        # 按章节拆分的开发者查阅手册
├─ configs/
│  ├─ *.yaml                    # 顶层实验配置
│  ├─ runners/                  # Runner 配置
│  ├─ algorithms/               # 算法配置
│  ├─ policies/                 # 策略网络配置
│  ├─ environments/             # 向量环境配置
│  ├─ simulators/               # 仿真器配置
│  └─ tasks/                    # 任务及各 Manager 配置
├─ envs/                        # 环境、任务、仿真器和各类 term
├─ rl/                          # 算法、策略、网络模块、经验存储
├─ runners/                     # 训练循环和回调
├─ assets/                      # 机器人 MJCF、URDF、USD 资源
├─ checkpoints/                 # 训练输出（运行时自动创建）
├─ requirements-mujoco.txt      # MuJoCo 环境依赖入口
├─ requirements-isaacsim.txt    # Isaac Sim 环境依赖入口
├─ requirements/                # 分环境依赖及约束文件
└─ tests/                       # 自动化测试
```

配置采用“两层结构”：顶层实验文件负责组合组件和阶段；组件的具体参数分别存放在 `configs/<component>/` 中。

## 2. 安装与运行

建议为 MuJoCo 和 Isaac Sim 分别创建 Python 环境。

```powershell
# MuJoCo
python -m pip install -r requirements-mujoco.txt

# Isaac Sim
python -m pip install -r requirements-isaacsim.txt

# 开发与测试依赖（可选）
python -m pip install -r requirements/dev.txt -c requirements/constraints/dev.txt
```

### 首次试跑

在已激活的 Python 3.12 环境中按上文安装 MuJoCo 依赖，然后在项目根目录运行仓库自带的 [Go1 CPU 试跑配置](configs/go1_trial.yaml)。

```powershell
python -m tools.run_app go1_trial
```

脚本依次训练、保存和录制 GIF。这次试跑使用一个环境采集 256 步，再执行一次 PPO 更新。打开 `checkpoints/go1_trial_<时间>/`，检查 `logs/training.log`、`configs/` 配置快照、`checkpoints/stage_000/latest.pt` 和 `videos/` 中的 GIF。原配置的阶段条件要求多次更新，因此日志中的 `Stage 0 timeout.` 符合这次试跑的预期。一次更新只用于验证流程，评估行走效果请使用正式训练的检查点。

### 运行已有实验

在项目根目录执行下列命令。将参数换成 `configs/` 下的顶层 YAML 文件名，省略 `.yaml` 后缀。脚本会依次训练、保存检查点和播放。

```powershell
python -m tools.run_app unitree_go1_mujoco_cpu_velocity_only
```

也可以执行 `python main.py`。该文件当前选择同一个实验，可修改其中的 `app_name` 切换实验。

可以通过 Python 单独训练和保存检查点。

```python
from app.application_entry import ApplicationEntry

with ApplicationEntry("unitree_go1_mujoco_cuda_velocity") as application:
    application.train()
    application.save()
```

现有示例包括 `unitree_go1_mujoco_cpu_velocity_only`、`unitree_go1_mujoco_cpu_velocity`、`unitree_go1_mujoco_cuda_velocity`、`unitree_go1_isaac_cuda_velocity` 和 `unitree_go1_isaac_cuda_test`。`ApplicationEntry` 未传 `train_time` 时创建新运行；传入历史运行目录名末尾的时间时读取该运行的配置和检查点。

常用方法：

```python
application.train()                         # 训练
application.test(num_episodes=1000)         # 确定性策略测试
application.save()                          # 保存当前阶段的 latest.pt
application.play(num_steps=500, formats="gif", num_plays=3)
```

`play()` 默认最多运行 3 段、每段最多 500 步。`formats` 可为 `"gif"`、`"mp4"` 或两者组成的列表或元组；录制需要仿真器的 `render_mode: rgb_array` 和 `requirements/demo.txt` 中的视频依赖。文件保存在运行目录的 `videos/` 下。

播放已有运行的检查点时，可在项目根目录执行：

```powershell
python -m tools.play_checkpoint unitree_go1_mujoco_cpu_velocity_only_2026-10-04_22-55-48 --num-plays 5 --format mp4
```

将示例中的目录名替换为实际存在的运行目录。省略运行目录时选择最新的带时间戳运行；`--num-plays` 默认 `3`，`--format` 默认 `gif`，也可选 `mp4`。VS Code 的 `Tool: Play Checkpoint (Video)` 任务只提供运行目录选择；需要指定这两个参数时使用终端。`Run: Select App` 任务可通过顶层配置名启动新的训练。

### 查看指标与评估检查点

实验启用了 TensorBoard 回调时，可在项目根目录运行下列命令。用实际运行目录名替换示例中的名称。TensorBoard 会占用当前终端，按 Ctrl+C 退出。

```powershell
python -m tools.launch_tensorboard go1_trial_2026-11-19_10-20-30
```

要评估某次训练，按运行目录拆出实验名和时间。例如目录名为 go1_trial_2026-11-19_10-20-30 时，使用下面的写法；实际运行时替换为自己的名称和时间。

```python
from app import ApplicationEntry

with ApplicationEntry("go1_trial", train_time="2026-11-19_10-20-30") as app:
    app.test(num_episodes=10)
```

### 通过键盘控制历史检查点

键盘通过 ROS 2 控制历史检查点时，使用单独的入口：

```powershell
python -m tools.play_human_ros2 unitree_go1_mujoco_cpu_velocity_only_2026-10-04_22-55-48 --playback-config locomotion_velocity
```

也可运行 VS Code 的 `Tool: Play Checkpoint (ROS2 Keyboard)` 任务：先输入运行目录，通过下拉菜单选择播放配置。运行目录留空时选择最新运行；播放配置可选择上次使用的配置，或从 `configs/playback/` 中选择另一个 YAML 文件。在终端使用 `--playback-config` 时，可输入该目录下的文件名（`.yaml` 可省略），也可输入相对项目根目录的路径或绝对路径。

此入口调用现有 `play(mode="human_ros2")`，由 `play()` 创建和清理 ROS 2 节点及键盘窗口，不生成视频。键盘窗口需保持焦点；每按一次正向或反向键升降一挡，松键后保持当前值，空格将全部控制值归零，失焦或关闭窗口也会归零。控制消息类型为 `geometry_msgs/Twist`。所用 Python 环境需能导入 `rclpy` 和 `geometry_msgs`。MuJoCo 可在播放时切换 human 窗口；已经以 headless 模式启动的 Isaac Sim 无法在同一进程切换到 human。

播放配置位于 `configs/playback/`：`locomotion_phase` 适用于 Anymal C 和 Unitree Go1 的 phase 任务，`locomotion_velocity` 适用于 Go1 速度跟踪任务。站立任务的三个速度 command 在主配置中均固定为零，因此不提供键盘速度播放配置。文件中的 `topic` 和 `ros_commands` 指定 ROS 话题、Twist 字段、键位和递增的 `levels` 挡位列表（须包含零挡及正负挡）；挡位范围不超过对应主任务配置的 command 范围。`fixed_commands` 指定其余 command 的固定值（向量按各分量列出）。每个任务 command 必须恰好由其中一种方式提供；启动播放时会检查名称和维度。若两个任务的 command 名称相同但范围不同，仍须自行选择与检查点任务匹配的播放配置。配置文件路径会写入播放日志。现有配置使用 `/cmd_vel`，键位为 `W/S`、`A/D`、`Q/E`；修改挡位只需编辑播放配置，不影响训练配置。

ROS 2 Jazzy 的 Conda 包列在 `requirements/ros2.txt`。在项目根目录、运行仿真的 Conda 环境中执行：

```powershell
conda activate quadruped_rl
conda install -c conda-forge -c robostack-jazzy --file requirements/ros2.txt
```

此文件供 `conda install --file` 使用，不属于 pip requirements。

### 从历史检查点续训

在 VS Code 中执行 **Tasks: Run Task → Run: Resume Training**，依次输入：

1. 历史运行目录名或路径，例如 `unitree_go1_mujoco_cpu_velocity_only_2026-10-04_22-55-48`；留空选择时间最新的运行目录。
2. 检查点路径，例如 `stage_000/checkpoint_00003500.pt`；留空按下文规则自动选择。
3. 输出方式：`fork` 创建新的续训目录（默认），`inplace` 写回原运行目录。

此 Task 使用当前选择的 Python 解释器，读取历史运行归档的配置，续训结束后保存检查点。也可在项目根目录直接执行：

```powershell
python -m tools.resume_training unitree_go1_mujoco_cpu_velocity_only_2026-10-04_22-55-48
```

命令行可通过 `--checkpoint stage_000/checkpoint_00003500.pt` 指定检查点，通过 `--resume-mode inplace` 写回原目录。

`train_time` 是 `checkpoints/<app_name>_<time>/` 目录名中的 `<time>`，格式为 `YYYY-MM-DD_HH-MM-SS`。历史目录需要包含 `configs/<app_name>.yaml`、所引用的组件配置以及可读取的检查点。默认 `resume_mode="fork"`：调用 `train()` 时创建新的时间目录，复制选定检查点为新目录的 `checkpoints/resume.pt`，归档本次使用的配置，并在新目录写入训练日志和后续检查点；原目录不会被续训写入。

```python
with ApplicationEntry(
    "unitree_go1_mujoco_cuda_velocity",
    train_time="2026-09-15_23-11-41",
    device="cuda",                         # 可选：仅覆盖本次运行设备
    checkpoint="stage_000/checkpoint_00000100.pt",  # 可选：指定检查点
) as application:
    application.train()
```

`checkpoint` 可写成相对 `checkpoints/` 的路径（如上例）、相对运行目录且以 `checkpoints/` 开头的路径，或该运行目录 `checkpoints/` 内的绝对路径；指定它时必须同时指定 `train_time`。省略时依次选择：编号最大的 `stage_XXX/latest.pt`、运行级 `checkpoints/latest.pt`、`checkpoints/resume.pt`。`XXX` 必须是三位数字；选择时会读取检查点内的阶段编号，确认它在配置阶段范围内，且与 `stage_XXX` 目录编号一致。指定的文件不存在或内容不合要求会报错，不会改选其他文件。

需要写回历史运行目录时，传 `resume_mode="inplace"`。fork 产生的目录仍按 `<app_name>_<time>` 命名，因此把新目录的时间再次作为 `train_time` 即可续训：设为 `inplace` 就在该目录继续写入，保持默认 `fork` 则再创建一个独立运行。只有调用 `train()` 才会创建 fork；单独调用 `test()` 或 `play()` 不会创建新目录。若新目录名或其准备目录已存在，fork 会报错。

阶段是否完成由检查点中保存的 `StageCallback.stop_training` 状态判断；旧检查点若含 `runner.stage_completed` 字段，也会读取该字段。目录中的 `stage_completed` 标记不参与判断。fork 的 `resume_manifest.json` 记录来源检查点和创建信息，仅供查看，不参与检查点选择或恢复校验。`device` 覆盖只影响本次运行，不改写已归档的配置；运行仍需相应项目代码、仿真资源和外部依赖。

普通续训会在环境 reset 前自动恢复 Python、NumPy、PyTorch CPU 及已初始化的 CUDA 全局随机数状态，无需额外配置。评估和阶段推进不恢复检查点中的随机数状态；CUDA 设备数量变化时，仅恢复当前可用的对应设备状态。不含随机数状态的检查点继续使用本次运行的随机数状态。

`RuntimeContext` 位于 `utils/runtime.py`；`app/utils/run_workspace.py` 的 `select_checkpoint()` 负责选取文件，`inspect_checkpoint_metadata()` 从 PyTorch 检查点读取阶段元数据。runner 的 `RestoreMode.RESUME / ADVANCE_STAGE / EVALUATE` 分别用于续训、阶段推进与评估；阶段条件通过 `config_update(transition=...)` 传入，`train()` 返回 `TrainResult`。新检查点的阶段完成状态保存在阶段回调状态中，不再另存顶层 `stage_completed` 或 `checkpoint_version`。

输出位于：

```text
checkpoints/<app_name>_<time>/
├─ configs/                      # 本次运行使用的配置快照
├─ logs/<file_name>
├─ tensorboard/                  # 启用 TensorBoard 时
├─ videos/                       # 调用 play() 且成功取得 RGB 帧时
├─ resume_manifest.json          # 仅 fork 运行生成，记录来源
└─ checkpoints/
   ├─ resume.pt                   # 仅 fork 运行生成，归档选定的起始检查点
   └─ stage_000/
      ├─ latest.pt
      └─ checkpoint_00000100.pt  # 达到定期保存条件时
```

## 3. 完整组合示例

可直接参考：

- MuJoCo CPU：`configs/unitree_go1_mujoco_cpu_velocity.yaml`
- MuJoCo CPU 单阶段速度任务：`configs/unitree_go1_mujoco_cpu_velocity_only.yaml`
- MuJoCo CUDA 策略计算：`configs/unitree_go1_mujoco_cuda_velocity.yaml`
- Isaac Sim CUDA：`configs/unitree_go1_isaac_cuda_velocity.yaml`
- Isaac Sim 快速测试：`configs/unitree_go1_isaac_cuda_test.yaml`
- 站立任务：`configs/tasks/locomotion_unitree_go1_standing.yaml`
- 速度跟踪任务：`configs/tasks/locomotion_unitree_go1_velocity.yaml`

顶层示例中的 `component` 字段展示了各组件配置的组合方式。调整环境数量时，也应选择相应的 `vector_env_<数量>` 配置，并按资源情况调整 runner 的 rollout 长度和 PPO 的 batch 数。

新增实验的一般流程：

1. 在对应子目录复制并修改 runner、algorithm、policy、environment、simulator、task 配置；
2. 在 `configs/` 新建顶层实验 YAML，通过 `component` 组合它们；
3. 为每个训练阶段配置 `max_iterations` 和 transition；
4. 使用 `python -m tools.run_app <实验名>` 启动新实验；
5. 先用较小 `num_envs`、`rollout_length` 和 `max_iterations` 验证，再扩大训练规模。

## 4. 校验与常见问题

运行测试：

```powershell
pytest -q
```

- **找不到配置**：`ApplicationEntry` 名称和 `configs/<name>.yaml` 必须一致；组件 `config` 名称也必须对应其默认目录中的文件。
- **第一阶段缺组件**：stage 继承只能复用已经构建的实例，首个 stage 必须提供完整组件集合。
- **阶段无法切换**：检查 metric 是否真实产生、各条件的 `window` 是否已积累满，以及所有 transition 是否同时满足。Stage 不支持通配符；相关数据出现后会检查已配置控制指标，有效性为真而主指标缺失时也会报错。
- **课程命令报错**：检查 `curriculum` 引用的实例是否存在；若联合空间过大，应减少各维度的格数或拆分实例。
- **CUDA 不可用**：`runtime.device: cuda` 会严格检查 `torch.cuda.is_available()`，不会自动回退 CPU。
- **确定性算法报错**：部分 CUDA/Isaac 操作不支持确定性实现，可将 `deterministic_ops` 设为 `false`。
- **录制失败**：使用 `render_mode: rgb_array`，并确保所选仿真后端已配置相机或能返回 RGB 帧。
- **续训失败**：确认历史目录有 `configs/<app_name>.yaml` 及所引用的组件配置，并检查 `checkpoints/stage_XXX/latest.pt`、`checkpoints/latest.pt` 或 `checkpoints/resume.pt`。检查点须是包含 `runner.stage_index` 等训练状态的框架产物，单独的模型权重文件不足以恢复训练；如指定了 `checkpoint`，还要检查路径及文件内的阶段编号。
