# LeggedRobot RL Framework

一个面向四足机器人运动控制的模块化强化学习框架。项目当前提供 PPO、可配置 Actor-Critic、向量化环境、MuJoCo/Isaac Sim 后端、多阶段训练、课程学习、检查点、TensorBoard 日志及视频导出。

本文档以当前代码为准，说明运行方式、配置组织及常用字段。

## 1. 项目结构

```text
.
├─ main.py                 # 当前运行入口
├─ tools/                  # 运行实验、播放检查点和检查工具
├─ app/                    # 应用生命周期、运行时上下文、多阶段训练
├─ configs/
│  ├─ *.yaml               # 顶层实验配置
│  ├─ runners/             # Runner 配置
│  ├─ algorithms/          # 算法配置
│  ├─ policies/            # 策略网络配置
│  ├─ environments/        # 向量环境配置
│  ├─ simulators/          # 仿真器配置
│  └─ tasks/               # 任务及各 Manager 配置
├─ envs/                   # 环境、任务、仿真器和各类 term
├─ rl/                     # 算法、策略、网络模块、经验存储
├─ runners/                # 训练循环和回调
├─ assets/                 # 机器人 MJCF、URDF、USD 资源
├─ checkpoints/            # 训练输出（运行时自动创建）
├─ requirements-mujoco.txt # MuJoCo 环境依赖入口
├─ requirements-isaacsim.txt # Isaac Sim 环境依赖入口
├─ requirements/           # 分环境依赖及约束文件
└─ tests/                  # 自动化测试
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

`main.py` 当前使用 `unitree_go1_mujoco_cpu_velocity_only`。按所用仿真环境，将 `app_name` 改为 `configs/` 下的顶层 YAML 文件名（不带 `.yaml`），然后运行：

```powershell
python main.py
```

例如：

```python
from app.application_entry import ApplicationEntry

with ApplicationEntry("unitree_go1_mujoco_cuda_velocity") as application:
    application.train()
    application.save()
    application.play(num_steps=500, formats="gif")
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

将示例中的目录名替换为实际存在的运行目录。省略运行目录时选择最新的带时间戳运行；`--num-plays` 默认 `3`，`--format` 默认 `gif`，也可选 `mp4`。VS Code 的 `Tool: Play Checkpoint` 任务只提供运行目录选择；需要指定这两个参数时使用终端。`Run: Select App` 任务可通过顶层配置名启动新的训练。

### 从历史检查点续训

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

## 3. 顶层实验配置

顶层配置必须包含 `runtime`、`stage`、`component`；`logging` 可省略。

```yaml
runtime:
  device: cuda             # torch 设备：cpu、cuda、cuda:0 等
  dtype: float32           # float32 | float64
  num_threads: 4           # >0 时设置 torch CPU 线程数；0 表示不设置
  seed: null               # 非负整数；null 或字符串 "none" 表示随机种子
  deterministic_ops: false # 是否启用 torch 确定性算法

logging:
  file_name: training.log  # 默认 training.log，必须是非空字符串
  console: true            # 是否同时输出到控制台

stage:
  - standing:
      max_iterations: 500
      transition:
        - metric: rollout/mean_len
          method: mean
          operator: ">="
          threshold: 900.0
          window: 20

component:
  runner:
    - {type: on_policy, config: mujoco_rollout_256}
  algorithm:
    - {type: ppo, config: ppo_num_batch_8}
  policy:
    - {type: actor_critic, config: actor_critic_direct_actuation}
  environment:
    - {type: vector_env, config: vector_env_64}
  simulator:
    - {type: mujoco, config: mujoco_unitree_go1_rgb_array}
  task:
    - {type: locomotion, config: locomotion_unitree_go1_standing}
```

### 3.1 stage 写法

每个 stage 必须是只含一个名称的映射，且内部必须恰好有：

- `max_iterations`：正整数；
- `transition`：非空条件列表，所有条件同时满足后进入下一阶段。

每条 transition 必须恰好包含：

| 字段 | 可用值 | 含义 |
|---|---|---|
| `metric` | 指标名或唯一通配模式 | 如 `rollout/mean_len`、`*/approx_kl` |
| `method` | `mean`、`max`、`min` | 对最近窗口聚合 |
| `operator` | `>`、`>=`、`<`、`<=`、`==`、`!=` | 与阈值比较 |
| `threshold` | 数值 | 目标阈值 |
| `window` | 正整数 | 连续记录窗口长度 |

向量指标会先在环境维度取平均。通配模式必须最多匹配一个指标，否则会报错。

### 3.2 component 写法与多阶段继承

当前可用类型：

| 组件 | `type` 可用值 | 默认配置目录 |
|---|---|---|
| runner | `on_policy` | `configs/runners/` |
| algorithm | `ppo` | `configs/algorithms/` |
| policy | `actor_critic` | `configs/policies/` |
| environment | `vector_env` | `configs/environments/` |
| simulator | `mujoco`、`isaac_sim` | `configs/simulators/` |
| task | `base`、`locomotion` | `configs/tasks/` |

组件条目支持三种定位方式：

```yaml
# 1. 默认目录 + 文件名（推荐，可省略 .yaml）
- type: ppo
  config: ppo_num_batch_8

# 2. 自定义目录 + 文件名
- type: ppo
  config_dir: ./my_configs/algorithms
  config: ppo_custom.yaml

# 3. 完整路径；设置后 config/config_dir 被忽略
- type: ppo
  config_path: ./my_configs/algorithms/ppo_custom.yaml
```

每类组件使用列表是为了对应多个 stage：

- 列表只有一项时，后续 stage 继承该项；
- 第 `N` 项从第 `N` 个 stage 开始生效；
- 相邻项完全相同时不会重建该组件；
- 组件条目数可以少于或等于 stage 数，不能多于 stage 数；
- 第一阶段必须能组合出完整的 runner、algorithm、policy、environment、simulator、task。

## 4. Runner 配置

`configs/runners/*.yaml`：

```yaml
rollout_length: 256                # 必填，单次更新前每个环境采样步数
rollout_length_history_size: 50    # 默认 50，统计平均 episode 长度的历史数量
callbacks:
  - progress_bar                   # 无参数写法
  - keyboard_interrupt
  - logging:
      log_interval: 1
  - checkpoint:
      save_iter_interval: 100
      directory_name: checkpoints
      save_on_train_end: false
```

回调可写成字符串（使用默认参数），也可写成仅含一个回调名的映射。

### 4.1 全部回调

| 回调 | 参数 |
|---|---|
| `progress_bar` | `width=30`，`refresh_interval=0.2` 秒 |
| `keyboard_interrupt` | 无参数；在交互终端按 Ctrl+X，请求在本次更新结束后停止训练 |
| `logging` | `log_interval=1` |
| `checkpoint` | `save_iter_interval=100`，`directory_name="checkpoints"`，`save_on_train_end=false` |
| `adaptive_learning_rate` | 必填 `monitor`、`allowed_range: [lower, upper]`、`factor`；可选 `min_learning_rate=null`、`max_learning_rate=null`、`buffer_len=10` |
| `early_stopping` | 必填 `monitor`、`max_no_improve_iters`、`mode: min\|max`、`min_delta`；可选 `warmup_iters=0` |
| `tensorboard` | 见下例 |

```yaml
- tensorboard:
    log_dir: tensorboard
    step_log_interval: 10
    histogram_step_interval: 10000
    flush_secs: 10
    initial_load_timeout: 600.0
    max_reload_threads: 4
    step_metrics:
      "reward": [value]
      "reward/*": [mean]
      "termination/*": [mean]
      "command/*": [histogram]
```

`step_metrics` 的键是指标名/通配模式；值支持 `value`、`mean`、`histogram`（具体是否适用取决于指标形状）。

### 4.2 播放时的速度追踪图

Runner 配置中的可选 `playback` 列表定义视频右侧的追踪图，每项把 `info` 中的一个目标指令与一个状态分量对应起来：

```yaml
playback:
  - target: "command/lin_vel_x"
    measured: "state/base_lin_vel_body[0]"
    label: "X linear velocity (m/s)"
    limits: [-2.8, 2.8]
  - target: "command/lin_vel_y"
    measured: "state/base_lin_vel_body[1]"
    label: "Y linear velocity (m/s)"
    limits: [-0.8, 0.8]
  - target: "command/ang_vel_z"
    measured: "state/base_ang_vel_body[2]"
    label: "Z angular velocity (rad/s)"
    limits: [-2.0, 2.0]
```

`target` 必须是 `command/` 开头的 `info` 键；`measured` 使用 `state/<name>[index]` 形式。`label` 是图标题，`limits` 是纵轴下限和上限。各图按列表顺序横向排列。曲线分别表示目标值、原始实测值和实测值的滑动平均；图例右侧的 `T +00.00` 显示从视频开始经过的秒数。平滑窗口目前在 `runners/utils/tracking_overlay.py` 的 `SMOOTH_WINDOW` 常量中设置，单位为帧；窗口对应的时间为 `SMOOTH_WINDOW / 视频帧率`。当前窗口为 25 帧，在 50 FPS 下约为 0.5 秒。

## 5. PPO 算法配置

`configs/algorithms/*.yaml`：

```yaml
init_learning_rate: 5.0e-4 # > 0
gamma: 0.99                # [0, 1]
gae_lambda: 0.95           # [0, 1]
clip_range: 0.2            # > 0
entropy_coef: 1.0e-3       # >= 0
value_coef: 0.5            # >= 0
max_grad_norm: 0.5         # > 0
num_epochs: 3              # 正整数
num_mini_batches: 8        # 正整数
advantage_norm: true       # 是否标准化 advantage，未写时沿用类默认值
```

`num_envs × rollout_length` 应能合理划分为 `num_mini_batches`；批数越多，单个 mini-batch 越小。

## 6. Actor-Critic 策略配置

`configs/policies/*.yaml`：

```yaml
distribution:
  type: diagonal_gaussian
  initial_std: 0.5
  min_std: 0.1
  max_std: 1.0

actor:
  - {type: linear, in_features: obs.shape, out_features: 256}
  - {type: elu}
  - {type: linear, in_features: 256, out_features: action.shape}

critic:
  - {type: linear, in_features: obs.shape, out_features: 256}
  - {type: elu}
  - {type: linear, in_features: 256, out_features: 1}
```

`actor`、`critic`、`distribution` 均必填。维度变量支持：

- `obs.shape`、`obs.shape[-1]`：完整 observation 维度；
- `action.shape`、`action.shape[-1]`：动作维度；
- `obs.<term>.shape`、`obs.<term>.shape[-1]`：某 observation term 的维度。

模块通用写法：

```yaml
- type: linear
  name: encoder                 # 可选；默认使用列表索引
  inputs: [obs]                 # 可选；默认取前一模块，首层默认 obs
  inherit: false                # 是否从同 type 的保存工件继承配置/权重
  trainable: true               # false 时冻结参数
  params:                       # 可选；也可把参数直接平铺在同一级
    in_features: obs.shape
    out_features: 128
```

同一参数不能同时出现在顶层和 `params`。`name` 只能包含字母、数字、`_`、`-`，且不能重名。`inputs` 可引用：

- `obs`：全部 observation；
- `obs[0:12]`：按索引切片；
- `obs.<term>` 或 `obs.<term>[start:stop]`：按 observation term 取值；
- 已命名的前序模块，或其 `[start:stop]` 切片；
- 多个输入会在最后一维拼接。

已注册模块：

| `type` | 常用参数 |
|---|---|
| `linear` | `in_features`、`out_features`、`bias=true` |
| `relu` | `inplace=false` |
| `tanh` | 无 |
| `elu` | `alpha=1.0`、`inplace=false` |
| `gelu` | `approximate="none"` |
| `silu` | `inplace=false` |
| `layer_norm` | `normalized_shape`、`eps=1e-5`、`elementwise_affine=true`、`bias=true` |
| `dropout` | `p=0.5`、`inplace=false` |
| `identity` | 无 |
| `gru` | `input_size`、`hidden_size`、`num_layers=1`、`bias=true`、`dropout=0.0`、`bidirectional=false`；当前只允许 actor 使用且必须单向 |
| `add` | 无参数；用 `inputs` 指定至少两个同形状来源 |
| `sequential` | `modules`：把一组子模块封装成一个具名复合模块 |

### 6.1 将多个模块封装为一个新模块

可以使用 `type: sequential` 和 `modules` 把多层网络统一命名为一个复合模块：

```yaml
actor:
  - name: velocity_encoder
    type: sequential
    inputs: obs
    modules:
      - {type: linear, in_features: obs.shape, out_features: 256}
      - {type: elu}
      - {type: linear, in_features: 256, out_features: 128}
      - {type: elu}

  - type: linear
    in_features: velocity_encoder.shape
    out_features: action.shape
```

这里的 `velocity_encoder` 同时具有三种用途：

- 它是当前网络中的模块名，可由后续层通过 `inputs: velocity_encoder` 引用；
- 它会产生维度变量 `velocity_encoder.shape`、`velocity_encoder.shape[-1]` 和 `velocity_encoder.out_features`；
- 保存策略时，它会作为一个完整模块导出，可在后续 stage 中把该名称直接当作 `type` 使用。

多分支组合也可以引用具名模块：

```yaml
actor:
  - name: motor_action
    inputs: obs
    type: linear
    in_features: obs.shape
    out_features: action.shape

  - name: actuator_shift
    inputs: obs
    type: linear
    in_features: obs.shape
    out_features: action.shape

  - name: final_action
    inputs: [motor_action, actuator_shift]
    type: add
```

普通模块收到多个 `inputs` 时会把它们沿最后一维拼接；只有 `add` 会逐元素相加，因此参与 `add` 的输入维度必须完全相同。

### 6.2 跨 stage 复用复合模块

第一阶段先定义一个具名复合模块：

```yaml
actor:
  - name: custom1
    type: sequential
    modules:
      - {type: linear, in_features: obs.shape, out_features: 128}
      - {type: elu}
  - {type: linear, in_features: 128, out_features: action.shape}
```

后续 stage 的策略配置可以直接把 `custom1` 当成模块类型。框架会从上一阶段导出的模块工件中恢复其内部结构和参数：

```yaml
actor:
  - type: custom1
    # 可选的接口断言；不写则使用保存的结构
    in_features: obs.shape
    out_features: 128
  - {type: linear, in_features: 128, out_features: action.shape}
```

也可以不封装结构，只让同名模块继承上一阶段参数：

```yaml
- name: encoder
  type: linear
  in_features: obs.shape
  out_features: 128
  inherit: true
  trainable: false
```

`inherit: true` 要求上一阶段存在同名、同类型且参数形状一致的模块；`trainable: false` 会冻结该模块。Actor 和 Critic 的命名空间相互独立。复合模块内部仍可使用 `name`、`inputs`、`gru` 等全部网络写法。

分布当前仅有：

```yaml
distribution:
  type: diagonal_gaussian
  initial_std: 0.5    # > 0
  min_std: 0.1
  max_std: 1.0
```

`action_dim` 由环境自动注入，不应写入 distribution。高级的 `module_artifacts` 用于为 `actor.<name>`/`critic.<name>` 提供已保存模块工件，常规训练无需手写。

## 7. 环境配置

`configs/environments/*.yaml`：

```yaml
num_envs: 64             # 必填，正整数
max_episode_steps: 1000  # 必填，正整数；达到后 truncated
```

MuJoCo 会为每个环境建立实例，数量过大会明显增加 CPU/内存开销；Isaac Sim 适合大规模 GPU 并行环境。

## 8. 仿真器配置

### 8.1 MuJoCo

```yaml
model_path: ./assets/unitree_go1/MJCF/scene.xml # 必填
sim_dt: 0.002                                  # 必填，> 0
frame_skip: 10                                 # 必填，正整数
step_workers: 8                                # 可选，并行推进环境的线程数；默认 1
render_mode: rgb_array                         # null | human | rgb_array
foot_geom_names: [FR, FL, RR, RL]              # 足端 geom 名称
floor_geom_names: [floor]                      # 地面 geom 名称
foot_contact_force_threshold: 15.0             # >= 0
reset_keyframe: home                           # 可选，MJCF keyframe 名
```

控制周期为 `sim_dt × frame_skip`。`rgb_array` 用于无窗口录制，`human` 用于交互显示。

### 8.2 Isaac Sim

```yaml
model_path: ./assets/unitree_go1/USD/go1/go1.usda
ros_package_paths:                             # 导入 URDF/MJCF 时可选
  - {go1_description: ./assets/unitree_go1/URDF}
sim_dt: 0.002
frame_skip: 10
render_mode: rgb_array                         # null | human | rgb_array
train_render_mode: null                        # 可选，训练期间的渲染模式
env_spacing: 2.0                               # > 0
robot_prim_path: /Robot
base_body_prim_path: /Geometry/base/trunk
foot_body_prim_paths:
  - /Geometry/base/trunk/FR_hip/FR_thigh/FR_calf/FR_foot
  - /Geometry/base/trunk/FL_hip/FL_thigh/FL_calf/FL_foot
floor_prim_paths: []                           # 必须为绝对 Prim path
foot_contact_force_threshold: 15.0             # >= 0
camera_prim_path: null                         # 可选相机 Prim
camera_resolution: [640, 480]                  # 两个正整数
merge_fixed_joints: false
allow_self_collision: false
joint_stiffness: 100.0                         # 标量或 {joint_name: value}
joint_damping: 2.0                             # 标量或 {joint_name: value}
reset_state:
  base_position: [0.0, 0.0, 0.27]             # 长度 3
  base_orientation: [1.0, 0.0, 0.0, 0.0]      # 长度 4，非全零四元数
  joint_positions:                             # joint 名 -> 初始位置
    FR_hip_joint: 0.0
```

`model_path` 支持 `.usd`、`.usda`、`.usdc`、`.urdf`、MJCF `.xml`。`robot_prim_path`、body path 会被规范为以 `/` 开头的相对机器人路径；`floor_prim_paths` 则必须是完整绝对路径。

#### Isaac Sim 退出异常与训练产物

在 Windows、Isaac Sim 6.1.0.0 的当前测试环境中，`SimulationApp.close()` 返回后，Python 进程仍可能在原生插件的退出清理阶段以 `0xC0000005`（PowerShell 中通常显示为 `-1073741819`）结束。最小化的 `SimulationApp` 脚本也能复现；原生转储将故障定位到 `isaacsim.core.simulation_manager.plugin.dll + 0x1c5fe`。这属于尚未解决的退出异常，**不能将非零退出码当作正常成功**，也不能仅凭退出码断定训练阶段没有完成。诊断过程见 [手动测试说明](tests/manual/README.md#isaac-sim-关闭诊断)。

判断一次训练是否已经留下可用产物，应检查同一次运行的 `checkpoints/<app_name>_<time>/`：`logs/<file_name>` 中出现 `All training stages completed.`，对应 `checkpoints/stage_XXX/latest.pt` 中的阶段回调状态记录完成，且 `configs/` 有本次配置快照。阶段完成时项目会保存包含回调状态的检查点；目录中的 `stage_completed` 标记不参与恢复判断。若只是达到最大迭代数或被回调停止，不能把日志中已运行的迭代数当作阶段完成；若需要确认检查点可用于继续训练，应在受信任的本地产物上实际执行一次续训加载。播放结果另看 `videos/` 下生成的文件和播放脚本的完成标记。即使这些检查都通过，退出码异常仍须单独记录，自动化任务也应将该进程视为退出失败。

## 9. locomotion 任务配置

任务文件必须提供以下八个顶层字段。`event_manager_config` 和 `randomization_manager_config` 当前为预留项，尚未构建对应 Manager，写 `{}` 即可。

```yaml
action_manager_config: {terms: {}}
command_manager_config: {terms: {}}
observation_manager_config: {terms: {}}
reward_manager_config: {terms: {}}
termination_manager_config: {terms: {}}
curriculum_manager_config: {terms: {}}
event_manager_config: {}
randomization_manager_config: {}
```

可用 YAML anchor 复用常量：

```yaml
constants:
  base_target_height: &base_target_height 0.27
reward_manager_config:
  terms:
    base_height_l2:
      weight: -0.2
      params: {target_height: *base_target_height}
```

`constants` 仅由 YAML 自身解析，框架不会读取它，因此可安全用作 anchor 容器。

### 9.1 Action Manager

Action term 会按 YAML 从上到下执行：第一项最靠近策略原始动作，最后一项的输出直接发送给仿真器。框架会在初始化时反向推导各层维度，因此不要随意交换有维度映射作用的 term：

```yaml
action_manager_config:
  terms:
    hard_clamp: {min_value: -1.0, max_value: 1.0}
    tanh_map: {}
    keyframe_centered_linear_map: {}
```

已注册 term：

| 名称 | 参数/作用 |
|---|---|
| `linear_map` | 无手写参数；将归一化动作线性映射到 actuator 控制范围 |
| `tanh_map` | 无；对输入应用 `tanh` |
| `keyframe_centered_linear_map` | 无；以默认/keyframe 控制为中心线性映射 |
| `hard_clamp` | 必填 `min_value`、`max_value` |

### 9.2 Command Manager

```yaml
command_manager_config:
  terms:
    lin_vel_x:
      type: UniformOnReset
      params: {min_value: -1.0, max_value: 1.0}
  constraints:
    lin_vel_x:
      operator: ">="
      expression: "-{lin_vel_y}.abs()"
```

term 的键是命令维度名称；`type` 使用类名（区分大小写）：

| `type` | `params` |
|---|---|
| `UniformOnReset` | 必填 `min_value`、`max_value` |
| `LpacSampleOnReset` | 必填 `group`、`min_value`、`max_value`、`num_cell` |
| `SymmetricLpacSampleOnReset` | 同上，但 min/max 必须非负，采样后随机取正负 |
| `FastLpacSampleOnReset` | 必填 `group`、`min_value`、`max_value`、`num_cell` |
| `SymmetricFastLpacSampleOnReset` | 同上，但 min/max 必须非负 |

LPAC term 必须搭配同类 curriculum term。`group` 相同的命令组成联合课程空间；`num_cell` 为该维度划分格数，且各空间总格数不能超过 curriculum 的 `max_cells`。

约束的 `operator` 支持 `<=`、`>=`、`<`、`>`、`!=`；表达式用 `{命令名}` 引用其他维度，可使用 `torch` 和 `abs`。`!=` 还支持 `direction: positive|negative`（默认 `positive`）决定相等时向哪一侧偏移。课程空间内的约束不能跨 `group`。

### 9.3 Observation Manager

```yaml
observation_manager_config:
  clip: 3.0                   # 可选，>0；最终拼接后裁剪到 [-clip, clip]
  terms:
    base_linear_velocity: {scale: 0.5}
    command: {scale: 0.5}
```

term 顺序决定最终 observation 的拼接顺序。所有 observation term 都支持 `scale`（默认 `1.0`）。已注册名称：

| 名称 | 输出 |
|---|---|
| `base_linear_velocity` | base 坐标系线速度 |
| `base_position` | base 世界坐标位置 |
| `base_height` | base 高度 |
| `base_angular_velocity` | base 坐标系角速度 |
| `projected_gravity` | 重力在 base 坐标系中的投影 |
| `joint_position` | 关节位置 |
| `joint_velocity` | 关节速度 |
| `joint_position_diff` | 关节位置相对默认位置的偏差 |
| `actuator_force` | 执行器力 |
| `foot_height` | 足端高度 |
| `foot_contact_normal_force` | 足端法向接触力 |
| `foot_contact_state` | 足端接触状态 |
| `foot_duration_tanh` | 每只足的触地状态保持时间经 `tanh(alpha × 时间)` 压缩；触地为正、离地为负，默认 `alpha=1.0` |
| `command` | 所有命令拼接结果 |
| `track_linear_velocity_x_error_integral` | 最近 `integral_length` 步的有符号 X 速度误差积分 |
| `track_linear_velocity_y_error_integral` | 最近 `integral_length` 步的有符号 Y 速度误差积分 |
| `track_angular_velocity_z_error_integral` | 最近 `integral_length` 步的有符号 Z 角速度误差积分 |
| `track_linear_velocity_x_error_integral_tanh` | 对 X 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `track_linear_velocity_y_error_integral_tanh` | 对 Y 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `track_angular_velocity_z_error_integral_tanh` | 对 Z 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `last_action` | 最近 `lags` 步策略动作，按从新到旧排列（默认 `lags: 1`） |

### 9.4 Reward Manager

统一写法：

```yaml
reward_manager_config:
  terms:
    <reward_name>:
      weight: 1.0             # 可省略，默认 1.0
      params: {}              # 可省略
```

已注册 reward 及其可写 `params`：

| 名称 | 参数（未注明即无参数） |
|---|---|
| `action_diff_l2` | `max_lag=1`、`decay=0.5`。在最大 lag 范围内惩罚当前动作与历史动作的均方差，权重按 `decay**(lag-1)` 衰减；历史按环境独立维护，并在 reset 时清空。 |
| `illegal_contact_l1` | `geom_legal_names=null` |
| `joint_velocity_l2` | — |
| `joint_position_diff_l2` | — |
| `joint_limit_violation_l1` | `lower_limits=null`、`upper_limits=null` |
| `joint_power_l1` | — |
| `base_linear_velocity_z_l2` | — |
| `base_linear_velocity_z_l2_xy_speed_weighted` | `min_speed=0.5` |
| `base_height_l2` | 必填 `target_height`；`std=1.0`、`max_normalized_error=null` |
| `base_height_l2_xy_speed_weighted` | 必填 `target_height`；`min_speed=0.5` |
| `base_angular_velocity_xy_l2` | — |
| `projected_gravity_xy_l2` | — |
| `track_linear_velocity_xy_l2_exp` | `x_std=1.0`、`y_std=1.0`、`command_names=[lin_vel_x, lin_vel_y]` |
| `track_linear_velocity_xy_l2_exp_and_logcosh` | 上述参数 + `logcosh_weight=0.5` |
| `track_linear_velocity_x_l2_exp` | `std=1.0`、`command_names=[lin_vel_x]` |
| `track_linear_velocity_y_l2_exp` | `std=1.0`、`command_names=[lin_vel_y]` |
| `track_linear_velocity_x_l2_exp_and_logcosh` | `std=1.0`、`command_names=[lin_vel_x]`、`logcosh_weight=0.5` |
| `track_linear_velocity_y_l2_exp_and_logcosh` | `std=1.0`、`command_names=[lin_vel_y]`、`logcosh_weight=0.5` |
| `track_linear_velocity_x_error_integral_l2` | 最近 `integral_length` 步的 X 有符号速度误差取平均后平方；窗口未满时只除以有效步数，reset 后重新累计；默认 `integral_length=100` |
| `track_linear_velocity_y_error_integral_l2` | 最近 `integral_length` 步的 Y 有符号速度误差取平均后平方；窗口未满时只除以有效步数，reset 后重新累计；默认 `integral_length=100` |
| `track_angular_velocity_z_l2_exp` | `std=1.0`、`command_names=[ang_vel_z]` |
| `track_angular_velocity_z_l2_exp_and_logcosh` | `std=1.0`、`command_names=[ang_vel_z]`、`logcosh_weight=0.5` |
| `track_angular_velocity_z_error_integral_l2` | 最近 `integral_length` 步的 Z 有符号角速度误差取平均后平方；窗口未满时只除以有效步数，reset 后重新累计；默认 `integral_length=100`、`command_names=[ang_vel_z]` |
| `trot_loop_duration_tanh` | `command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`growth_rate=1.0`；指令范数低于 `0.1` 时要求四足着地 |
| `quadrupedal_gait_phase_l2_exp` | 必填 `target_height`；`sigma=1.0`；从命令 `half_period_duration` 读取半周期时长并按 `step_dt` 换算为步数 |
| `quadrupedal_command_adaptive_gait_phase_l2_exp` | 必填 `target_height`；`sigma=1.0`、`command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`alpha=0.5`、`beta=0.2`；半周期时长为 `beta * exp(alpha / max(command_norm, 0.1))`，命令范数低于 `0.1` 时不推进足端相位 |
| `foot_state_duration_command_weighted_exp` | 整组足端触地状态保持时间；`command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`sigma=1.0` |
| `foot_state_duration_cubic_command_weighted_exp` | 整组足端触地状态保持时间的三次方版本；参数同上 |
| `foot_state_duration_cubic_command_tanh_weighted_exp` | 整组足端触地状态保持时间的三次方版本，指令范数先取 `tanh`；参数同上 |
| `foot_duration_cubic_command_tanh_weighted_exp` | 分别统计每只足的触地状态保持时间，取三次方并按 `tanh` 指令范数加权后求平均；参数同上 |
| `foot_state_switch` | `hold_time=0.2`；触地状态在保持时间内切换时返回惩罚值 |
| `foot_sliding_velocity_l2` | — |
| `foot_lift_height_diff_command_weighted_exp` | 必填 `target_height`；`height_std=0.03`、`command_std=0.5`、`command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`；仅对摆动脚计算高度奖励，零指令时为零 |
| `foot_lift_height_diff_command_gated_l2` | 必填 `target_height`；`height_std=null` 时使用 `target_height`；命令范数 `<=0.1` 时所有足端目标高度为 0，运动时支撑脚目标为 0、摆动脚目标为 `target_height`；使用负权重作为高度误差惩罚 |
| `quadrupedal_foot_velocity_diff_l2` | — |
| `foot_contact_without_command` | `command_names=[lin_vel_x, lin_vel_y, ang_vel_z]` |

名称由注册类名自动转为 snake_case；大小写缩写会被正确拆分，例如 `TrackAngularVelocityZL2Exp` → `track_angular_velocity_z_l2_exp`。

### 9.5 Termination Manager

```yaml
termination_manager_config:
  terms:
    body_contact:
      body_names: [trunk, FR_thigh, FL_thigh, RR_thigh, RL_thigh]
    base_height:
      min_height: 0.16
```

| 名称 | 参数 |
|---|---|
| `body_contact` | 必填 `body_names`，任一指定 body 发生接触即终止 |
| `base_height` | 必填 `min_height`，低于阈值即终止 |

Episode 达到 `max_episode_steps` 属于 `truncated`，不是 termination term。

### 9.6 Curriculum Manager

```yaml
curriculum_manager_config:
  terms:
    fast_lpac_command_reward:
      temperature: 0.01
      exploration: 0.1
      max_cells: 100000
      min_samples_per_cell: 5
      min_coverage: 0.8
```

| 名称 | 参数 |
|---|---|
| `lpac_command_reward` | `temperature=0.01`（>0）、`exploration=0.1`（[0,1]）、`max_cells=1000`、`min_samples_per_cell=5`（正整数） |
| `fast_lpac_command_reward` | 同上，并增加 `min_coverage=0.8`（(0,1]） |

`lpac_command_reward` 对应 `LpacSampleOnReset`/`SymmetricLpacSampleOnReset`；`fast_lpac_command_reward` 对应 Fast 版本。没有课程学习时写 `terms: {}`。

## 10. 完整组合示例

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
4. 在 `main.py` 将 `ApplicationEntry("...")` 指向新实验名；
5. 先用较小 `num_envs`、`rollout_length` 和 `max_iterations` 验证，再扩大训练规模。

## 11. 校验与常见问题

运行测试：

```powershell
pytest -q
```

- **找不到配置**：`ApplicationEntry` 名称和 `configs/<name>.yaml` 必须一致；组件 `config` 名称也必须对应其默认目录中的文件。
- **第一阶段缺组件**：stage 继承只能复用已经构建的实例，首个 stage 必须提供完整组件集合。
- **阶段无法切换**：检查 metric 是否真实产生、通配符是否唯一、`window` 是否已积累满，以及所有 transition 是否同时满足。
- **课程命令报错**：LPAC command 与 curriculum 类型必须配对；同组命令的格点乘积不能超过 `max_cells`。
- **CUDA 不可用**：`runtime.device: cuda` 会严格检查 `torch.cuda.is_available()`，不会自动回退 CPU。
- **确定性算法报错**：部分 CUDA/Isaac 操作不支持确定性实现，可将 `deterministic_ops` 设为 `false`。
- **录制失败**：使用 `render_mode: rgb_array`，并确保所选仿真后端已配置相机或能返回 RGB 帧。
- **续训失败**：确认历史目录有 `configs/<app_name>.yaml` 及所引用的组件配置，并检查 `checkpoints/stage_XXX/latest.pt`、`checkpoints/latest.pt` 或 `checkpoints/resume.pt`。检查点须是包含 `runner.stage_index` 等训练状态的框架产物，单独的模型权重文件不足以恢复训练；如指定了 `checkpoint`，还要检查路径及文件内的阶段编号。

## 12. 扩展注册表

项目通过注册表扩展。新增实现后应在相应 registry 中注册或添加映射：

- Runner：`runners/registry.py`
- Callback：`runners/callbacks/registry.py`
- Algorithm：`rl/algorithms/registry.py`
- Policy：`rl/policies/registry.py`
- Network module：`rl/policies/modules/registry.py`
- Environment/Simulator/Task：`envs/*/registry.py`
- Action、Command、Observation、Reward、Termination、Curriculum term：各 manager 的 `terms/registry.py`

Manager term 使用装饰器注册后，配置名通常由类名转换得到：action/observation/reward/termination/curriculum 使用 snake_case；command 使用类名本身。新增字段应放在 term 的构造函数参数中，然后在 YAML 的 term 配置或 `params` 中传入。
