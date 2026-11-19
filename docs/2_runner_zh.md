# 2. Runner 配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

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

## 2.1 全部回调

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

step_metrics 的键可以是指标全名或通配模式。记录标量（包括只有一个元素的张量）用 `value`；对于逐环境张量，选用 `mean`、`std`、`min`、`max` 或 `histogram`。例如要比较各环境的跟踪表现，可以把 `"reward/*": [mean]` 改为 `"reward/*": [mean, std]`，在 TensorBoard 中同时观察均值和离散程度。

自适应学习率的 monitor 需匹配一个支持的指标：`rollout/approx_kl`、`rollout/clip_fraction` 或 `rollout/entropy`。以示例 Runner 的 approx_kl 为例，窗口均值超过上限时学习率乘以 factor，低于下限时除以 factor。两个边界须递增，factor 须在 0 与 1 之间。

## 2.2 播放时的速度追踪图

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

`target` 必须是 `command/` 开头的 `info` 键；`measured` 使用 `state/<name>[index]` 形式。`label` 是图标题，`limits` 是纵轴下限和上限。各图按列表顺序横向排列。曲线分别表示目标值、原始实测值和实测值的滑动平均。

若替换了任务中的 command，请同步更新 playback 的 target。可先录一段短视频，检查目标和实测曲线的方向、分量及纵轴范围。
