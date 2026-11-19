# 1. 顶层实验配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

复制已有的顶层实验 YAML。需要调整组件时，再复制对应的组件 YAML，并更新顶层 `component` 中的 `config`。例如，修改 [Go1 速度任务](../configs/tasks/locomotion_unitree_go1_velocity.yaml)中的奖励项后，检查 [阶段切换条件](#11-stage-写法)引用的指标名和阈值是否仍然适用。

顶层配置必须包含 `runtime`、`stage` 和 `component`；`logging` 可省略。

```yaml
runtime:
  device: cuda             # torch 设备：cpu、cuda、cuda:0 等
  dtype: float32           # float32 | float64
  num_threads: 4           # >0 时设置 torch CPU 线程数；0 表示不设置
  seed: null               # 非负整数；null 或字符串 "none" 表示随机种子
  deterministic_ops: false # 是否启用 torch 确定性算法

logging:
  file_name: training.log  # 默认 training.log，必须是非空字符串
  console: true            # 是否同时输出到控制台，默认为 true

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

## 1.1 stage 写法

每个 stage 必须是只含一个名称的映射，且内部必须恰好有：

- `max_iterations`：正整数；
- `transition`：非空条件列表，所有条件同时满足后进入下一阶段。

达到 max_iterations 时若条件仍未全部满足，训练在当前阶段结束并记录 timeout。要依次训练多个阶段，应为每个阶段选用该阶段实际输出的指标，并留出足够的更新次数来填满窗口。

每条 transition 必须包含 `metric`、`method`、`operator`、`threshold` 和 `window`，还可显式关联更新版本 `revision_metric` 与有效性指标 `validity_metric` ：

| 字段 | 可用值 | 含义 |
|---|---|---|
| `metric` | 完整指标名，不支持通配符 | 如 `rollout/mean_len`、`reward/action_diff_l2` |
| `method` | `mean`、`max`、`min` | 对最近窗口聚合 |
| `operator` | `>`、`>=`、`<`、`<=`、`==`、`!=` | 与阈值比较 |
| `threshold` | 有限数值 | 目标阈值 |
| `window` | 正整数 | 连续记录窗口长度 |
| `revision_metric`（可选） | 完整指标名，默认 `null` | 只在版本变化时记录一次；版本回退时清空窗口 |
| `validity_metric`（可选） | 完整指标名，默认 `null` | 值为真或 `1` 时才记录；为假或 `0` 时清空窗口 |

每条条件维护自己的缓存窗口，所有条件同时满足后才切换阶段。目标向量指标在环境维度取平均； `revision_metric` 与 `validity_metric` 对应的两个控制指标必须是标量。未配置控制指标时，每次收到目标指标都记录。

Stage 只读取回调 `info` 中配置指定的字段，不识别课程路径、不推导关联字段，也不访问 task 或 manager。三个指标名均禁止通配符；指标存在性在收到相关数据后检查。所有条件都会在初始化时建立缓存。

课程 cell 评价可按如下方式接入（阈值仅示意，需按奖励来源和 episode 归一化设置校准）：

```yaml
transition:
  - metric: curriculum/locomotion/reward_min
    method: min
    operator: ">="
    threshold: 3000.0
    window: 3
    revision_metric: curriculum/locomotion/statistics_revision
    validity_metric: curriculum/locomotion/all_cells_valid
```

上例要求全体 cell 都有有效评价，且最近三次统计更新中的最弱 cell 分数均不低于 `3000.0`。`reward_min` 取有效 cell 分数的最小值，`method: min` 再取窗口内三次评价的最小值。配置版本指标后，`window` 按不同版本计数；同一份统计不会重复入窗，各 cell 的评价仍可能来自不同时刻。其他来源同样可以显式提供自己的版本与有效性指标，无需采用课程指标的命名。

窗口的保留与清空规则如下：

- 本次回调没有目标及已配置的控制指标时，保持窗口不变，等待后续数据。一旦出现相关字段，缺少任一已配置控制指标会明确报错。
- 版本必须是有限的非负整数。相同版本不重复入窗，版本回退时清空旧窗口。
- 有效性必须是布尔值或数值 `0/1`。无效时清空窗口；若同时配置版本指标，该无效版本仍会被记录，从下一次有效的新版本重新积累。
- 有效性为真但目标指标缺失时明确报错；未配置有效性指标时，允许目标尚未产生，清空窗口后继续等待。目标指标出现 NaN/Inf 时清空窗口。
- 检查点保存目标及控制指标引用、窗口和最后版本。恢复时核对三者引用；未恢复环境状态时，依赖控制指标的窗口重新积累。旧开关与缺少指标引用元数据的旧检查点不做兼容。

## 1.2 component 写法与多阶段继承

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
