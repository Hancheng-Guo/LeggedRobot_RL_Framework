# 4. Actor-Critic 策略配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

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

连续动作使用 diagonal_gaussian。initial_std 为初始标准差，min_std 与 max_std 为标准差的下、上限，配置时取 0 < min_std ≤ initial_std ≤ max_std。action_dim 由环境提供。

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

## 4.1 将多个模块封装为一个新模块

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

## 4.2 跨 stage 复用复合模块

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

修改任务 observation 的顺序或缩放会改变已经训练好的策略接收的输入，即使总宽度相同也会影响播放结果。回放已有策略时应使用该次运行的配置快照；设计新策略时，首层用 obs.shape、actor 末层用 action.shape，critic 末层输出一个状态价值。

高级的 `module_artifacts` 用于为 `actor.<name>`/`critic.<name>` 提供已保存模块工件，常规训练无需手写。
