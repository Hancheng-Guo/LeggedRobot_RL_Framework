# 7. locomotion 任务配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

建议从[完整 Go1 速度任务](../configs/tasks/locomotion_unitree_go1_velocity.yaml)复制修改。下例只列出字段层级，实际训练需填入与模型和策略匹配的动作、观测、奖励等 term。以下六项 manager 配置必填，`task_state_manager_config` 未启用时填写 `{}`：

```yaml
task_state_manager_config: {}
action_manager_config: {terms: {}}
command_manager_config: {terms: {}}
observation_manager_config: {terms: {}}
reward_manager_config: {terms: {}}
termination_manager_config: {terms: {}}
```

`curriculum_manager_config`、`event_manager_config` 和 `randomization_manager_config` 可省略、写 `null` 或写 `{}`，表示停用相应功能。event manager 当前只接受空配置。课程采样条目必须引用一个已配置的课程实例。

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

## 7.1 Task State Manager

`task_state_manager_config` 必须显式提供，未启用时填写 `{}`；配置的 task state term 在每个仿真步更新一次，局部 reset 时只清理对应环境。reward 和 observation 可通过 `TaskContext` 读取同一份状态。

```yaml
task_state_manager_config:
  terms:
    quadrupedal_gait_phase: {omega_min: 4.0, omega_max: 10.0, command_gain: 1.0}
```

已注册 task state term：

| 名称 | 参数 | 状态与行为 |
|---|---|---|
| `quadrupedal_gait_phase` | `command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`omega_min=4.0`、`omega_max=10.0`、`command_gain=1.0` | 维护 `[num_envs, 2]` 的两组对角足相位；命令范数低于 `0.1` 时分别推进到下一次触地位置后停止。相位观测项和高度奖励项可以单独启用，但都依赖此 term。 |

## 7.2 Action Manager

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

## 7.3 Command Manager

```yaml
command_manager_config:
  terms:
    lin_vel_x:
      type: Uniform
      params: {min_value: -1.0, max_value: 1.0}
    lin_vel_y:
      type: Uniform
      params: {min_value: -0.5, max_value: 0.5}
  constraints:
    lin_vel_x:
      operator: ">="
      expression: "-{lin_vel_y}.abs()"
```

term 的键是命令维度名称；`type` 使用类名（区分大小写）：

| `type` | `params` |
|---|---|
| `Uniform` | 必填 `min_value`、`max_value` |
| `Curriculum` | 必填 `curriculum`（课程实例名）、`min_value`、`max_value`、`num_cell` |
| `SymmetricCurriculum` | 同上，但 min/max 必须非负，采样后随机取正负 |

课程命令必须显式通过 `curriculum` 指定课程实例，缺失、`null` 或空字符串会报错。引用同一实例的命令组成一个联合空间；引用不同实例时独立采样。`num_cell` 为该维度划分格数，空间过大时会触发内部资源保护上限，可减少各维度的格数。命令采样类型与课程算法解耦，使用普通或对称的通用课程采样类型。

约束的 `operator` 支持 `<=`、`>=`、`<`、`>`、`!=`；表达式用 `{命令名}` 引用其他维度，可使用 `torch` 和 `abs`。`!=` 还支持 `direction: positive|negative`（默认 `positive`）决定相等时向哪一侧偏移。课程空间内的约束不能跨课程实例。

## 7.4 Observation Manager

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
| `base_height` | base 离地高度；`height_reference=ground`，可设为 `world` |
| `base_angular_velocity` | base 坐标系角速度 |
| `projected_gravity` | 重力在 base 坐标系中的投影 |
| `joint_position` | 关节位置 |
| `joint_velocity` | 关节速度 |
| `joint_position_diff` | 关节位置相对默认位置的偏差 |
| `actuator_force` | 执行器力 |
| `foot_height` | 各足 geom 中心离地高度；`height_reference=ground`，可设为 `world` |
| `foot_contact_normal_force` | 足端法向接触力 |
| `foot_contact_state` | 足端接触状态 |
| `foot_duration_tanh` | 每只足的触地状态保持时间经 `tanh(alpha × 时间)` 压缩；触地为正、离地为负，默认 `alpha=1.0` |
| `quadrupedal_gait_phase` | 读取任务状态中的两组对角足相位，输出 `[sin(phase_0), sin(phase_1), cos(phase_0), cos(phase_1)]` |
| `command` | 所有命令拼接结果 |
| `track_linear_velocity_x_error_integral` | 最近 `integral_length` 步的有符号 X 速度误差积分 |
| `track_linear_velocity_y_error_integral` | 最近 `integral_length` 步的有符号 Y 速度误差积分 |
| `track_angular_velocity_z_error_integral` | 最近 `integral_length` 步的有符号 Z 角速度误差积分 |
| `track_linear_velocity_x_error_integral_tanh` | 对 X 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `track_linear_velocity_y_error_integral_tanh` | 对 Y 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `track_angular_velocity_z_error_integral_tanh` | 对 Z 误差积分计算 `tanh(alpha * integral)`；默认 `alpha=1.0` |
| `last_action` | 最近 `lags` 步策略动作，按从新到旧排列（默认 `lags: 1`） |

## 7.5 Reward Manager

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
| `joint_position_diff_l2` | `std=1.0` |
| `base_joint_position_diff_l2` | `std=1.0`；仅对未列入 `leg_joint_names` 的受控关节取均方偏差，应使用负权重。 |
| `leg_joint_position_diff_l2` | `std=1.0`；仅对列入 `leg_joint_names` 的受控关节取均方偏差，应使用负权重。与`base_joint_position_diff_l2`均为分别取均值，直接沿用原 `joint_position_diff_l2` 的权重通常不会得到相同的合计惩罚。 |
| `joint_limit_violation_l1` | `lower_limits=null`、`upper_limits=null` |
| `joint_power_l1` | — |
| `base_linear_velocity_z_l2` | — |
| `base_linear_velocity_z_l2_xy_speed_weighted` | `min_speed=0.5` |
| `base_height_l2` | 必填 `target_height`；`std=1.0`、`max_normalized_error=null`、`height_reference=ground`（可选 `world`） |
| `base_height_l2_xy_speed_weighted` | 必填 `target_height`；`min_speed=0.5`、`height_reference=ground`（可选 `world`） |
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
| `quadrupedal_command_adaptive_gait_phase_height_l2` | 必填 `landing_distance`（落地时足端到 base 平面的距离幅值，必须为正数）；`lift_height=0.08`（相对落地位置的非负举升高度）、`std=0.08`。读取 `quadrupedal_gait_phase` task state term 的相位，并根据 `foot_phase_real/imag` 命令的四足分量（每个 `[num_envs, 4]`）逐足添加 `atan2(imag, real)` 相位偏移；相对 base 平面的有符号落地目标为 `-landing_distance`，摆动最高点为 `-landing_distance + lift_height`；返回四足归一化高度误差的平方均值，应使用负权重。 |
| `quadrupedal_command_adaptive_gait_phase_height_l2_exp` | 必填 `landing_distance`；`lift_height=0.08`、`std=0.08`。目标高度与相位偏移同上；逐足计算 `exp(-((目标高度 - 实际高度) / std)^2)` 后取四足均值，应使用正权重。 |
| `foot_state_duration_command_weighted_exp` | 整组足端触地状态保持时间；`command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`sigma=1.0` |
| `foot_state_duration_cubic_command_weighted_exp` | 整组足端触地状态保持时间的三次方版本；参数同上 |
| `foot_state_duration_cubic_command_tanh_weighted_exp` | 整组足端触地状态保持时间的三次方版本，指令范数先取 `tanh`；参数同上 |
| `foot_duration_cubic_command_tanh_weighted_exp` | 分别统计每只足的触地状态保持时间，取三次方并按 `tanh` 指令范数加权后求平均；参数同上 |
| `foot_state_switch` | `hold_time=0.2`；触地状态在保持时间内切换时返回惩罚值 |
| `foot_sliding_velocity_l2` | 触地足端 geom 中心世界 xyz 速度平方，按全部足及三个速度分量取均值，非触地足贡献为零；包含法向运动，不是严格的接触切向滑移 |
| `foot_lift_height_diff_command_weighted_exp` | 必填 `target_height`；`height_std=0.03`、`command_std=0.5`、`command_names=[lin_vel_x, lin_vel_y, ang_vel_z]`、`height_reference=ground`（可选 `world`）；仅对摆动脚计算高度奖励，零指令时为零 |
| `foot_lift_height_diff_command_gated_l2` | 必填 `target_height`；`height_std=null`（默认使用 `target_height`）、`height_reference=ground`（可选 `world`）；命令范数 `<=0.1` 或支撑脚时贡献为零，仅对运动中的摆动脚计算相对 `target_height` 的误差；使用负权重作为惩罚 |
| `quadrupedal_foot_velocity_diff_l2` | — |
| `foot_contact_without_command` | `command_names=[lin_vel_x, lin_vel_y, ang_vel_z]` |

名称由注册类名自动转为 snake_case；大小写缩写会被正确拆分，例如 `TrackAngularVelocityZL2Exp` → `track_angular_velocity_z_l2_exp`。

## 7.6 Termination Manager

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
| `base_height` | 必填 `min_height`；`height_reference=ground`（可选 `world`），低于阈值即终止 |
| `terrain_bounds` | 无参数。机身或任一足端离开出生地块时终止 |
| `terrain_map_bounds` | 无参数。机身或任一足端离开整张地面覆盖范围时终止 |

普通平地不会触发这两个地形终止项。跨地块行走时可只启用 `terrain_map_bounds`。达到 `max_episode_steps` 的 episode 会标记为 `truncated`。地面覆盖范围见 [Terrain](#781-terrain)。

## 7.7 Curriculum Manager

### 实例配置

```yaml
curriculum_manager_config:
  terms:
    locomotion:
      type: fast_lpac_reward
      params:
        temperature: 0.01
        exploration: 0.1
        min_samples_per_cell: 5
        normalize_by_episode_steps: false
        min_coverage: 0.8
        reward_sources:
          - reward/track_linear_velocity_x_l2_exp_and_logcosh
          - reward/track_angular_velocity_z_l2_exp_and_logcosh: 0.5
```

| 名称 | 参数 |
|---|---|
| `lpac_reward` | `temperature=0.01`（>0）、`exploration=0.1`（[0,1]）、`min_samples_per_cell=5`（正整数）、`normalize_by_episode_steps=false`（布尔值） |
| `fast_lpac_reward` | 同上，并增加 `min_coverage=0.8`（(0,1]） |

课程条目必须显式填写 `type`，算法参数放在 `params` 中。每个实例有一个采样空间，并独立维护奖励统计和采样概率。

### 独立采样与联合采样

让平移和转向引用不同的实例，即可分别学习各自的采样概率。

```yaml
curriculum_manager_config:
  terms:
    translation: {type: fast_lpac_reward, params: {reward_sources: [reward/track_linear_velocity_x_l2_exp_and_logcosh]}}
    rotation: {type: fast_lpac_reward, params: {reward_sources: [reward/track_angular_velocity_z_l2_exp_and_logcosh]}}

command_manager_config:
  terms:
    lin_vel_x: {type: Curriculum, params: {curriculum: translation, min_value: -1.0, max_value: 1.0, num_cell: 10}}
    ang_vel_z: {type: Curriculum, params: {curriculum: rotation, min_value: -1.0, max_value: 1.0, num_cell: 5}}
```

联合规划平移和转向时，让两个命令引用同一实例，并为该实例配置共同的 `reward_sources`。

```yaml
curriculum_manager_config:
  terms:
    locomotion:
      type: fast_lpac_reward
      params:
        reward_sources:
          - reward/track_linear_velocity_x_l2_exp_and_logcosh
          - reward/track_angular_velocity_z_l2_exp_and_logcosh

command_manager_config:
  terms:
    lin_vel_x: {type: Curriculum, params: {curriculum: locomotion, min_value: -1.0, max_value: 1.0, num_cell: 10}}
    ang_vel_z: {type: Curriculum, params: {curriculum: locomotion, min_value: -1.0, max_value: 1.0, num_cell: 5}}
```

上例将两个命令维度划分为 `10 × 5 = 50` 个联合 cell，并用两项奖励共同评价每个速度组合，学习对应的采样概率。无需课程学习时，可省略 `curriculum_manager_config` 或配置为空，command 使用独立的 `Uniform` 采样。

联合空间过大时会报错。可减少各命令维度的格数，或将维度分到不同的课程实例。地形类别、地形难度和摩擦倍率也可加入联合空间，配置见 [Terrain](#781-terrain) 和 [Friction](#782-friction)。

### 奖励来源与回合计分

`reward_sources` 为该课程实例的必填非空列表，元素为来源字符串或单项字典：

- `reward/<name>`：使用该项原始奖励 × 训练权重。
- `reward/<name>: c`：使用该项原始奖励 × 覆盖系数 `c`，替换训练权重，不是额外乘一次。允许有限的负数和零；显式 `null` 不表示省略系数，应改用字符串形式。
- `reward`：使用逐环境的所有奖励按训练权重加权后的总和。整体奖励不支持覆盖系数，可写作 `reward_sources: [reward]`。

以上来源均不做整体 `dt` 缩放；同一实例的来源直接求和，不自动归一化。仅接受完整名称，同一实例不能重复配置同一来源，不能全部为显式零系数。`reward` 可以和逐项来源混合，选中的逐项会再次计入，不做隐式去重。

LPAC/FastLPAC 通过 `normalize_by_episode_steps` 决定 episode 结束时如何向 cell 计分：

- `false`（默认）：直接使用 episode 累计奖励，不除以实际步数，将存活时长和奖励所衡量的表现（例如跟踪质量）共同纳入评价。
- `true`：累计奖励除以该 episode 的实际步数，使用平均每步奖励，减弱 episode 长度本身的影响。

两种模式下，cell 分数均会再除以采集的 episode 数，分别得到“episode 累计奖励的均值”或“episode 每步平均奖励的均值”。后者是 episode 等权平均，不是所有采样步合在一起求平均；尚未采集步骤的 episode 不计入统计。以下存活时长相关注意事项针对 `false` 的累计模式。

- 当每步奖励主要为正时，相近质量下更长的 episode 通常得到更高累计分数；当每步奖励主要为负时，更早结束可能因少累计负奖励而得到更高分。例如每步均为 `-0.1`，1000 步累计为 `-100`，100 步累计为 `-10`，后者分数更高。应根据所选奖励的组合判断它是否符合预期的存活偏好。
- learning progress 通常使用当前 cell 分数与历史分数之差，不是 PPO advantage。上述例子会产生 `+90` 的差分；减去历史基线不能消除由 episode 长度变化带来的影响。FastLPAC 首次初始化另有 baseline 填充规则。
- 累计分数的尺度会受到 episode 长度、最大步数和奖励权重影响；修改这些设置后，应重新检查采样 `temperature` 及使用相关分数的阈值，避免直接沿用每步均值口径下的数值。
- curriculum 使用未乘 `control_dt` 的课程奖励，累计模式对其逐步求和，归一化模式再除以实际步数；训练传给 PPO 的总奖励仍由环境统一乘 `control_dt`，两条链路的缩放相互独立。切换来源、归一化模式或控制步长后，应重新检查 `temperature` 和相关阈值。

### Cell 评价与训练指标

回合结束后，课程通过 `finalize_episodes()` 结算奖励，再通过 `resample()` 为新回合分配 cell。命令随后在分配的区间内采样。

每个课程实例使用独立的 cell 评价缓冲，每个 cell 收集至少 `min_samples_per_cell` 个已结束 episode 后，保存这批 episode 分数的均值，再清空该批累计；下一批未完成时保留上次评价。episode 分数受 `reward_sources` 和 `normalize_by_episode_steps` 控制。FastLPAC 为未采样 cell 填充的 baseline 不计入真实评价，也不会提高覆盖率。

指标前缀为 `curriculum/<instance>/`，例如 `curriculum/locomotion/coverage`：

| 指标 | 含义 |
|---|---|
| `coverage` | 各 cell 最后一次完成的批次评价中，有效 cell 数 / 经过 constraints 过滤后的总 cell 数；评价需有足量真实 episode，且均值有限。尚未完成首批评价的 cell 不计入，未完成的新批次不影响已有评价的有效性 |
| `valid_cell_count` | 有效评价 cell 数 |
| `num_cells` | 总 cell 数 |
| `all_cells_valid` | 全部 cell 均有有效评价时为 1，否则为 0 |
| `statistics_revision` | 本实例每次有一批 cell 评价完成时递增；仅收集指标或无新步骤的 reset 不递增 |
| `reward_mean` | 有效 cell 最新批次均值的等权平均，不按 cell 的采样次数加权 |
| `reward_min` | 有效 cell 最新评价的最小值 |
| `reward_max` | 有效 cell 最新评价的最大值 |
| `reward_std` | 有效 cell 最新评价的总体标准差；只有一个有效 cell 时为 0 |
| `learning_progress` | 各 cell 的学习进度向量，沿用现有 TensorBoard 直方图记录 |
| `probabilities` | 各 cell 当前的采样概率向量，按相同 cell 顺序记录为直方图 |

初始化时覆盖率为 0，尚无有效 cell 时不导出 `reward_*` 指标，避免用零分或估计值冒充真实结果。有限评价不会因采样窗口清空或新批次尚未完成而失效；若新一批评价为 NaN/Inf，则该 cell 标记为无效。各 cell 使用自己最后一次完成的批次评价，可能来自不同时刻。任务指标在本步 done 环境 reset、cell 结算完成后统一收集，返回值均为独立快照；评估时也会导出冻结的学习进度与采样概率。

Stage 可将 `revision_metric` 指向实例的 `statistics_revision`，避免重复计入同一份结果，并将 `validity_metric` 指向 `all_cells_valid`，要求该实例的全体 cell 评价有效；配置方式见 [1.1 节](1_experiments_zh.md#11-stage-写法)。

### 评估、回放与续训

评估与回放时冻结 cell 评价和采样概率，命令仍可从冻结的分布生成。

训练检查点会保存 curriculum 的采样状态、cell 统计，以及 Stage 条件窗口和已采样的统计版本，续训时会校验配置并恢复；配置 `revision_metric` 时，同一统计版本不会重复计入窗口。

未结束 episode 的累计数据不会恢复。涉及 terrain 的课程会保存当前 cell 和独立采样随机状态，恢复后的首次 reset 保留该分配并重建地形，后续回合才重新采样；纯命令课程按恢复后的概率开始新 episode。检查点不保存完整模拟器动力学状态，因此不保证续训轨迹与连续训练逐步一致。不含环境状态的检查点不会恢复 curriculum 数据，配置 `revision_metric` 或 `validity_metric` 的条件窗口也会重新积累。

## 7.8 Randomization Manager

MuJoCo 和 Isaac Sim 都支持地形随机化（`terrain`）和摩擦倍率随机化（`friction`）。两项均配置在 task YAML 的 `randomization_manager_config.terms` 下，可以独立启用，也可以组合使用。省略整个配置或写 `{}` 时，使用仿真场景原有的地面和接触参数。

### 7.8.1 Terrain

#### 快速配置

下面的示例让每个环境按权重选择类别，并独立均匀抽取难度。只配置 `plane` 时，使用仿真场景原有的平地。

```yaml
randomization_manager_config:
  terms:
    terrain:
      category:
        type: Weighted
        params:
          values:
            plane: {weight: 1}
            slope: {weight: 1, max_slope: 0.15}
            rough: {weight: 1, max_height: 0.04}
      difficulty:
        type: Uniform
        params:
          values: [0.25, 0.6, 1.0]
      patch_size: [4, 4]
      resolution: 0.1
      spawn_size: [1.5, 1.2]
      seed: 0
```

#### 地形类别与公共参数

- **`plane`**：平坦地面。和其他类别混用时，以零高度地块参与采样。
- **`slope`**：坡地，并在中央留出平坦出生区，区外沿 X、Y 两轴的正负方向同时抬升或降低，外缘回到零高度。实际采样系数在 `[-max_slope × difficulty, +max_slope × difficulty)` 内均匀抽取。`max_slope` 控制采样系数上限，双轴叠加后的实际坡度另由网格决定。
- **`rough`**：粗糙地面，在出生区及其相邻一圈网格之外，使各顶点高度在 `[-max_height × difficulty, +max_height × difficulty)` 内独立均匀采样。外缘回到零高度，与周围地面衔接。

| `terrain` 下的参数 | 含义 | 默认值 |
|---|---|---|
| `patch_size` | 单块地形沿 X、Y 的全尺寸，单位为米 | `[4.0, 4.0]` |
| `resolution` | 最大网格间距，单位为米 | `0.1` |
| `spawn_size` | 中央平坦出生区的全尺寸，须容纳 reset 姿态下机器人的完整碰撞投影 | `[0.5, 0.5]` |
| `spawn_margin` | 校验出生区尺寸时，每侧预留的最小边距，单位为米 | `0.5` |
| `seed` | 地形种子，可为非负整数。留空时使用运行种子 | `null` |

#### 类别、难度与课程

`category` 和 `difficulty` 是两个独立采样轴，各用自己的 `type` 和 `params.values`。类别候选以地形名为键。难度候选是 `[0, 1]` 内互不重复的实际数值，抽到的值会缩放当前地形的起伏。省略类别时使用固定平地，省略难度时使用固定值 `1.0`。

| `type` | `category` | `difficulty` |
|---|---|---|
| `Fixed` | 只填一个类别 | 只填一个难度 |
| `Weighted` | 按类别的 `weight` 抽样，权重默认是 `1` | — |
| `Uniform` | — | 均匀抽取一个难度候选 |
| `Curriculum` | 把类别作为离散课程轴 | 把难度作为离散课程轴 |

类别的 `weight` 仅用于 `Weighted`。`Curriculum` 轴在 `params.curriculum` 填课程实例名。命令、类别和难度引用同一实例时联合采样，引用不同实例时各自采样。可以只让其中一个地形轴加入课程。下面两段写在 `terrain` 下，会把类别和难度都接到 `locomotion`。

```yaml
category:
  type: Curriculum
  params:
    curriculum: locomotion
    values:
      plane: {}
      slope: {max_slope: 0.15}
      rough: {max_height: 0.04}
difficulty:
  type: Curriculum
  params:
    curriculum: locomotion
    values: [0.25, 0.6, 1.0]
```

完整联合课程配置见 [ANYmal C 地形任务](../configs/tasks/locomotion_anymal_c_phase.yaml)。

#### 地面布局与越界

| | MuJoCo | Isaac Sim |
|---|---|---|
| **仿真器 YAML** | `floor_geom_names` 指向一个未旋转、启用碰撞的 world-body 平面。启用几何地形时，场景只能有这一个平面 | 启用几何地形时将 `floor_prim_paths` 留空 |
| **地面布局** | 每个环境独立分配一个高度场，四周各延伸 `1000` 米平地 | 启动时把预建地块拼成一张共享静态网格，机器人之间保持碰撞隔离 |
| **换地方式** | 更新所选环境的高度场 | 把所选机器人 reset 到预建区域的出生位置 |
| **越出出生地块** | 进入该环境自己的外围平地 | 可以走到相邻地块，回合成绩仍计入出生时分配的课程条件 |

Isaac 的 `terrain_atlas` 写在 **simulator YAML** 中。以下是默认布局。`variants` 是每个类别和难度组合的预建版本数，`region_tiles` 表示每个版本沿 X、Y 方向重复铺几块，`border_width` 是整图每侧外围平地的宽度（米）。例如单块地形为 `4 × 4` 米时，默认每个变体占 `12 × 4` 米。增大区域或提高网格密度都会增加三角形数，整图上限为 200 万个三角形。

```yaml
terrain_atlas:
  variants: 2
  region_tiles: [3, 1]
  border_width: 6.0
```

可按出生地块或整张地面覆盖范围设置越界终止。整张地面覆盖范围包含外围平地。配置见 [Termination Manager](#76-termination-manager)。

#### 重采样、恢复与回放

- **`resample_patch`** 默认为 `false`，类别和难度未变时复用地块。设为 `true` 后，每次训练 reset 时，MuJoCo 会生成同条件下的新高度网格并写入现有高度场，Isaac 会切换到同条件下的下一个预建变体。
- **`resample_spawn`** 默认为 `false`，出生位置由当前地块的种子确定。设为 `true` 可以在保留同一地块时重新抽取安全出生位置。
- **续训**需要相同的地形配置、候选顺序、种子和环境数。Isaac 还需要相同的 `terrain_atlas`。恢复后的首次 reset 复现快照中的分配和出生位置，后续训练 reset 再继续采样。
- **评估与回放**复用当前地块、出生位置和已有课程 cell，并冻结课程统计。

#### 地面状态

`get_state()` 提供 `base_ground_height` 和 `foot_ground_heights`。带 `height_reference` 参数的高度条目可用 `ground`（默认）或 `world`，分别表示离地高度和世界 Z 值。`base_position` 仍表示世界位置。

`terrain_in_bounds` 表示机身及足端是否都在出生地块内，`ground_valid` 表示它们下方是否都有可查询的地面。在两个后端中，机器人走上外围平地后可以同时出现 `terrain_in_bounds=False` 和 `ground_valid=True`。几何地面覆盖范围之外的高度查询返回 NaN，`SimulatorState` 会将对应高度信号清零。Isaac 刚 reset 时接触力为零，完成下一次物理步后再读取新接触力。

复跑入口见 [MuJoCo 仿真配置](../configs/simulators/mujoco_anymal_c_rgb_array.yaml) 和 [Isaac Sim 几何地形验收](../tests/manual/README.md#isaac-sim-几何地形)。

### 7.8.2 Friction

#### 配置与倍率

`friction` 为每个环境采样一个倍率。**目标足地接触系数 = 初始足地接触系数 × 倍率**。倍率 `1.0` 对应初始接触系数，`0.0` 对应零摩擦目标。

```yaml
randomization_manager_config:
  terms:
    friction:
      type: Uniform
      params:
        factors: [0.5, 1.0, 1.5]
      seed: 17
```

| `type` | 配置与采样规则 |
|---|---|
| `Fixed` | `params.factors` 包含一个倍率 |
| `Uniform` | 在候选中等概率选择 |
| `Curriculum` | 在 `params.curriculum` 填课程实例名，`params.factors` 作为离散轴 `friction` 的候选 |

候选倍率须为有限、非负、互不重复的数值。省略 `seed` 时使用运行种子。若摩擦、地形和命令引用同一课程实例，课程会联合分配这些维度。

#### 两个后端如何应用倍率

| 后端 | 应用方式 | 影响范围 |
|---|---|---|
| MuJoCo | 将本环境的地面与足端滑动摩擦系数分别乘以倍率，沿用模型的接触优先级。同优先级时取两侧较大值 | 地面包含高度场和外围平地。足端与其他物体、机身与地面的接触也可能受材料变化影响 |
| Isaac Sim | 分别计算原始足地静摩擦和动摩擦，各乘一次倍率，再反推本环境足端材质应写入的系数 | 共享地面保留原材质，倍率随机器人跨地块移动。足端碰到其他物体也会使用补偿后的材质 |

Isaac 计算基线时沿用原接触的 `average`、`min`、`multiply` 或 `max` 规则。正摩擦地面通常通过 `multiply` 实现补偿，足端写入「目标接触系数 ÷ 地面系数」。所有地面的静、动摩擦均为零时，通过 `max` 合成，足端直接写入目标系数。

#### 地面材质与模型限制

- **MuJoCo** 的实际接触滑动摩擦下限为 `1e-5`。启用全局接触覆盖，或使用涉及地面、足端的显式接触对时，绑定会报错。
- **Isaac 内置地形**的平地、坡地、粗糙地块和外围平地共用一份物理材质，静摩擦和动摩擦均为 `0.5`。
- **Isaac 自定义地面**如果有多种物理材质，同一足端必须能用一组系数实现所有地面的目标。正摩擦地面的 `max` 合成优先于足端的 `multiply`，当前实现会在绑定时报错。需要对零摩擦通道产生正接触系数且无法使用 `max`，或各地面要求的补偿系数不同，也会报错。

#### Reset、恢复与验收

每次应用倍率都从初始值计算。局部 reset 只更新选中的环境。评估、回放和续训首次 reset 复用已保存的倍率。reset 失败时恢复所选环境的摩擦、物理状态和采样进度。

可复跑验证见 [MuJoCo 摩擦验收](../tests/manual/README.md#摩擦倍率随机化) 和 [Isaac Sim 摩擦验收](../tests/manual/README.md#isaac-sim-摩擦倍率随机化)。
