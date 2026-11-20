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

`curriculum_manager_config`、`event_manager_config` 和 `randomization_manager_config` 可省略或写 `null`，分别表示不启用课程、事件和随机化，也可显式写 `{}`。event manager 当前仍为预留项，不接受非空配置。未启用随机化时使用 simulator 原有地面。命令若引用了未配置的课程，仍会明确报错，不自动退回独立采样。

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
| `terrain_bounds` | 无参数；机身或任意足端离开随机地形区域时终止；未配置有限地形时不会触发 |

Episode 达到 `max_episode_steps` 属于 `truncated`，不是 termination term。

## 7.7 Curriculum Manager

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

课程条目必须显式填写 `type`，算法参数放在 `params` 中（无参数时可省略）。一个实例对应一个联合采样空间，独立持有奖励统计与采样概率。同一种算法可以构建多个实例，例如分别学习平移与转向：

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

若需要联合规划这两个命令维度，就让它们引用同一个实例，并为该实例配置共同的 `reward_sources` :

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

回合结束后，课程通过 `finalize_episodes()` 结算该回合的奖励，再通过 `resample()` 为新回合分配 cell，command 随后在对应区间内采样。评估与播放时冻结 cell 评价和采样概率，仍允许从冻结分布生成命令。

联合空间过大时会报错，应减少各维度的格数或拆分课程实例。terrain 的 `category`／`difficulty` 轴可通过 `type: Curriculum` 和 `params: {curriculum: locomotion}` 引用同一实例，将地形类型、难度与命令组成联合空间；引用另一实例则分别学习。地形采样方式见 [7.8.1 Terrain](#781-terrain)。

`reward_sources` 为该课程实例的必填非空列表，元素为来源字符串或单项字典：

- `reward/<name>`：使用该项原始奖励 × 训练权重。
- `reward/<name>: c`：使用该项原始奖励 × 覆盖系数 `c`，替换训练权重，不是额外乘一次。允许有限的负数和零；显式 `null` 不表示省略系数，应改用字符串形式。
- `reward`：使用逐环境的所有奖励按训练权重加权后的总和。整体奖励不支持覆盖系数，可写作 `reward_sources: [reward]`。

以上来源均不做整体 `dt` 缩放；同一实例的来源直接求和，不自动归一化。仅接受完整名称，同一实例不能重复配置同一来源，不能全部为显式零系数。`reward` 可以和逐项来源混合，选中的逐项会再次计入，不做隐式去重。

**Episode 累计奖励的统计口径与注意事项：**

LPAC/FastLPAC 通过 `normalize_by_episode_steps` 决定 episode 结束时如何向 cell 计分：

- `false`（默认）：直接使用 episode 累计奖励，不除以实际步数，将存活时长和奖励所衡量的表现（例如跟踪质量）共同纳入评价。
- `true`：累计奖励除以该 episode 的实际步数，使用平均每步奖励，减弱 episode 长度本身的影响。

两种模式下，cell 分数均会再除以采集的 episode 数，分别得到“episode 累计奖励的均值”或“episode 每步平均奖励的均值”。后者是 episode 等权平均，不是所有采样步合在一起求平均；尚未采集步骤的 episode 不计入统计。以下存活时长相关注意事项针对 `false` 的累计模式。

- 当每步奖励主要为正时，相近质量下更长的 episode 通常得到更高累计分数；当每步奖励主要为负时，更早结束可能因少累计负奖励而得到更高分。例如每步均为 `-0.1`，1000 步累计为 `-100`，100 步累计为 `-10`，后者分数更高。应根据所选奖励的组合判断它是否符合预期的存活偏好。
- learning progress 通常使用当前 cell 分数与历史分数之差，不是 PPO advantage。上述例子会产生 `+90` 的差分；减去历史基线不能消除由 episode 长度变化带来的影响。FastLPAC 首次初始化另有 baseline 填充规则。
- 累计分数的尺度会受到 episode 长度、最大步数和奖励权重影响；修改这些设置后，应重新检查采样 `temperature` 及使用相关分数的阈值，避免直接沿用每步均值口径下的数值。
- curriculum 使用未乘 `control_dt` 的课程奖励，累计模式对其逐步求和，归一化模式再除以实际步数；训练传给 PPO 的总奖励仍由环境统一乘 `control_dt`，两条链路的缩放相互独立。切换来源、归一化模式或控制步长后，应重新检查 `temperature` 和相关阈值。

**Cell 评价统计：**

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

训练检查点会保存 curriculum 的采样状态、cell 统计，以及 Stage 条件窗口和已采样的统计版本，续训时会校验配置并恢复；配置 `revision_metric` 时，同一统计版本不会重复计入窗口。

未结束 episode 的累计数据不会恢复。涉及 terrain 的课程会保存当前 cell 和独立采样随机状态，恢复后的首次 reset 保留该分配并重建地形，后续回合才重新采样；纯命令课程按恢复后的概率开始新 episode。检查点不保存完整模拟器动力学状态，因此不保证续训轨迹与连续训练逐步一致。不含环境状态的检查点不会恢复 curriculum 数据，配置 `revision_metric` 或 `validity_metric` 的条件窗口也会重新积累。

## 7.8 Randomization Manager

`RandomizationManager` 通过注册表构建和调度随机化 term。MuJoCo 支持 `terrain` 和 `friction`，可单独启用或组合。配置属于 **task YAML**，省略 `randomization_manager_config` 或设为 `{}` 时沿用模型的地面和接触参数。

### 7.8.1 Terrain

**配置与支持范围**

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

MuJoCo 支持三种地形（`plane`、`slope` 和 `rough`），Isaac Sim 当前只支持平地，非平地配置会明确报错。可配对使用 [ANYmal C 地形任务](../configs/tasks/locomotion_anymal_c_phase.yaml) 与 [MuJoCo 仿真配置](../configs/simulators/mujoco_anymal_c_rgb_array.yaml)；仿真器 YAML 只保存模型和物理参数。

`category` 和 `difficulty` 是 terrain 内部的两个采样子 term，均采用 `type + params`。`category.params.values` 按地形名保存候选及参数，仅 `Weighted` 允许填写 `weight`，默认为 1，决定类别采样概率；`Fixed` 和 `Curriculum` 均拒绝显式权重。`difficulty.params.values` 保存 `[0, 1]` 内不重复的难度候选，与地形类别不按位置对应。省略 category 时使用固定 plane；省略 difficulty 时使用固定难度 `1.0`。尺寸、分辨率、出生区、种子及重采样开关属于 terrain。

**`plane`**：平坦地面，高度相对原地面为零。category 仅含 plane 时沿用原有无限平面；与其他类型混合配置时，使用同尺寸的有限高度场，所有顶点高度为零。

**`slope`**：中央出生区保持平坦，区外沿 ±x、±y 四个方向同号起伏：正 `slope` 均抬升，负值均降低，接近外边界时回到零高度，与外围平地衔接。系数均匀采样于 `[-max_slope × difficulty, +max_slope × difficulty)`；`max_slope` 限制采样系数，不限制双轴合成后的实际坡度。

**`rough`**：各网格顶点在 `[-max_height × difficulty, +max_height × difficulty)` 内独立均匀采样，形成随机起伏；中央出生区及其一圈网格保持平坦，最外圈顶点高度为零。网格间由三角形连接，`resolution` 控制最大网格间距；该类型不受 `max_slope` 约束。

选中的实际难度数值缩放 slope 的采样系数和 rough 的高度幅度；默认难度 `1.0` 使用配置的完整幅度，plane 始终为零高度。

MuJoCo 非平地要求 `floor_geom_names` 指定一个未旋转、启用碰撞且属于 world body 的 plane，原位替换为高度场并保留地面配置。`spawn_size` 是中央平坦区的全尺寸，须容纳 reset 姿态下机器人的完整碰撞投影；ANYmal C 示例使用 `[1.5, 1.2]`。不支持额外 plane、倾斜地面或不适配的出生区。

**采样策略**

两个轴支持的策略如下，候选放在各自的 `params.values` 中：

| 策略 | category | difficulty |
|---|---|---|
| `Fixed` | `params.values` 必须且只能包含一个类别，直接选择它；拒绝 `weight` | `params.values` 必须且只能包含一个难度值 |
| `Weighted` | 每次训练 reset 按类别权重抽样 | 不支持 |
| `Uniform` | 不支持 | 每次训练 reset 均匀抽取难度候选 |
| `Curriculum` | 将类别候选编号作为离散课程轴，拒绝 `weight` | 将难度候选编号作为离散课程轴 |

课程轴必须填写 `params.curriculum: 实例名`。引用同一实例时联合规划，引用不同实例时分别学习；只有配置为 Curriculum 的轴贡献课程维度。课程处理离散编号，子 term 将编号映射回类别或难度；用户配置的难度值不是索引。独立采样不消费训练随机流。ANYmal 示例将两个轴与命令共同接入 locomotion：

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
resample_patch: false
resample_spawn: false
```

仅让难度参与课程时，将 category 的 type 改为 `Weighted` 或其他非 `Curriculum` type 并移除其 `params` 中的 `curriculum` ；仅让类别参与时，可省略 difficulty，或将其配置为其他非 `Curriculum` type 。固定 `seed` 可复现场景；`seed: null` 跟随运行种子。旧的 `types`、`difficulty_levels` 和 `sampling` 配置不再接受。

类别和难度未变时复用当前 patch；设置 terrain 下的 `resample_patch: true` 才在每次训练 reset 主动更换同条件下的网格。出生位置默认由 patch 种子确定；`resample_spawn: true` 可独立重新抽取安全出生位置。两项默认均为 false。同条件换网格和换出生位置也可分别请求 `request_reallocation(env_ids)`、`request_spawn(env_ids)`；难度轴未使用课程时可通过 `set_difficulty(env_ids_tensor, difficulties)` 请求候选中的实际难度数值，难度轴使用课程时由课程决定。

**检查点、评估与播放**

续训时保持地形配置、候选顺序、种子和环境数与快照一致。恢复后的首次 reset 会重建同一块地形和出生位置，后续训练 reset 再按配置采样。评估和播放固定当前地形、出生位置及对应课程 cell，课程统计保持冻结。

**地面信号与使用边界**

simulator 在 `get_state()` 中批量查询机身及足端地面高度，并将 `base_ground_height`、`foot_ground_heights`、`ground_valid` 和 `terrain_in_bounds` 作为可选物理字段一并构造到 `SimulatorState`，供 task、观测、奖励和终止读取；高度条目默认 `height_reference: ground`，可显式设为 `world`；`base_position`、机身平面相对的步态信号和世界速度项保留原含义。普通平地不执行这项查询。

MuJoCo 几何地形外围每侧延伸 1000 米平地，高度为原地面世界 z，不覆盖内部负高度地形。高度查询按机身/足端各点分别选择高度场或外围平地，跨界后观测、奖励与接触信号仍正常计算。

`ground_valid` 表示机身及全部足端的地面高度是否有效，`terrain_in_bounds` 表示它们是否全部处于随机区域内，两者均为 `[env]`。`termination_manager_config.terms.terrain_bounds: {}` 可选择越出随机区域即终止；当前地形示例未启用该项，可在外围继续运动；按需添加该项后，越出随机区域会终止。查询超出外围覆盖的位置返回 NaN，状态采集将该环境标记为 `ground_valid=False` 并屏蔽无定义高度信号。

### 7.8.2 Friction

MuJoCo 的 `friction` term 为每个环境采样一个倍率。该环境中地面与足端几何体的滑动摩擦系数分别按「模型初始系数 × 采样倍率」计算，地面包含启用高度场时的外围平地。

```yaml
randomization_manager_config:
  terms:
    friction:
      type: Uniform
      params:
        factors: [0.5, 1.0, 1.5]
      seed: 17
```

| `type` | 采样规则 |
|---|---|
| `Fixed` | `params.factors` 包含一个倍率 |
| `Uniform` | 在候选中等概率选择 |
| `Curriculum` | 通过 `params.curriculum` 引用课程实例，候选作为离散轴 `friction` |

倍率必须为有限、非负、互不重复的数值。`1.0` 对应模型初始值。`0.0` 将目标几何体的滑动摩擦设为零，MuJoCo 生成接触时仍会将滑动摩擦限制在 `1e-5` 或以上。

省略 `seed` 时使用运行种子。地形、摩擦和命令引用同一实例时联合采样。评估、回放和续训首次 reset 复用已保存的倍率。每次计算都以模型初始值为基准。reset 失败时，所选环境恢复到 reset 前的几何体摩擦系数、物理状态和采样进度。

MuJoCo 按接触优先级选用一侧的系数，优先级相同时取两侧较大值。两侧共用一个倍率，因此地面或足端占优时，足地接触滑动摩擦都会随倍率变化。全局接触覆盖和涉及目标几何体的显式接触对都会覆盖几何体摩擦，启用该 term 时会报错。当前该 term 支持 MuJoCo。

可复跑验证见[手动诊断与验收](../tests/manual/README.md#摩擦倍率随机化)。
