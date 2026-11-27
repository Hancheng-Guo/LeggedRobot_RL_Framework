# 6. 仿真器配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。两端均可通过 `simulator_state_manager_config.terms.foot_terrain_scan` 启用平地足端扫描。安装点按足序填写，三种测距模式与完整示例见[足端扫描配置](plans/terrain_sensor_development_plan_zh.md#三职责分配与配置)。

## 6.1 MuJoCo 基础配置

```yaml
model_path: ./assets/unitree_go1/MJCF/scene.xml # 必填
sim_dt: 0.002                                  # 必填，> 0
frame_skip: 10                                 # 必填，正整数
step_workers: 8                                # 可选，并行推进环境的线程数；默认 1
render_mode: rgb_array                         # null | human | rgb_array
foot_geom_names: [FR, FL, RR, RL]              # 足端 geom 名称
leg_joint_names: [FR_thigh_joint, FR_calf_joint, FL_thigh_joint, FL_calf_joint, RR_thigh_joint, RR_calf_joint, RL_thigh_joint, RL_calf_joint] # 大腿与小腿关节名
floor_geom_names: [floor]                      # 地面 geom 名称
foot_contact_force_threshold: 15.0             # >= 0
reset_keyframe: home                           # 可选，MJCF keyframe 名
```

控制周期为 `sim_dt × frame_skip`。`rgb_array` 用于无窗口录制，`human` 用于交互显示。

## 6.2 Isaac Sim 基础配置

下面节选 Go1 的字段说明，reset_state 的关节表仅展示一项。运行时请复制[完整的 Go1 配置](../configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml)，其中包含十二个关节的初始位置。

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
leg_joint_names: [FR_thigh_joint, FR_calf_joint, FL_thigh_joint, FL_calf_joint, RR_thigh_joint, RR_calf_joint, RL_thigh_joint, RL_calf_joint] # DOF 名称，不是 Prim path
foot_body_prim_paths:
  - /Geometry/base/trunk/FR_hip/FR_thigh/FR_calf/FR_foot
  - /Geometry/base/trunk/FL_hip/FL_thigh/FL_calf/FL_foot
  - /Geometry/base/trunk/RR_hip/RR_thigh/RR_calf/RR_foot
  - /Geometry/base/trunk/RL_hip/RL_thigh/RL_calf/RL_foot
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

两种仿真器的 `leg_joint_names` 均可省略（默认为空）；启用 `leg_joint_position_diff_l2` 时，至少须配置一个模型中已受控的腿关节名称。构建 `ModelContext` 时会解析成 `leg_joint_qpos_ids`，名称不存在时会报错。`base_joint_position_diff_l2` 中的 `base` 指未列入名单的受控关节，在当前 Go1 配置中为四个髋关节，不是浮动基座的位置或姿态。

### 6.2.1 Isaac Sim 关闭与训练产物

Windows 上的 Isaac Sim 6.1 默认应用曾在关闭后访问已卸载的 DLL，PhysX 原生克隆也会在解释器退出时访问已失效的 Python 线程状态。使用 6.1 pip 包时，项目自动采用自己的 [Kit 配置](../envs/simulators/isaac_sim_windows.kit)，通过 USD 克隆和显式碰撞过滤隔离环境。Kit 仍逐个关闭扩展，相关 DLL 保持加载到进程完成退出。复验应同时核对训练产物和原始退出码，步骤见 [手动测试说明](../tests/manual/README.md#isaac-sim-关闭诊断)。

共享地形会为重叠的机器人和批量 reset 预留 PhysX 碰撞对缓冲区。`human` 模式会加载 Stage、Property 和视口菜单，并在 Kit 停止前显式关闭窗口，以便调用方仍持有的窗口对象能安全释放。

判断一次训练是否已经留下可用产物，应检查同一次运行的 `checkpoints/<app_name>_<time>/`：`logs/<file_name>` 中出现 `All training stages completed.`，对应 `checkpoints/stage_XXX/latest.pt` 中的阶段回调状态记录完成，且 `configs/` 有本次配置快照。阶段完成时项目会保存包含回调状态的检查点；目录中的 `stage_completed` 标记不参与恢复判断。若只是达到最大迭代数或被回调停止，不能把日志中已运行的迭代数当作阶段完成；若需要确认检查点可用于继续训练，应在受信任的本地产物上实际执行一次续训加载。播放结果另看 `videos/` 下生成的文件和播放脚本的完成标记。即使这些检查都通过，退出码异常仍须单独记录，自动化任务也应将该进程视为退出失败。

## 6.3 地形配置

地形随机化由 task 的 Randomization Manager 控制，配置见 [7.8.1 Terrain](7_task_zh.md#781-terrain)。
