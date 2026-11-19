# 5. 环境配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

`configs/environments/*.yaml`：

```yaml
num_envs: 64             # 必填，正整数
max_episode_steps: 1000  # 必填，正整数；达到后 truncated
```

按 Go1 示例的 sim_dt=0.002 秒、frame_skip=10，每个环境步为 0.02 秒，1000 步对应最长 20 秒仿真时间。任务终止条件（例如躯干触地）可以提前结束单个环境的 episode。rollout_length 决定何时更新策略，一段 rollout 可以跨越 episode。若将 max_episode_steps 调低到 900 以下，也要修改依赖 rollout/mean_len ≥ 900 的阶段条件，否则阶段无法达标。

MuJoCo 会为每个环境建立实例，数量过大会明显增加 CPU/内存开销；Isaac Sim 适合大规模 GPU 并行环境。
