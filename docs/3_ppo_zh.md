# 3. PPO 算法配置

[开发者手册目录](guide.md) · [项目首页](../README.md)

以下配置路径和命令均以项目根目录为基准。

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
advantage_norm: true       # 是否标准化 advantage，省略时为 false
```

gamma 决定未来奖励的折扣，gae_lambda 控制优势估计的时间跨度。clip_range 限制 PPO 策略目标中概率比的裁剪区间，value_coef 和 entropy_coef 分别设置价值损失与熵项的权重，max_grad_norm 限制梯度范数。init_learning_rate 设置优化器初始学习率；若 Runner 启用了 adaptive_learning_rate，训练中学习率还会随监控指标调整。

一次更新先从每个环境收集 rollout_length 步，再对同一批样本训练 num_epochs 轮。前馈策略按单步样本分批，num_mini_batches 最大为 num_envs × rollout_length；循环策略需要保留整段时间序列，按环境分批，上限为 num_envs。两种方式都允许不整除，余数会分摊到各批。

例如 Go1 CPU 单阶段速度配置使用 32 个环境、256 步、4 个小批量和 3 轮更新。前馈策略每批有 2048 个样本，每次采样后执行 12 次优化器更新。切换到单环境 GRU 试跑时，要同时把 num_mini_batches 改成 1。
