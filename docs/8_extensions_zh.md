# 8. 扩展注册表

[开发者手册目录](guide.md) · [项目首页](../README.md)

项目通过注册表扩展。新增实现后应在相应 registry 中注册或添加映射：

- Runner：`runners/registry.py`
- Callback：`runners/callbacks/registry.py`
- Algorithm：`rl/algorithms/registry.py`
- Policy：`rl/policies/registry.py`
- Network module：`rl/policies/modules/registry.py`
- Environment：[envs/registry.py](../envs/registry.py)
- Simulator 和 Task：分别见 [simulator 注册表](../envs/simulators/registry.py)和 [task 注册表](../envs/tasks/registry.py)
- Action、Command、Observation、Reward、Termination、Curriculum term：各 manager 的 `terms/registry.py`

Manager term 使用装饰器注册后，配置名通常由类名转换得到：action/observation/reward/termination/curriculum 使用 snake_case；command 使用类名本身。新增字段应放在 term 的构造函数参数中，然后在 YAML 的 term 配置或 `params` 中传入。

## 以新增奖励项为例

1. 在 reward/terms 下参考同类奖励实现，继承 BaseRewardTerm，使用 @register_reward 装饰类。compute 接收 TaskContext，返回每个环境的奖励张量。
2. 若新建了 Python 模块，在 [terms 包入口](../envs/tasks/managers/reward/terms/__init__.py)中导入它，使装饰器能在加载包时执行。遇到 Unknown reward term 时先核对该导入和 YAML 中的 snake_case 名称。
3. 在任务副本的 reward_manager_config.terms 中加入该名称，设置 weight，需要传给构造函数的自定义参数放在 params 中。在顶层实验中选用这个任务副本。
4. 按 [README 首次试跑](../README.md#首次试跑)运行一次短训练，在 Runner 的 TensorBoard 配置中按名称记录新奖励：单环境使用 value，多环境使用 mean。检查该奖励的输出及其量级，再决定权重和阶段阈值。
