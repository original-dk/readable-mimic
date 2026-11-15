# Readable Mimic

按 Motion → command / obs → history → Actor → MuJoCo → Buffer，以及
Buffer → WM 和 imagined rollout → Actor / Critic 两条分支组织的简化实现。

## 安装与运行

```powershell
cd D:\pycharm_project\readable-mimic
uv sync
uv run python tools/download_data.py
uv run python -m readable_mimic.train --config config.json
uv run pytest -q
```

配置对照和已知差异见 `CONFIG_PARITY.md`。运行 `uv run pytest -q -s` 可看到
原生 MuJoCo CPU 采样后两轮训练的实时损失；该 CPU 环境仅用于测试，不替代 G1 仿真后端。

默认训练配置需要 NVIDIA CUDA 设备。CPU 测试使用小型 MuJoCo 机器人和生成的运动数据，
覆盖实际 FK、WM 多步更新、可微 imagined rollout、Actor / Critic 更新及保存恢复，
不代表已完成 G1 训练或复现论文指标。

## 模块

| 文件 | 职责 |
|---|---|
| config.json | 训练、奖励、observation_noise、domain_randomization；训练超参数保留上游键名 |
| trainer.py | 构建模块、采样、学习率、日志、保存恢复 |
| actor.py | command encoder 和 history decoder |
| critic.py | 状态价值网络 |
| world_model.py | 局部增量预测、结构化状态积分 |
| updates.py | WM 回归、imagined Actor 更新、独立 Critic 回归 |
| buffer.py | 时间×环境的循环 Buffer 和历史恢复 |
| environment.py | MuJoCo-Warp 采样，均匀 bin reset |
| motion.py | 参考动作重采样及 command |
| data_transforms.py | obs、command、history、Critic 输入、Actor 回报、价值目标和状态转换 |
| task.py | 跟踪奖励和终止条件 |
| simulation_parameters.py | 摩擦、关节偏移、质心偏移的采样、应用和归一化 |
| kinematics.py | 上游 FK 算法，重命名状态参数 |
| rotations.py | 四元数、旋转矩阵、SO(3) 运算 |
| mirror.py | 左右镜像增强 |
| optimizers.py | Muon 隐藏矩阵 + Adam 边界层 |
| g1/g1_constants.py | G1 固定模型描述、关节顺序、驱动器定义 |

Critic 输入由 invariant state、参考关节差、身体/根位置误差、相对姿态、
normalized simulation_parameters **和 command** 拼接。Actor 输入只有 history 和 command。
Actor 更新期间冻结 WM / Critic 参数，但保留对状态和动作的梯度。

## 命名

`base_lin_vel` / `base_ang_vel` 表示机体系速度，`_w` 表示世界系，
`_heading` 表示 yaw 坐标系。`body_*_w` 是各 link 原点的运动学量，
不是质心位置或质心线速度。动作历史统一为 `last_action`。

## 与上游的边界

参考 fuyichen0725/blinkmimic，commit `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`。
FK、旋转、参考处理、奖励与网络动力学公式从其 MIT 代码改编；保留 LICENSE 和第三方声明。

默认数值来自 `scripts/presets.sh` 的 main：WM 3×1024、Actor encoder/decoder
2048-1024-512-256、lr=4e-4、WM mirror=0.5、EMA=0.995、结构化 WM、参考 skip、Muon。
不包含失败优先 / return 优先 / evaluation-probe 采样；reset 在时间 bin 间均匀采样。
不包含 AMP、torch.compile、CUDA Graph、fused Adam 或异步 checkpoint。
因此采样分布、浮点舍入和速度与论文运行不同，不保证原论文的时间和回报。
Muon 内部正交化遵循 torch.optim.Muon 原实现，FP32 指网络与状态计算不使用 autocast。

checkpoint 恢复网络、标准化统计、优化器、迭代和学习率控制状态；
与上游一样，环境与 replay 重新 warm-up，属于 warm restart，不是逐步精确续跑。
数据由外部下载：LAFAN1 遵循 CC BY-NC-ND 4.0，机器人资产遵循其原许可证。

## 当前数据流中的对象

环境直接保存 `num_envs`、`num_joints`、`last_action`、`episode_step` 和 `simulation_parameters`。
`data_transforms.py` 统一转换数据；Trainer 保存 history。
不再包含 ObsPipeline、XiLayout、传感滤波/延迟状态或动态参数注册。

`simulation_parameters` 是网络和 Buffer 使用的数值向量，顺序固定为：
`[friction (1), joint_pos_offset (num_joints), center_of_mass_offset (3)]`。
只有到 WM / Critic 的输入处才归一化，物理仿真和 Buffer 保存原始值。
向量切片仅用于这三个明确字段，不再维护额外的 layout 对象。

本次修改了网络参数名称和 observation_noise 配置结构。旧版代码由 Git 历史保留，
旧版 checkpoint 不直接加载到新命名版本；需要续跑旧模型时先切回对应提交。

## 数据字段约定

参考数据统一 `ref_`，根身体统一 `base_`；姿态字段用 `rot`，速度用 `lin_vel` / `ang_vel`。
误差用 `_error`，奖励用 `_reward`，惩罚用 `_penalty`。
奖励详情字典的键与变量名称一致，不保留 `track`、`e_*` 等旧键。
训练超参数仍采用上游的配置键；奖励与观测数据相关配置使用本项目名称，数值不变。
外部 mjlab API（如 write_root_link_pose_to_sim）保持依赖库规定的名称。

## 数据转换与训练更新

所有张量输入、历史与训练目标转换集中在 `data_transforms.py`：
`compute_observation`、`compute_command`、`compute_observation_history`、
`update_observation_history`、`compute_replay_observation_history`、`compute_critic_input`、
`compute_actor_return`、`compute_value_targets`、状态打包与参数归一化。
这些函数的输入都通过参数传入，不读取 Trainer 或更新器的内部状态。

`Trainer` 不再继承更新类。它通过关键字参数创建 `TrainingUpdates`，
后者在自身构造函数中定义所有模型、优化器、Buffer 和配置依赖。
训练调用为 `self.updates.update_world_model()` 和 `self.updates.update_actor()`。
删除仅承载一个函数的 objective.py，以及转换逻辑分散的 observation.py。

## 随机起点可视化回放

```bash
uv run --no-sync python -m readable_mimic.evaluate \
  --config config.json \
  --checkpoint runs/main/last.pt \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 30 \
  --ghost-base follow
```

轨迹名对应 motion_dir 中的 CSV 文件名，可省略 .csv；列表必须加引号。
`--duration` 是每条轨迹的模拟播放秒数；播放到末尾或触发 termination 后，
在同一条轨迹内重新随机起步。列表按给定顺序播放一次，结束后关闭窗口。
`--motions all` 播放全部轨迹。`--seed` 控制随机起点，默认 0。

`--ghost-base follow`：ghost 的 xy 和 yaw 对齐当前机器人，保留参考高度和倾斜。
`--ghost-base reference`：ghost 保留参考动作原本的世界轨迹。
Ghost 是透明蓝色的 viewer 几何体，不加入物理模型、不参与碰撞。

回放优先读取 checkpoint 中的 EMA Actor，关闭动作/观测/reset 噪声和域随机化，
摩擦设为 1、关节与质心偏移为 0。模型结构使用 checkpoint 保存的配置，
本机资产路径和设备使用 --config 指定的配置。旧 checkpoint 的权重键名必须与当前 Actor 一致。
窗口通过 MuJoCo viewer 打开，需要在有图形界面的会话运行。

这是随机起点的可视化检查，不输出整段动作的标准评测分数。
