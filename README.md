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
| observation.py | 直接拼接当前 obs，定义观测噪声范围 |
| task.py | 解析奖励、终止、Critic features |
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
`observation.py` 从当前状态拼接 obs；Trainer 保存 history。
不再包含 ObsPipeline、XiLayout、传感滤波/延迟状态或动态参数注册。

`simulation_parameters` 是网络和 Buffer 使用的数值向量，顺序固定为：
`[friction (1), joint_pos_offset (num_joints), center_of_mass_offset (3)]`。
只有到 WM / Critic 的输入处才归一化，物理仿真和 Buffer 保存原始值。
向量切片仅用于这三个明确字段，不再维护额外的 layout 对象。

本次修改了网络参数名称和 observation_noise 配置结构。旧版代码由 Git 历史保留，
旧版 checkpoint 不直接加载到新命名版本；需要续跑旧模型时先切回对应提交。
