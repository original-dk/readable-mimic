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
| env.py | MuJoCo-Warp 采样，均匀 bin reset |
| motion.py | 参考动作重采样及 command |
| data_transforms.py | obs、command、history、Critic 输入、Actor 回报、价值目标和状态转换 |
| task.py | 跟踪奖励和终止条件 |
| sim.py | 仿真配置、scene/仿真器创建、时间常量及物理参数随机化 |
| kinematics.py | 上游 FK 算法，重命名状态参数 |
| rotations.py | 四元数、旋转矩阵、SO(3) 运算 |
| mirror.py | 左右镜像增强 |
| optimizers.py | Muon 隐藏矩阵 + Adam 边界层 |
| g1/g1_constants.py | G1 固定模型描述、关节顺序、驱动器定义 |

Critic 输入由 invariant state、参考关节差、身体/根位置误差、相对姿态、
normalized_xi **和 command** 拼接。Actor 输入只有 history 和 command。
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

环境直接保存 `num_envs`、`num_joints`、`last_action`、`episode_step` 和 `xi`。
`data_transforms.py` 统一转换数据；Trainer 保存 history。
不再包含 ObsPipeline、XiLayout、传感滤波/延迟状态或动态参数注册。

`xi` 是网络和 Buffer 使用的数值向量，顺序固定为：
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

## 随机起点视频导出

```bash
uv run --no-sync python -m readable_mimic.evaluate \
  --checkpoint runs/main/last.pt \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 30 \
  --ghost-base follow \
  --output-dir videos
```

每条轨迹输出一个 MP4，文件名为轨迹名称。列表必须加引号，轨迹名可省略 .csv；
--motions all 导出全部轨迹。--duration 是每条视频的模拟时长（秒），输出固定为 50 FPS、960×720。
仿真不等待墙钟时间，生成过程可能慢于实时，但视频播放速度由编码帧率确定。

每条轨迹随机起步；片段结束或摔倒后，在同一条轨迹重新随机起步。
--ghost-base follow 对齐机器人 xy/yaw，reference 保留参考世界轨迹。
Ghost 仅为透明渲染几何体，不参与物理计算。默认读取 EMA Actor，关闭训练噪声和域随机化。

不启动 MuJoCo viewer，也不弹出预览窗口。Linux 无 DISPLAY 时自动选择 EGL；
如需手动指定，可在命令前加 MUJOCO_GL=egl。编码依赖 imageio[ffmpeg]，更新项目后先执行 uv sync。
若之前为解决 MuJoCo 混装手工修复过环境，应保留该处理，再使用 --no-sync 运行导出命令。

这是可视化视频导出，不是完整动作的标准评测分数。

## Checkpoint 与继续训练

所有设置仍放在 config.json：

```json
{
  "ckpt_every": 100,
  "resume": null,
  "iterations": 3500
}
```

每 100 个训练迭代保存 log_dir/checkpoints/iteration_000100.pt 等独立文件，
并更新 log_dir/last.pt。训练正常结束时也保存，即使最后一次迭代不是保存间隔的倍数。
文件先写临时文件再替换，避免半写入的 checkpoint 覆盖有效文件。

每个文件包含 Actor、EMA Actor、Critic、WM、各自标准化统计、三个优化器
（包含 Muon 与 Adam 子状态）、完整配置、已完成迭代数、采样余数、学习率触发状态，
以及 Python、NumPy、PyTorch CPU 和 CUDA 随机状态。不保存 Replay Buffer 和仿真环境。

继续训练时修改 config.json：

```json
{
  "resume": "runs/main/checkpoints/iteration_001500.pt",
  "iterations": 5000,
  "ckpt_every": 100
}
```

然后执行原训练命令：

```bash
uv run --no-sync python -m readable_mimic.train --config config.json
```

iterations 是总目标，不是追加轮数：上例从完成的第 1500 次迭代继续，训练到第 5000 次。
模型结构与算法配置从 checkpoint 自动恢复；当前配置中的 device、num_threads、
motion_dir、log_dir、resume、iterations、保存/日志间隔和本次时长限制仍然有效。
如果回退到较早的 checkpoint，建议修改 log_dir，避免与已有日志混在一起。

Buffer 默认不保存。默认 4096 环境 × 1000 步的原始 Buffer 约 3.6 GiB，
而它可以通过重新采样恢复；继续训练会创建新环境并执行 warmup_steps，再恢复随机数状态进入更新。
因此这是恢复模型/优化器后的 warm restart，不保证与不间断运行逐步完全一致。
现有同版本字段命名的旧 last.pt 仍可加载；不兼容历史字段重命名之前的权重。

也可不修改配置，直接使用命令行：uv run --no-sync python -m readable_mimic.train --config config.json --resume runs/main/last.pt。--resume 优先于配置中的 resume，路径相对于当前工作目录；配置文件可以省略 resume 字段。总训练目标仍由 iterations 指定。

## 固定数据的初始化

关节偏移范围在 Trainer 和 TrackingEnv 各自构造时计算，不在 update 或 reset 时重新匹配关节正则。
Trainer 初始化 xi_center / xi_scale，并显式传给更新器。
数据转换仍位于 data_transforms.py，归一化只做 (xi-xi_center)/xi_scale。

以下小张量同样只创建一次：command 的未来帧偏移、Actor 目标的折扣序列、
Buffer 的 WM 窗口偏移和 history 回看偏移。视频导出时 command 偏移每条视频创建一次。
Buffer 的窗口长度和历史长度由构造参数确定，采样接口不再重复传入这两个固定长度。
WM rollout 的整批仿真参数在进入逐步预测前归一化一次。

配置改变后重新构建 Trainer / Env；不使用全局缓存，不缓存随机采样结果、当前状态特征或正在更新的标准化统计。
缓存由配置推导，不额外写入 checkpoint，不改变权重和优化器的存储格式。


仿真相关代码统一位于 sim.py：create_simulation、EagerSimulation、
CONTROL_DT / DECIMATION / PHYSICS_DT、DomainRandomizationConfig 和参数采样函数。
env.py 通过 create_simulation 创建 scene/仿真器，只负责环境状态、reset、observe、step。
参数归一化等纯数据转换仍位于 data_transforms.py，保留已有模块职责。


物理上下文统一命名为 xi：xi_dim 为维度，normalized_xi 为归一化结果，
xi_center / xi_scale 为初始化生成的归一化常量。
xi 固定顺序为 [friction, joint_pos_offset, center_of_mass_offset]。
模型 parameters() 指 PyTorch 权重参数，与 xi 无关；权重和 checkpoint 格式保持不变。


create_simulation 直接接收 config.simulation 字典，内部完成配置转换、scene 创建和仿真器初始化；不再提供 make_sim 中间接口。

训练阶段计时：config.json 的 profile_speed 为 true 时，日志及 metrics.jsonl 输出两次日志之间的每轮平均毫秒数。
prepare_ms 为学习率和采样步数准备，collect_ms 为真实环境采样及数据入库/标准化，world_model_ms 包含全部 WM 更新。
actor_prepare_ms 为 replay 起点采样和历史/command 准备，actor_rollout_ms 为 imagined rollout，reward_ms 为轨迹整理、奖励和终止概率计算。
actor_return_ms 为 critic bootstrap 和 actor return，actor_backward_ms 包含反向传播、梯度裁剪、优化器和 EMA；critic_ms 包含 critic 特征/目标计算及全部更新。
other_ms 为剩余轮内工作；iteration_ms 为以上阶段之和，不包含日志打印和 checkpoint。各项均按整个日志区间的轮数平均。
CUDA 阶段边界会同步，以避免异步执行造成耗时归属错误，因此计时模式会影响 it/s；profile_speed=false 恢复普通训练。
继续训练时 profile_speed 以当前配置为准，计时状态不写入 checkpoint。首次日志可能包含运行时初始化开销，应比较后续稳定日志。

细分计时保留 collect_ms、world_model_ms、actor_backward_ms 汇总，它们由各自子项相加得到，iteration_ms 不会重复计入汇总项。
collect_prepare_ms：观测、history、command 及 Actor 标准化；collect_actor_ms：Actor 推理和动作噪声。
collect_physics_ms：状态读取、设置目标和全部物理子步；collect_reward_ms：奖励、终止和超时判断。
collect_finalize_ms：transition 打包、episode 统计和 reset；collect_buffer_ms：Buffer 写入、WM 数据转换及标准化、回报统计。
collect_env_other_ms / collect_other_ms：环境返回和循环剩余开销。
world_model_prepare_ms / forward_ms / backward_ms / clip_ms / optimizer_ms / other_ms：窗口采样与镜像、前向及 loss、清梯度与反传、裁剪、优化器、结果整理及其余开销（各字段均带 world_model_ 前缀）。
actor_backward_gradients_ms / clip_ms / optimizer_ms / ema_ms / other_ms：清梯度与反传、裁剪、有限性检查及优化器、EMA、恢复梯度开关（各字段均带 actor_backward_ 前缀）。
所有子项仍按每轮平均，WM 5 次更新、真实采样多个控制步等重复调用的耗时会先累计；不是单次调用耗时。
