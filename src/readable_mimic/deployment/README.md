# ONNX → native MuJoCo sim2sim

此目录负责部署，不调用 TrackingEnv、mjlab 或 Genesis。`play_mujoco.py` 不导入 Torch；
它直接读取原生 MuJoCo 状态、构建 NumPy 输入、运行 ONNX Runtime 并推进物理。
导出阶段会复用已验证的 Actor、FK 和 motion 预处理。

## 安装

在项目根目录执行，使用单独环境，不需要安装 mujoco 或 genesis extra：

```bash
UV_PROJECT_ENVIRONMENT=.venv-deploy uv sync --extra deploy --locked
```

deploy extra 包含 ONNX 导出器和 CPU ONNX Runtime；原生 MuJoCo 来自项目基础依赖。
环境中安装 Torch 是为了导出，播放器本身不使用它。

## 导出

```bash
UV_PROJECT_ENVIRONMENT=.venv-deploy uv run --extra deploy \
  python -m readable_mimic.deployment.export_onnx \
  --checkpoint runs/main/last.pt \
  --config runs/main/config.json \
  --output exports/policy.onnx
```

需先准备 G1 MJCF/meshes 与 CSV 动作。config 中的 motion_dir 必须指向当前机器的有效路径，
其中的训练 resume 会忽略；Actor 结构与权重取自指定 checkpoint。
默认优先导出 EMA Actor，否则导出 Actor。标准化统计及 reference joint pose skip 均在 ONNX 中，
运行端不要再做标准化，也不要再次添加参考关节角。
导出会检查 ONNX，并用 3 批输入核对 ONNX Runtime 与 PyTorch 的动作一致性；不做量化。

输出三个配套文件，必须一起保留：

| 文件 | 内容 |
|---|---|
| policy.onnx | batch=1 的 Actor，输入 observation_history 和 command，输出 action |
| policy.json | 关节顺序、默认姿态、动作缩放、PD/力矩/armature、物理设置与资产路径 |
| policy.motion.npz | 50 Hz 参考轨迹、参考速度、旋转矩阵及 clip 边界，运行时无需 Torch/FK 预处理 |

MJCF 和其引用的 meshes 不复制进 bundle。保留资产相对位置；迁移模型后可编辑 JSON 的 mjcf_path，
或播放时指定 `--mjcf /path/to/g1_29dof.xml`。不需要训练 checkpoint、Buffer、Critic 或 WM 来播放。

## 原生 MuJoCo 播放

```bash
UV_PROJECT_ENVIRONMENT=.venv-deploy uv run --extra deploy \
  python -m readable_mimic.deployment.play_mujoco \
  --model exports/policy.onnx \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 30 --ghost-base follow
```

默认打开原生 MuJoCo passive viewer，尽量按 50 Hz 墙钟速度播放。
`--duration` 是每条轨迹的模拟秒数；`--motions all` 使用所有导出的轨迹。
随机起点由 `--seed` 控制；片段结束或达到高度/倾斜终止条件后，在同一片段重新起步，清空 last_action/history。
`--ghost-base follow/reference/off` 控制参考 ghost；ghost 为 user scene 几何，不参与碰撞。
此命令是交互播放，不导出 MP4；原生视频导出仍使用 readable_mimic.evaluate。

无显示窗口或自动测试时：

```bash
UV_PROJECT_ENVIRONMENT=.venv-deploy uv run --extra deploy \
  python -m readable_mimic.deployment.play_mujoco \
  --model exports/policy.onnx --motions "walk1_subject1" --duration 5 --headless
```

headless 不创建 viewer 或 GL 场景，不等待墙钟时间，每条轨迹打印控制步、物理步和 reset 数。

## 输入与物理约定

- observation_history：float32，`[1, history_length, 3*num_joints+12]`，最旧观测在前。
- command：float32，`[1, command_frames*(15+2*num_joints)]`，未来帧在 clip 末端截断。
- action：float32，`[1, num_joints]`，顺序与 policy.json 的 joint_names 一致。
- 观测包含 joint_pos-default_joint_pos、joint_vel、机体系 base_ang_vel、projected_gravity、last_action、相对参考姿态 6D。
- 不输入 xi，不添加训练噪声和域随机化；joint_pos_offset 默认为零。
- ONNX 内执行标准化和 reference skip；MuJoCo 目标仅为 `default_joint_pos + action_scale * action`。
- ONNX 按控制周期更新目标；每个物理子步显式计算 tau=kp*(target_joint_pos-joint_pos)-kd*joint_vel，按力矩范围裁剪后 data.ctrl[:]=tau，再 mj_step。
- 基座线速度使用世界系，MuJoCo free-joint 角速度使用机体系；reset 时显式转换。

可在 policy.json 中修改 friction、kp_scale、kd_scale，或逐关节的 stiffness/damping/armature/effort，
测试部署动力学偏差；这些数值不会作为额外信息输入 Actor。
默认 friction=1.0、kp_scale=kd_scale=1.0，50 Hz 控制和 200 Hz 物理。

## 验证边界

NumPy obs/command 与训练实现有数值对照；测试包含 EMA、标准化、ONNX 输出、PD/armature、
ghost 不修改物理模型，以及独立子进程不导入 Torch/训练 Env 的实际 MuJoCo 步进。
另使用真实 G1 29DoF 模型完成过 headless 100 次控制/400 次物理步 smoke test。
该 smoke test 使用生成的测试权重和参考姿态，不代表用户训练 checkpoint 的跟踪质量已经验证。

主循环保留 ONNX、显式 PD、mj_step、history 和 viewer；不再包含分阶段计时。ghost 的几何更新独立放在 ghost.py。显式 PD 与训练中的隐式/近似隐式控制离散化不同，sim2sim 跟踪结果可能变化。

播放等待按累计 deadline 对齐墙钟时间，减少逐步 sleep 的唤醒误差累积；物理步长不变。片段末尾仅打印一次 wall_seconds、work_ms（不含主动 sleep）和 rtf，方便区分计算超过 20 ms 的慢放与策略跟踪差异。
