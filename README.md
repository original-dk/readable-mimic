# Readable Mimic

按 `Motion → command / obs → history → Actor → 仿真 → ReplayBuffer`，以及
`ReplayBuffer → WM → imagined rollout → Actor / Critic` 组织的动作跟踪实现。

本分支为 `compile_speed_genesis`，同时支持 Genesis 与 MuJoCo/mjlab。
`compile_speed` 保留迁移 Genesis 之前的实现；`pure_mimic` 是更早的基线。

## 1. 安装：两个后端分别使用独立 uv 环境

以下命令在项目根目录执行，示例使用 Linux Bash。需要 uv、NVIDIA GPU 和可用的 CUDA 驱动。
Python 支持 3.11–3.13，本地验证使用 3.11；Linux 的 PyTorch 2.9.0 从 cu128 源安装。
CPU 小模型测试不代表已经验证 GPU 上的训练速度或论文复现指标。

```bash
cd ~/readable-mimic
git switch compile_speed_genesis
uv --version

UV_PROJECT_ENVIRONMENT=.venv-genesis uv sync --extra genesis --locked
UV_PROJECT_ENVIRONMENT=.venv-mujoco uv sync --extra mujoco --locked
```

首次新建环境可在 sync 后加 `--python 3.11`。已有可用环境无需为了文档示例更换 Python。
不要使用 `--all-extras`：两个后端的第三方 Pillow 要求不兼容，extra 已声明互斥。
每次运行都同时指定环境目录和 extra；目录名称本身不会选择物理后端，后端由配置的 `simulator` 决定。
`uv run` 正常自动同步依赖，不需要手动激活环境或常规使用 `--no-sync`。

Windows PowerShell 使用相同命令，但先单独设置环境变量，例如：

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-genesis"
uv sync --extra genesis --locked
uv run --extra genesis python -m readable_mimic.train --config config.json
```

固定版本：`torch==2.9.0`、`mujoco==3.11.0`、`genesis-world==1.4.5`、
`mjlab==1.6.0`、`mjviser==0.0.16`、`warp-lang==1.16.0`；以 `uv.lock` 为准。
mjviser 0.0.17 会引入与 MuJoCo 3.11 文件冲突的 mujoco-core 4.x，因此不使用它。

### 已有损坏环境的恢复

旧环境可能曾混装 mujoco/mujoco-core。更新锁文件不会自动修复已经丢失的包文件。
对发生错误的那个环境执行一次：

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv sync --extra genesis --reinstall-package mujoco --no-cache
# MuJoCo 环境出错时改用下面这一条：
UV_PROJECT_ENVIRONMENT=.venv-mujoco uv sync --extra mujoco --reinstall-package mujoco --no-cache
```

不要再单独安装 mujoco-core，也不需要卸载整个 PyTorch 或删除项目数据。
如果仍报动态库错误，保留完整输出，检查实际包路径和动态库，不能只凭包版本判断安装正确。

### 安装检查

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis python -c \
  "import torch,mujoco,genesis; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(mujoco.__file__,mujoco.__version__,mujoco.MjModel); print(genesis.__version__)"

UV_PROJECT_ENVIRONMENT=.venv-mujoco uv run --extra mujoco python -c \
  "import torch,mujoco,mjlab; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(mujoco.__file__,mujoco.__version__,mujoco.MjModel)"
```

训练配置默认 `device: "cuda:0"`；若 CUDA 不可用，先检查 `nvidia-smi` 与上面打印的 PyTorch/CUDA 信息。
Genesis 也需要 MuJoCo 绑定来加载原版 FK，但它的物理和视频渲染均由 Genesis 完成。

## 2. 准备模型和动作数据

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis python tools/download_data.py
```

此脚本下载 LAFAN1 G1 动作、URDF 和 meshes，再运行 `tools/urdf_to_mjcf.py` 生成 MJCF。
两后端共用同一份数据，无需各下载一次。已有完整文件会跳过，下载中的内容写入 `.part` 文件。
训练前应存在：

```text
data/g1_description/g1_29dof_rev_1_0.urdf
data/g1_description/g1_29dof.xml
data/g1_description/meshes/
data/lafan1_g1/*.csv
```

数据不存入 Git。LAFAN1 与机器人资产分别遵循其原许可证，见第三方声明。

## 3. 训练配置与启动

所有实验配置放在 JSON 中。仓库 `config.json` 默认 `simulator: "genesis"`。

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis \
  python -m readable_mimic.train --config config.json
```

如果训练 MuJoCo，先在项目根目录复制配置，修改其中的 `simulator` 为 `mujoco`，
并给 `log_dir` 设置单独的目录，例如 `runs/mujoco`，避免覆盖另一个实验：

```bash
cp config.json config.mujoco.json
# 编辑 config.mujoco.json：simulator=mujoco，log_dir=runs/mujoco
UV_PROJECT_ENVIRONMENT=.venv-mujoco uv run --extra mujoco \
  python -m readable_mimic.train --config config.mujoco.json
```

`motion_dir`、`log_dir`、配置中的 `resume` 相对于配置文件目录解析。
不要直接将根目录配置移到 `runs/main/` 后继续使用原来的相对路径。
训练保存的 `runs/main/config.json` 包含已解析的绝对路径；复制到另一台机器时要检查这些路径。

主要配置：

| 字段 | 含义 |
|---|---|
| simulator | genesis 或 mujoco；必须与运行 extra 匹配 |
| num_envs / iterations | 并行环境数 / 总训练轮数 |
| motion_dir / log_dir | 动作目录 / 本次实验输出目录 |
| simulation | 50 Hz 控制、4 个物理子步及求解器参数 |
| genesis | Genesis 特有选项；默认关闭自碰撞，与 G1 MuJoCo 设置对齐 |
| domain_randomization / observation_noise / reward | 随机化、观测噪声、奖励参数 |
| compile_world_model / compile_actor | CUDA 训练中的 WM、Actor 路径编译开关 |
| allow_tf32 | 允许 float32 矩阵使用 TF32；不是严格 FP32 精度，不使用 BF16/autocast |
| profile_speed | 分阶段同步计时，会增加诊断开销 |

两后端复用 G1 关节顺序、PD 增益、力矩限制、armature、奖励和学习超参数。
摩擦和质心随机化实际写入引擎；积分器和接触求解实现仍有差别，不能视为数值等价。
MuJoCo 专用选项不会全部映射到 Genesis，具体边界见 `CONFIG_PARITY.md`。

## 4. 日志、checkpoint 与继续训练

`log_dir` 下保存完整配置、`metrics.jsonl`、`last.pt` 和独立的编号 checkpoint：

```text
runs/main/config.json
runs/main/metrics.jsonl
runs/main/last.pt
runs/main/checkpoints/iteration_000100.pt
```

`ckpt_every` 控制保存间隔；正常结束也会保存。checkpoint 包含四个模型及标准化统计、
三个优化器、配置、轮次、学习率控制和随机状态；不保存 Buffer 或仿真环境。
恢复后会重新采样 warmup，是 warm restart，不保证与不间断训练逐步完全一致。

将当前配置的 `iterations` 改为大于 checkpoint 轮数的总目标后运行：

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis \
  python -m readable_mimic.train --config config.json --resume runs/main/last.pt
```

也可设置配置中的 `resume`。CLI 的 `--resume` 优先，路径相对于当前工作目录。
网络和算法设置从 checkpoint 恢复；当前配置中的设备、路径、总轮数、保存/日志间隔、
计时/编译/TF32 开关和仿真后端设置仍有效。回退旧 checkpoint 时建议换 log_dir。
历史字段重命名前的 checkpoint 不保证兼容；换物理后端继续训练也不等于精确续跑。

日志中 loss 显示四位小数，其他浮点数三位，`metrics.jsonl` 保留原始精度。
`sim_r` 是训练采样得到的归一化单步奖励，`episode_return` 是已结束 episode 的归一化回报，二者统计口径不同。
计时字段为日志窗口内平均每轮毫秒数：collect、world_model、actor_backward 的汇总由子项相加。
`iteration_ms` 不重复计入汇总，不含日志和 checkpoint；`it/s` 按日志之间的实际墙钟时间计算。
第一次前向/反传可能触发编译，应比较稳定后的日志；正常测速时可关闭 profile_speed。

## 5. Genesis 原生视频评估

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis \
  python -m readable_mimic.evaluate \
  --config runs/main/config.json \
  --checkpoint runs/main/last.pt \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 30 --ghost-base follow --output-dir videos/genesis
```

确认评估配置的 simulator 为 genesis。评估只使用 `--checkpoint` 指定的权重，
不会跟随配置中的训练 resume 再加载另一个 checkpoint。
网络结构取自 checkpoint，EMA Actor 有效时优先使用，否则使用 Actor。
环境、物理后端与数据路径取自评估配置。

Genesis 使用原生相机及默认地面/光照，机器人和透明 ghost 放在独立显示场景，
不改变实际 rollout 的接触或摩擦；使用 PyAV 进程内编码。无需启动 viewer。

## 6. 用 MuJoCo 评估同一个 checkpoint

从根配置复制为 `config.mujoco.json`，设置 simulator=mujoco，并确认 motion_dir 有效。
checkpoint 可以是 Genesis 训练的，无需修改权重：

```bash
UV_PROJECT_ENVIRONMENT=.venv-mujoco uv run --extra mujoco \
  python -m readable_mimic.evaluate \
  --config config.mujoco.json \
  --checkpoint runs/main/last.pt \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 30 --ghost-base follow --output-dir videos/mujoco
```

这是真正的 MuJoCo 物理与渲染，不是只换输出目录。MuJoCo 视频使用棋盘地面。
跨后端评估用于检查差异，不应由画面相似推断动力学一致。

两后端都每条轨迹输出一个 MP4，固定 50 FPS、960×720。
`--duration` 是每条视频的模拟秒数；`--motions all` 导出全部，列表中的 `|` 必须加引号。
每条轨迹随机起点，摔倒或片段结束后在同一轨迹重新随机起步。
`--ghost-base follow` 对齐机器人 xy/yaw，`reference` 保留原始世界轨迹。
评估关闭训练噪声和随机化：摩擦为 1.0，关节偏移和质心偏移为零。
视频生成不等待实时；此命令是可视化 rollout，不是完整动作的标准评测分数。
Linux 无 DISPLAY 时使用离屏渲染；MuJoCo 可显式设置 `MUJOCO_GL=egl`，Genesis 原生相机使用其离屏后端。

## 7. 独立 ONNX sim2sim

独立部署代码位于 `src/readable_mimic/deployment/`，详见该目录的 README。
播放器只用 NumPy、ONNX Runtime 和原生 MuJoCo，不调用训练 Env，不导入 Torch。

```bash
UV_PROJECT_ENVIRONMENT=.venv-deploy uv sync --extra deploy --locked
UV_PROJECT_ENVIRONMENT=.venv-deploy uv run --extra deploy \
  python -m readable_mimic.deployment.export_onnx \
  --checkpoint runs/main/last.pt --config runs/main/config.json --output exports/policy.onnx
UV_PROJECT_ENVIRONMENT=.venv-deploy uv run --extra deploy \
  python -m readable_mimic.deployment.play_mujoco \
  --model exports/policy.onnx --motions "walk1_subject1|dance1_subject1" --duration 30 --ghost-base follow
```

导出 ONNX、JSON 控制配置和 NPZ 参考动作三个配套文件；默认使用 EMA，标准化与 reference skip 包含在 ONNX 中。
MJCF/meshes 仍需保留，迁移后可用 --mjcf 指定资产路径。播放默认打开 viewer，服务器使用 --headless。
可编辑导出的 JSON 中 friction、kp_scale、kd_scale 测试参数偏差；不把这些物理参数输入 Actor。
这是独立部署测试，不替换已有 evaluate 的原生视频路径。

## 8. 验证

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis pytest -q
UV_PROJECT_ENVIRONMENT=.venv-mujoco uv run --extra mujoco pytest -q
uv lock --check
```

可选后端未安装时，对应测试会跳过。测试包含 FK 数值与梯度、编译图、Replay 窗口、
训练/续训、速度统计、视频帧数与 reset；Genesis 测试实际运行 CPU 物理。
原生 Genesis 相机和 MP4 编解码另已在本机执行过真实 smoke test；不代表已验证 Linux/5090 全部驱动组合。

## 模块与数据约定

| 文件 | 职责 |
|---|---|
| trainer.py / updates.py | 采样与训练调度 / WM、Actor、Critic 更新 |
| env.py | 共用 obs、reset、reward、transition 流程 |
| sim.py / sim_mujoco.py / sim_genesis.py | 时间常量与 xi / 各物理后端 |
| evaluate.py / render_genesis.py | 评估、MuJoCo 视频 / Genesis 原生视频 |
| data_transforms.py | obs、command、history、Critic 输入与训练目标转换 |
| actor.py / critic.py / world_model.py | 三个网络 |
| motion.py / kinematics.py / task.py | 参考动作 / 原版 FK / 奖励和终止 |
| buffer.py / mirror.py / optimizers.py / timing.py | 回放 / 镜像 / 优化器 / 计时 |
| robot.py / g1/g1_constants.py | 通用机器人描述 / G1 关节与驱动参数 |

数据使用 `ref_`、`base_`、`joint_pos`、`joint_vel`、`last_action` 等名称；
`_w` 表示世界系，`_heading` 表示 yaw 坐标系，`body_*_w` 对应 link 原点。
xi 顺序为 `[friction, joint_pos_offset, center_of_mass_offset]`，Buffer 保存原值，
WM/Critic 输入处归一化；Actor 使用 history 与 command。
固定索引、偏移范围和归一化尺度在初始化构建，不缓存随机采样或正在更新的统计。

参考 BlinkMimic commit `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`，保留 LICENSE 和第三方声明。
未实现失败优先/probe reset、AMP、手工 CUDA Graph、fused Adam 或异步 checkpoint。
更多超参数及算法边界见 `CONFIG_PARITY.md`。
