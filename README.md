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

### 100STYLE → G1 训练数据

离线重定向使用独立 uv 环境，CPU 即可；训练/评估环境无需安装重定向依赖。

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv sync --extra retarget --locked
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget python tools/download_100style.py
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python -m readable_mimic.retargeting.convert_100style --config retarget_100style.json
```

下载器获取并校验官方原始 BVH 压缩包（约 1.5 GB）和 `Frame_Cuts.csv`，解压到 `data/100style_bvh`。
已有数据可直接设置 `retarget_100style.json` 中的 `input_dir` 和 `frame_cuts`，跳过下载。
机器人模型仍使用前面的 `data/g1_description/g1_29dof.xml`。
完整数据包含 100 个风格、810 条片段；源数据和转换结果均不提交 Git。

重定向的路径、风格筛选、动作类型筛选、进程数、比例和 IK 权重统一在 `retarget_100style.json`。
`styles: []`、`movements: []` 表示全部；例如 `styles: ["Aeroplane", "Angry"]`、
`movements: ["FW", "FR"]`。`workers` 控制 CPU 并行数，默认 6。
所有路径相对此配置文件解析。训练读取的 CSV 固定为 30 Hz。
做子集实验时同时设置新的 `output_dir`；筛选只控制本次转换，不会删除目录中已有的其他 CSV。

流程为：按官方标注裁剪首尾 → BVH FK/坐标转换 → 两阶段 Mink IK → 支撑阶段足底高度校正 → CSV。
BVH 的通道旋转顺序逐关节读取；厘米转换为米，Y-up 转为 Z-up；不镜像、不人为改变播放速度。
IK 的身体对应关系与初始权重参考 GMR，手臂坐标偏移和比例按本项目的 G1 模型调整；不需要安装完整 GMR。
裁剪范围按零起点 `[START, STOP)` 解释，60 Hz 输入每隔一帧求解得到 30 Hz。
根据源动作脚趾高度和垂直速度识别支撑阶段；支撑帧的最低足底碰撞面校正到地面下 0.1 mm，以形成接触。
腾空段用 PCHIP 插值高度补偿，保留跳跃；阈值统一放在 grounding 配置中。接触识别为启发式，极端风格仍需检查。
已有旧 CSV 可执行 `UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget python tools/reground_100style.py --config retarget_100style.json`，无需重跑 IK。
原始 CSV 和元数据会备份到输出目录同级的 `100style_g1_before_grounding`；此命令只修改基座高度。
已更新本地全部 810 条数据，原有格式、四元数、关节限位及训练 FK 检查通过。
步行、展臂步行、双脚跳和左右单脚跳的代表片段，支撑帧均检测到足底—地面接触，跳跃保留无接触腾空帧。
该修正保证支撑帧的最低足底贴地，不等价于双脚锁定、防滑或动力学稳定性验证。

输出到 `data/100style_g1/*.csv`：每行 `base_pos_w[3], base_quat_xyzw[4], joint_pos[29]`，
无表头，米/弧度，关节顺序复用 `G1_JOINT_NAMES`，与 LAFAN1 训练数据格式一致。
每条轨迹旁的 JSON 保存来源、裁剪区间、帧数、关节顺序、最大关节速度和校验值。
汇总及失败详情在 `conversion_report.json`；有失败时命令以非零状态退出。
相同源数据、模型和转换设置会复用已经校验的输出；改变 IK 设置时请使用新的 `output_dir`。

训练时将实验配置的 `motion_dir` 改为 `data/100style_g1`，并设置单独的 `log_dir`，
然后使用前述 Genesis 或 MuJoCo 训练命令；训练代码会照常重采样到 50 Hz 并计算 FK。
这里完成的是运动学重定向；极端风格可能触及关节限位或需要筛选，并不保证 PD 跟踪和动力学可行性。

本地全量验证：100 个风格、810 条片段、2,028,184 帧（30 Hz），转换失败为 0。
所有 CSV 通过 36 列格式、有限值、校验值、四元数和关节限位检查；
飞机、举手、屈膝、普通步态和双脚跳共 5 类片段通过现有训练读取器与 FK。
重复执行已验证 810 条全部复用。详情保存在输出目录的 `validation_report.json`。
少数片段的关节速度较高（例如 `LimpLeft_BR` 约 53.5 rad/s），尚未进行动力学可行性筛选。

数据遵循 [100STYLE 的 CC BY 4.0](https://www.ianxmason.com/100style/)，
GMR 对应关系遵循 MIT，引用和许可证见 `THIRD_PARTY_NOTICES.md`。

### 直接查看 CSV 动作（运动学播放）

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/play_motion.py \
  --motion data/100style_g1/Aeroplane_FW.csv --loop
```

直接把 CSV 姿态写入 MuJoCo `data.qpos`，调用 `mj_forward` 后显示；不运行策略、PD 或 `mj_step`。
CSV 的基座四元数是 `xyzw`，赋值前转成 MuJoCo 的 `wxyz` 并归一化。
关节列按项目 G1 MJCF 的顺序使用。默认 30 fps，也适用于现有 LAFAN1 CSV。
`--loop` 循环播放，省略则播放一遍；`--fps` 指定源帧率，`--mjcf` 指定模型。
`--headless` 不创建窗口、不等待墙钟时间，用于快速检查数据。关窗或 Ctrl+C 可退出。
该工具只需要 NumPy 和原生 MuJoCo，也可在已有的 Genesis、MuJoCo 或 deploy uv 环境运行。

### AMASS / SMPL-X neutral → G1 候选动作

在 AMASS 官网下载 **SMPL-X N**，不要混入 SMPL-X G、SMPL+H 或 Render。
下载需要本人已获得的账号和各子集授权；工具不保存账号、密码或浏览器凭据。
身体模型使用授权下载的 `SMPLX_NEUTRAL.npz`，放在 `data/body_models/smplx/`。
模型和数据均不随 Git 分发。

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv sync --extra retarget --locked
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/prepare_amass.py --config retarget_amass.json --archives /path/to/downloads
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python -m readable_mimic.retargeting.convert_amass --config retarget_amass.json
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/screen_amass.py --config retarget_amass.json
```

Windows 的压缩包目录可以传 `--archives D:/Downloads`。只处理下载完整的 `.tar.bz2`，
不会读取 `.crdownload`；解压检查路径边界，并把压缩包 SHA256 和状态记入 `data/amass/download_manifest.json`。
原始压缩包留在下载目录，不额外复制一整份。已有文件按清单续跑，新完成的下载可再次执行上述命令。

配置统一放在 `retarget_amass.json`；`datasets` 同时限定解压和转换的子集（空列表表示全部）。
当前已排除并删除 ACCAD、BMLmovi、BMLrub、DFaust、EKUT、HumanEva、Eyes_Japan_Dataset、KIT、MoSh，不再导入这些子集。

AMASS 的 IK 按 GMR 原版设置运行：身高缩放、单位基座四元数/零关节初始姿态、离线收紧限位、足尖目标坐标系与迭代停止条件。训练机器人的 XML 和限位不变；足底支撑修正仍只作用于输出的 base z。
修复前候选保存在 `data/amass_g1_candidates_before_gmr_fix`，不要用于训练。
对 CMU `01_01`（688 帧）与 `05_06`（222 帧），相同人体输入下、落地修正前的关节结果与原版 GMR 对照，均方根误差分别为 3.93e-13 度与 1.17e-11 度。这验证了转换一致性，不代表动态跟踪可行性。

筛选子集时填写解压后的顶层目录名称（部分官方目录使用 MPI_HDM05、MPI_mosh 等别名）。
默认匹配 `*_stageii.npz`，排除 `neutral_stagei` 等辅助文件。
读取人体形状及前 22 个身体关节，进行 FK；平移线性插值、关节旋转 SLERP，按真实源帧率重采样到 30 Hz。
不计算手指、表情和软组织变形，也不需要在运行端生成完整人体网格。
G1 使用项目原模型和共用的 Mink IK、支撑阶段贴地；足尖任务换算到项目实际存在的 ankle_roll_link。

三个阶段独立保存：

| 目录 | 内容 |
|---|---|
| data/amass | 原始 AMASS 参数及下载清单 |
| data/amass_g1_candidates | 重定向候选 CSV、转换报告和 screening_report.json |
| data/amass_g1_train | 人工复核及仿真跟踪验证后选入的训练动作 |

**转换成功不等于动作适合 G1。** 初筛报告检查格式、四元数、关节限位和速度、基座高度与倾斜，
抽样检查非足部地面接触，并提示可能依赖道具/地面支撑的动作名称及数据集。
它只标记风险，所有动作仍是 `needs_review`，不会自动复制到训练目录；无标记也不代表动力学可行。
筛选阈值和训练目录统一放在 `screening` 中，实际训练只指向复核后的目录。

运动学预览：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/play_motion.py \
  --motion data/amass_g1_candidates/CNRS__283__-01_L_1_stageii.csv --loop
```

输出仍为无表头的 36 列、30 Hz CSV：`base_pos_w[3], base_quat_xyzw[4], joint_pos[29]`。
文件名前缀保留子集与受试者路径，避免不同受试者同名动作相互覆盖。
来源、身体模型、配置、源码和 CSV 校验值记录在报告中；重复转换会校验后复用已有结果。
修改 IK 或身体模型时使用新候选输出目录，避免混用不同版本。

历史初始验证使用 HumanEva 28 条、15,264 帧，14,396 个识别出的支撑帧均有足底—地面接触；该子集现已按实测结果删除，这些历史验证不代表动作可用于训练。
人体 FK 与官方 SMPL-X 的实测最大关节位置差约 `1.74e-7 m`；训练读取和 G1 FK 已验证。
这些结果只证明格式、运动学和接触检查通过，尚未证明策略能稳定跟踪。

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
  --model exports/policy.onnx --motions "walk1_subject1|dance1_subject1" --duration 30
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
