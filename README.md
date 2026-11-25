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

### 100STYLE / AMASS → G1 数据

两个数据集各用一个目录，训练输入与中间结果分开：

```text
data/
  100style/
    raw/                 # BVH、Frame_Cuts.csv、archives/100STYLE.zip
    candidates/          # 30 Hz G1 CSV 与逐动作 JSON
    cache/               # reports/、旧实验和落地修正备份
    train/               # 通过统一筛选的 CSV，只有 CSV
  amass/
    raw/                 # 各子集 NPZ、archives/、download_manifest.json
    candidates/          # 全部子集共用的 G1 候选输出
    cache/               # reports/、中间缓存
    train/               # 通过同一套筛选的 CSV，只有 CSV
```

`raw`、候选和许可证数据不提交 Git。机器人和授权 SMPL-X 模型仍作为共享资源保存在
`data/g1_description/`、`data/body_models/smplx/`。旧试验集中在 `100style/cache/legacy/`；
旧 AMASS 修复前候选已删除，不能作为训练输入。

离线重定向使用独立 uv 环境，CPU 即可：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv sync --extra retarget --locked
```

100STYLE 的下载、官方帧裁剪、BVH 轴向与 IK 配置在 `retarget_100style.json`：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget python tools/download_100style.py
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python -m readable_mimic.retargeting.convert_100style --config retarget_100style.json
```

官方数据为 100 种风格、810 条片段。BVH 按每个关节声明的旋转顺序读取，厘米转米、Y-up 转 Z-up，
按 `[START, STOP)` 裁剪并重采样为 30 Hz，不镜像、不改变播放速度。
转换器只执行重采样和 IK，保持 IK 的基座高度，不做足底修正或平滑。
AMASS 转换器遵守同样的阶段划分；统一由 `tools/process_motion_csv.py` 做后处理。
新转换元数据包含 `grounding_applied: false`、`processing_stage: retargeted`。
本地统一第一阶段结果位于 `data/100style/candidates` 和 `data/amass/candidates`，
各家输入格式由各自转换器处理，输出都遵守同一个 30 Hz G1 CSV 格式。
旧版已做接地修正的候选保存在各数据集的 `cache/legacy/grounded_candidates_previous/`，
不会与新候选混用。本次只更新第一阶段结果，未重新生成或覆盖正式 train。
如需变更 IK 并保留本批候选，转换器支持 `--output-dir <新候选目录>`。

AMASS 需要本人获得各子集与 SMPL-X 模型授权。只使用 **SMPL-X N**，不混入 G、SMPL+H 或 Render。
从官网下载的完整压缩包统一放入 `data/amass/raw/archives/`：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/prepare_amass.py --config retarget_amass.json --archives data/amass/raw/archives
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python -m readable_mimic.retargeting.convert_amass --config retarget_amass.json
```

**CMU 与其他 AMASS 子集只使用同一个 `convert_amass.py`。** `retarget_amass.json` 的 `datasets`
限定处理范围，当前保留 CMU、CNRS、HDM05、SOMA、Transitions、SFU、PosePrior；其他已下载原始库仍可留存，
不会因此自动参与转换。已放弃的 ACCAD、BMLmovi、BMLrub、DFaust、EKUT、HumanEva、
Eyes_Japan_Dataset、KIT、MoSh 已删除并排除，勿用空列表重新导入。
转换读取 `*_stageii.npz` 的身体形状与前 22 个关节，不计算手指、表情或软组织；按真实帧率插值为 30 Hz。
AMASS IK 对齐 GMR 的身高缩放、关节零位种子、离线限位、足尖坐标系及迭代停止条件。
**处理第一帧前，基座位置和朝向对齐到该帧的机器人目标，包含 SMPL→G1 旋转偏移**；
固定世界零朝向在初始方向差较大时可能把肩肘 IK 推到反折局部解。
仅在离线模型副本添加足尖任务帧，训练机器人 XML 和限位不变。
过去 CMU `01_01`、`05_06` 与 GMR 默认固定初始化的数值一致性属于旧版本验证；
当前比较原版算法时也需要使用同样的首帧初始化，不能要求与错误分支的旧轨迹逐值相同。
人体 FK 与官方 SMPL-X 的位置误差实测约 1.74e-7 m；这些是运动学一致性证据，不是动态跟踪保证。

### 第一阶段统一 CSV 结果

第一阶段只做源格式读取、必要的官方片段裁剪、重采样和 G1 重定向，不做动作质量剔除、平滑或足底修正。
CSV 无表头，每行 36 列，30 Hz，位置单位米、关节单位弧度：
`base_pos_w[3], base_quat_xyzw[4], joint_pos[29]`。四元数为 xyzw，关节顺序为项目 G1 的 `joint_names`。
不同来源的 CSV 可直接交给同一个 `tools/process_motion_csv.py`。
LAFAN1 原有 `data/lafan1_g1` 的 40 条 CSV 已是同一布局，保持上游文件不变；不对其上游处理历史作新的判定。

| 目录/前缀 | 原格式 | CSV 数量 |
|---|---|---:|
| `data/100style/candidates/` | BVH | 810 |
| `data/amass/candidates/CMU__*` | SMPL-X NPZ | 1,983 |
| `data/amass/candidates/CNRS__*` | SMPL-X NPZ | 79 |
| `data/amass/candidates/HDM05__*` | SMPL-X NPZ | 215 |
| `data/amass/candidates/SOMA__*` | SMPL-X NPZ | 69 |
| `data/amass/candidates/Transitions__*` | SMPL-X NPZ | 110 |
| `data/amass/candidates/SFU__*` | SMPL-X NPZ | 44 |
| `data/amass/candidates/PosePrior__*` | SMPL-X NPZ | 35 |

逐动作 JSON 保留来源和格式信息，并明确 `processing_stage: retargeted`、`grounding_applied: false`。
各自的 `cache/reports/unified_csv_validation.json` 记录全量文件、来源覆盖和校验值验证结果。
100STYLE 利用经源文件、配置、模型指纹验证的早期备份，去除其精确记录的常量高度偏移恢复 IK 输出；
与新转换器重新求解的三个样本最大差异为 1e-8（CSV 舍入精度）。AMASS 由新转换器完整重新求解。
所有已放弃子集维持排除；尚未导入的 GRAB 等下载包不属于本批结果。

### HDM05 首帧初始化修复复测

回归样例为 `HDM05__tr__HDM_tr_01-01_01_120_stageii` 和
`HDM05__mm__HDM_mm_08-01_01_120_stageii`。前者旧结果的左肩俯仰约 99.8% 帧贴限位，
后者右肩/肘也进入反折分支。修正首帧基座初始化后，两条完整轨迹的肩部限位饱和比例均为零。
测试增加了相同任务旋转至 0、145、−85 度世界朝向时关节解不变的检查，防止同类回归。

HDM05 全部 215 条修复版原始候选在 `data/amass/cache/hdm05_fixed/candidates/`，
再次经过 CSV 后处理的全部候选在 `data/amass/cache/hdm05_fixed/processed/candidates/`，
筛选通过的测试 CSV 在 `processed/train/`。
已将 215 条修复版 HDM05 同步到正式 `data/amass/candidates`，旧候选及旧后处理缓存备份在
`data/amass/cache/legacy/hdm05_before_promotion/`。按用户原命令以 `data/amass/train` 作为输出根重跑后，
真正的导出 CSV 位于 `data/amass/train/train/`（输出根不是最终 train），其中 `mm_08-01_01` 已与修复版逐字节一致。
旧播放器需关闭重开才能加载更新文件；代码更新不会自动替换另一台机器上已有的 CSV，跨机器需同步更新的数据。
`arm_tracking_comparison.json`、`initialization_regression.json` 保留完整对照数值。
该修复在统一 AMASS 转换器中生效；旧 CSV 不会自动改变，其他库重转时应使用新的候选目录。

### 统一筛选并导出 train

两类候选均为无表头、36 列、米/弧度的 CSV：
`base_pos_w[3], base_quat_xyzw[4], joint_pos[29]`，关节顺序为项目的 `G1_JOINT_NAMES`。
逐动作 JSON 保存 30 Hz、帧数、来源、关节顺序、落地修正及 CSV SHA256；报告不放进 train。
相同源与配置会复用校验过的结果；改变 IK 或模型时使用新的候选目录，避免混用版本。

统一处理入口是 **`tools/process_motion_csv.py`**，内部复用 `tools/prepare_training_data.py` 的筛选与导出实现。两类数据共享
默认规则 **`motion_filter_level_1.json`**（原 `motion_filter_keep_fallup.json`）。
需要使用原严格规则时，显式传入 `--rules motion_filter_level_0.json`。

`--output-dir` 是处理根目录。若输入是 `data/amass/candidates` 且输出根设为 `data/amass`，
处理后的候选写入 `data/amass/processed_candidates`，训练 CSV 写入 `data/amass/train`。

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/process_motion_csv.py --input-dir data/100style/candidates \
  --output-dir data/100style/cache/csv_processed_v2 --config retarget_100style.json
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/process_motion_csv.py --input-dir data/amass/candidates \
  --output-dir data/amass/cache/csv_processed_v2 --config retarget_amass.json
```

检查列数、有限值、四元数、30 Hz、帧数与校验值、关节顺序、限位和速度、基座高度和倾角。
**逐帧**用实际足底碰撞几何检查平地支撑：足底高于 2 cm 视为腾空，连续超过 1 秒拒绝；
超过 0.1 秒的短时腾空还必须具有起落地边界，且整机质心的重力抛物线拟合最大误差不超过 6 cm。
足底穿地超过 3 mm 也拒绝。这里检查的是碰撞面的最低点，不是踝关节原点；保留正常短时跳跃，
楼梯、悬停及依赖高处支撑的参考不能因为“没有穿地”就通过。各阈值集中在 `motion_filter_level_1.json`。
每 5 帧另外检查非足部地面穿透；可能依赖道具或其他支撑的名称也会被标记。
报告记录悬空比例、最长悬空时长、足底高度、各腾空区间与拟合误差，以及筛选源码 SHA256。
`CMU__13__13_37` 是本地回归样例：连续悬空约 8.77 秒，必须被拒绝，不通过名字黑名单绕过检测。
只有全部检查通过的动作复制到 train，其他候选保留供复核，原因见各自
`cache/reports/screening_report.json`。单独复核已经后处理的候选时，可运行
`tools/prepare_training_data.py --config <输出根目录>/cache/reports/filter_config.json --rules motion_filter_level_1.json`；
加 `--export` 才更新 train。这个独立筛选命令不做平滑和接地，不能替代统一后处理入口。
自动筛选不证明无抖动、无自碰撞或策略能稳定跟踪，实际训练前仍应预览并评估。
手动测试后不满意的动作，把不带扩展名的完整文件名加入 `motion_filter_level_1.json` 的
`excluded_motions`，再执行导出；同一规则会阻止它再次进入 train。
`cache/reports/training_export.json` 记录导出校验值；后续筛选只清理此前工具导出的旧文件，
遇到手工修改的训练文件会报错保留。导出临时文件放 cache，train 只有最终 CSV。

训练配置的 `motion_dir` 指向本次输出目录的 `train`，例如 `data/amass/cache/csv_processed_v2/train`；
加载器照常重采样至 50 Hz 并做 FK。原有正式 train 目录保留，不自动替换。
不要把 raw、candidates 或 cache 指定为正式训练目录。

### Level 1 楼梯筛选

`motion_filter_level_1.json` 的 `stair_support` 独立定义楼梯规则。先完成足底修正，再检查处理后的 CSV。
同一只脚的速度低于 0.15 m/s、两脚脚底都高于地面 8 cm、骨盆比起始参考高度抬升超过 12 cm，
三项连续同时满足至少 0.3 秒时标记 `stair_support`。起始参考高度取五帧中值滤波后的前九帧中位数。
明确的地形名称可单独触发 `stair_terrain_label`，骨盆漂移指标单独记录供复核。

在 2,535 条处理后的 AMASS 候选上，楼梯专项命中 104 条。已知楼梯 `CMU__36__36_01` 和
`CMU__13__13_37` 均命中；翻跟头 `CMU__90__90_03`、跳绳
`HDM05__tr__HDM_tr_03-10_02_120` 和单腿支撑 `PosePrior__03099__lar3` 均未命中。
专项名单位于 `data/amass/cache/reports/stair_combined_hits.csv`，不含椅子规则的结果。

### CSV → CSV 批量平滑与筛选（推荐入口）

只用原 CSV 即可，不要求输入同名 JSON、AMASS NPZ 或 SMPL-X 模型。
输入固定为项目的 **30 Hz、36 列 G1 CSV**：`base_pos_w[3], base_quat_xyzw[4], joint_pos[29]`。
仍需要项目的 G1 MJCF 来计算足底碰撞面、关节限位和质心。

Linux：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget python tools/process_motion_csv.py \
  --input-dir data/amass/candidates \
  --output-dir data/amass/cache/csv_processed_v2 \
  --pattern "CMU__*.csv"
```

Windows PowerShell：

```powershell
$env:UV_PROJECT_ENVIRONMENT=".venv-retarget"
uv run --extra retarget python tools/process_motion_csv.py --input-dir data/amass/candidates --output-dir data/amass/cache/csv_processed_v2 --pattern "CMU__*.csv"
```

`--pattern "CMU__*.csv"` 只处理 CMU；省略 `--pattern` 处理输入目录中全部 CSV。
也可将输入换成 `data/amass/train` 或 `data/100style/candidates`，并使用新的输出目录。
选择范围和路径以命令参数为准。
默认 `--config retarget_amass.json` 提供 G1 模型和足部名称，
`--rules motion_filter_level_1.json` 提供平滑、接地修正参数和全部筛选阈值。

流程：**原 CSV → 平滑 → 按原 CSV 的接地分段修正足底高度 → 共同筛选 → 新 CSV**。
关节和基座位置采用二阶 5 Hz 双向零相位滤波；朝向先校正四元数符号连续性，滤波后归一化。
滤波越限用连续幅度收缩处理。`grounding` 配置集中管理接地：先以足底高度的 2% 分位估计整体
偏差，只允许 5 cm 内的基准对齐（超出则不做整体搬移），再识别支撑分段。
支撑帧由对齐后原 CSV 中实际足底碰撞面位于
`[-max_foot_penetration, foot_contact_height]` 的区间确定；无支撑时保留轨迹交给筛选，
不会把持续悬空或深度穿地的动作整体搬到地面。
**此 CSV-only 分段不读取人体标签，与旧 NPZ 辅助平滑的输出不保证逐值一致。**

当前共同限制是：基座倾角 70 度、连续离地 1 秒、质心重力轨迹拟合误差 6 cm；
原有格式、四元数、限位、速度、最低基座高度和接地检查继续生效。
最大关节改动超过 `smoothing.max_joint_change_degrees`（30 度）的平滑结果标记为
`large_smoothing_change`，留在候选供复核，不直接进入 train。所有参数在 `motion_filter_level_1.json`。

输出结构：

```text
data/amass/cache/csv_processed_v2/
  candidates/             # 全部处理成功的新 CSV 与元数据
  train/                  # 筛选通过的新 CSV，只有 CSV
  cache/reports/
    processing_report.json # 来源、参数、处理失败、改变量
    screening_report.json  # 每条通过/拒绝原因与异常时段
    training_export.json   # train 文件校验值
```

脚本不改输入文件，拒绝输入与输出目录相同或相互嵌套；递归输入会把相对路径用 `__` 合成唯一文件名。
相同输入和处理参数校验后复用缓存，并重新执行筛选。改变平滑参数或输入内容时使用新的输出目录；
只调整筛选限制时可以复用原输出目录。处理失败时会记录错误、以非零状态退出，不更新 train。
快动作仍可能被平滑削弱，请预览新 CSV；筛选通过不代表动力学可行。

旧的 `data/amass/cache/cmu_smoothed` 是使用原始人体支撑标签生成的历史测试集，保留用于对照。
新的 CSV-only 批处理使用独立输出目录，避免混用不同接地分段版本。

### 直接查看 CSV 动作

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget \
  python tools/play_motion.py --motion data/100style/candidates/Aeroplane_FW.csv --loop
```

直接赋值 MuJoCo `data.qpos` 并调用 `mj_forward`；不运行策略、PD 或物理步进。
CSV 四元数 `xyzw` 在播放时转换为 MuJoCo 的 `wxyz`。默认 30 fps，`--headless` 可快速检查加载，
关窗或 Ctrl+C 退出；`--loop` 循环播放，`--fps` 和 `--mjcf` 可指定帧率和模型。
100STYLE 遵循 CC BY 4.0，GMR 遵循 MIT，引用与许可见 `THIRD_PARTY_NOTICES.md`。

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

`motion_dir` 支持单个目录或目录列表，例如
`"motion_dir": ["data/lafan2/train", "data/amass/train"]`。
多个目录合并为同一个动作库，按配置顺序加载，各目录内按文件名排序。
重复指向同一文件时只加载一次；不同文件使用同一动作名时会报错，需先修改文件名。
训练、评估和 ONNX 导出均使用这套读取方式。采样按合并后的时间 bin 分配，
目录之间的采样比例取决于动作时长和当前困难度采样。
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

评估时可用 `--motion-dir` 覆盖配置里的动作目录。省略该参数时沿用配置中的 `motion_dir`。
覆盖仅对本次评估生效，训练配置文件不会被改写。`--motions` 从评估使用的目录中按动作名匹配。

例如，用 BONES-SEED 训练的 checkpoint 评估 LAFAN1 动作：

```bash
UV_PROJECT_ENVIRONMENT=.venv-genesis uv run --extra genesis \
  python -m readable_mimic.evaluate \
  --config runs/boneseed/config.json \
  --checkpoint runs/boneseed/last.pt \
  --motion-dir data/lafan1_g1 \
  --motions "walk1_subject1|dance1_subject1" \
  --duration 60 --ghost-base follow --output-dir videos/genesis/boneseed
```
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


CSV 筛选中的 `unsupported_support` 只针对近静止坐姿，不再直接拒绝动态质心出界。
必须同时满足：双侧大腿接近水平（竖直分量/大腿长度不超过 0.65）、有脚底接近地面、
质心速度不超过 0.15 m/s、质心加速度不超过 1.5 m/s²、两脚速度均不超过 0.12 m/s，
且质心水平投影超出脚底支撑区域 0.05 米，连续超过 0.5 秒。
所有阈值和大腿身体名称统一位于 `motion_filter_level_1.json`；腾空、运动或离开坐姿会中断计时。
这是外部支撑依赖的保守几何启发式，不代表完整动力学验证，也不能识别所有坐椅子动作。
报告保留最大质心出界距离，以及满足上述全部条件的连续区间和最长时长。
在 `process_motion_csv.py` 的共用筛选阶段执行，已有平滑缓存仍会重新筛选。
重跑原命令后，先前导出且未被手工修改的不合格 CSV 会从输出的 `train/` 移除；原始候选保留。
基座倾角与短时腾空的抛物线拟合偏差只作为报告指标，不作为拒绝条件。
仍拒绝超过允许时长的连续悬空、首尾无法核验的较长腾空、穿地、外部支撑依赖以及数据格式/关节限位等错误。

### BONES-SEED G1

从项目根目录运行。输入是解包后的 G1 CSV 目录，支持递归查找。

```bash
uv run --extra retarget python tools/prepare_boneseed.py --input-dir /path/to/g1/csv --output-dir data/boneseed
```

先试跑时加 `--pattern '*walk*.csv' --limit 5`，并使用单独的输出目录。只做格式转换时加
`--convert-only`。`converted/` 是 30 Hz、无表头的 36 列 G1 CSV，`processed/train/`
是通过共用平地筛选的训练 CSV。训练时把运动目录指向 `processed/train/`。
`boneseed_conversion.csv` 记录名称拒绝原因，`processed/cache/reports/screening_report.json`
记录几何筛选结果。仅转换或全部被名称拦截时只生成转换阶段输出。

名称规则参考 [Uni-SuperMimic BONES-SEED 筛选](https://github.com/Motphys/Uni-SuperMimic/blob/main/src/unisupermimic/data/boneseed.py)，
仅选用楼梯、外物支撑等相关词，配置在 `motion_filter_level_1.json` 的 `boneseed` 段。
几何检查沿用本项目的持续悬空、落地和坐姿支撑规则。无标签楼梯的识别取决于这些几何条件。


### 执行器目标延迟

训练配置 `domain_randomization.action_delay_max_ms` 支持 0、5、10、15、20 ms，
默认 20 ms。每个并行环境在 reset 时均匀采样一个物理步延迟，episode 内保持固定。
设为 0 可关闭随机延迟，并使用原有 world model / critic 输入维度。

Policy 每 20 ms 输出一次动作。Genesis 使用 dt=0.005、substeps=1，
每个控制周期外部推进四步；MuJoCo 同样每周期推进四个 5 ms 物理步。
延迟 10 ms 时，前两步沿用上一位置目标，后两步执行新目标。
Reset 的旧目标取实际初始关节姿态。

启用延迟后，G1 的 xi 从 40 维变为 41 维，末项为物理步数 0–4；
buffer 额外保存 previous_target，world model 和 critic 接收该目标相对当前关节的位置误差。
四个物理步均结束后，当前下发目标成为下一周期的 previous_target。
观测中的 last_action 仍表示上一条下发动作。启用延迟需要重新训练 world model 和 critic。

评估与 ONNX 播放使用 `--dr dr.json`，其中 `action_delay_ms` 为固定延迟值，
例如 10.0。省略该配置时使用标称零延迟。模型困难度排名使用标称零延迟。


#### 基于 MorphMimic 的延迟建模改进

执行器延迟的采样方式参考 MorphMimic。该项目在物理步尺度延迟位置目标，
默认范围为 0–20 ms，每个 episode 内保持固定，标称评估使用零延迟。
本项目在此基础上增加显式延迟状态，作为 world model 与 critic 的输入，
改进目标是减少隐藏执行器状态给动力学预测和价值估计带来的不确定性。
参考版本为提交 `770394b4a0cad9effbf807a6c03435d1699733d0`。
[MorphMimic 延迟配置](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/env/env_cfg.py)

MorphMimic 的 world model 使用当前物理状态、当前动作、33 维 xi 和可选形态参数；
xi 包含摩擦、编码器偏置和躯干 COM 偏移。其 world model 与 critic 输入未显式包含
采样到的执行器延迟或待执行的位置目标队列。本项目为 xi 增加延迟步数，
并在 replay 中记录 `previous_target`，将其相对当前关节位置的误差提供给
world model 和 critic。
[MorphMimic world model](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/model/world_model.py) ·
[MorphMimic critic](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/model/policy.py)

设计依据是执行器目标缓冲会影响后续运动。例如，两条样本的当前关节位置、
速度和新目标相同，但旧目标或延迟不同，下一控制周期的状态仍可能不同。
在最大延迟不超过一个控制周期的当前设置下，上一条目标和延迟步数能够描述
这一段目标切换过程。Reset 时使用实际初始关节姿态初始化旧目标；
imagined rollout 中沿用相同的目标历史更新规则，并保留跨步梯度。
Critic 也读取该历史状态，以区分后续执行目标不同的状态。

这是相对于参考实现的建模扩展，其实际训练收益需要消融实验验证。
新增输入增加了 replay 存储、模型输入维度和训练成本，启用时需要重新训练
world model 与 critic。验证时应固定物理延迟分布、数据集和训练预算，
比较“仅仿真延迟”和“仿真延迟加显式延迟状态”两种设置，
检查多步预测误差、梯度稳定性、跟踪质量和达到目标成功率所需的训练时间。
