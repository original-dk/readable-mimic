# 配置对照与复现边界

参考上游 commit `d264658d6c9ce61fb52f9f0f8985a369f3ccc409` 的
`train.py` 默认参数与 `scripts/presets.sh` 中 `PRESET[main]`。

| 设置 | 当前值 | 对照 |
|---|---|---|
| num_envs / buffer_steps / utd | 4096 / 1000 / 0.4 | 一致 |
| horizon / goal_frames / hist | 32 / 10 / 10 | 一致 |
| wm_window / wm_batches / wm_batch | 8 / 5 / 2048 | 一致 |
| pi_batch | 2048 | 一致 |
| wm_hidden | 1024,1024,1024 | 一致 |
| enc_hidden / dec_hidden | 2048,1024,512,256 | 一致 |
| lr / gamma / lam | 4e-4 / 0.99 / 0.95 | 一致 |
| critic_epochs / critic_minibatches | 5 / 4 | 一致 |
| action_noise / pi_beta1 / pi_clip_value | 0.1 / 0.9 / 0.1 | 一致 |
| pi_ema / warmup_steps | 0.995 / 64 | 一致 |
| wm_structured / wm_mirror | true / 0.5 | 一致 |
| pi_opt / wm_opt | muon / muon | 一致，使用 eager 实现 |
| pi_ref_skip / wm_only_iters | true / 0 | 一致 |
| contact_reward / ref_contact_fix | 0.25 / true | 一致 |
| pi_lr_mult / lr_decay_ret | 4 / 0.75 | 一致 |
| lr_decay_len / lr_decay_by | 300 / 1400 | 一致 |
| iterations / ckpt_every | 3500 / 100 | 一致 |
| 奖励、随机化、观测噪声 | config.json 对应小节 | 从上游 dataclass 默认值导出 |
| MuJoCo 参数 | config.json 的 simulation | 与上游相同，50 Hz 控制 / 200 Hz 物理 |
| 失败、回报优先和 probe reset | 不实现，均匀时间 bin | 按本项目要求简化 |
| WM rollout / loss compile | CUDA 训练默认开启 | compile_world_model 控制，分别编译两个已有方法 |
| AMP / fused / CUDA Graph | 关闭 | 按本项目要求简化 |

配置名沿用上游，数据名称采用本项目完整名称。G1 固定关节顺序、驱动器和模型结构
在 g1_constants.py 中，属于机器人描述；实验参数在 config.json 中。
当前只实现原论文使用的观测噪声、摩擦、关节偏移与质心偏移。
未启用的 IMU、滤波、传感延迟、质量/PD 随机化配置和代码已删除。
观测噪声小节为 observation_noise，数值保持上游一致。

学习率回报触发与上游一致，先取每次采样完成 episode 的平均回报，再取日志窗口均值；
触发检查发生在日志边界，最晚触发 iteration 1400。
所有 checkpoint 同步完整写入；不包含上游的轻量异步写入。

本机测试环境：Windows、Intel Graphics、torch 2.9.0 CPU、mjlab 1.6.0、warp-lang 1.16.0。
Linux 安装通过 uv 的平台条件源使用 torch 2.9.0 CUDA 12.8。
GPU G1 学习曲线和最终回报尚未验证，不能据此宣称已复现论文结果。

## 数据相关配置名称映射（数值保持不变）

| 上游名称 | 本项目名称 |
|---|---|
| body_ori_std | body_rot_std |
| body_linvel_std / body_angvel_std | body_lin_vel_std / body_ang_vel_std |
| root_pos_std / root_ori_std | base_pos_std / base_rot_std |
| w_action_rate / w_joint_limit | action_rate_weight / joint_limit_weight |
| term_height / term_tilt | termination_height_threshold / termination_tilt_threshold |
| noise_joint_pos / noise_joint_vel | joint_pos_noise / joint_vel_noise |
| noise_ang_vel / noise_gravity | base_ang_vel_noise / projected_gravity_noise |
| noise_ref_ori | ref_base_rot_noise |
| joint_offset | joint_pos_offset |

以上更名已同步到 config.json、配置类和调用处；旧配置文件需使用该映射迁移。
