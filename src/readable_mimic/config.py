"""One JSON file is the source of experiment configuration."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class ObservationNoiseConfig:
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.5
    base_ang_vel_noise: float = 0.2
    projected_gravity_noise: float = 0.05
    ref_base_rot_noise: float = 0.05


def load_config(path="config.json", resume=None, *, restore_training_config=True):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    values.setdefault("resume", None)
    values.setdefault("dkwbc_feedforward", False)
    values.setdefault("profile_speed", False)
    values.setdefault("compile_world_model", True)
    values.setdefault("compile_actor", True)
    values.setdefault("allow_tf32", True)
    values.setdefault("simulator", "mujoco")
    values.setdefault(
        "genesis",
        {
            "enable_self_collision": False,
            "enable_adjacent_collision": False,
            "enable_mujoco_compatibility": True,
            "use_hibernation": False,
        },
    )
    if resume is not None:
        values["resume"] = str(Path(resume).resolve())
    for key in ("motion_dir", "log_dir", "resume"):
        if values[key] is not None:
            values[key] = str((path.parent / values[key]).resolve())
    if restore_training_config and values["resume"] is not None:
        import torch

        checkpoint = torch.load(values["resume"], map_location="cpu", weights_only=False)
        saved_config = checkpoint["config"]
        runtime_fields = (
            "device",
            "num_threads",
            "motion_dir",
            "log_dir",
            "resume",
            "iterations",
            "ckpt_every",
            "ckpt_minutes",
            "max_hours",
            "log_every",
            "profile_speed",
            "compile_world_model",
            "compile_actor",
            "allow_tf32",
            "simulator",
            "genesis",
        )
        runtime_config = {name: values[name] for name in runtime_fields}
        values = {**values, **saved_config, **runtime_config}
    config = SimpleNamespace(**values)
    assert config.simulator in ("mujoco", "genesis"), "simulator must be mujoco or genesis"
    if config.dkwbc_feedforward and (config.simulator != "genesis" or config.robot != "g1"):
        raise ValueError("dkwbc_feedforward requires Genesis and G1")
    assert config.amp == "none", "Only FP32 is supported."
    assert config.fail_reset == 0, "Only uniform reset sampling is supported."
    assert config.warmup_steps >= config.wm_window
    assert config.buffer_steps >= config.warmup_steps
    assert config.horizon * config.pi_batch >= config.critic_minibatches
    assert config.utd > 0
    assert config.pi_ema > 0 and config.pi_ema < 1
    assert config.simulation["control_dt"] == 0.02 and config.simulation["decimation"] == 4, (
        "Reference resampling uses 50 Hz and four physics substeps."
    )
    assert config.ckpt_every > 0 and config.log_every > 0
    assert config.critic_epochs > 0 and config.wm_batches > 0
    assert 0 <= config.wm_mirror <= 1
    return config
