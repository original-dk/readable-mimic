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
    values.setdefault("grad_optimization", {})
    values.setdefault("resume", None)
    values.setdefault("failure_reset_fraction", values.get("fail_reset", 0.0))
    values.setdefault("ground_supported_reset", False)
    values.setdefault("reset_foot_clearance_cm", 3.0)
    values.setdefault("reset_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"])
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
    motion_directories = values["motion_dir"]
    if isinstance(motion_directories, list):
        if not motion_directories or any(not isinstance(item, str) or not item.strip() for item in motion_directories):
            raise ValueError("motion_dir must contain at least one nonempty directory")
        values["motion_dir"] = [str((path.parent / item).resolve()) for item in motion_directories]
    elif isinstance(motion_directories, str) and motion_directories.strip():
        values["motion_dir"] = str((path.parent / motion_directories).resolve())
    else:
        raise ValueError("motion_dir must be a directory string or a nonempty list of directories")
    for key in ("log_dir", "resume"):
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
            "ground_supported_reset",
            "failure_reset_fraction",
            "reset_foot_clearance_cm",
            "reset_foot_bodies",
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
        # Older checkpoints resume their original update rule.
        values["grad_optimization"] = saved_config.get("grad_optimization", {})
    config = SimpleNamespace(**values)
    from .grad_optimization import GradientOptimizationConfig
    gradient_config = GradientOptimizationConfig(**config.grad_optimization)
    if gradient_config.model_resets and config.failure_reset_fraction:
        raise ValueError("Choose model-ranked resets or failure_reset_fraction")

    assert config.simulator in ("mujoco", "genesis"), "simulator must be mujoco or genesis"
    assert config.amp == "none", "Only FP32 is supported."
    if not 0 <= config.failure_reset_fraction <= 1:
        raise ValueError("failure_reset_fraction must be in [0, 1]")
    if config.failure_reset_fraction and not config.ground_supported_reset:
        raise ValueError("Failure reset requires ground_supported_reset")
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
