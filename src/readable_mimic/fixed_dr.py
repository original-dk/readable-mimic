"""Fixed dynamics configuration shared by native deployment and training evaluation."""

import json
from pathlib import Path
import numpy as np


def load_dr(path, settings):
    """Load fixed physical values. Scale gains are relative to the exported joint gains."""
    values = {
        "friction": settings["friction"], "ground_friction": settings["friction"],
        "joint_pos_offset": 0.0, "center_of_mass_body": settings.get("center_of_mass_body", "torso_link"),
        "center_of_mass_offset": [0.0, 0.0, 0.0],
        "mass_scale": 1.0, "inertia_scale": 1.0, "armature_scale": 1.0,
        "passive_damping_add": 0.0, "frictionloss_add": 0.0,
        "kp_scale": settings["kp_scale"], "kd_scale": settings["kd_scale"],
        "action_delay_ms": 0.0,
    }
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))["domain_randomization"]
        unknown = supplied.keys() - values.keys()
        if unknown:
            raise ValueError(f"Unknown DR keys: {sorted(unknown)}")
        values.update(supplied)
    for name, value in values.items():
        if name == "center_of_mass_body":
            if not isinstance(value, str) or not value:
                raise ValueError("center_of_mass_body must be a body name")
            continue
        array = np.asarray(value, dtype=float)
        expected_shape = (3,) if name == "center_of_mass_offset" else ()
        if name == "joint_pos_offset":
            if array.shape not in ((), (len(settings["joint_names"]),)):
                raise ValueError("joint_pos_offset must be a scalar or one value per joint")
        elif array.shape != expected_shape:
            raise ValueError(f"Invalid shape for {name}: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        if name == "action_delay_ms" and (
            float(array) not in (0.0, 5.0, 10.0, 15.0, 20.0)
        ):
            raise ValueError("action_delay_ms must be one of 0, 5, 10, 15, 20")
        if name.endswith("_scale") and np.any(array <= 0):
            raise ValueError(f"{name} must be positive")
        if name in ("friction", "ground_friction", "passive_damping_add", "frictionloss_add") and np.any(array < 0):
            raise ValueError(f"{name} must be nonnegative")
    return values

