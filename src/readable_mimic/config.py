"""One JSON file is the source of experiment configuration."""
import json
from pathlib import Path
from types import SimpleNamespace


def load_config(path='config.json'):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding='utf-8'))
    for key in ('motion_dir', 'log_dir', 'resume'):
        if values[key] is not None:
            values[key] = str((path.parent / values[key]).resolve())
    config = SimpleNamespace(**values)
    assert config.amp == 'none', 'Only FP32 is supported.'
    assert config.fail_reset == 0, 'Only uniform reset sampling is supported.'
    assert config.warmup_steps >= config.wm_window
    assert config.buffer_steps >= config.warmup_steps
    assert config.horizon * config.pi_batch >= config.critic_minibatches
    assert config.utd > 0
    return config
