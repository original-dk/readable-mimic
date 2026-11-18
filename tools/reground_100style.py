import hashlib
import json
from pathlib import Path
import shutil
import argparse
import mujoco
import numpy as np
from readable_mimic.retargeting.bvh import load_bvh
from readable_mimic.retargeting.ik import detect_support_frames, ground_motion

parser = argparse.ArgumentParser(
    description="Reground existing 100STYLE CSVs without repeating IK; preserve originals."
)
parser.add_argument("--config", default="retarget_100style.json")
config = Path(parser.parse_args().config).resolve()
root = config.parent
settings = json.loads(config.read_text(encoding="utf-8"))
import readable_mimic.retargeting.ik as ik_module

package = Path(ik_module.__file__).parent
settings["converter_sha256"] = hashlib.sha256(
    b"".join((package / name).read_bytes() for name in ["bvh.py", "ik.py", "convert_100style.py"])
).hexdigest()
for key in ["input_dir", "output_dir", "frame_cuts", "mjcf_path"]:
    settings[key] = str((root / settings[key]).resolve())
source_files = {path.stem: path for path in Path(settings["input_dir"]).rglob("*.bvh")}
model = mujoco.MjModel.from_xml_path(settings["mjcf_path"])
directory = Path(settings["output_dir"])
backup = directory.with_name(directory.name + "_before_grounding")
backup.mkdir(exist_ok=True)
files = sorted(directory.glob("*.csv"))
if not files:
    raise ValueError("No CSV files found")
results = []
for index, path in enumerate(files):
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    for source in [path, metadata_path]:
        destination = backup / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    raw = np.loadtxt(backup / path.name, delimiter=",")
    current = np.loadtxt(path, delimiter=",")
    if current.shape != raw.shape or not np.allclose(
        current[:, [0, 1, *range(3, 36)]], raw[:, [0, 1, *range(3, 36)]], atol=1e-8, rtol=0
    ):
        raise ValueError(f"Original backup no longer matches joint poses: {path}")
    qpos = raw.copy()
    qpos[:, 3:7] = raw[:, [6, 3, 4, 5]]
    source = source_files[path.stem]
    cuts = [metadata["source_start_frame"], metadata["source_stop_frame"]]
    human = load_bvh(source, *cuts, 30, settings["bvh_position_scale"], settings["bvh_axis_order"])
    support = detect_support_frames(human, settings["grounding"])
    correction = ground_motion(model, qpos, settings["foot_bodies"], settings["ground_clearance"], support)
    raw[:, 2] = qpos[:, 2]
    temporary = path.with_suffix(".csv.part")
    np.savetxt(temporary, raw, delimiter=",", fmt="%.8f")
    temporary.replace(path)
    fingerprint = hashlib.sha256(
        source.read_bytes()
        + json.dumps(
            {
                k: v
                for k, v in settings.items()
                if k
                not in (
                    "workers",
                    "styles",
                    "movements",
                    "input_dir",
                    "output_dir",
                    "frame_cuts",
                    "mjcf_path",
                )
            },
            sort_keys=True,
        ).encode()
        + Path(settings["mjcf_path"]).read_bytes()
        + json.dumps(cuts).encode()
    ).hexdigest()
    metadata.update(
        grounding_version=2,
        ground_offset=correction,
        fingerprint=fingerprint,
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results.append(metadata)
    if (index + 1) % 50 == 0:
        print("Grounded", index + 1, "/", len(files), flush=True)
if files:
    (directory / "conversion_report.json").write_text(
        json.dumps(dict(config=settings, clips=results, failures=[]), indent=2), encoding="utf-8"
    )
print("Updated", len(files), "clips; originals preserved in", backup, flush=True)
