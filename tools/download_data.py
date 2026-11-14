"""Download the same public motion and G1 assets as BlinkMimic; resumable by file."""

import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASET = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main"


def list_files(folder):
    with urllib.request.urlopen(f"{API}/{folder}", timeout=60) as response:
        return [entry["path"] for entry in json.load(response) if entry["type"] == "file"]


def download(remote, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(remote, flush=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(f"{DATASET}/resolve/main/{remote}", timeout=120) as response:
        with temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    temporary.replace(destination)


def main():
    for folder, local in [("g1", "lafan1_g1"), ("robot_description/g1/meshes", "g1_description/meshes")]:
        for remote in list_files(folder):
            download(remote, ROOT / "data" / local / Path(remote).name)
    download("robot_description/g1/g1_29dof_rev_1_0.urdf", ROOT / "data/g1_description/g1_29dof_rev_1_0.urdf")
    subprocess.run([sys.executable, str(ROOT / "tools/urdf_to_mjcf.py")], check=True)


if __name__ == "__main__":
    main()
