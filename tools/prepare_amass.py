"""Safely unpack completed official SMPL-X neutral downloads and record provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

NEUTRAL_DATASETS = [
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "CNRS",
    "DFaust",
    "EKUT",
    "EyesJapanDataset",
    "GRAB",
    "HDM05",
    "HumanEva",
    "KIT",
    "MoSh",
    "PosePrior",
    "SFU",
    "SOMA",
    "SSM",
    "TCDHands",
    "TotalCapture",
    "Transitions",
    "WEIZMANN",
    "MOYO",
    "LARa",
]


def prepare_archives(config_path, archive_dir):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    output = (config_path.parent / settings["input_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prepared = []
    for name in NEUTRAL_DATASETS:
        archive = Path(archive_dir) / (name + ".tar.bz2")
        if not archive.is_file():
            manifest.setdefault(name, dict(status="not_downloaded"))
            continue
        stat = archive.stat()
        old = manifest.get(name, {})
        if (
            old.get("status") == "extracted"
            and old.get("size") == stat.st_size
            and old.get("mtime_ns") == stat.st_mtime_ns
        ):
            continue
        try:
            count = 0
            print(f"Extracting {name}...", flush=True)
            with tarfile.open(archive, mode="r|bz2") as source:
                for member in source:
                    if not member.isfile() or not member.name.endswith(".npz"):
                        continue
                    target = (output / member.name).resolve()
                    if not target.is_relative_to(output):
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".npz.part")
                    with source.extractfile(member) as data, temporary.open("wb") as destination:
                        shutil.copyfileobj(data, destination)
                    temporary.replace(target)
                    count += 1
            if not count:
                raise ValueError("No NPZ motion/model files in archive")
            with archive.open("rb") as data:
                checksum = hashlib.file_digest(data, "sha256").hexdigest()
            manifest[name] = dict(
                status="extracted",
                archive=str(archive.resolve()),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=checksum,
                npz_files=count,
            )
            prepared.append(name)
            print(f"{name}: {count} NPZ files", flush=True)
        except Exception as error:
            manifest[name] = dict(status="failed", error=str(error), archive=str(archive.resolve()))
            print(f"FAILED {name}: {error}", flush=True)
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_amass.json")
    parser.add_argument(
        "--archives", required=True, help="Folder containing completed official .tar.bz2 downloads"
    )
    args = parser.parse_args()
    manifest, _ = prepare_archives(args.config, args.archives)
    print(
        "Extracted",
        sum(v["status"] == "extracted" for v in manifest.values()),
        "of",
        len(NEUTRAL_DATASETS),
        "neutral subsets",
    )
