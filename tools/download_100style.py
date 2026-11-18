"""Download the official raw 100STYLE BVH archive and frame cuts (CC BY 4.0)."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ARCHIVE_URL = "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
ARCHIVE_MD5 = "3cf627852fd8192024c04a8d0ef49583"
CUTS_URL = (
    "https://drive.usercontent.google.com/download?id=1d0VM8k4UjA4dDmaviuZMjAf-WQNLUwnZ&export=download"
)


def download(url, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...", flush=True)
    downloaded = 0
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded % (64 * 1024 * 1024) == 0:
                print(f"  {downloaded // 1024**2} MiB", flush=True)
    temporary.replace(destination)


def prepare_data(config_path):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = (config_path.parent / settings["input_dir"]).resolve()
    cuts_path = (config_path.parent / settings["frame_cuts"]).resolve()
    archive_path = cuts_path.parent / "archives/100STYLE.zip"
    download(ARCHIVE_URL, archive_path)
    with archive_path.open("rb") as file:
        checksum = hashlib.file_digest(file, "md5").hexdigest()
    if checksum != ARCHIVE_MD5:
        raise ValueError(f"Archive checksum mismatch: {archive_path}; expected {ARCHIVE_MD5}, got {checksum}")
    download(CUTS_URL, cuts_path)
    if not cuts_path.read_text(encoding="utf-8-sig").startswith("STYLE_NAME,"):
        raise ValueError(f"Expected official Frame_Cuts.csv, got {cuts_path}")
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name = Path(entry.filename.replace("\\", "/")).name
            if not name.lower().endswith(".bvh") or name.startswith("._"):
                continue
            # Flatten known dataset basenames; never extract untrusted archive paths.
            destination = input_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != entry.file_size:
                temporary = destination.with_suffix(".bvh.part")
                with archive.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(destination)
            count += 1
    print(f"Verified archive; prepared {count} BVH clips in {input_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_100style.json")
    prepare_data(parser.parse_args().config)
