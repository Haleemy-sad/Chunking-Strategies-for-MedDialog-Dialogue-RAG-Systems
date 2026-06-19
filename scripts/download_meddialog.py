"""
Utility to help committee members obtain the MedDialog dataset file.

Usage:
  - To download from a URL:
      python scripts/download_meddialog.py --url <DIRECT_DOWNLOAD_URL>

  - To only verify and move a local file into place:
      python scripts/download_meddialog.py --local /path/to/meddialog.json

The script will save the file to `data/downloaded/meddialog.json` and will not add it to git.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


OUT_PATH = Path(__file__).parents[1] / "data" / "downloaded" / "meddialog.json"


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    if requests is None:
        # fallback to urllib
        from urllib.request import urlopen

        with urlopen(url) as response, open(dest, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    else:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


def move_local(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def human_size(path: Path) -> str:
    try:
        s = path.stat().st_size
    except Exception:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if s < 1024.0:
            return f"{s:.2f} {unit}"
        s /= 1024.0
    return f"{s:.2f} TB"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download or place MedDialog dataset into data/downloaded/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Direct download URL for meddialog.json")
    group.add_argument("--local", help="Path to an existing local meddialog.json to copy into the project")
    parser.add_argument("--verify-size", type=int, help="Minimum expected file size in bytes (optional)")

    args = parser.parse_args(argv)

    try:
        if args.url:
            print("This may take a while depending on your connection.")
            download_url(args.url, OUT_PATH)
        else:
            src = Path(args.local).expanduser()
            move_local(src, OUT_PATH)

        print(f"Saved dataset to: {OUT_PATH}")
        print(f"File size: {human_size(OUT_PATH)}")

        if args.verify_size:
            actual = OUT_PATH.stat().st_size
            if actual < args.verify_size:
                print(f"Warning: downloaded file ({actual} bytes) is smaller than expected ({args.verify_size} bytes)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
