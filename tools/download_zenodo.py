"""
tools/download_zenodo.py

Automated download and extraction of the Lipi dataset from Zenodo.
Usage:
  python tools/download_zenodo.py --record 14123456
  python tools/download_zenodo.py --url https://zenodo.org/records/14123456/files/lipi_dataset_v1.0.zip?download=1
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

def download_with_progress(url: str, output_path: Path):
    print(f"Downloading from Zenodo:\n  {url}")
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb:.1f} / {total_mb:.1f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print("\n[✓] Download complete.")

def main():
    parser = argparse.ArgumentParser(description="Download Lipi dataset from Zenodo")
    parser.add_argument("--record", help="Zenodo record ID (e.g. 14123456)")
    parser.add_argument("--url", help="Direct Zenodo file download URL")
    parser.add_argument("--filename", default="lipi_dataset_v1.0.zip", help="Zip file name on Zenodo")
    args = parser.parse_args()

    if args.url:
        url = args.url
    elif args.record:
        url = f"https://zenodo.org/records/{args.record}/files/{args.filename}?download=1"
    else:
        # Default placeholder or prompt
        raise SystemExit("Error: provide either --record <id> or --url <zenodo-direct-url>")

    zip_target = REPO / "lipi_zenodo_download.zip"
    try:
        download_with_progress(url, zip_target)
        print(f"Extracting into {DATA_DIR} ...")
        with zipfile.ZipFile(zip_target, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"[✓] Extracted successfully. data/canonical is ready!")
    finally:
        if zip_target.exists():
            zip_target.unlink()

if __name__ == "__main__":
    main()
