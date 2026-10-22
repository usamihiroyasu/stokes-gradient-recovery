"""Install hash-pinned IEEE Access template files for a local manuscript build."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip"
ARCHIVE_SHA256 = "60c7efc9db8ac9e8bdb31c550ad4e03cb6f258a878ececc0bc690b6203e45a67"
PREFIX = "ACCESS_latex_template_20260513/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded template ZIP")
    args = parser.parse_args()
    if args.archive:
        payload = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            payload = response.read(20 * 1024 * 1024 + 1)
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("Template archive hash mismatch; no files installed")
    hashes = json.loads((ROOT / "scripts/template_files.json").read_text())
    pending = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name, expected in hashes.items():
            if Path(name).name != name:
                raise SystemExit("Invalid template file name")
            data = archive.read(PREFIX + name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f"Template file hash mismatch: {name}")
            target = ROOT / "latex" / name
            if target.exists() and target.read_bytes() != data:
                raise SystemExit(f"Existing local file differs: {name}")
            pending[target] = data
    for target, data in pending.items():
        target.write_bytes(data)
    print(f"Installed {len(pending)} verified template files in latex/")


if __name__ == "__main__":
    main()
