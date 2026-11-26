"""Fetch, verify and safely extract the Zenodo "Ferrographic Image Dataset" (Li Zhenlin 2026, CC-BY-4.0,
doi 10.5281/zenodo.22997504) into data/real/zenodo_ferro/{raw,extracted}/.

Usage (run with -I; nothing from the archive is ever executed):
    .venv/Scripts/python.exe -I scripts/open_ferro_fetch.py [--root "<project root>"] [--list]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://zenodo.org/api/records/22997504/files/Ferrographic-%20Image-%20Dataset.zip/content"
MD5 = "46a5def7fafc4ded2d0e2cb7bc7b893e"  # from https://zenodo.org/api/records/22997504 (files[0].checksum)
ZIP_NAME = "ferrographic_image_dataset.zip"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """(member, relative path) for regular files; raises on absolute paths, '..', drive letters or symlinks."""
    out = []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        rel = PurePosixPath(name)
        if name.startswith("/") or ":" in name or ".." in rel.parts:
            raise ValueError(f"unsafe member path {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"symlink member {info.filename!r}")
        if not info.is_dir():
            out.append((info, rel))
    return out


def extract(zip_path: Path, dest: Path) -> int:
    """Extract regular files member by member (paths validated first); returns the number of files."""
    with zipfile.ZipFile(zip_path) as zf:
        members = safe_members(zf)
        root = dest.resolve()
        for info, rel in members:
            target = (dest / Path(*rel.parts)).resolve()
            if root not in target.parents:
                raise ValueError(f"member escapes the destination: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return len(members)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--list", action="store_true", help="only list the archive members")
    a = ap.parse_args()
    base = Path(a.root) / "data" / "real" / "zenodo_ferro"
    raw, ext = base / "raw", base / "extracted"
    raw.mkdir(parents=True, exist_ok=True)
    zp = raw / ZIP_NAME
    if not zp.exists():
        tmp = zp.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(zp)
    got = md5sum(zp)
    if got != MD5:
        raise SystemExit(f"md5 mismatch: {got} != {MD5}")
    if a.list:
        with zipfile.ZipFile(zp) as zf:
            for info, rel in safe_members(zf):
                print(str(rel).encode("ascii", "replace").decode(), info.file_size)
        return
    n = extract(zp, ext)
    print(f"md5 ok; extracted {n} files to {ext}")


if __name__ == "__main__":
    main()
