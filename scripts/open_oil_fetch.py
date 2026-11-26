"""Fetch and verify the open oil-debris sources of scripts/open_oil_analysis.py into data/real/<name>/raw/.

Zenodo 10.5281/zenodo.17287193 (CC-BY-4.0), Harvard Dataverse 10.7910/DVN/SO9SXL (CC0 1.0) and NASA NTRS reports
(US government works, NTRS "public use permitted"). Files are only downloaded and checksummed: nothing is executed or
extracted (the xlsx files are later read as data with openpyxl). Existing files with the right checksum are kept.
Sources, licences and citations: docs/open_data_oil.md. data/ is gitignored and must never be redistributed.

Usage: .venv/Scripts/python.exe -I scripts/open_oil_fetch.py [--root "<project root>"] [--skip-dataverse]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

NTRS = "https://ntrs.nasa.gov/api/citations/{}"
# md5 of the NTRS PDFs the NASA tables in wearpg.open_oil were transcribed from (NTRS publishes no checksum)
NTRS_PDF_MD5 = {
    "20000120403": "64155a12e65eabc2fb95fd4d4540f08a",  # NASA/TM-2000-210371 (checked, not used)
    "20020009087": "26362b8bd1e16d8d2b2018b4ac8d43c6",  # NASA/TM-2001-210936 Tables 2-4 (cross-check)
    "20020070653": "9d0a5028884ce9537cb486ae40cb9fa7",  # NASA/TM-2002-211126 (checked, not used)
    "20030056593": "b79c3978634fd9fa69a8837c0c3b4cb3",  # NASA/TM-2003-211307 Tables 3.7, 3.9-3.24 (primary)
    "20040082336": "a54a8d74143be6aa5066150960cfca27",  # NASA/TM-2004-212883 (checked, not used)
    "20090039501": "ff86b052771b435fe88bd069ae453cc7",  # NASA/TM-2009-215667 (debris only plotted: not used)
}
# (directory under data/real, file name, url, expected md5 or None, md5 published by the repository?)
FILES = [
    ("zenodo_wear_debris", "zenodo_record_17287193.json", "https://zenodo.org/api/records/17287193", None, False),
    ("zenodo_wear_debris", "wear_derbis.xlsx",
     "https://zenodo.org/api/records/17287193/files/wear%20derbis.xlsx/content", "565f3ca72da39b3489e66e07c0994c94", True),
    ("dataverse_lubricating_oil", "dataverse_SO9SXL.json",
     "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/SO9SXL", None, False),
    ("dataverse_lubricating_oil", "Lubricating_Oil_Parameters_1.xlsx",
     "https://dataverse.harvard.edu/api/access/datafile/6402623", "8780a311b83d1b33c11877adab0bb9ec", True),
    *[("nasa_spur_gear", f"ntrs_{i}.json", NTRS.format(i), None, False) for i in NTRS_PDF_MD5],
    *[("nasa_spur_gear", f"{i}.pdf", NTRS.format(i) + f"/downloads/{i}.pdf", m, False) for i, m in NTRS_PDF_MD5.items()],
]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, md5: str | None, strict: bool) -> str:
    """Download url to dest unless dest exists with the expected md5; returns 'kept' | 'downloaded' | 'md5 differs'.
    A mismatch against a repository-published md5 (strict) raises and leaves only a .part file behind."""
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        return "kept"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wearpg-open-data-fetch"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    got = md5sum(tmp)
    if md5 is not None and got != md5 and strict:
        raise SystemExit(f"md5 mismatch for {url}: {got} != {md5} (left {tmp})")
    tmp.replace(dest)
    return "downloaded" if md5 is None or got == md5 else "md5 differs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--skip-dataverse", action="store_true", help="skip the 32 MB Dataverse workbook")
    a = ap.parse_args()
    for name, fname, url, md5, strict in FILES:
        if a.skip_dataverse and name == "dataverse_lubricating_oil":
            continue
        status = fetch(url, Path(a.root) / "data" / "real" / name / "raw" / fname, md5, strict)
        note = " (NTRS re-rendered the PDF? check the transcribed pages)" if status == "md5 differs" else ""
        print(f"{status:11s} {name}/raw/{fname}{note}")


if __name__ == "__main__":
    main()
