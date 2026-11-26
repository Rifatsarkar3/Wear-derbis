"""Digitise the published experiments that pair oil debris with measured wear (docs/paired_data_candidates.md) into
data/real/paired/<id>/paired.csv (schema: wearpg.paired) + study.json, and write every second reading to
outputs/open_data/paired/digitisation_check.csv. No statistic is computed here.

Order of preference: born-digital text and tables (typed here with the page), vector paths (PyMuPDF), pixels. Numbers
that were read twice (another figure of the same quantity, another table, or another procedure) are in `debris_2` /
`wear_2`; the relative disagreement is the digitisation error. PDFs are untrusted: checksums are asserted, nothing in
them is executed, run with -I. Re-running overwrites only the files named above.
Usage: .venv/Scripts/python.exe -I scripts/paired_digitise.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import match_template
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd

from wearpg.paired import validate  # noqa: E402

DATA, OUT = ROOT / "data" / "real" / "paired", ROOT / "outputs" / "open_data" / "paired"
CHECK: list[dict] = []  # one row per twice-read number
DATE = "2026-11-24"


# ------------------------------------------------------------------------------------------------------- helpers
def open_pdf(sid: str) -> fitz.Document:
    src = json.loads((DATA / sid / "raw" / "source.json").read_text(encoding="utf-8"))
    f = DATA / sid / src["file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != src["sha256"]:
        raise ValueError(f"{f}: checksum differs from source.json")
    return fitz.open(f)


def table(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.strip()), sep="|")


def lin(p0, v0, p1, v1):
    return lambda p: v0 + (np.asarray(p, float) - p0) * (v1 - v0) / (p1 - p0)


def log_axis(p0, e0, p1, e1):
    f = lin(p0, e0, p1, e1)
    return lambda p: 10 ** f(p)


def quad_centres(page, box) -> list[tuple[float, float]]:
    """Centres of stroked diamonds (one quad item) inside box = (x0, y0, x1, y1)."""
    out = []
    for g in page.get_drawings():
        if g["type"] == "s" and len(g["items"]) == 1 and g["items"][0][0] == "qu":
            q = g["items"][0][1]
            c = [(q.ul.x + q.ur.x + q.ll.x + q.lr.x) / 4, (q.ul.y + q.ur.y + q.ll.y + q.lr.y) / 4]
            if box[0] < c[0] < box[2] and box[1] < c[1] < box[3]:
                out.append(tuple(c))
    return sorted(out)


def polyline(page, box, colour=(0, 0, 0), min_width=0.55, max_width=1.5) -> list[tuple[float, float]]:
    """Distinct vertices of stroked lines of one colour and width range inside box, sorted by x."""
    pts = set()
    for g in page.get_drawings():
        if g["type"] != "s" or not min_width <= (g["width"] or 0) <= max_width:
            continue
        if tuple(round(c, 1) for c in g["color"]) != tuple(round(c, 1) for c in colour):
            continue
        for it in g["items"]:
            if it[0] == "l":
                for p in it[1:3]:
                    if box[0] < p.x < box[2] and box[1] < p.y < box[3]:
                        pts.add((round(p.x, 2), round(p.y, 2)))
    return sorted(pts)


def markers(page, box, colour, size=(1.5, 6.0)) -> list[tuple[float, float]]:
    out = []
    for g in page.get_drawings():
        if "f" not in g["type"] or g.get("fill") is None:
            continue
        r = g["rect"]
        if tuple(round(c, 1) for c in g["fill"]) == colour and size[0] < r.width < size[1] and size[0] < r.height < size[1]:
            c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            if box[0] < c[0] < box[2] and box[1] < c[1] < box[3]:
                out.append(c)
    return sorted(out)


def raster(page, box, dpi):
    pix = page.get_pixmap(dpi=dpi, clip=fitz.Rect(*box))
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    to_pt = lambda x, y: (box[0] + (x + 0.5) * 72 / dpi, box[1] + (y + 0.5) * 72 / dpi)
    return a, to_pt


def dark_blobs(page, box, dpi=400, opening_px=11) -> list[tuple[float, float]]:
    """Second reading of vector plots from pixels: dark marker cores after an opening that removes thin lines."""
    a, to_pt = raster(page, box, dpi)
    core = ndi.binary_opening(a.sum(2) < 250, structure=np.ones((opening_px, opening_px)))
    lab, n = ndi.label(core)
    return sorted(to_pt(x, y) for y, x in ndi.center_of_mass(core, lab, range(1, n + 1)))


def image(doc, xref) -> np.ndarray:
    pix = fitz.Pixmap(doc, xref)
    if pix.colorspace is None or pix.n - pix.alpha not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return a.astype(int)


def colour_mask(a, col, tol):
    return (np.abs(a[:, :, :3] - np.array(col)).sum(2) < tol).astype(float)


def check(study, what, first, second, how):
    first, second = np.atleast_1d(np.asarray(first, float)), np.atleast_1d(np.asarray(second, float))
    ok = np.isfinite(first) & np.isfinite(second)
    d = second[ok] - first[ok]
    pos = (first[ok] > 0) & (second[ok] > 0)
    lg = np.log10(second[ok][pos] / first[ok][pos])
    CHECK.append({"study": study, "quantity": what, "n_read_twice": int(ok.sum()), "n_first_only": int((~ok).sum()),
                  "rms_diff": float(np.sqrt((d**2).mean())) if ok.any() else np.nan,
                  "max_abs_diff": float(np.abs(d).max()) if ok.any() else np.nan,
                  "median_abs_log10": float(np.median(np.abs(lg))) if pos.any() else np.nan,
                  "rms_log10": float(np.sqrt((lg**2).mean())) if pos.any() else np.nan,
                  "max_abs_log10": float(np.abs(lg).max()) if pos.any() else np.nan, "second_reading": how})


def rel(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(second / first - 1)


def save(sid: str, df: pd.DataFrame, sets: dict, **meta):
    src = json.loads((DATA / sid / "raw" / "source.json").read_text(encoding="utf-8"))
    df = df.copy()
    df.insert(0, "study", sid)
    validate(df)
    df.to_csv(DATA / sid / "paired.csv", index=False, lineterminator="\n", float_format="%.6g")
    spec = {"id": sid, "citation": src["citation"], "doi": src["doi"], "url": src["landing"], "rights": src["rights"],
            "spec_date": DATE, "added_after_first_run": False, **meta, "sets": sets}
    (DATA / sid / "study.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{sid}: {len(df)} rows")


# --------------------------------------------------------------------------------------------- FEUP (Martins 2008)
# Paper B (PDF pp. 129-131) and Paper E (PDF pp. 172-173): data labels of mass loss, CPUC, ISUC; pinion / wheel split Paper E
FEUP_MP = """
unit|stage|order|adi|ester|cpuc|isuc|mass|pinion|wheel|cycles|hertz|page|item
CM_mp|K3+K7|1|0|0|277|5.6e4|27|16|11|1.53|1180|129-131|Figs 2-4, Table 6 (Paper B); Fig. 7, Table 7 (Paper E)
CM_mp|K9-1|2|0|0|480|1.7e5|45|30|16|1.44|1505|129-131|Figs 2-4, Table 6 (Paper B); Fig. 7, Table 7 (Paper E)
CM_mp|K9-2|3|0|0|340|8.9e4|42|||1.44|1505|129-130|Figs 2-4 (Paper B)
CE_mp|K3+K7|1|0|1|396|1.2e5|31|17|14|1.53|1180|129-131|Figs 2-4, Table 6 (Paper B); Fig. 7, Table 7 (Paper E)
CE_mp|K9-1|2|0|1|419|1.2e5|58|36|22|1.44|1505|129-131|Figs 2-4, Table 6 (Paper B); Fig. 7, Table 7 (Paper E)
CE_mp|K9-2|3|0|1|364|9.5e4|38|||1.44|1505|129-130|Figs 2-4 (Paper B)
AM_mp|K3+K5|1|1|0|237|4.3e4|35|19|16|1.53|760|173|Figs 8-9, Table 7 (Paper E)
AM_mp|K7|2|1|0|756|3.9e5|155|69|86|1.44|1046|173|Figs 8-9, Table 7 (Paper E)
AE_mp|K3+K5|1|1|1|255|4.5e4|20|13|7|1.53|760|173|Figs 8-9, Table 7 (Paper E)
AE_mp|K7|2|1|1|649|2.7e5|132|61|72|1.44|1046|173|Figs 8-9, Table 7 (Paper E)
"""
# Table 6 (PDF p. 131, d = 0.1) and text of PDF p. 174: readings on the diluted sample, the check of the split below
FEUP_DL = {("CM_mp", "K3+K7"): (23.9, 3.8), ("CM_mp", "K9-1"): (42, 6), ("CE_mp", "K3+K7"): (34.8, 4.8),
           ("CE_mp", "K9-1"): (35.8, 6.1), ("AM_mp", "K7"): (63.5, None), ("AE_mp", "K7"): (53.4, None)}
# Paper F (PDF pp. 183-191): mass loss in g -> mg; loaded wheel cycles (1e6); CM last stage stopped after 133 of 300 min
FEUP_PF = """
unit|adi|ester|coated|mass|cycles|hertz|page
CM_pl|0|0|0|923|4.90|1781|187
CE_pl|0|1|0|105|5.40|1781|187
MM_pl|0|0|1|106|2.70|1781|189
ME_pl|0|1|1|133|5.40|1781|189
AM_pl|1|0|0|150|3.60|1230|191
AE_pl|1|1|0|73|3.60|1230|191
"""
# Paper I Table 7 (PDF p. 242); loaded wheel cycles 5.76 - 1.44; 1.50 GPa thesis p. 19 (PDF p. 43)
FEUP_PI = """
unit|ester|cpuc|mass
M1_pI|0|364|55
E1_pI|1|8780|525
E2_pI|1|167|7
E3_pI|1|650|72
"""
# Paper H Tables 2-3 (PDF p. 217); viscosity at 75 C (cSt) PDF p. 218
FEUP_FB = """
unit|oil|visc|wsd1|track|DL|DS|cpuc|isuc
M0_1|M0|31.47|376|324|4.9|2.7|7.6|16.7
M0_2|M0|31.47|396|420|3.5|2.4|5.9|6.49
E1_1|E1|27.97|404|360|6.6|1.6|8.2|41.1
E1_2|E1|27.97|392|396|1.3|0.8|2.1|1.1
E2_1|E2|32.57|364|276|3.7|2.7|6.4|6.4
E2_2|E2|32.57|342|252|2.3|0.6|2.9|4.9
"""


def split_dl_ds(cpuc, isuc):
    """DL/d and DS/d from CPUC = (DL + DS)/d and ISUC = (DL^2 - DS^2)/d^2; DS clipped at 0 (ISUC has two digits)."""
    r = np.minimum(isuc / cpuc, cpuc)
    return (cpuc + r) / 2, (cpuc - r) / 2


def feup():
    doc = open_pdf("feup_martins2008")
    rows = []
    mp = table(FEUP_MP)
    dl, ds = split_dl_ds(mp.cpuc, mp.isuc)
    printed = [FEUP_DL.get((u, s), (None, None))[0] for u, s in zip(mp.unit, mp.stage)]
    check("feup_martins2008", "DL/d derived from CPUC and ISUC against printed DL (Table 6, p. 174)",
          dl, [10 * v if v else np.nan for v in printed], "printed DL readings / dilution 0.1")
    check("feup_martins2008", "mass loss against pinion + wheel (Paper E Figs 7-8)", mp.mass, mp.pinion + mp.wheel,
          "sum of the two printed parts")
    for r, a, b in zip(mp.itertuples(index=False), dl, ds):
        rows.append({"set": "fzg", "unit": r.unit, "order": r.order, "mode": "interval", "stage": r.stage, "adi": r.adi,
                     "ester": r.ester, "cyc": r.cycles, "hertz": r.hertz, "debris": r.cpuc, "wear": r.mass, "last": r.cpuc,
                     "isuc": r.isuc, "wear_2": (r.pinion + r.wheel), "debris_2": np.nan, "DL": a, "DS": b,
                     "page": r.page, "item": r.item, "extraction": "text (figure data labels)", "coated": 0, "note":
                     "fresh oil after every weighing; the first row of a gear pair includes the K3 running-in in both columns"})
    # Paper F: CPUC (left) and ISUC (right) plots, one page per material; y axes: (box, zero, top, top value)
    cfg = {187: {"C": ((126, 367, 280.5, 482), 481.35, 367.70, 8000), "I": ((353, 362, 490, 491.5), 490.93, 363.08, 4e7),
                 "units": ("CM_pl", "CE_pl")},
           189: {"C": ((128, 351, 277.5, 464), 463.13, 351.44, 800), "I": ((353, 347, 489.5, 474.5), 473.77, 347.54, 3.5e5),
                 "units": ("MM_pl", "ME_pl")},
           191: {"C": ((133, 358, 277.5, 471), 470.43, 358.74, 3000), "I": ((353, 350.5, 490, 478.5), 477.95, 351.16, 4.5e6),
                 "units": ("AM_pl", "AE_pl")}}
    series: dict[str, dict[str, list[float]]] = {}
    for p, c in cfg.items():
        page = doc[p - 1]
        for key in "CI":
            box, y0, y1, top = c[key]
            val = lin(y0, 0.0, y1, top)
            legend = {187: 381, 189: 365, 191: 375}[p] if key == "C" else {187: 378, 189: 363, 191: 366}[p]
            legend_x = box[0] + 75
            keep = lambda pts: [q for q in pts if not (q[1] < legend and q[0] < legend_x)]
            mineral = keep(polyline(page, box, min_width=1.1, max_width=1.3))
            ester = keep(quad_centres(page, box))
            series.setdefault(c["units"][0], {})[key] = [float(val(y)) for _, y in mineral]
            series.setdefault(c["units"][1], {})[key] = [float(val(y)) for _, y in ester]
            blobs = np.array(dark_blobs(page, box))
            second = []
            for q in mineral + ester:
                d = np.hypot(*(blobs - np.array(q)).T)
                second.append(float(val(blobs[d.argmin()][1])) if d.min() < 3 else np.nan)
            check("feup_martins2008", f"{'CPUC' if key == 'C' else 'ISUC'} power loss Paper F, PDF p. {p}",
                  [float(val(y)) for _, y in mineral + ester], second, "marker centroids in a 400 dpi rendering")
    # Paper D Fig. 3 (PDF p. 155): the two ester tests on a finer CPUC axis (0-800)
    page = doc[154]
    val = lin(609.94, 0.0, 518.58, 800.0)
    unc = [float(val(y)) for _, y in polyline(page, (101, 520, 235, 609), colour=(0.1, 0.1, 0.1), min_width=1.0, max_width=1.1)]
    coa = [float(val(y)) for _, y in quad_centres(page, (101, 520, 235, 609))]
    check("feup_martins2008", "CPUC ester tests: Paper F (p. 187, 189) against Paper D Fig. 3 (p. 155)",
          series["CE_pl"]["C"] + series["ME_pl"]["C"], unc + coa, "same tests in another figure with a finer axis")
    series["CE_pl"]["C2"], series["ME_pl"]["C2"] = unc, coa
    for r in table(FEUP_PF).itertuples(index=False):
        s = series[r.unit]
        cp = s["C2"] if r.unit == "CE_pl" else s["C"]  # p. 187 has an 8000 axis (1 pt = 70 units): finer figure first
        other = s["C"] if r.unit == "CE_pl" else s.get("C2", [np.nan])
        q = 80 / 1500
        deb = cp[-1] + q * sum(cp[:-1])
        a, b = split_dl_ds(np.array(cp[-1]), np.array(s["I"][-1]))
        rows.append({"set": "fzg", "unit": r.unit, "order": 1, "mode": "cumulative", "stage": "power loss F", "adi": r.adi,
                     "ester": r.ester, "cyc": r.cycles, "hertz": r.hertz, "debris": deb, "wear": r.mass, "last": cp[-1],
                     "isuc": s["I"][-1], "wear_2": np.nan, "debris_2": other[-1] + q * sum(other[:-1]), "DL": float(a),
                     "DS": float(b), "page": str(r.page) + (", 155" if r.unit in ("CE_pl", "ME_pl") else ""),
                     "item": "Paper F Figs 3-4 / 6-7 / 9-10", "extraction": "vector (CPUC, ISUC); text (mass)",
                     "coated": r.coated, "note": f"samples CPUC = {', '.join(f'{v:.0f}' for v in cp)}; q = 80/1500 per earlier sample"})
    # Paper I Figs 4-5 (PDF pp. 238-239), log axes; final CPUC from Table 7
    cp_ax, is_ax = log_axis(590.83, 2, 558.0, 3), log_axis(191.57, 4, 172.76, 5)
    col = {"M1_pI": (0, 0, 0), "E1_pI": (1, 0, 0), "E2_pI": (0, 0, 1), "E3_pI": (0, 0.5, 0)}
    firsts, seconds = [], []
    for r in table(FEUP_PI).itertuples(index=False):
        cps = [float(cp_ax(y)) for _, y in polyline(doc[237], (385, 526, 489, 623), col[r.unit], 0.6, 0.66)
               if not (365 < _ < 404 and y < 557)]
        iss = [float(is_ax(y)) for _, y in polyline(doc[238], (158.5, 116.5, 252, 210), col[r.unit], 0.6, 0.66)]
        firsts.append(r.cpuc)
        seconds.append(cps[-1])
        q = 40 / 1500
        a, b = split_dl_ds(np.array(float(r.cpuc)), np.array(iss[-1]))
        rows.append({"set": "fzg", "unit": r.unit, "order": 1, "mode": "cumulative", "stage": "power loss I", "adi": 0,
                     "ester": r.ester, "cyc": 4.32, "hertz": 1500, "debris": r.cpuc + q * sum(cps[:-1]), "wear": r.mass,
                     "last": r.cpuc, "isuc": iss[-1], "wear_2": 77 if r.unit == "E3_pI" else np.nan,
                     "debris_2": cps[-1] + q * sum(cps[:-1]), "DL": float(a), "DS": float(b), "page": "238-239, 242",
                     "item": "Paper I Table 7, Figs 4-5", "extraction": "text (Table 7); vector (earlier samples, ISUC)",
                     "coated": 0, "note": f"samples CPUC = {', '.join(f'{v:.0f}' for v in cps)}; q = 40/1500, replaced by fresh oil"
                     + ("; text p. 214 of the thesis says 77 mg (wear_2)" if r.unit == "E3_pI" else "")
                     + ("; scuffing failure in the last stage" if r.unit == "E1_pI" else "")})
    check("feup_martins2008", "final CPUC Paper I: Table 7 against vector Fig. 4", firsts, seconds, "log axis, vector")
    f = pd.DataFrame(rows)
    out = pd.DataFrame({"set": f.set, "unit": f.unit, "row": [f"{u}:{s}" for u, s in zip(f.unit, f.stage)], "order": f.order,
                        "mode": f["mode"], "rig": "FZG gear test rig, type C gears, dip lubrication, 1.5 l",
                        "material_pair": np.where(f.adi == 1, "ADI gear pair", np.where(f.coated == 1,
                                                  "20MnCr5 carburised, MoS2/Ti coated", "20MnCr5 carburised gear pair")),
                        "lubricant": np.where(f.ester == 1, "biodegradable ester gear oil", "mineral gear oil"),
                        "cond_cycles_M": f.cyc, "cond_hertz_MPa": f.hertz, "cond_ester": f.ester, "cond_adi": f.adi,
                        "debris": f.debris, "debris_metric": "direct-reading ferrometer CPUC = (DL + DS)/d, run inventory",
                        "debris_unit": "index", "debris_type": "index", "wear": f.wear,
                        "wear_metric": "weighed gear pair mass loss (pinion + wheel)", "wear_unit": "mg",
                        "resolved": 2 * f.DL + f.DS, "class_DL": f.DL, "class_DS": f.DS, "class_isuc": f.isuc,
                        "class_scuffed": f.unit.isin(["CM_pl", "E1_pI"]).astype(int),
                        "alt_debris_last_sample": f["last"], "alt_debris_isuc": f.isuc, "debris_2": f.debris_2,
                        "wear_2": f.wear_2, "debris_err": rel(f.debris, f.debris_2), "wear_err": 1.0 / f.wear,
                        "page": f.page, "item": f.item, "extraction": f.extraction, "note": f.note})
    fb = table(FEUP_FB)
    R, P = 6.35, 40 * 9.80665 * 0.408248
    a_h = (3 * P * (R / 2) / (4 * 210e3 / (2 * (1 - 0.3**2)))) ** (1 / 3)  # mm, two equal steel balls, N and MPa
    vol = lambda d, dh: 3 * np.pi * ((d / 1000) ** 4 - dh**4) / (64 * R)
    out2 = pd.DataFrame({"set": "fourball", "unit": fb.unit, "row": fb.unit, "order": 1, "mode": "interval",
                         "rig": "four-ball machine ASTM D4172 (40 kgf, 1200 rpm, 60 min, 75 C)",
                         "material_pair": "bearing steel balls 12.7 mm", "lubricant": fb.oil, "cond_visc75_cSt": fb.visc,
                         "debris": fb.cpuc, "debris_metric": "direct-reading ferrometer CPUC of the whole oil plus rinse",
                         "debris_unit": "index", "debris_type": "index", "wear": vol(fb.wsd1, 2 * a_h),
                         "wear_metric": "wear volume of the three lower balls from the mean scar diameter minus the Hertz elastic cap",
                         "wear_unit": "mm3", "resolved": 2 * fb.DL + fb.DS, "class_DL": fb.DL, "class_DS": fb.DS,
                         "class_isuc": fb.isuc, "alt_wear_no_elastic": vol(fb.wsd1, 0.0), "alt_wear_scar_um": fb.wsd1,
                         "debris_2": fb.DL + fb.DS, "wear_2": np.nan, "debris_err": rel(fb.cpuc, fb.DL + fb.DS),
                         "wear_err": 4 * 10 / fb.wsd1, "page": "216-218", "item": "Paper H Tables 2-3",
                         "extraction": "text (tables)", "note": f"Hertz diameter {2 * a_h:.4f} mm; scar resolution 0.01 mm"})
    check("feup_martins2008", "four-ball CPUC against DL + DS (Table 3)", fb.cpuc, fb.DL + fb.DS, "sum of printed readings")
    save("feup_martins2008", pd.concat([out, out2], ignore_index=True), {
        "fzg": {"conditions": [["cond_cycles_M", "log10"], ["cond_hertz_MPa", "log10"], ["cond_ester", "id"], ["cond_adi", "id"]],
                "resolved_def": "size-resolved, not mechanism: 2 DL/d + DS/d (area x nominal size 5 um : 2.5 um)",
                "group": "cond_adi",
                "sensitivity": {"without_scuffed": {"drop": ["CM_pl", "E1_pI"]}, "rows_not_units": {"unit": "row"},
                                "interval_rows_only": {"mode": "interval"}, "cumulative_rows_only": {"mode": "cumulative"},
                                "last_sample_no_q": {"debris": "alt_debris_last_sample"}, "isuc": {"debris": "alt_debris_isuc"},
                                "wear_second_reading": {"wear": "wear_2", "fallback": True},
                                "debris_second_reading": {"debris": "debris_2", "fallback": True}}},
        "fourball": {"conditions": [["cond_visc75_cSt", "log10"]],
                     "resolved_def": "size-resolved: 2 DL + DS", "sensitivity": {"no_elastic_correction": {"wear": "alt_wear_no_elastic"}}}})


# ------------------------------------------------------------------------------------------------------ Raadnui 2015
def raadnui():
    sid = "phm_raadnui2015"
    doc = open_pdf(sid)
    t = np.array([3, 10, 15, 20, 30.0])
    mean = np.array([0.8333, 0.9333, 2.4670, 5.4000, 10.830])  # Table 1, exponential mean of three fractions
    total = 3 * mean
    fig10 = sorted(markers(doc[3], (354, 480, 573, 616), (1.0, 0.0, 0.0)))
    vol = lin(364.13, 0, 569.32, 9)([x for x, _ in fig10])
    mass10 = lin(613.8, 0, 482.58, 35)([y for _, y in fig10])
    check(sid, "total debris mass: Table 1 (3 x mean) against vector Fig. 10", total, mass10, "another figure, vector")
    a, to_pt = raster(doc[3], (354, 480, 573, 616), 500)
    lab, n = ndi.label((a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80))
    cm = sorted(to_pt(x, y) for y, x in ndi.center_of_mass(lab > 0, lab, range(1, n + 1)))
    if len(cm) == 5:
        check(sid, "wear volume Fig. 10: vector against pixels", vol, lin(364.13, 0, 569.32, 9)([x for x, _ in cm]),
              "red-dot centroids at 500 dpi")
    # Fig. 9 (raster, PDF p. 4): mean wear scar diameter, second reading of the wear volume through the cap formula
    scar = {3: 1.49, 10: 1.83, 15: 3.06, 20: 3.24, 30: 4.81}  # read on the zoomed page image (Fig. 9), +- 0.03 mm
    d = np.array([scar[int(k)] for k in t])
    R = 6.35
    check(sid, "printed wear volume against pi d^4 / (32 R) with the scar of Fig. 9", vol, np.pi * d**4 / (32 * R),
          "formula reconstruction; scar read by eye +- 0.03 mm")
    h = R - np.sqrt(R**2 - (d / 2) ** 2)
    three_caps = 3 * np.pi * h**2 * (3 * R - h) / 3
    # Fig. 12 (vector): three fraction masses per test, log axis; smallest = >100 um, middle = 40-100, largest = 0.45-40 (Fig. 8)
    page = doc[4]
    dec = log_axis(*(lambda x: (x["0.100"], -1, x["10.000"], 1))({w[4]: (w[0] + w[2]) / 2 for w in page.get_text("words")
                                                              if w[4] in ("0.100", "10.000")}))
    groups = {3: (0.0, 0.0, 0.0), 10: (1.0, 0.0, 0.0), 15: (0.4, 0.8, 0.4), 20: (0.0, 0.0, 1.0), 30: (1.0, 0.6, 0.0)}
    frac = {}
    for k, c in groups.items():
        pts = sorted(((y, x) for x, y in markers(page, (343, 485, 491, 545), c, (1.2, 6))), reverse=True)
        ranks = [[q[1] for q in pts if abs(q[0] - yr) < 6] for yr in (534.9, 512.1, 495.2)]  # 20.6, 50, 79.4 percent
        frac[k] = sorted(float(dec(np.mean(r))) for r in ranks)
    f = np.array([frac[int(k)] for k in t])
    check(sid, "sum of the three fractions (Fig. 12) against Table 1 total", total, f.sum(1), "vector log axis")
    df = pd.DataFrame({"set": "dry_fourball", "unit": [f"{int(k)}min" for k in t], "row": [f"{int(k)}min" for k in t], "order": 1,
                       "mode": "interval", "rig": "four-ball machine, 40 kgf, 1470 rpm, DRY; debris rinsed with heptane",
                       "material_pair": "AISI 52100 balls 12.7 mm", "lubricant": "none (dry)", "cond_min": t,
                       "debris": total, "debris_metric": "debris mass on three filter patches (0.45-40, 40-100, >100 um)",
                       "debris_unit": "mg", "debris_type": "mass", "wear": vol,
                       "wear_metric": "wear volume as printed (Fig. 10; equals pi d^4/(32 R))", "wear_unit": "mm3",
                       "resolved": np.nan, "class_fine_mg": f[:, 2], "class_mid_mg": f[:, 1], "class_coarse_mg": f[:, 0],
                       "alt_wear_three_caps": three_caps, "alt_debris_fig10": mass10, "debris_2": mass10,
                       "wear_2": np.pi * d**4 / (32 * R), "debris_err": rel(total, mass10),
                       "wear_err": rel(vol, np.pi * d**4 / (32 * R)), "page": "4-6", "item": "Table 1, Figs 9, 10, 12",
                       "extraction": "text (Table 1); vector (Figs 10, 12)", "note": "severity by the author: 3 min less severe, "
                       "10 min transition, 15-30 min severe"})
    save(sid, df, {"dry_fourball": {"conditions": [["cond_min", "log10"]], "resolved_def": None,
                                    "absolute": {"factor": 1 / 7.80, "note": "D / (7.80 mg/mm3 x W): wear volume to steel mass"},
                                    "fractions": ["class_fine_mg", "class_mid_mg", "class_coarse_mg"],
                                    "sensitivity": {"three_ball_caps": {"wear": "alt_wear_three_caps"},
                                                    "debris_fig10": {"debris": "alt_debris_fig10"}}}})


# ------------------------------------------------------------------------------------------------------ Higashi 2003
def higashi():
    sid = "jsme_higashi2003"
    doc = open_pdf(sid)
    p2, p1 = image(doc, 19)[:, :, 0], image(doc, 10)[:, :, 0]
    s = 3296 / 1075
    crop = lambda a, b: a[int(b[1] * s):int(b[3] * s), int(b[0] * s):int(b[2] * s)]
    f6, f5 = crop(p2, (630, 1110, 945, 1335)) < 128, crop(p2, (630, 865, 945, 1085)) < 128
    f2 = crop(p1, (580, 1000, 950, 1250)) < 128
    holes = lambda b: [r for r in regionprops(label(~b, connectivity=1)) if 150 <= r.area <= 420 and
                       r.bbox[3] - r.bbox[1] < 30 and r.bbox[2] - r.bbox[0] < 30 and r.eccentricity < 0.6 and
                       not (r.centroid[1] < 650 and r.centroid[0] < 232)]  # legend

    def grid(b):  # frame lines of the plot: longest rows and columns
        rows, cols = b.sum(1), b.sum(0)
        return int(np.flatnonzero(rows > 0.6 * rows.max())[0]), int(np.flatnonzero(rows > 0.6 * rows.max())[-1]), \
            int(np.flatnonzero(cols > 0.6 * cols.max())[0]), int(np.flatnonzero(cols > 0.6 * cols.max())[-1])

    top, bot, left, right = grid(f6)
    X, Y = lin(left, 0, right, 40), lin(bot, 0, top, 40)
    km = np.arange(5, 45, 5)
    open_c = [(float(X(r.centroid[1])), float(Y(r.centroid[0]))) for r in holes(f6)]
    bal = np.array([next((v for x, v in open_c if abs(x - k) < 0.8), np.nan) for k in km])
    # triangles: template (base 20, height 19 px) slid vertically at the known abscissa; legend blanked
    n = 11
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    tt = (yy + 9.5) / 19
    tpl = ((tt >= 0) & (tt <= 1) & (np.abs(xx) <= 10 * tt)).astype(float)
    f6t = f6.copy()
    f6t[88:214, 185:650] = False
    resp = match_template(f6t.astype(float), tpl, pad_input=True)
    cnt = []
    for k in km:
        c = int(round(left + (right - left) * k / 40))
        col = resp[:, max(c - 2, 0):c + 3].max(1)
        col[: top + 2] = -1
        col[bot:] = -1
        cnt.append(float(Y(col.argmax())))
    cnt = np.array(cnt)
    by_eye = lin(601, 0, 73, 40)([593, 577, 555, 516, 469, 403, 340, 266])  # zoomed crop of Fig. 6, read by eye
    top2, bot2, left2, right2 = grid(f2)
    X2, Y2, W2 = lin(left2, 0, right2, 40), lin(bot2, 0, top2, 40), lin(bot2, 0, top2, 1.5)
    vol2 = [(float(X2(r.centroid[1])), float(Y2(r.centroid[0]))) for r in holes(f2)]
    bal2 = np.array([next((v for x, v in vol2 if abs(x - k) < 0.8), np.nan) for k in km])
    disc = ndi.binary_opening(f2, structure=np.ones((13, 13)))
    lab, m = ndi.label(disc)
    rate = [(float(X2(x)), float(W2(y))) for y, x in ndi.center_of_mass(disc, lab, range(1, m + 1)) if not (x < 560 and y < 232)]
    rate = np.array([next((v for x, v in rate if abs(x - k) < 0.8), np.nan) for k in km])  # 1e-6 mm3/mm
    check(sid, "balance volume: Fig. 6 against Fig. 2", bal, bal2, "another figure, pixel holes")
    check(sid, "balance interval volume: Fig. 6 increments against wear rate of Fig. 2 x 5 km", np.diff(bal, prepend=0), rate * 5,
          "another quantity, pixel discs")
    check(sid, "counter volume Fig. 6: template against reading by eye", cnt, by_eye, "zoomed crop by eye")
    dW, dD = np.diff(bal, prepend=0), np.diff(cnt, prepend=0)
    df = pd.DataFrame({"set": "plate_intervals", "unit": "run1", "row": [f"{k - 5}-{k}km" for k in km], "order": np.arange(1, 9),
                       "mode": "interval", "rig": "cylinder end face on plate, oil 4 l, 300 N, 0.6 m/s, VG32, 40 C, in-line counters",
                       "material_pair": "S55C steel on WJ2 white metal (plate weighed)", "lubricant": "VG32 mineral oil",
                       "cond_mid_km": km - 2.5, "debris": dD,
                       "debris_metric": "increment of debris volume from outlet minus inlet particle counters (11-45 um, thickness law)",
                       "debris_unit": "mm3", "debris_type": "volume", "wear": dW,
                       "wear_metric": "increment of the plate volume loss from the precision balance", "wear_unit": "mm3",
                       "resolved": np.nan, "class_cum_debris": cnt, "class_cum_wear": bal, "debris_2": np.diff(by_eye, prepend=0),
                       "wear_2": np.diff(bal2, prepend=0), "debris_err": rel(dD, np.diff(by_eye, prepend=0)),
                       "wear_err": rel(dW, np.diff(bal2, prepend=0)), "page": "212 (Fig. 6), 211 (Fig. 2)", "item": "Figs 2, 6",
                       "extraction": "scan (400 dpi)", "note": "one test; counter window 9-46 um; plate only is weighed"})
    save(sid, df, {"plate_intervals": {"conditions": [["cond_mid_km", "log10"]], "resolved_def": None,
                                       "absolute": {"factor": 1.0, "note": "both in mm3"},
                                       "sensitivity": {"second_reading": {"debris": "debris_2", "wear": "wear_2"}}}})
    # Fig. 5: particle thickness against size (16 classes of 5 um)
    top5, bot5, left5, right5 = grid(f5)
    X5, Y5 = lin(left5, 0, right5, 80), lin(bot5, 0, top5, 6)
    disc = ndi.binary_opening(f5, structure=np.ones((11, 11)))
    lab, m = ndi.label(disc)
    pts = sorted((float(X5(x)), float(Y5(y))) for y, x in ndi.center_of_mass(disc, lab, range(1, m + 1)))
    th = pd.DataFrame([(d, t) for d, t in pts if t > 0.5 and 3 < d < 82], columns=["size_um", "thickness_um"])
    eye = {5: 0.94, 10: 1.49, 15: 1.65, 20: 2.89, 25: 1.89, 30: 2.35, 35: 3.05, 40: 2.39, 45: 2.51, 50: 3.79, 55: 3.31, 60: 3.16,
           65: 4.22, 70: 3.65, 75: 3.69, 80: 4.98}  # zoomed crop read by eye
    th["thickness_2"] = [eye.get(int(5 * round(d / 5)), np.nan) for d in th.size_um]
    check(sid, "particle thickness Fig. 5: pixels against reading by eye", th.thickness_um, th.thickness_2, "zoomed crop by eye")
    th.to_csv(DATA / sid / "thickness.csv", index=False, lineterminator="\n", float_format="%.4g")
    print(f"{sid}: thickness {len(th)} classes")


# -------------------------------------------------------------------------------------------- NASA/TM-2011-217117
# Fig. 10 marker abscissae read by eye on a 5x enlargement (left crop starts at px 75, right crop at px 300 of the 600-px raster)
NASA_X5 = {"27": ([123, 191, 223, 660, 748, 890, 1008, 1088, 1137, 1240], [243, 343, 397, 460, 635, 738, 826, 906, 968, 1020, 1052,
                                                                         1093, 1157, 1219, 1277]),
           "33": ([47, 645, 705, 767, 828, 897, 938, 988, 1045, 1157, 1265], [232, 298, 358, 447, 494, 530, 583, 623, 670, 682, 730,
                                                                            755, 832]),
           "35": ([230, 339, 433, 462, 542, 607, 637, 672, 702, 737, 787, 823, 845, 867, 913, 962, 1005, 1048, 1090, 1128, 1142], []),
           "36": ([57, 387, 482, 552, 598, 707, 762, 790, 825, 857, 920, 967, 1007, 1060, 1112, 1152, 1213, 1262, 1322],
                  [243, 592, 632, 715, 732, 798, 885])}
NASA_AX = {"27": (32, 79, 589, 200, 295, 8, 1000, 20, []), "33": (34, 86, 647, 250, 351, 64, 1000, 25, [(116.5, 8.4), (180.5, 26.6)]),
           "35": (38, 88, 650, 200, 357, 95, 600, 15, [(79.0, 9.1)]), "36": (40, 87, 648, 350, 352, 76, 800, 20, [])}
NASA_LOAD = {"27": 125, "33": 100, "35": 150, "36": 150}  # Table 2 (PDF p. 11)
NASA_END = {"27": 16.2, "33": 27.8, "35": 15.2, "36": 19.2}


def nasa_bearing():
    from skimage.feature import peak_local_max

    sid = "nasa_tm2011_217117"
    doc = open_pdf(sid)
    disk = lambda r, n: (lambda g: ((g[0] ** 2 + g[1] ** 2) <= r * r).astype(float))(np.mgrid[-n:n + 1, -n:n + 1])
    f10 = image(doc, 52)
    xv = lambda px: (px - 82) / (575 - 82) * 1000
    shape = {"27": (70, 114, 168), "33": (228, 108, 10), "35": (178, 20, 158), "36": (79, 98, 40)}
    rows, M1, M2, L1, L2 = [], [], [], [], []
    for cup, (xref, x0, x1, t1, y0, yt, mt, lt, extra) in NASA_AX.items():
        a = image(doc, xref)
        g = colour_mask(a, (119, 147, 60), 70)
        g[25:108, 85:298] = 0  # legend
        r = match_template(g, disk(5.5, 6), pad_input=True)
        pts = [(X, Y) for Y, X in map(tuple, peak_local_max(r, min_distance=2, threshold_abs=0.5))]
        for t, L in extra:  # two / one overlapping markers below the detection threshold, located by eye, refined by template
            X, Y = round(x0 + t / t1 * (x1 - x0)), round(y0 - L / lt * (y0 - yt))
            dy, dx = np.unravel_index(r[Y - 3:Y + 4, X - 3:X + 4].argmax(), (7, 7))
            pts.append((X + dx - 3, Y + dy - 3))
        pts.sort()
        left, right = NASA_X5[cup]
        mass10 = [xv(75 + v / 5) for v in left] + [xv(300 + v / 5) for v in right]
        if len(mass10) != len(pts):
            raise ValueError(f"cup {cup}: {len(mass10)} markers in Fig. 10 against {len(pts)} in the time plot")
        blue = colour_mask(a, (0, 112, 192), 90)
        blue[25:108, 85:298] = 0
        m10 = colour_mask(f10, shape[cup], 110)
        m10[12:164, 95:220] = 0
        resp = match_template(m10, disk(5.0, 6), pad_input=True)
        hw = 5 / (y0 - yt) * mt  # half marker height in mg
        for i, ((X, Y), m) in enumerate(zip(pts, mass10), 1):
            t, L = (X - x0) / (x1 - x0) * t1, (y0 - Y) / (y0 - yt) * lt
            ys = np.flatnonzero(blue[:, X - 1:X + 2].any(1))
            ys = ys[ys < y0 + 8]
            upper = (y0 - ys.min()) / (y0 - yt) * mt - hw
            px, py = int(round(82 + m / 1000 * 493)), int(round(319 - L / 30 * 311))
            win = resp[py - 6:py + 7, max(px - 3, 0):px + 4]
            dy, _ = np.unravel_index(win.argmax(), win.shape)
            l10 = (319 - (py - 6 + dy)) / 311 * 30 if win.max() > 0.55 and cup != "36" else np.nan
            rows.append({"set": "cups", "unit": f"cup{cup}", "row": f"cup{cup}:{i}", "order": i, "mode": "series",
                         "rig": "bearing spall propagation rig, in-line inductive debris monitor", "material_pair":
                         "tapered roller bearing, UH-60M tail gearbox (service bearings)", "lubricant": "not stated",
                         "cond_time_h": t, "cond_load_pct": NASA_LOAD[cup], "debris": m,
                         "debris_metric": "accumulated oil-debris-monitor mass (particles 125-1016 um as spheres)",
                         "debris_unit": "mg", "debris_type": "mass", "wear": L, "wear_metric": "measured spall length on the cup",
                         "wear_unit": "mm", "resolved": np.nan, "debris_2": upper, "wear_2": l10,
                         "debris_err": abs(upper / m - 1), "wear_err": abs(l10 / L - 1) if np.isfinite(l10) else np.nan,
                         "page": "13 (Fig. 10), 9-10 (Figs 4-7)", "item": "Figs 4-7, 10",
                         "extraction": "raster (lossless): mass by eye on Fig. 10, length by template on Figs 4-7", "note": ""})
            M1.append(m), M2.append(upper), L1.append(L), L2.append(l10)
        check(sid, f"cup {cup} end spall length against Table 2", [(y0 - pts[-1][1]) / (y0 - yt) * lt], [NASA_END[cup]], "printed, Table 2")
    check(sid, "debris mass at the inspections: Fig. 10 by eye against the mass curve of Figs 4-7", M1, M2,
          "upper end of the mass curve at the inspection time")
    check(sid, "spall length: template on Figs 4-7 against template on Fig. 10 (cups 27, 33, 35)", L1, L2, "another figure")
    save(sid, pd.DataFrame(rows), {"cups": {"conditions": [["cond_time_h", "id"]], "resolved_def": None, "analysis": "series_cups",
                                            "sensitivity": {"mass_second_reading": {"debris": "debris_2"}}}})


# ------------------------------------------------------------------------------------------------------- Torres 2011
def torres():
    sid = "bu_torres2011"
    doc = open_pdf(sid)
    page = doc[155]
    X, Y = lin(139.7, 0, 420.5, 90), lin(455.1, 0, 312.1, 1)
    curves = {}
    for g in page.get_drawings():
        c = tuple(round(v, 1) for v in (g.get("color") or ()))
        if g["type"] == "s" and c in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) and len(g["items"]) > 500:
            pts = [(it[1].x, it[1].y) for it in g["items"] if it[0] == "l"]
            curves.setdefault(c, []).extend(pts)
    ticks = sorted({round(it[1].x, 1) for g in page.get_drawings() if g["type"] == "s" and len(g["items"]) == 1
                    for it in g["items"] if it[0] == "l" and abs(it[1].x - it[2].x) < 0.01 and 451 < min(it[1].y, it[2].y) < 453})
    X = lin(ticks[0], 0, ticks[-1], 10 * (len(ticks) - 1))
    info = {(1.0, 0.0, 0.0): ("Mobil DTE 13M", 93, 7.8117, 7.8022, 32), (0.0, 0.0, 1.0): ("Repsol VG46", 148, 7.6021, 7.5905, 46)}
    rows, store = [], {}
    for c, (oil, peak, w0, w1, visc) in info.items():
        h, v = X([p[0] for p in curves[c]]), Y([p[1] for p in curves[c]])
        order = np.argsort(h, kind="stable")
        h, v = h[order], v[order]
        store[oil] = (h, v * peak)
        a, to_pt = raster(page, (139.7, 310, 429, 456), 400)
        col = (a[:, :, 0] - a[:, :, 2] > 90) if c[0] else (a[:, :, 2] - a[:, :, 0] > 90)  # hairlines, anti-aliased
        ys, xs = np.nonzero(col)
        hp = X(to_pt(xs, ys)[0])
        med2 = [float(np.median(Y(to_pt(xs, ys)[1])[np.abs(hp - t) < 0.25])) for t in (20, 40, 60)]
        med1 = [float(np.median(v[np.abs(h - t) < 0.25])) for t in (20, 40, 60)]
        check(sid, f"normalised density {oil} at 20, 40, 60 h", med1, med2, "pixels of the curve colour at 400 dpi")
        end = float(np.median(v[h > h.max() - 1])) * peak
        rows.append({"set": "pin_totals", "unit": oil, "row": oil, "order": 1, "mode": "cumulative",
                     "rig": "pin on disc, 300 N, 0.26 m/s, 60 C, 125 ml oil circulated at 4 ml/min", "material_pair":
                     "mild steel pin on EN24T disc (pin weighed)", "lubricant": oil, "cond_visc40_cSt": visc,
                     "debris": peak * 0.125, "debris_metric": "peak ferrous density (ppm = mg/l) x 0.125 l oil",
                     "debris_unit": "mg", "debris_type": "mass", "wear": (w0 - w1) * 1000, "wear_metric": "pin mass loss after 92 h",
                     "wear_unit": "mg", "resolved": np.nan, "class_peak_h": float(h[v.argmax()]), "class_end_mg": end * 0.125,
                     "debris_2": np.nan, "wear_2": np.nan, "debris_err": 5 / peak, "wear_err": 0.14 / ((w0 - w1) * 1000), "page": "156",
                     "item": "Fig. 4.28 (legend and table)", "extraction": "text", "note": "sensor repeatability 5 ppm; weights to 0.1 mg; "
                     f"end-of-test density {end:.0f} ppm; disc wear not measured"})
    t = np.arange(10, 130, 10.0)  # Table 4.5 (PDF p. 158), other run, same nominal conditions (Mobil)
    wt = np.array([7.7983, 7.7975, 7.7967, 7.7962, 7.7954, 7.7948, 7.7941, 7.7936, 7.7932, 7.7929, 7.7926, 7.7925, 7.7924])
    h, d = store["Mobil DTE 13M"]
    for i, k in enumerate(t, 1):
        sel = np.abs(h - k / 60) < 0.06
        rows.append({"set": "pin_weight_series", "unit": "Mobil DTE 13M (other run)", "row": f"{int(k)}min", "order": i, "mode": "series",
                     "rig": "pin on disc, interrupted run, 12 x 10 min", "material_pair": "mild steel pin on EN24T disc",
                     "lubricant": "Mobil DTE 13M", "cond_visc40_cSt": 32, "cond_min": k, "debris": max(float(np.median(d[sel])) * 0.125, 0.0),
                     "debris_metric": "ferrous density of the uninterrupted run at the same time x 0.125 l", "debris_unit": "mg",
                     "debris_type": "mass", "wear": (wt[0] - wt[i]) * 1000, "wear_metric": "cumulative pin mass loss", "wear_unit": "mg",
                     "resolved": np.nan, "debris_2": np.nan, "wear_2": np.nan, "debris_err": 0.625 / max(float(np.median(d[sel])) * 0.125, 0.625),
                     "wear_err": 0.14 / ((wt[0] - wt[i]) * 1000), "page": "156, 158", "item": "Fig. 4.28, Table 4.5",
                     "extraction": "vector (density); text (weights)", "note": "debris and wear come from two runs under the same nominal conditions"})
    save(sid, pd.DataFrame(rows), {"pin_totals": {"conditions": [], "resolved_def": None, "absolute": {"factor": 1.0, "note": "mg of debris in the oil / mg of pin loss"}},
                                   "pin_weight_series": {"conditions": [["cond_min", "id"]], "resolved_def": None, "absolute": {"factor": 1.0, "note": "mg / mg"}}})


# ------------------------------------------------------------------------------------------------------- Jordan 1995
def jordan():
    sid = "uct_jordan1995"
    doc = open_pdf(sid)

    def diamonds(page, clip):
        a, _ = raster(doc[page - 1], clip, 300)
        core = ndi.binary_opening(a.sum(2) < 330, structure=np.ones((9, 9)))
        lab, n = ndi.label(core)
        c = ndi.center_of_mass(core, lab, range(1, n + 1))
        ar = ndi.sum(core, lab, range(1, n + 1))
        return a, [(x, y) for (y, x), s in zip(c, ar) if s > 190]

    a9, p9 = diamonds(50, (55, 100, 455, 370))
    X, Y = lin(218, 0, 1308, 0.004), lin(886, 0, 234, 16)
    p9 = sorted((float(X(x)), float(Y(y))) for x, y in p9 if x < 1320 and y > 200)
    _, p13 = diamonds(52, (55, 370, 455, 640))  # left axis is skewed by the scan: x = 220.2 at y = 509, 225 at y = 987; 225.25 px per 0.001
    x13 = sorted((x - (220.2 + (y - 509) / 478 * 4.8)) / 225.25 * 1e-3 for x, y in p13 if x < 1150 and y > 480)
    rate, wi = np.array([q[0] for q in p9]), np.rint([q[1] for q in p9])
    check(sid, "wear index (integer) read from Fig. 9", wi, [p[1] for p in p9], "unrounded reading")
    second = np.array([min(x13, key=lambda v: abs(v - r)) if min(abs(v - r) for v in x13) < 1.2e-4 else np.nan for r in rate])
    check(sid, "wear rate: Fig. 9 against Fig. 13 (same samples, maximum-size plot)", rate, second, "another scanned figure")
    # Laboratory sheets App. 7 (PDF pp. 115-127): sample: (WI, AVE code, MAX code, Fe ppm); grease samples 3-6 and 12-14
    sheets = {3: (11, 3, 4, 9), 4: (9, 3, 9, 8), 5: (15, 6, 3, 8), 6: (12, 4, 2, 3), 12: (1, 1, 1, 8), 13: (11, 9, 4, 7), 14: (1, 1, 1, 6)}
    ave_um = {1: 2.5, 2: 7.5, 3: 15, 4: 25, 5: 35, 6: 45, 7: 55, 8: 65, 9: 75}
    match, left = [], dict(sheets)
    for r, w in zip(rate, wi):  # unique index values first; the two samples with 11 are told apart by AVE (Fig. 11), those with 1 are equal
        cand = [s for s, v in left.items() if v[0] == w]
        if w == 11:
            cand = [13] if r > 0.003 else [3]
        match.append(cand[0])
        left.pop(cand[0])
    ave = np.array([sheets[s][1] for s in match])
    df = pd.DataFrame({"set": "grease", "unit": [f"sample{s}" for s in match], "row": [f"sample{s}" for s in match], "order": 1,
                       "mode": "interval", "rig": "fretting rig (false brinelling), roller on flat, 56.9 N, 6 h", "material_pair":
                       "1% C 1.5% Cr bearing steel", "lubricant": "lithium grease NLGI 2 (whole bath analysed)", "debris": wi,
                       "debris_metric": "ferrous wear index (PQ2000) of the whole lubricant bath", "debris_unit": "index",
                       "debris_type": "index", "wear": rate, "wear_metric": "wear rate from Talysurf profiles", "wear_unit": "mg/h",
                       "resolved": wi * np.array([ave_um[v] for v in ave]), "class_ave_code": ave,
                       "class_max_code": [sheets[s][2] for s in match], "class_fe_ppm": [sheets[s][3] for s in match],
                       "class_floor": (wi <= 1).astype(int), "debris_2": wi, "wear_2": second, "debris_err": np.nan,
                       "wear_err": rel(rate, second), "page": "50 (Fig. 9), 51-52 (Figs 11, 13), 115-127 (App. 7)", "item": "Fig. 9, App. 7",
                       "extraction": "scan", "note": "index 1 is the reading floor; samples 12 and 14 (both index 1) cannot be told apart; the MAX "
                       "codes plotted in Fig. 13 do not agree with the laboratory sheets (pairing of the thesis uncertain)"})
    save(sid, df, {"grease": {"conditions": [], "resolved_def": "index x mean particle size (class midpoint), the product named by the author",
                              "sensitivity": {"without_floor_readings": {"drop": list(df.unit[df.class_floor == 1])},
                                              "wear_second_reading": {"wear": "wear_2", "fallback": True},
                                              "iron_ppm": {"debris": "class_fe_ppm"}}}})


# ------------------------------------------------------------------------------------------- Yoshinaga 1998, Quinn 1983
def yoshinaga():
    sid = "jsme_yoshinaga1998"
    open_pdf(sid)
    rows = []
    # Table 2 (p. 665): share of particles >= 40 um among those >= 10 um; specific wear rate read on Fig. 5 (two tests per load)
    for P, share, k in ((30, 0.0, (1.43e-8, 8.2e-9)), (100, 2.1, (2.3e-6, 1.84e-6)), (500, 18.1, (4.86e-6, 4.23e-6))):
        rows.append({"set": "filter_table2", "unit": f"{P}N", "cond_load_N": P, "debris": share, "wear": float(np.sqrt(k[0] * k[1])),
                     "debris_metric": "share of particles >= 40 um (membrane filter, microscope, >= 10 um counted)", "page": "665",
                     "item": "Table 2, Fig. 5", "extraction": "table; scan (log axis)", "wear_err": 0.1})
    for P, share, k in ((500, 24.1, 5.6e-6), (1000, 11.8, 4.8e-7)):  # Tables 4-5 (p. 667), on-line images >= 40 um
        rows.append({"set": "online_table4", "unit": f"{P}N_online", "cond_load_N": P, "debris": share, "wear": k,
                     "debris_metric": "share of particles > 100 um (on-line images, >= 40 um counted)", "page": "667",
                     "item": "Tables 4-5", "extraction": "table", "wear_err": 0.02})
    df = pd.DataFrame(rows)
    df = df.assign(row=df.unit, order=1, mode="rate", rig="cylinder end face on plate, VG32, 0.6 m/s", resolved=np.nan,
                   material_pair="S55C steel on lead bronze LBC3", lubricant="ISO VG32", debris_unit="%", debris_type="share",
                   wear_metric="specific wear rate", wear_unit="mm3/(N m)", debris_2=np.nan, wear_2=np.nan, debris_err=0.03,
                   note="relative size shares, no debris amount; two counting methods, not poolable")
    cols = ["set", "unit", "row", "order", "mode", "rig", "material_pair", "lubricant", "cond_load_N", "debris", "debris_metric",
            "debris_unit", "debris_type", "wear", "wear_metric", "wear_unit", "resolved", "debris_2", "wear_2", "debris_err", "wear_err",
            "page", "item", "extraction", "note"]
    save(sid, df[cols], {"filter_table2": {"conditions": [], "resolved_def": None, "analysis": "table_only"},
                         "online_table4": {"conditions": [], "resolved_def": None, "analysis": "table_only"}})


# Fig. 3 of NASA CR-3686 (PDF p. 113, rotated), px on a 1.5x enlargement of three crops that start at (480, y0) of the 300-dpi page
QUINN = {2: (400, 203, 19.02, (641, 445, 244, 63), (12, 30), [(285, 596), (339, 592), (391, 553), (398, 577), (482, 591), (486, 606),
             (580, 580), (667, 556), (760, 557), (853, 435)]),
         3: (840, 210, 19.12, (628, 436, 241, 51), (7.7, 41.3), [(289, 607), (342, 599), (395, 602), (582, 552), (671, 576), (773, 552),
             (865, 520), (960, 548), (1042, 303), (1140, 449), (1233, 91)]),
         4: (1270, 215, 19.07, (638, 442, 243, 57), (None, 41.8), [(294, 623), (347, 612), (400, 622), (495, 605), (597, 595), (784, 578),
             (868, 569), (960, 559), (1055, 359), (1245, 183), (1338, 145)])}


def quinn():
    sid = "nasa_cr3686_quinn1983"
    doc = open_pdf(sid)
    pix = doc[112].get_pixmap(dpi=300)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, 0]
    b = np.rot90(a, -1) < 128
    centres = [(r.centroid[1], r.centroid[0]) for r in regionprops(label(~b, connectivity=1)) if 20 <= r.area <= 200 and
               r.bbox[3] - r.bbox[1] <= 18 and r.bbox[2] - r.bbox[0] <= 18]
    rows, w1, w2 = [], [], []
    for v, (y0, x0, px_per_N, yt, (t34, tfeo), pts) in QUINN.items():
        scale = lambda y: float(np.interp(-y, [-q for q in yt], [0, 1, 2, 3]))
        for i, (x, y) in enumerate(pts, 1):
            load, w = (x - x0) / px_per_N, scale(y)
            near = min(((math.hypot((cx - 480) * 1.5 - x, (cy - y0) * 1.5 - y), cy) for cx, cy in centres), default=(99, 0))
            second = scale((near[1] - y0) * 1.5) if near[0] < 8 else np.nan
            regime = "FeO" if load >= tfeo else ("Fe3O4" if t34 is not None and load >= t34 else ("Fe2O3" if t34 is not None else "Fe3O4"))
            rows.append({"set": "en8_dry", "unit": f"{v}m/s:{i}", "row": f"{v}m/s:{i}", "order": 1, "mode": "rate",
                         "rig": "pin on disc, dry, unlubricated", "material_pair": "EN8 steel pin on steel disc", "lubricant": "none (dry)",
                         "cond_load_N": load, "cond_speed_ms": v, "debris": np.nan, "debris_metric": "dominant oxide of the collected debris "
                         "(X-ray powder diffraction); no amount", "debris_unit": "class", "debris_type": "class", "wear": w * 1e-12,
                         "wear_metric": "wear rate", "wear_unit": "m3/m", "resolved": np.nan, "class_regime": regime, "debris_2": np.nan,
                         "wear_2": second * 1e-12, "debris_err": np.nan, "wear_err": abs(second / w - 1), "page": "113", "item": "Fig. 3",
                         "extraction": "scan (by eye on an enlargement)", "note": "regime from the two arrows of the panel; at 4 m/s only the FeO arrow is "
                         "drawn, lower loads taken as Fe3O4"})
            w1.append(w), w2.append(second)
    check(sid, "wear rate (1e-12 m3/m): by eye against the enclosed circle centre", w1, w2, "connected components at 300 dpi")
    save(sid, pd.DataFrame(rows), {"en8_dry": {"conditions": [["cond_load_N", "log10"], ["cond_speed_ms", "log10"]], "resolved_def": None,
                                               "analysis": "regime"}})


# ------------------------------------------------------------------------------------------------------ Bolander 2009
def bolander():
    sid = "phm_bolander2009"
    doc = open_pdf(sid)
    a = image(doc, 31)
    m = colour_mask(a, (0, 0, 128), 60) > 0
    lab, n = ndi.label(m)
    c = ndi.center_of_mass(m, lab, range(1, n + 1))
    s = ndi.sum(m, lab, range(1, n + 1))
    pts = sorted((x, y) for (y, x), k in zip(c, s) if k > 40 and x < 450 and not (95 < x < 110 and y < 40))
    X, Y = lin(74, 0, 459.3, 4), lin(202.3, 0, 202.3 - 150 * 1.1775, 150)
    eye = [(222, 607), (273, 591), (426, 579), (614, 450), (927, 306), (951, 366), (987, 237), (1315, 196)]
    xs, ys = X([p[0] for p in pts]), Y([p[1] for p in pts])
    check(sid, "debris count (axis divisions)", xs, X([q[0] / 3 for q in eye]), "3x enlargement by eye")
    check(sid, "spall size (% of critical)", ys, Y([q[1] / 3 for q in eye]), "3x enlargement by eye")
    df = pd.DataFrame({"set": "rig_tests", "unit": [f"point{i}" for i in range(1, 9)], "row": [f"point{i}" for i in range(1, 9)],
                       "order": 1, "mode": "cumulative", "rig": "aircraft engine bearing rig tests (several)", "material_pair": "not stated",
                       "lubricant": "not stated", "debris": np.maximum(xs, 0), "debris_metric": "oil-debris count, relative axis (divisions; zero "
                       "assumed at the axis crossing)", "debris_unit": "divisions", "debris_type": "count", "wear": np.maximum(ys, 0),
                       "wear_metric": "measured spall size, % of critical", "wear_unit": "%", "resolved": np.nan,
                       "debris_2": np.maximum(X([q[0] / 3 for q in eye]), 0), "wear_2": np.maximum(Y([q[1] / 3 for q in eye]), 0),
                       "debris_err": np.nan, "wear_err": np.nan, "page": "3", "item": "Fig. 2", "extraction": "raster (lossless)",
                       "note": "axis without numbers (Low ... High); points of several unidentified rig tests"})
    df.loc[0, ["debris", "wear"]] = 0.0  # the marker sits on the axis crossing
    save(sid, df, {"rig_tests": {"conditions": [], "resolved_def": None, "units_unknown": True}})


# --------------------------------------------------------------------------------------------------------- Craig 2010
# Gravimetric tables (PDF pp. 109, 120, 135, 143, 156, 218, 224/225, 231): pre / post weights in g; cup_mg / cone_mg = printed differences
CRAIG_W = """
test|bearing|cup_pre|cone_pre|cup_post|cone_post|cup_mg|cone_mg|table
1|1|61.1334|116.65044|61.13224|116.64409|1.16|6.35|5.2
1|2|61.02045|116.51866|61.01939|116.50891|1.06|9.75|5.2
1|3|61.09645|116.52825|61.09607|116.52714|0.38|1.11|5.2
1|4|61.05285|117.17384|61.05252|117.15682|0.33|17.02|5.2
3|1|61.111|116.887|61.1108|116.879|0.2|7.93|5.3
3|2|61.1432|116.72|61.1428|116.718|0.36|2.36|5.3
3|3|61.1507|116.965|61.1504|116.964|0.28|0.77|5.3
3|4|61.117|116.381|60.9309|116.324|186.1|57.79|5.3
5|1|61.20675|116.65536|61.2034|116.65387|3.35|1.49|5.4
5|2|61.19786|116.85761|61.1949|116.83431|2.96|23.3|5.4
5|3|61.20769|116.60531|61.20625|116.60303|1.44|2.28|5.4
5|4|61.20627|116.59734|61.20277|116.59669|3.5|0.65|5.4
7|1|61.2271|116.6283|61.2253|116.6187|1.8|9.6|6.2
7|2||194.93959||194.90139||38.2|6.2
7|3||195.21048||195.20868||1.8|6.2
7|4|61.1948|116.5385|61.1926|116.5317|2.2|6.8|6.2
10|1|61.271|116.1926|61.6295|116.1895|1.5|3.1|6.3
10|2||194.87958||194.80868||70.9|6.3
10|3||195.19438||195.19178||2.6|6.3
10|4|61.0728|116.3196|61.0701|116.3133|2.7|6.3|6.3
2|1|61.2977|116.73273|61.1322|116.727|165.5|5.73|E.1
2|2|61.04|116.44562|61.0194|115.986|20.6|459.62|E.1
2|3|61.2773|116.6689|61.0961|116.669|181.2|0.1|E.1
2|4|61.21|116.65419|61.0525|116.741|157.5|86.81|E.1
4|1|61.14753|116.68499|61.13813|116.67809|0.94|0.69|F.1
4|2|61.05145|116.63897|61.01305|116.63521|38.4|3.76|F.1
4|3|61.13258|116.70185|61.13048|116.69995|2.1|1.9|F.1
4|4|61.14958|116.59376|61.14588|116.59266|3.7|1.1|F.1
6|1|61.33856|116.76418|61.3352|116.7616|3.36|2.58|G.1
6|2|61.38479|116.69253|61.38004|116.67722|4.75|15.31|G.1
6|3|61.36454|116.71359|61.36111|116.7102|3.43|3.39|G.1
6|4|61.37994|116.70726|61.37775|116.70548|2.19|1.78|G.1
"""
CRAIG_FAILED = {1: None, 2: 459.0, 3: 186.0, 4: 38.4, 5: 23.3, 6: 15.31, 10: 70.9}  # author, Tables 7.1-7.9
CRAIG_MAXCOUNT = {2: 30000, 3: 105000, 4: 5000, 5: 1500, 6: 500, 10: 5100}  # author, Tables 7.3 and 7.9 (wear out)
CRAIG_END = {1: 80.0, 2: 78.0, 3: 62.5, 4: 142.0, 5: 76.0, 6: 165.0, 9: 115.0, 10: 23.6}  # h, Tables 7.6-7.8
CH = {"5-15": (0, 0, 0), "15-25": (1, 0, 0), "25-50": (0, 1, 0), "50-100": (0, 0, 1), ">100": (0, 1, 1)}
# page: (box of the PODS panel, time axis, lower y axis, upper y axis or None, y of the break, SI box, SI axis, has markers)
CRAIG_FIG = {1: (106, (170.1, 257.7, 479.8, 362.4), (170.04, 0, 470.42, 80), (362.28, 0, 323.76, 300), (309.0, 1000, 257.64, 6000), 310.0,
                 (170.1, 362.5, 479.8, 467.1), (467.04, 0, 371.88, 0.10), True),
             3: (116, (177.9, 326.7, 478.7, 399.9), (177.84, 0, 461.28, 60), (399.84, 0, 375.84, 6000), (361.68, 30000, 335.4, 90000), 363.0,
                 (177.9, 399.9, 478.7, 473.1), (473.04, 0, 436.92, 1.0), True),
             5: (132, (169.1, 422.7, 484.2, 532.4), (169.08, 0, 484.32, 80), (532.32, 0, 489.24, 4000), (476.52, 40000, 433.44, 60000), 477.5,
                 (169.1, 532.4, 484.2, 642.1), (642.0, 0, 542.28, 0.10), True),
             9: (148, (167.8, 192.4, 475.4, 299.4), (167.76, 0, 475.44, 120), (299.28, 0, 240.12, 4000), (223.56, 12000, 207.96, 14000), 224.5,
                 (167.8, 299.5, 475.4, 406.3), (406.2, 0, 307.56, 0.06), True),
             10: (154, (167.2, 158.2, 483.4, 228.0), (167.16, 0, 483.48, 25), (227.88, 0, 164.52, 5000), None, 0,
                  (167.2, 228.1, 483.4, 297.6), (297.48, 0, 239.4, 0.20), True),
             2: (216, (164.7, 76.4, 465.3, 138.1), (164.64, 71, 465.36, 79), (138.0, 0, 121.92, 1000), (103.44, 10000, 80.76, 30000), 113.0,
                 (165.2, 138.1, 465.8, 199.9), (199.8, 0, 153.24, 1.5), False),
             4: (224, (179.2, 81.3, 481.3, 148.6), (179.16, 0, 481.32, 150), (148.44, 0, 81.24, 7000), None, 0,
                 (179.2, 148.7, 481.3, 215.9), (215.76, 0, 159.12, 0.16), False),
             6: (229, (172.9, 110.0, 486.8, 177.1), (172.8, 0, 486.84, 170), (177.0, 0, 109.92, 4000), None, 0,
                 (172.9, 177.1, 486.8, 244.2), (244.08, 0, 183.12, 0.10), False)}
# legend entries inside the panels (thin boxes where data points come close)
LEGEND = {1: [(230, 312, 420, 336)], 3: [(190, 342, 292, 375)], 5: [(186, 436, 212, 492)], 9: [(314, 256, 416, 290)],
          10: [(x, y - 0.6, x + 25, y + 0.6) for x, y in ((309, 196.08), (387, 196.08), (309, 207.24), (387, 207.24), (309, 218.4))],
          2: [(160, 88, 334, 108)],
          4: [(x, y - 0.5, x + 22, y + 0.5) for x, y in ((340, 89.04), (402, 89.04), (340, 98.64), (406, 98.64), (340, 108.12))],
          6: [(400, 118, 430, 168)]}


# SOAP iron panels: (box, axis) of the tests whose figure has one (ppm; tests 1, 5, 9: no iron detected, text)
CRAIG_FE = {3: ((177.9, 473.1, 478.7, 546.2), (546.12, 0, 504.36, 0.2)), 4: ((179.2, 215.9, 481.3, 283.0), (282.96, 0, 259.8, 0.1)),
            10: ((167.2, 297.6, 483.4, 367.3), (367.2, 0, 326.16, 0.1)), 2: ((165.2, 199.9, 465.8, 261.5), (261.36, 0, 240.12, 0.1))}


def in_box(p, b):
    return b[0] < p[0] < b[2] and b[1] < p[1] < b[3]


def craig_series(doc, test):
    page_no, box, tx, lo, hi, brk, si_box, si_ax, has_markers = CRAIG_FIG[test]
    page = doc[page_no - 1]
    T = lin(*tx)
    val = lambda y: float(lin(*hi)(y)) if hi is not None and y < brk else float(lin(*lo)(y))
    grab = lambda c, b: [q for q in (markers(page, b, tuple(float(v) for v in c), (1.5, 4.0)) if has_markers and not
                         (test == 10 and c != (0, 0, 0)) else polyline(page, b, c, 0.25, 0.35)) if not any(in_box(q, k) for k in LEGEND[test])]
    wide = (box[0] - 0.1, box[1] - 0.1, box[2] + 0.1, box[3] + 0.05)
    base = grab((0, 0, 0), wide)
    times = np.array([float(T(x)) for x, _ in base])
    tol = 0.6 * (tx[3] - tx[1]) / (tx[2] - tx[0])  # 0.6 pt on the time axis

    def near(pts):  # value of the same sample in another series; no marker or vertex = on the zero line
        return np.array([min(((abs(t - k), v) for t, v in pts if abs(t - k) < tol), default=(0, 0.0))[1] for k in times])

    out = {"5-15": np.array([val(y) for _, y in base])}
    for ch, c in list(CH.items())[1:]:
        c = (1, 0, 1) if test == 2 and ch == ">100" else c
        pts = [(float(T(x)), val(y)) for x, y in grab(c, wide)]
        out[ch] = near(pts)
    si_pts = grab((0, 0, 0), (si_box[0] - 0.1, si_box[1], si_box[2] + 0.1, si_box[3]))
    T2 = lin(165.12, 71, 465.84, 79) if test == 2 else T
    si = [(float(T2(x)), float(lin(*si_ax)(y)) if not (test == 3 and y < 429.0) else float(lin(420.6, 18, 405.0, 21)(y))) for x, y in si_pts]
    out["SI"] = near(si)
    out["Fe"] = np.zeros(len(times))
    if test in CRAIG_FE:
        b, ax = CRAIG_FE[test]
        b = (b[0] - 0.1, b[1], b[2] + 0.1, b[3] + 0.05)
        out["Fe"] = near([(float(T2(x)), float(lin(*ax)(y))) for x, y in grab((0, 0, 0), b)])
    return times, {k: np.maximum(v, 0.0) for k, v in out.items()}


def integral(t, v, end):
    """Time integral 0..end: trapezoid between samples, constant before the first and after the last sample."""
    return float(v[0] * t[0] + np.trapezoid(v, t) + v[-1] * max(end - t[-1], 0.0))


def craig():
    sid = "soton_craig2010"
    doc = open_pdf(sid)
    w = table(CRAIG_W)
    for part in ("cup", "cone"):
        d = (w[f"{part}_pre"] - w[f"{part}_post"]) * 1000
        ok = ~w.test.eq(2) & (d > -10)
        check(sid, f"{part} mass difference: printed against weights (without Table E.1 and the mistyped cup of Table 6.3)",
              w[f"{part}_mg"][ok], d[ok], "recomputed from pre and post weights")
        w[f"{part}_d"] = np.where(d < -10, w[f"{part}_mg"], d)  # an impossible gain is a typing error of a weight -> printed value
    unreliable = w.test.eq(2)  # Table E.1: cup columns repeat the post weights of Table 5.2, bearing 4 cone gains 87 mg
    tot = {int(t): float(np.nansum(g[["cup_d", "cone_d"]].to_numpy())) for t, g in w[~unreliable].groupby("test")}
    tot_printed = {int(t): float(np.nansum(g[["cup_mg", "cone_mg"]].to_numpy())) for t, g in w[~unreliable].groupby("test")}
    tot[2] = tot_printed[2] = 5.73 + 459.62  # cones 1 and 2 only
    S = {t: craig_series(doc, t) for t in CRAIG_FIG}
    d_mid = {"5-15": math.sqrt(75), "15-25": math.sqrt(375), "25-50": math.sqrt(1250), "50-100": math.sqrt(5000), ">100": 126.0}
    v_um3 = {k: 0.3 * (math.pi * d * d / 4) ** 1.5 for k, d in d_mid.items()}  # the paper's volume prior
    prof = [{"test": t, "time_h": tt, **{k: float(v[i]) for k, v in s.items()}} for t, (tm, s) in S.items() for i, tt in enumerate(tm)]
    pd.DataFrame(prof).to_csv(DATA / sid / "oil_samples.csv", index=False, lineterminator="\n", float_format="%.6g")
    pods = {1: (5250, 5250), 3: (7500, 105000), 5: (60000,), 9: (15000,), 10: (5100,), 2: (30000,), 4: (6500,)}
    check(sid, "largest 5-15 um count per test: vector against the author's tables and text", [S[t][1]["5-15"].max() for t in pods],
          [max(v) for v in pods.values()], "printed (rounded) values, Tables 7.1-7.3")
    check(sid, "largest iron concentration (ppm): vector against Tables 7.2-7.3", [S[t][1]["Fe"].max() for t in (2, 3, 4, 10)],
          [0.2, 0.3, 0.1, 0.1], "printed")
    check(sid, "test 1: counts at 1.5, 3, 5.5, 10 h (5-15 um) against the text of p. 89", S[1][1]["5-15"][:4], [5250, 2700, 2500, 1500],
          "printed (rounded) values")
    tests = {"T1": (1,), "T3": (3,), "T4": (4,), "T5": (5,), "T6": (6,), "T9+10": (9, 10), "T2": (2,)}
    rows = []
    for name, parts in tests.items():
        last = parts[-1]
        I = {k: sum(integral(S[p][0], S[p][1][k], max(CRAIG_END[p], S[p][0][-1])) for p in parts) for k in [*CH, "SI", "Fe"]}
        hours = sum(max(CRAIG_END[p], S[p][0][-1]) for p in parts)
        first = min(S[p][0][0] for p in parts)
        cover = sum(S[p][0][-1] - S[p][0][0] for p in parts) / hours
        press = {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.0, 10: 1.6}[last]
        rows.append({"set": "bearings", "unit": name, "row": name, "order": 1, "mode": "interval",
                     "rig": "Timken bearing test rig, 4 bearings, 2500 rpm, 4 l/min, 25 l sump, 5 um full-flow filter",
                     "material_pair": "deep groove ball bearings MM206K" if last == 10 else "tapered roller bearings LM67010/LM67048",
                     "lubricant": "Shell Vitrea 32", "cond_hours": hours, "cond_pressure_GPa": press, "debris": I["5-15"],
                     "debris_metric": "time integral of the off-line optical particle count 5-15 um (count x h)", "debris_unit": "count h",
                     "debris_type": "count", "wear": tot[last], "wear_metric": "mass loss of the four bearings (cups and cones)",
                     "wear_unit": "mg", "resolved": sum(I[k] * v_um3[k] for k in CH), **{f"class_int_{k}": I[k] for k in CH},
                     "class_fe_ppm_h": I["Fe"], "class_scored": int(cover >= 0.5), "class_coverage": cover, "alt_debris_si_integral": I["SI"],
                     "alt_debris_total_count": sum(I[k] for k in CH), "alt_debris_author_max": CRAIG_MAXCOUNT.get(last, np.nan),
                     "alt_wear_failed_element": CRAIG_FAILED.get(last) if CRAIG_FAILED.get(last) else np.nan,
                     "alt_wear_printed": tot_printed[last], "debris_2": np.nan, "wear_2": tot_printed[last], "debris_err": 0.02,
                     "wear_err": rel(tot[last], tot_printed[last]), "page": ", ".join(str(CRAIG_FIG[p][0]) for p in parts) +
                     "; gravimetry " + {1: "109", 2: "218", 3: "120", 4: "225", 5: "135", 6: "231", 10: "156"}[last],
                     "item": "off-line PODS figure; gravimetric table", "extraction": "vector (counts); text (weights)",
                     "note": f"first sample {first:.1f} h, samples span {cover:.2f} of the test" + ("; Table E.1 unusable except cones 1-2"
                             if name == "T2" else "") + ("; Table 6.3 cup weight of bearing 1 mistyped, printed difference used" if last == 10 else "")
                     + ("; Table F.1 printed differences of bearing 1 are 10x smaller than its weights" if last == 4 else "")})
    df = pd.DataFrame(rows)
    df.loc[df.unit == "T2", "class_scored"] = 0
    save(sid, df, {"bearings": {"conditions": [["cond_hours", "log10"], ["cond_pressure_GPa", "id"]],
                                "resolved_def": "size-resolved: sum over five channels of the count integral x 0.3 A^1.5 (mid diameter)",
                                "exclude_primary": ["T2"], "sensitivity": {
                                    "with_test_2": {"include": ["T2"]},
                                    "author_numbers": {"debris": "alt_debris_author_max", "wear": "alt_wear_failed_element", "include": ["T2"]},
                                    "failed_element_mass": {"wear": "alt_wear_failed_element"},
                                    "severity_index_integral": {"debris": "alt_debris_si_integral"},
                                    "total_count_integral": {"debris": "alt_debris_total_count"},
                                    "printed_differences": {"wear": "alt_wear_printed"}}}})
    for t, (tm, s) in S.items():
        print(f"   test {t}: {len(tm)} samples, 5-15 um {s['5-15'].round(0).astype(int).tolist()}, hours {np.round(tm, 1).tolist()}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (feup, raadnui, higashi, nasa_bearing, torres, jordan, yoshinaga, quinn, bolander, craig):
        fn()
    pd.DataFrame(CHECK).to_csv(OUT / "digitisation_check.csv", index=False, lineterminator="\n", float_format="%.5g")
    print(pd.DataFrame(CHECK)[["study", "quantity", "n_read_twice", "rms_diff", "rms_log10", "max_abs_log10"]].to_string())


if __name__ == "__main__":
    main()
