"""Write data/real/paired/{jet495,tribint2009,ilt2011,pij2011}/{paired.csv, study.json} from reading 1 and reading 2 (plan part 5 of PAIRED.md).
Printed data labels are used as they are; read values are the geometric mean of the two readings when they agree within 10 %, else reading 1 is kept and the discrepancy is
listed (no print exists for these curves).

    python -I outputs/open_data/paired/holdout/build_holdout_sources.py <reading2.csv>
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "data" / "real" / "paired"
R1 = json.loads((HERE / "reading1_part5.json").read_text(encoding="utf-8"))
R2 = {}
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    R2[(r["source"], r["series"], r["point"], r["quantity"])] = float(r["value"])

RIGHTS = "Publisher copy supplied by the user; numbers read with citation only, the PDF and its figures are not redistributed."
CITE = {
    "jet495": ("Cardoso N.F.R., Martins R.C., Seabra J.H.O. (2009). Micropitting of carburized gears lubricated with biodegradable low-toxicity oils. Proc. IMechE Part J 223(3), 481-495.", "10.1243/13506501JET495"),
    "tribint2009": ("Cardoso N.F.R., Martins R.C., Seabra J.H.O., Igartua A., Rodriguez J.C., Luther R. (2009). Micropitting performance of nitrided steel gears lubricated with mineral and ester oils. Tribology International 42(1), 77-87.", "10.1016/j.triboint.2008.05.010"),
    "ilt2011": ("Martins R., Locatelli C., Seabra J. (2011). Evolution of tooth flank roughness during gear micropitting tests. Industrial Lubrication and Tribology 63(1), 34-45.", "10.1108/00368791111101821"),
    "pij2011": ("Martins R.C., Seabra J.H.O., Ruis-Moron L.F. (2011). Influence of oil formulation on gear micropitting and power loss performance. Proc. IMechE Part J 225(6), 429-439.", "10.1177/1350650110397260"),
}
LOG = []


def both(source, series, point, quantity, v1):
    """Geometric mean of the two readings of one curve value (cumulative value as plotted)."""
    v2 = R2.get((source, series, point, quantity))
    if v2 is None:
        LOG.append(f"{source} {series} {point} {quantity}: no second reading, reading 1 kept")
        return v1, float("nan"), float("nan")
    rel = abs(math.log10(v2 / v1)) if v1 > 0 and v2 > 0 else 9
    ok = rel <= math.log10(1.10)
    if not ok:
        LOG.append(f"{source} {series} {point} {quantity}: {v1:.2f} and {v2:.2f} differ by more than 10 %, reading 1 kept")
    return (math.sqrt(v1 * v2) if ok else v1), v2, 100 * (v2 / v1 - 1)


def row(study, unit, label, mode, mat, lub, cyc, hz, ester, d, d1, d2, w, w2, wprint, page, item, note, order):
    return {"study": study, "set": "fzg_new", "unit": unit, "row": f"{unit}:{label}", "order": order, "mode": mode, "rig": "FZG gear test rig, type C gears, dip lubrication",
            "material_pair": mat, "lubricant": lub, "cond_cycles_M": cyc, "cond_hertz_MPa": hz, "cond_ester": ester, "cond_adi": 0, "debris": round(d, 3),
            "debris_metric": "direct-reading ferrometer CPUC = (DL + DS)/d, increment between stage ends", "debris_unit": "index", "debris_type": "index", "wear": round(w, 3),
            "wear_metric": "weighed gear pair mass loss (pinion + wheel)", "wear_unit": "mg", "resolved": "", "debris_2": "" if d2 != d2 else round(d2, 3), "wear_2": "" if w2 != w2 else round(w2, 3),
            "debris_err": "" if d1 != d1 or d2 != d2 or d1 <= 0 or d2 <= 0 else round(abs(math.log10(d2 / d1)), 4), "wear_err": "", "page": page, "item": item, "extraction": "vector path (reading 1), raster (reading 2)",
            "note": note, "alt_wear_printed": wprint}


def stages(source, oil, cum1, cumW, mat, lub, ester, page, item, labels, wlabel, hz, cyc, note):
    """Three stages of a micropitting test: increments of the cumulative CPUC; W from the printed labels."""
    out = []
    v = []
    for k, st in enumerate(("K6", "K8", "K9")):
        v.append(both(source, oil, st, "cpuc", cum1[k + 1]))
    prev, prev1, prev2 = 0.0, 0.0, 0.0
    for k, st in enumerate(labels):
        d, r2, _ = v[k]
        d1 = cum1[k + 1]
        inc, inc1 = d - prev, d1 - prev1
        inc2 = r2 - prev2 if r2 == r2 else float("nan")
        out.append(row(source, oil, st, "interval", mat, lub, cyc[k], hz[k], ester, inc, inc1, inc2, wlabel[k], float("nan"), wlabel[k], page, item, note, k + 1))
        prev, prev1 = d, d1
        prev2 = r2 if r2 == r2 else float("nan")
    return out


def write(sid, rows, sets, page, extra=""):
    d = DATA / sid
    d.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with open(d / "paired.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    cite, doi = CITE[sid]
    spec = {"id": sid, "citation": cite, "doi": doi, "rights": RIGHTS, "spec_date": "2026-11-25", "added_after_first_run": True, "holdout": "fzg", "holdout_sets": ["fzg_new"],
            "sets": {"fzg_new": {"conditions": [["cond_cycles_M", "log10"], ["cond_hertz_MPa", "log10"], ["cond_ester", "id"], ["cond_adi", "id"]], "sensitivity": {}}}, "note": extra}
    (d / "study.json").write_text(json.dumps(spec, indent=1), encoding="utf-8", newline="\n")
    print(sid, len(rows), "rows")


c = R1["cumulative"]
W_JET = {"M1": [3, 10, 5], "E1": [23, 21, 26], "E2": [13, 7, 31]}
W_TRI = {"M1": [5, 10, 27], "E1": [9, 14, 41], "E2": [7, 12, 25]}
HZ, CY = (953, 1256, 1408), (1.53, 1.44, 1.44)
LAB = ("K3+K6", "K8", "K9")
NOTE3 = "K3 running-in and K6 share one oil fill and one weighing (both columns); fresh oil after every weighing; CPUC increments of the cumulative curve"
jet, tri = [], []
for o in ("M1", "E1", "E2"):
    jet += stages("jet495", o, c["jet495"][o], None, "20MnCr5 carburised gear pair", "mineral oil" if o == "M1" else "ester oil", int(o != "M1"), "488-489", "Fig. 5 (labels), Fig. 7", LAB, W_JET[o], HZ, CY, NOTE3)
    tri += stages("tribint2009", o, c["tribint2009"][o], None, "34CrNiMo6 nitrided gear pair", "mineral oil" if o == "M1" else "ester oil", int(o != "M1"), "82", "Fig. 7 (labels), Fig. 9", LAB, W_TRI[o], HZ, CY, NOTE3)
write("jet495", jet, ["fzg_new"], "488-489")
write("tribint2009", tri, ["fzg_new"], "82")

ilt, cc, ww = [], c["ilt2011"]["cpuc"], c["ilt2011"]["mass_mg"]
HZI, CYI, LI = (514, 953, 1256, 1408), (0.09, 1.44, 1.44, 1.44), ("K3", "K6", "K8", "K9")
pd, pw, pd1, pw1, pd2, pw2 = 0, 0, 0, 0, 0, 0
for k, st in enumerate(LI):
    d, r2, _ = both("ilt2011", "gear", st, "cpuc", cc[k])
    w, s2, _ = both("ilt2011", "gear", st, "mass_mg", ww[k])
    ilt.append(row("ilt2011", "E2", st, "interval", "20MnCr5 carburised gear pair", "ester oil", CYI[k], HZI[k], 1, d - pd, cc[k] - pd1, (r2 - pd2) if r2 == r2 else float("nan"), w - pw, (s2 - pw2) if s2 == s2 else float("nan"), "",
                   "39", "Fig. 6 (mass loss of the gear against CPUC)", "one gear pair, interrupted test; fresh oil at the first period of each stage; gear (pinion + wheel) cumulative loss from Fig. 6 (labels of Fig. 4 are the pinion only)", k + 1))
    pd, pw, pd1, pw1 = d, w, cc[k], ww[k]
    pd2, pw2 = r2, s2
write("ilt2011", ilt, ["fzg_new"], "39")

pij, WP = [], {"P1": 16, "P4": 11}
for o in ("P1", "P4"):
    vals = [both("pij2011", o, f"t{t}", "cpuc", c["pij2011"][o][k]) for k, t in enumerate((12, 24, 36, 48))]
    dfin = vals[3][0] + 40 / 1500 * sum(v[0] for v in vals[:3])
    d1 = c["pij2011"][o][3] + 40 / 1500 * sum(c["pij2011"][o][:3])
    d2 = vals[3][1] + 40 / 1500 * sum(v[1] for v in vals[:3]) if all(v[1] == v[1] for v in vals) else float("nan")
    pij.append(row("pij2011", o, "end", "cumulative", "20MnCr5 carburised gear pair", "PAO + ester oil (10 % ester)", 2.52, 1500, 0, dfin, d1, d2, WP[o], float("nan"), WP[o], "435-436", "Fig. 6 (labels), Fig. 7",
                   "power-loss test, 1.5 l, one oil fill, four samples of 40 ml (12, 24, 36, 48 h); D = last CPUC + 40/1500 x sum of the earlier ones; P3: mass loss 0 (unmeasurable), excluded; base oil counted as not ester", 1))
write("pij2011", pij, ["fzg_new"], "435-436", "P3 excluded (no measurable mass loss); P2 not tested (value from earlier work only).")
for m in LOG:
    print("NOTE", m)
