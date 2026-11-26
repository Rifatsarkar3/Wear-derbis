"""Scale bars burnt into the Zenodo 22997504 ferrograms (POST HOC, after an audit: the manuscript had said that no pixel size
is given). Finds the red bar with its label in every image, reads 10 or 20 um from the first digit (template), gives the
pixel size, and re-scores the classical segmenters on the test images with the annotation (bar and label) left out, because
a threshold marks the red characters as debris.

    .venv/Scripts/python.exe -I scripts/open_ferro_scalebar.py      # -> outputs/open_data/ferro/scalebar/{images.csv, summary.json}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from open_ferro_eval import DATA, methods, predict  # noqa: E402
from wearpg.metrics import volume_consistency  # noqa: E402
from wearpg.open_data import load_zenodo_ferro, pixel_scores, read_mask  # noqa: E402
from wearpg.segmentation import load_image  # noqa: E402
from wearpg.stats import cluster_bootstrap  # noqa: E402
from wearpg.utils import load_json  # noqa: E402

FER = ROOT / "outputs" / "open_data" / "ferro"
OUT = FER / "scalebar"
MIN_BAR, MAX_BAR = 50, 75                                    # px; the bars are about 60 px long


def red_mask(rgb):
    a = np.asarray(rgb).astype(int)
    return (a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] < 110) & (a[..., 0] - a[..., 1] > 80)


def find_bar(red):
    """(row, start, end) of the longest horizontal red run of bar length, or None."""
    best = None
    for y in np.where(red.any(1))[0]:
        d = np.diff(np.r_[0, red[y].astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if MIN_BAR <= e - s <= MAX_BAR and (best is None or e - s > best[2] - best[1]):
                best = (int(y), int(s), int(e))
    return best


def digit(red, bar):
    """The first character below the bar, 16 x 10 px, for the 10 / 20 decision."""
    y, s, e = bar
    box = red[y + 6:y + 42, max(0, s - 10):e + 10]
    lab, n = ndi.label(box)
    if n == 0:
        return None
    objs = [o for o in ndi.find_objects(lab) if (o[0].stop - o[0].start) >= 10]
    if not objs:
        return None
    o = min(objs, key=lambda o: o[1].start)
    c = box[o].astype(float)
    return np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((10, 16))) / 255.0


def main():
    idx = load_zenodo_ferro(DATA).reset_index(drop=True)
    rows, digits = [], {}
    for r in idx.itertuples():
        red = red_mask(Image.open(r.image_path).convert("RGB"))
        bar = find_bar(red)
        rows.append({"image_id": r.image_id, "split": r.split, "bar_px": bar[2] - bar[1] if bar else np.nan,
                     "bar_row": bar[0] if bar else np.nan, "bar_col": bar[1] if bar else np.nan})
        if bar:
            digits[r.image_id] = digit(red, bar)
    d = pd.DataFrame(rows)
    ref1 = digits.get("trainO_1"), digits.get("trainO_10")   # labelled "10 um" (checked by eye)
    ref2 = digits.get("trainO_100"), digits.get("trainO_102")  # labelled "20 um" (checked by eye)
    t1 = np.mean([x for x in ref1 if x is not None], axis=0)
    t2 = np.mean([x for x in ref2 if x is not None], axis=0)

    def value(i):
        g = digits.get(i)
        if g is None:
            return np.nan
        return 10 if np.abs(g - t1).sum() < np.abs(g - t2).sum() else 20
    d["um"] = [value(i) if i in digits else np.nan for i in d.image_id]
    d["um_per_px"] = d.um / d.bar_px
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "images.csv", index=False)

    ms = methods(load_json(FER / "seg_tuned_params.json"))
    test = idx[idx.split == "testO"]
    out = []
    for r in test.itertuples():
        img, gt = load_image(r.image_path), read_mask(r.mask_path)
        red = red_mask(Image.open(r.image_path).convert("RGB"))
        ann = np.zeros_like(gt)
        b = d.loc[d.image_id == r.image_id].iloc[0]
        if np.isfinite(b.bar_px):
            y0, x0, x1 = int(b.bar_row), int(b.bar_col), int(b.bar_col + b.bar_px)
            ann[max(0, y0 - 6):y0 + 46, max(0, x0 - 14):x1 + 14] = True
            ann &= ndi.binary_dilation(red, iterations=3) | ann      # the box, plus red pixels around it
        for m in ("ours_tuned", "local_tuned", "otsu"):
            pred, _ = predict(img, *ms[m])
            for tag, keep in (("as_scored", np.ones_like(gt, bool)), ("without_annotation", ~ann)):
                out.append({"image_id": r.image_id, "method": m, "scoring": tag, "has_bar": bool(np.isfinite(b.bar_px)),
                            **pixel_scores(pred & keep, gt & keep)})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "per_image_test.csv", index=False)
    summ = {"status": "post hoc, after an audit", "n_images": len(d), "n_with_bar": int(d.bar_px.notna().sum()),
            "n_with_bar_by_split": d[d.bar_px.notna()].groupby("split").size().to_dict(),
            "n_10um": int((d.um == 10).sum()), "n_20um": int((d.um == 20).sum()),
            "um_per_px_by_label": d.groupby("um").um_per_px.median().round(4).to_dict(),
            "bar_px_median": float(d.bar_px.median())}

    def met(g):
        lc = volume_consistency(np.log(g.area_pred.clip(lower=1)), np.log(g.area_gt))
        return {"iou": g.iou.mean(), "ccc_log_area": lc["ccc"], "abs_area_rel_err": g.area_rel_err.abs().mean(),
                "bias": g.area_rel_err.mean()}
    for (m, tag), g in s.groupby(["method", "scoring"]):
        summ[f"{m}|{tag}"] = {k: {"point": v["point"], "lo": v["lo"], "hi": v["hi"]}
                              for k, v in cluster_bootstrap(met, g, "image_id", n_boot=2000, seed=0).items()
                              if isinstance(v, dict) and "point" in v}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict) or "|" not in k}, indent=1))
    for k, v in summ.items():
        if "|" in k:
            print(k, {a: f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]" for a, b in v.items()})


if __name__ == "__main__":
    main()
