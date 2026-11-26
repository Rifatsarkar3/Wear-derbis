"""Renderer calibration statistics from Zenodo **trainO** images + GT masks only (testO / TCBGY never used).

Per image: background = pixels farther than 4 px from the GT mask; its median RGB / luma, fine-scale residual SD
(luma minus a sigma-6 Gaussian blur), dark-speckle density (residual < -0.06); inside the GT mask: luma quantiles
relative to the background luma, fraction darker (< 0.8 bg) / brighter (> 1.1 bg), mean RGB of the dark and of
the bright pixels relative to the background RGB; component orientation (mask moments) relative to the image axes.
Writes outputs/renderer/calibration_trainO.csv (per image) and calibration_trainO.json (quantiles).
Run: python -I outputs/renderer/calibrate_trainO.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label, regionprops

from wearpg.open_data import load_zenodo_ferro, read_mask
from wearpg.segmentation import load_image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "renderer"
LUMA = np.array([0.2125, 0.7154, 0.0721])


def stats(r) -> dict:
    img = load_image(r.image_path).astype(float) / 255
    m = read_mask(r.mask_path)
    g = img @ LUMA
    far = ndimage.distance_transform_edt(~m) > 4
    bg_rgb = np.median(img[far], axis=0)
    bg = float(bg_rgb @ LUMA)
    res = g - ndimage.gaussian_filter(g, 6)
    out = dict(image_id=r.image_id, bg_r=bg_rgb[0], bg_g=bg_rgb[1], bg_b=bg_rgb[2], bg_luma=bg,
               bg_res_sd=float(res[far].std()), bg_speckle=float((res[far] < -0.06).mean()),
               fg_fraction=float(m.mean()))
    if m.sum() > 50:
        rel = g[m] / max(bg, 1e-3)
        out.update({f"fg_rel_q{q}": float(np.quantile(rel, q / 100)) for q in (10, 25, 50, 75, 90)})
        dark, bright = rel < 0.8, rel > 1.1
        out.update(fg_dark=float(dark.mean()), fg_bright=float(bright.mean()))
        for name, sel in (("dark", dark), ("bright", bright)):
            if sel.sum() > 10:
                c = img[m][sel].mean(axis=0) / np.maximum(bg_rgb, 1e-3)
                out.update({f"{name}_rel_r": c[0], f"{name}_rel_g": c[1], f"{name}_rel_b": c[2]})
        props = [p for p in regionprops(label(m, connectivity=1)) if p.area >= 30 and p.axis_major_length > 0]
        if props:
            w = np.array([p.area for p in props], float)
            ang = np.array([p.orientation for p in props])  # rad, angle between row axis and major axis
            out["orient_vertical_frac"] = float(np.sum(w * (np.abs(np.sin(ang)) < np.sin(np.radians(20)))) / w.sum())
            out["comp_width_med"] = float(np.median([p.axis_minor_length for p in props]))
    return out


def main() -> None:
    idx = load_zenodo_ferro(ROOT / "data" / "real" / "zenodo_ferro" / "extracted")
    idx = idx[idx.split == "trainO"]
    df = pd.DataFrame([stats(r) for r in idx.itertuples()])
    df.to_csv(OUT / "calibration_trainO.csv", index=False)
    q = {c: [float(x) for x in df[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9])] for c in df.columns if c != "image_id"}
    json.dump({"n_images": len(df), "quantiles_10_25_50_75_90": q}, open(OUT / "calibration_trainO.json", "w",
                                                                          encoding="utf-8"), indent=1)
    print(pd.DataFrame(q, index=[10, 25, 50, 75, 90]).T.round(3).to_string())


if __name__ == "__main__":
    main()
