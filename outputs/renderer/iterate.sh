#!/usr/bin/env bash
# One tuning iteration: smoke config + renderer v2 -> scratch dataset, trainO-only score (frozen functions), montage.
# Usage: bash outputs/renderer/iterate.sh <tag> <scratch_dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="python"
TAG=$1; D=$2
rm -rf "$D"
"$PY" -m wearpg synth --config smoke --out "$D" --set imaging.renderer=v2 > /dev/null 2>&1
"$PY" -I outputs/renderer/score_renderer.py --syn "$D" --tag "$TAG" --tune 2>&1 | grep -v "^gap" | tail -6
"$PY" - "$TAG" "$D" << 'EOF'
import sys
import numpy as np, pandas as pd
from PIL import Image
tag, d = sys.argv[1], sys.argv[2]
o = f"outputs/renderer/{tag}"
a = pd.read_csv(f"{o}/tune_auc.csv")
print(a.pivot_table(index="b", columns="group", values="auc").round(3).to_string())
k = pd.read_csv(f"{o}/tune_ks.csv")
k = k[k.feature.isin(["aspect", "roundness", "solidity"])]
print(k.pivot_table(index="b", columns="feature", values="ks_D").round(3).to_string())
f = pd.read_csv(f"{o}/tune_features.csv")
g = f[(f.b == "synthetic_chain_fe") & f.group.isin(["texture", "colour", "shape_scale_free"])]
print(g.sort_values("effect", ascending=False).head(12)[["feature", "auc_univariate", "median_a", "median_b"]]
      .round(3).to_string(index=False))
ims = pd.read_csv(f"{d}/images.csv"); fov = ims[ims.modality == "ferrograph_optical"].set_index("image_id")
cnt = pd.read_csv(f"{d}/particles.csv").groupby("image_id").size().sort_values(ascending=False)
pick = list(cnt.index[:3]) + list(cnt.index[20:23])
t = [np.asarray(Image.open(f"{d}/{fov.loc[i, 'path']}").convert("RGB")) for i in pick]
Image.fromarray(np.concatenate([np.concatenate(t[:3], 1), np.concatenate(t[3:], 1)], 0)).save(f"{d}_montage.png")
EOF
