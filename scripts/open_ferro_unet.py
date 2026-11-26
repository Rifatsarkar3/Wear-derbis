"""POST-HOC learned baseline on the real ferrogram benchmark (Zenodo 22997504): a U-Net trained with and without the
training images that leak into the test split. Plan and verdict rules: outputs/agent_logs/UNET.md (written first).

Stages: train (one checkpoint-free run per arm and seed; each scored once on testO) -> unet/per_image.csv
        summarise -> unet/summary.json, unet/paired.csv (bootstrap over images, 2000 replicates, Holm)

    .venv/Scripts/python.exe scripts/open_ferro_unet.py              # ~30 min on the RTX 5070
    .venv/Scripts/python.exe scripts/open_ferro_unet.py --smoke      # 1 seed, 2 epochs, writes nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from open_ferro_eval import DATA, MIN_AREA, summary_metrics  # noqa: E402
from wearpg.metrics import volume_consistency  # noqa: E402
from wearpg.open_data import load_zenodo_ferro, mask_components, pixel_scores, read_mask  # noqa: E402
from wearpg.segmentation import load_image, segmentation_quality  # noqa: E402
from wearpg.stats import cluster_bootstrap, holm, paired_cluster_bootstrap  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

FER = ROOT / "outputs" / "open_data" / "ferro"
OUT = FER / "unet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, EPOCHS, BATCH, BASE, LR, VAL = (0, 1, 2), 60, 8, 32, 1e-3, 0.15


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, base=BASE, depth=4):
        super().__init__()
        w = [base * 2 ** k for k in range(depth + 1)]
        self.down = nn.ModuleList([Block(3, w[0])] + [Block(w[k], w[k + 1]) for k in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[k + 1], w[k], 2, 2) for k in range(depth)])
        self.dec = nn.ModuleList([Block(2 * w[k], w[k]) for k in range(depth)])
        self.out = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x if k == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        for k in reversed(range(len(self.up))):
            x = self.dec[k](torch.cat([self.up[k](x), skips[k]], 1))
        return self.out(x)


def load_all(idx):
    img = np.stack([np.asarray(load_image(p))[..., :3] for p in idx.image_path])
    img = (img * 255).round().astype(np.uint8) if img.dtype != np.uint8 else img
    msk = np.stack([read_mask(p) for p in idx.mask_path])
    return (torch.from_numpy(img).permute(0, 3, 1, 2).to(DEV), torch.from_numpy(msk)[:, None].to(DEV))


def augment(x, y, g):
    for k in range(x.shape[0]):
        r = int(torch.randint(0, 4, (1,), generator=g))
        x[k], y[k] = torch.rot90(x[k], r, (1, 2)), torch.rot90(y[k], r, (1, 2))
        if torch.rand(1, generator=g) < 0.5:
            x[k], y[k] = x[k].flip(2), y[k].flip(2)
    gain = 1 + 0.4 * (torch.rand(x.shape[0], 1, 1, 1, generator=g) - 0.5).to(DEV)
    bias = 0.2 * (torch.rand(x.shape[0], 1, 1, 1, generator=g) - 0.5).to(DEV)
    return (x * gain + bias).clamp(0, 1), y


def loss_fn(z, y):
    p = torch.sigmoid(z)
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(z, y) + dice.mean()


@torch.no_grad()
def predict(net, img, idx):
    net.eval()
    out = []
    for i in range(0, len(idx), BATCH):
        with torch.autocast(DEV, torch.float16, enabled=DEV == "cuda"):
            out.append(net(img[idx[i:i + BATCH]].float() / 255).sigmoid().float().cpu() > 0.5)
    return torch.cat(out)[:, 0].numpy()


def iou(pred, gt):
    inter, union = (pred & gt).sum((1, 2)), (pred | gt).sum((1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def drop_small(m):
    lab, n = ndi.label(m)
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= MIN_AREA
    keep[0] = False
    return keep[lab]


def train(img, msk, tr, seed, epochs):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    perm = np.random.default_rng(seed).permutation(tr)
    n_val = max(1, round(VAL * len(perm)))
    val, fit = perm[:n_val], perm[n_val:]
    net = UNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), LR)
    steps = epochs * (len(fit) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler(DEV, enabled=DEV == "cuda")
    gt_val = msk[val][:, 0].cpu().numpy()
    best, state = -1.0, None
    for ep in range(epochs):
        net.train()
        order = torch.randperm(len(fit), generator=g).numpy()
        for i in range(0, len(fit) - BATCH + 1, BATCH):
            j = fit[order[i:i + BATCH]]
            x, y = augment(img[j].float() / 255, msk[j].float(), g)
            with torch.autocast(DEV, torch.float16, enabled=DEV == "cuda"):
                z = net(x)
            loss = loss_fn(z.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        v = iou(predict(net, img, val), gt_val).mean()
        if v > best:
            best, state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    return net, float(best), len(fit), len(val)


def score(pred, idx_row):
    gt = read_mask(idx_row.mask_path)
    m = drop_small(pred)
    gcomp = mask_components(gt, MIN_AREA)
    q = segmentation_quality([c["contour"] for c in mask_components(m, MIN_AREA)], [c["contour"] for c in gcomp], gt.shape)
    return {"image_id": idx_row.image_id, **pixel_scores(m, gt), "precision": q["precision"], "recall": q["recall"],
            "f1": q["f1"], "n_pred": q["n_pred"], "n_gt": q["n_gt"],
            "count_rel_err": (q["n_pred"] - q["n_gt"]) / q["n_gt"] if q["n_gt"] else np.nan}


def run(smoke=False):
    idx = load_zenodo_ferro(DATA).reset_index(drop=True)
    tr_all = idx.index[idx.split == "trainO"].to_numpy()
    te = idx.index[idx.split == "testO"].to_numpy()
    leak = pd.read_csv(FER / "seg_leak_check.csv")
    leaky = set(leak.loc[leak.near_dup, "best_train_image"])
    assert int(leak.near_dup.sum()) == 6 and len(te) == 86 and len(tr_all) == 343, (len(te), len(tr_all))
    clean = np.array([i for i in tr_all if idx.image_id[i] not in leaky])
    print(f"device {DEV}; train {len(tr_all)} / clean {len(clean)}; leaky training images {sorted(leaky)}", flush=True)
    img, msk = load_all(idx)
    rows = []
    for arm, tr in (("FULL", tr_all), ("CLEAN", clean)):
        for seed in (SEEDS[:1] if smoke else SEEDS):
            t0 = time.time()
            net, v, n_fit, n_val = train(img, msk, tr, seed, 2 if smoke else EPOCHS)
            pred = predict(net, img, te)
            for p, i in zip(pred, te):
                rows.append({"arm": arm, "seed": seed, **score(p, idx.loc[i])})
            df = pd.DataFrame(rows)
            m = df[(df.arm == arm) & (df.seed == seed)]
            print(f"{arm} seed {seed}: val IoU {v:.3f} ({n_fit}/{n_val}), test IoU {m.iou.mean():.3f}, {time.time() - t0:.0f} s",
                  flush=True)
            if not smoke:
                OUT.mkdir(parents=True, exist_ok=True)
                df.to_csv(OUT / "per_image.csv", index=False)
            del net
    return pd.DataFrame(rows)


def single_net_metrics(d):
    """Metrics of a single network, averaged over the seeds (never areas averaged over networks: errors would cancel)."""
    ccc = [volume_consistency(np.log(g.area_pred.clip(lower=1)), np.log(g.area_gt))["ccc"] for _, g in d.groupby("seed")]
    return {"iou": d.iou.mean(), "abs": d.area_rel_err.abs().mean(), "ccc": float(np.mean(ccc)), "f1": d.f1.mean(),
            "bias": d.area_rel_err.mean(), "count_rel_err": d.count_rel_err.mean()}


def summarise():
    df = pd.read_csv(OUT / "per_image.csv")
    leak = pd.read_csv(FER / "seg_leak_check.csv").set_index("test_image").near_dup
    test = pd.read_csv(FER / "seg_per_image_test.csv")
    df["near_dup"] = df.image_id.map(leak).astype(bool)
    df["abs_err"] = df.area_rel_err.abs()
    test["near_dup"] = test.image_id.map(leak).astype(bool)
    test["abs_err"] = test.area_rel_err.abs()
    out = {"note": "POST-HOC learned baseline; plan outputs/agent_logs/UNET.md; metrics of a single network averaged over "
                   "three seeds (corrected after an audit: areas are no longer averaged over the networks); image = cluster",
           "n_seeds": int(df.seed.nunique()), "epochs": EPOCHS}
    for arm, g in df.groupby("arm"):
        for name, h in (("all86", g), ("no_dup80", g[~g.near_dup]), ("dup6", g[g.near_dup])):
            r = {"n": int(h.image_id.nunique()), **{k: float(v) for k, v in single_net_metrics(h).items()}}
            r["pooled_count_ratio"] = float(h.n_pred.sum() / h.n_gt.sum())
            if h.image_id.nunique() > 10:
                r["boot"] = cluster_bootstrap(single_net_metrics, h, "image_id", n_boot=2000, seed=0)
                r["ccc_log_area"], r["iou_mean"] = r["ccc"], r["iou"]
            out[f"{arm}_{name}"] = r
    avg = df.groupby(["arm", "image_id", "near_dup"], as_index=False)[["iou", "abs_err", "f1"]].mean()
    rows = []
    u = avg[(avg.arm == "CLEAN") & ~avg.near_dup]
    for tag in ("ours_tuned", "local_tuned", "otsu"):
        b = test[(test.method == tag) & ~test.near_dup]
        r = paired_cluster_bootstrap(lambda d: {"iou": d.iou.mean(), "abs_area_rel_err": d.abs_err.mean(), "f1": d.f1.mean()},
                                     u, b, "image_id", "image_id", n_boot=2000, seed=0)
        rows += [{"family": "Q1", "a": "unet_CLEAN", "b": tag, "metric": m, **r[m]} for m in r if m != "n_clusters"]
    w = avg.pivot(index="image_id", columns="arm", values="iou")
    d = (w.FULL - w.CLEAN).rename("d").reset_index()
    d["near_dup"] = d.image_id.map(leak).astype(bool)
    rng = np.random.default_rng(0)
    a, b = d[d.near_dup].d.to_numpy(), d[~d.near_dup].d.to_numpy()
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
    out["Q2_leak_effect_iou"] = {
        "point": float(a.mean() - b.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
        "dup_mean_gain": float(a.mean()), "other_mean_gain": float(b.mean()), "n_dup": len(a), "n_other": len(b),
        "mean_FULL_all86": float(w.FULL.mean()), "mean_CLEAN_all86": float(w.CLEAN.mean())}
    pr = pd.DataFrame(rows)
    pr["p_holm"] = pr.groupby("metric").p_value.transform(lambda p: holm(p.to_numpy()))
    pr.to_csv(OUT / "paired.csv", index=False)
    save_json(out, OUT / "summary.json")
    print(pr.to_string())
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "boot"},
                  {a: [round(c["point"], 3), round(c["lo"], 3), round(c["hi"], 3)] for a, c in v.get("boot", {}).items()
                   if isinstance(c, dict) and "lo" in c})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        print(run(smoke=True).groupby("arm").iou.mean())
    elif a.summarise:
        summarise()
    else:
        run()
        summarise()
