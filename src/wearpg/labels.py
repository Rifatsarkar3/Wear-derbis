"""Mechanism labels: inter-rater agreement, validity against reference raters, adjudication, soft labels.

`labels_df` is a `labels` table (particle_id, rater_id, label, ...). Reference raters (default
`schema.REFERENCE_RATERS` = ("SEMREF",), independent SEM/EDS assessment) are excluded from agreement,
adjudication and soft labels; they are only used by `validity_report`.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from wearpg.schema import MECHANISMS, REFERENCE_RATERS, coerce_table
from wearpg.utils import get_logger


def label_index(names, classes=MECHANISMS) -> np.ndarray:
    """Class index of each name (position in `classes`); unknown names -> -1. int64 (n,)."""
    return pd.Index(list(classes)).get_indexer(list(names)).astype(np.int64)


def cohens_kappa(a, b, labels=None) -> float:
    """Cohen's kappa of two aligned label sequences (sklearn). Pairs with a label outside `labels` are
    dropped. NaN when undefined (no pairs, or both raters use one single identical class)."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if labels is not None:
        keep = np.isin(a, list(labels)) & np.isin(b, list(labels))
        a, b = a[keep], b[keep]
    if a.size == 0 or len(set(a) | set(b)) < 2:  # expected agreement = 1 -> 0/0
        return float("nan")
    return float(cohen_kappa_score(a, b, labels=None if labels is None else list(labels)))


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from an (n_items, n_categories) table of rater counts per item and category.

    Every row must sum to the same number of raters n >= 2 (raters may differ between items).
    NaN when chance agreement is 1 (a single category used).
    """
    c = np.asarray(counts, dtype=float)
    n = c.sum(axis=1)
    if c.ndim != 2 or c.shape[0] == 0 or not np.allclose(n, n[0]) or n[0] < 2:
        raise ValueError("fleiss_kappa needs >= 1 item and the same number (>= 2) of raters per item")
    n = n[0]
    p_bar = ((c * (c - 1)).sum(axis=1) / (n * (n - 1))).mean()
    p_e = ((c.sum(axis=0) / c.sum()) ** 2).sum()
    return float("nan") if np.isclose(p_e, 1.0) else float((p_bar - p_e) / (1 - p_e))


def _counts(labels_df: pd.DataFrame, classes) -> pd.DataFrame:
    """(particles x classes) number of raters choosing each class."""
    return pd.crosstab(labels_df["particle_id"], labels_df["label"]).reindex(columns=list(classes), fill_value=0)


def _raters_only(labels_df: pd.DataFrame, classes, reference_raters) -> pd.DataFrame:
    return labels_df[~labels_df["rater_id"].isin(list(reference_raters)) & labels_df["label"].isin(list(classes))]


def krippendorff_alpha_nominal(labels_df) -> float:
    """Krippendorff's alpha (nominal) over all raters in `labels_df` (no rater filtering; handles missing
    ratings and varying rater counts). Units with < 2 ratings are not pairable. NaN if undefined.

    alpha = 1 - (n - 1) sum_{c != k} o_ck / sum_{c != k} n_c n_k   (coincidence matrix o, n pairable values)
    """
    cnt = pd.crosstab(labels_df["particle_id"], labels_df["label"]).to_numpy(float)
    m = cnt.sum(axis=1)
    cnt, m = cnt[m >= 2], m[m >= 2]
    n_c = cnt.sum(axis=0)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    o_diag = (cnt * (cnt - 1) / (m - 1)[:, None]).sum()  # sum_c o_cc; all coincidences sum to n
    d_e = n * n - (n_c**2).sum()
    return float("nan") if d_e == 0 else float(1.0 - (n - 1) * (n - o_diag) / d_e)


def agreement_report(labels_df, classes=MECHANISMS, reference_raters=REFERENCE_RATERS) -> dict:
    """Inter-rater agreement of the non-reference raters.

    Keys: n_items (particles with >= 1 rating), n_multi_rated (>= 2 raters), raters (sorted ids),
    pairwise_kappa {"A|B": Cohen kappa on the pair's common items} (pairs with common items only),
    kappa (Cohen on the two raters' common items if exactly 2 raters; Fleiss on the multi-rated items with
    the modal rater count (ties -> larger count) if >= 3; NaN if < 2), kappa_method ("cohen"|"fleiss"|"none"),
    n_kappa_items, krippendorff_alpha, per_class_agreement {cls: proportion of specific agreement
    sum_i n_ic (n_ic - 1) / sum_i n_ic (m_i - 1) over multi-rated items; NaN if the class never occurs},
    rater_confusion (C x C nested list of int, rows/cols in `classes` order: sum over rater pairs r1 < r2
    of the confusion r1 label (row) vs r2 label (col) on their common items).
    """
    lab = _raters_only(labels_df, classes, reference_raters)
    raters = sorted(lab["rater_id"].unique().tolist())
    wide = lab.pivot(index="particle_id", columns="rater_id", values="label")
    cnt = _counts(lab, classes)
    m = cnt.sum(axis=1).to_numpy()

    pairwise, conf = {}, np.zeros((len(classes), len(classes)), dtype=np.int64)
    for r1, r2 in combinations(raters, 2):
        both = wide[[r1, r2]].dropna()
        if len(both):
            pairwise[f"{r1}|{r2}"] = cohens_kappa(both[r1], both[r2], labels=classes)
            conf += confusion_matrix(both[r1], both[r2], labels=list(classes))

    kappa, method, n_kappa = float("nan"), "none", 0
    if len(raters) == 2:
        kappa, method, n_kappa = pairwise.get(f"{raters[0]}|{raters[1]}", float("nan")), "cohen", int((m >= 2).sum())
    elif len(raters) >= 3 and (m >= 2).any():
        freq = pd.Series(m[m >= 2]).value_counts()
        modal = max(freq.index[freq == freq.max()])
        rows = cnt.to_numpy()[m == modal]
        kappa, method, n_kappa = fleiss_kappa(rows), "fleiss", int(len(rows))

    multi = cnt.to_numpy(float)[m >= 2]
    num = (multi * (multi - 1)).sum(axis=0)
    den = (multi * (m[m >= 2] - 1)[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = num / den
    return {
        "n_items": int(len(cnt)),
        "n_multi_rated": int((m >= 2).sum()),
        "raters": raters,
        "pairwise_kappa": pairwise,
        "kappa": float(kappa),
        "kappa_method": method,
        "n_kappa_items": n_kappa,
        "krippendorff_alpha": krippendorff_alpha_nominal(lab),
        "per_class_agreement": {c: float(v) for c, v in zip(classes, per_class)},
        "rater_confusion": conf.tolist(),
    }


def validity_report(labels_df, labels_adjudicated, reference_raters=REFERENCE_RATERS, classes=MECHANISMS) -> dict:
    """Adjudicated labels vs reference-rater labels on their common particles.

    One comparison per (particle, reference rater) row, so n = number of common particles when there is a
    single reference rater. Returns {"kappa" (Cohen), "accuracy", "n"}; NaN metrics when n == 0.
    """
    ref = labels_df.loc[labels_df["rater_id"].isin(list(reference_raters)), ["particle_id", "label"]]
    both = ref.merge(labels_adjudicated[["particle_id", "label"]], on="particle_id", suffixes=("_ref", "_adj"))
    n = len(both)
    return {
        "kappa": cohens_kappa(both["label_adj"], both["label_ref"], labels=classes) if n else float("nan"),
        "accuracy": float((both["label_adj"] == both["label_ref"]).mean()) if n else float("nan"),
        "n": int(n),
    }


def adjudicate(labels_df, adjudicator=None, reference_raters=REFERENCE_RATERS) -> pd.DataFrame:
    """Final labels (labels_adjudicated schema) from the non-reference raters.

    1 rater -> "single"; all agree -> "unanimous"; strict majority (> n/2) -> "majority"; otherwise the
    adjudicator's decision -> "adjudicated". `adjudicator` is a rater_id in `labels_df` (its rows only resolve
    disagreements and are not votes) or a mapping/Series particle_id -> label. Disagreements without a
    decision stay unlabelled (omitted; count logged), never resolved arbitrarily.
    agreement = fraction of voting raters whose label equals the final label.
    """
    lab = labels_df[~labels_df["rater_id"].isin(list(reference_raters))]
    if isinstance(adjudicator, str):
        decisions = lab.loc[lab["rater_id"] == adjudicator].set_index("particle_id")["label"]
        lab = lab[lab["rater_id"] != adjudicator]
    else:
        decisions = pd.Series({} if adjudicator is None else adjudicator, dtype=object)
    cnt = _counts(_raters_only(lab, MECHANISMS, ()), MECHANISMS)
    c = cnt.to_numpy()
    n = c.sum(axis=1)
    top = c.max(axis=1)
    method = np.select([n == 1, top == n, 2 * top > n], ["single", "unanimous", "majority"], "adjudicated")
    label = np.where(method == "adjudicated", cnt.index.map(decisions).to_numpy(dtype=object),
                     np.asarray(MECHANISMS, dtype=object)[c.argmax(axis=1)])
    resolved = pd.notna(label)
    if not pd.Series(label[resolved]).isin(MECHANISMS).all():
        raise ValueError("adjudicator decisions must be labels in schema.MECHANISMS")
    if (~resolved).any():
        get_logger().warning("adjudicate: %d disagreements without an adjudicator decision left unlabelled",
                             int((~resolved).sum()))
    rows = np.flatnonzero(resolved)
    agreement = c[rows, label_index(label[resolved])] / n[rows]
    out = pd.DataFrame({"particle_id": cnt.index[rows], "label": label[resolved], "method": method[rows],
                        "n_raters": n[rows], "agreement": agreement})
    return coerce_table(out, "labels_adjudicated")


def soft_labels(labels_df, particle_ids, classes=MECHANISMS, smoothing=0.0, reference_raters=REFERENCE_RATERS) -> np.ndarray:
    """(len(particle_ids), C) rater vote distribution, rows sum to 1; particles without votes get the uniform
    row (mask them via the hard target). Label smoothing: (1 - smoothing) * votes + smoothing / C."""
    lab = _raters_only(labels_df, classes, reference_raters)
    c = _counts(lab, classes).reindex(index=list(particle_ids), fill_value=0).to_numpy(float)
    tot = c.sum(axis=1, keepdims=True)
    p = np.where(tot > 0, c / np.maximum(tot, 1.0), 1.0 / len(classes))
    return (1.0 - smoothing) * p + smoothing / len(classes)
