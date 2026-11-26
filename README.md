# Wear-derbis

Code, results and trained weights of the study "From Oil Debris to Gear Wear: How Far One Ferrograph Factor Carries
Between FZG Gear Pairs".

## Contents

| Path | What |
|---|---|
| `src/wearpg/` | Python package: data handling, features, particle graphs, models, training, statistics, paired debris and wear analysis |
| `scripts/` | Analyses of the open data (NASA oil debris, ferrograms with masks, published paired debris and wear data), diagnostics, simulations, figures |
| `tests/` | Unit tests (`pytest`) |
| `configs/` | Run configurations |
| `outputs/open_data/` | Results on the open and published data, including the frozen hold-out test (`paired/holdout/`) and the post-hoc analyses of the 23 gear pairs (`paired/pooled/`) |
| `outputs/pilot/`, `outputs/smoke/`, `outputs/simulation/` | Results on synthetic data, with the trained weights (`training/**/best.pt`) |
| `outputs/renderer/`, `outputs/diagnostics/` | Renderer calibration of synthetic ferrograms and diagnostic analyses |

## Use

```
pip install -e .[dev]
pytest
python scripts/paired_holdout.py      # the frozen hold-out test (do not re-run if the frozen result is to be kept)
python scripts/paired_pooled.py       # post-hoc analyses of the gear pairs
python scripts/make_paper_figures.py  # figures from the stored results
```

The raw data are not redistributed; obtain them from their sources: NASA/TM-2003-211307 and NASA/TM-2001-210936 (NTRS),
the ferrograms with masks of Li (2026, Zenodo, doi:10.5281/zenodo.22997504, CC BY 4.0), the tractor engine oil analyses
of Afsharnia et al. (2025, Zenodo, doi:10.5281/zenodo.17287193, CC BY 4.0), and the publications listed in the article
for the paired debris and wear values (the digitised values are in `outputs/open_data/paired/`). Images in
`outputs/open_data/ferro/` are drawn on the ferrograms of Li (2026), CC BY 4.0. Particles of the TCBGY images (He et
al. 2024, no licence stated) enter only as distribution statistics; their per-particle rows are not included. The
trained weights are PyTorch `state_dict` files of models trained on synthetic data only.
