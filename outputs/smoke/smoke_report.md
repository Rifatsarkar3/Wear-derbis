# wearpg smoke report

**SYNTHETIC DATA - PIPELINE VERIFICATION ONLY - NOT EVIDENCE FOR ANY SCIENTIFIC CLAIM**

**VERIFICATION ONLY - NO GO/NO-GO DECISION**

- config: configs/smoke.yaml (config_hash ab2719f99275)
- dataset: synthetic_smoke {"is_synthetic": true, "n_specimens": 12, "n_runs": 12, "n_samples": 36, "n_images": 391, "n_particles": 590, "n_views": 924, "n_labels": 1432, "n_labels_adjudicated": 590, "n_materials": 2, "n_conditions": 4, "n_labelled_particles": 590}
- runtime: 334 s (cv: 3 folds x seeds [0])
- results: outputs/smoke (metrics/, gates/, predictions/parts/)
- report: outputs/smoke/report/report.md

## Headline metrics, protocol cv (point [95% CI])

| model | variant | particle macro_f1 | sample qwk | run mae log10 k |
|---|---|---|---|---|
| conditions_only | default | - | 0.868 [0.621, 0.987] | 0.208 [0.131, 0.293] |
| contour_cnn | default | 0.769 [0.724, 0.801] | - | - |
| descriptor_agg_rf | default | - | 0.771 [0.668, 1.000] | 0.688 [0.426, 0.977] |
| fecnn | default | 0.722 [0.669, 0.757] | - | - |
| fourier_fractal_cart | default | 0.817 [0.774, 0.837] | - | - |
| fourier_fractal_svm | default | 0.814 [0.787, 0.835] | - | - |
| multiview_svm | default | 0.788 [0.747, 0.813] | - | - |
| particle_gnn | default | 0.660 [0.618, 0.698] | - | - |
| physics_only | default | - | 0.781 [0.000, 0.889] | 0.270 [0.201, 0.350] |
| resnet18 | default | 0.670 [0.628, 0.698] | - | - |
| wearpg | full | 0.729 [0.699, 0.751] | 0.759 [0.532, 0.900] | 0.290 [0.195, 0.402] |
| wearpg | no_physics | 0.744 [0.696, 0.784] | 0.744 [0.501, 0.889] | 0.592 [0.329, 0.825] |

## Gates

| gate | status | message |
|---|---|---|
| pilot | FAIL | (a) 4/4 conditions have >= 3 runs; (b) 0/4 run aggregates with p_adj < 0.05 and abs(rho) >= 0.3 (best frac_large: rho=0.77, p_adj=0.0528) |
| graph_value | FAIL | particle_gnn macro-F1 beats 0/2 baselines (lower CI bound > 0) |
| mechanism | PASS | agreement PASS (kappa=0.634 [cohen], 2 raters, 590 multi-rated items); validity PASS (kappa=0.920, n=252); SEM/EDS coverage PASS (0.427 of 590 labelled particles) |
| wear | NOT EVALUABLE | repeatability PASS (median SD log10 k = 0.160 over 4 regime-homogeneous conditions, 0 runs in regime-mixed conditions); external NOT EVALUABLE (no external run predictions) |
| portfolio | FAIL | particles macro_f1 FAIL (wearpg 0.7287 vs best baseline fourier_fractal_cart 0.8173); runs mae FAIL (wearpg 0.2898 vs best baseline conditions_only 0.2085) |
