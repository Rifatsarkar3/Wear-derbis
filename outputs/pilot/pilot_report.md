# wearpg pilot report

**SYNTHETIC DATA - PIPELINE VERIFICATION ONLY - NOT EVIDENCE FOR ANY SCIENTIFIC CLAIM**

**VERIFICATION ONLY - NO GO/NO-GO DECISION** (SIMULATION results)

- config: configs/pilot.yaml (config_hash ec2c4200509e)
- dataset: synthetic_pilot_v2 {"is_synthetic": true, "n_specimens": 54, "n_runs": 54, "n_samples": 270, "n_images": 6652, "n_particles": 10065, "n_views": 16103, "n_labels": 24321, "n_labels_adjudicated": 10065, "n_materials": 3, "n_conditions": 18, "n_labelled_particles": 10065}
- runtime: 6386 s (cv: 5 folds x seeds [0, 1, 2])
- results: outputs/pilot (metrics/, gates/, predictions/parts/)
- report: outputs/pilot/report/report.md
- primary endpoint (eval.primary_metric): run.mae

## Headline metrics, protocol cv (point [95% CI])

| model | variant | run mae log10 k (primary) | run within-cond. Spearman | particle macro_f1 | sample qwk |
|---|---|---|---|---|---|
| conditions_only | default | 0.423 [0.351, 0.495] | -0.110 [-0.480, 0.217] | - | 0.634 [0.515, 0.740] |
| contour_cnn | default | - | - | 0.871 [0.864, 0.878] | - |
| descriptor_agg_rf | default | 0.367 [0.303, 0.428] | 0.825 [0.452, 0.909] | - | 0.919 [0.894, 0.939] |
| fourier_fractal_svm | default | - | - | 0.881 [0.875, 0.887] | - |
| multiview_svm | default | - | - | 0.881 [0.875, 0.887] | - |
| particle_gnn | default | - | - | 0.875 [0.868, 0.881] | - |
| physics_conditions | default | 0.201 [0.168, 0.234] | 0.931 [0.740, 0.973] | - | 0.848 [0.753, 0.918] |
| physics_only | default | 0.200 [0.168, 0.233] | 0.934 [0.745, 0.974] | - | 0.848 [0.753, 0.918] |
| resnet18 | default | - | - | 0.859 [0.852, 0.867] | - |
| wearpg | full | 0.203 [0.160, 0.246] | 0.907 [0.517, 0.946] | 0.855 [0.846, 0.863] | 0.920 [0.895, 0.936] |
| wearpg | no_mechanism_link | 0.204 [0.164, 0.246] | 0.914 [0.583, 0.955] | 0.869 [0.863, 0.875] | 0.919 [0.894, 0.936] |
| wearpg | no_physics | 0.379 [0.317, 0.441] | 0.903 [0.642, 0.950] | 0.862 [0.855, 0.868] | 0.919 [0.894, 0.936] |
| wearpg | with_conditions | 0.133 [0.106, 0.160] | 0.908 [0.522, 0.945] | 0.860 [0.853, 0.866] | 0.920 [0.895, 0.936] |

## Gates

| gate | status | message |
|---|---|---|
| pilot | PASS | (a) 18/18 conditions have >= 3 runs; (b) 4/4 run aggregates with p_adj < 0.05 and abs(rho) >= 0.3 (best median_log_ecd: rho=0.83, p_adj=0.0008) |
| graph_value | FAIL | particle_gnn macro-F1 beats 0/2 baselines (lower CI bound > 0) |
| mechanism | PASS | agreement PASS (kappa=0.672 [cohen], 2 raters, 10065 multi-rated items); validity PASS (kappa=0.941, n=4191); SEM/EDS coverage PASS (0.416 of 10065 labelled particles) |
| wear | NOT EVALUABLE | repeatability PASS (median SD log10 k = 0.165 over 10 regime-homogeneous conditions, 24 runs in regime-mixed conditions); external NOT EVALUABLE (no external run predictions) |
| wear_rate | FAIL | wearpg/full: run MAE beats 1/2 baselines (upper CI bound < 0) FAIL; volume slope 0.907 [0.862, 0.955], CCC 0.978 [0.966, 0.986] FAIL; within-condition Spearman 0.907 [0.517, 0.946] PASS |
| portfolio | FAIL | runs mae FAIL (wearpg 0.2035 vs best baseline physics_only 0.2); particles macro_f1 FAIL (wearpg 0.8552 vs best baseline fourier_fractal_svm 0.8809) |

## Claim verdicts (paper/FRAMING.md sec. 3; criteria DESIGN sec. 15) - SIMULATION, not evidence

| claim | criterion | evidence | verdict |
|---|---|---|---|
| C1 | run MAE CI < 0 vs conditions_only and physics_only; slope CI contains 1; CCC >= min_volume_ccc | MAE diff vs conditions_only -0.220 [-0.308, -0.135]; MAE diff vs physics_only +0.003 [-0.048, +0.054]; slope 0.907 [0.862, 0.955]; CCC 0.978 [0.966, 0.986] | against_us |
| C1b | within-condition Spearman lower CI bound > 0 | 0.907 [0.517, 0.946] | for_us |
| C2 | wearpg/full run MAE CI < 0 vs physics_only and vs no_mechanism_link | vs physics_only/default +0.003 [-0.048, +0.054]; vs wearpg/no_mechanism_link -0.001 [-0.030, +0.029] | against_us |
| C3 | particle_gnn macro-F1 lower CI bound > 0 vs every graph-gate baseline | vs contour_cnn +0.003 [-0.004, +0.009]; vs fourier_fractal_svm -0.006 [-0.014, +0.001] | against_us |
| C4 | detection >= 0.8, false alarm <= 0.2, timing MAE <= 1 interval (point estimates) | detection 0.94, false alarm 0.04, MAE 0.27 intervals (oracle floor 0.25); 30 transition / 24 stable runs | for_us |
| H-F (C1 support) | physics_conditions/default run MAE CI < 0 vs both components; within-condition Spearman lower CI > 0 | vs physics_only/default +0.001 [-0.001, +0.003]; vs conditions_only/default -0.222 [-0.301, -0.145]; within-condition Spearman 0.931 [0.740, 0.973] | against_us |
| H-F with_conditions (secondary) | wearpg/with_conditions run MAE CI < 0 vs both components; within-condition Spearman lower CI > 0 | vs wearpg/full -0.070 [-0.107, -0.033]; vs conditions_only/default -0.290 [-0.369, -0.209]; within-condition Spearman 0.908 [0.522, 0.945] | for_us |
