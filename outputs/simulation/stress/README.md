# Stress tests of the run-level debris estimators

**SIMULATION - parameter-level synthetic campaigns (no rendering, renderer not calibrated to real ferrograms); design and software verification only, not evidence about real wear**

The generator and the physics_only observation model share their structure (aggregate Archard volume, observation fraction, capture efficiency by magnetic class), so these numbers measure internal consistency, identifiability and campaign power of the debris chain, not its external validity.

One factor at a time around configs/synthetic_pilot.yaml (54 runs = 3 materials x 3 loads x 2 speeds x 3 repeats, 5 samples per run unless varied); 20 campaigns per setting; each campaign = pipeline-identical grouped 5-fold cv (experiments.make_splits + run_run_baseline). Cells: mean [2.5 %, 97.5 %] over campaigns. Confocal thickness / volume noise is not a factor here: physics_only, conditions_only and physics_conditions never use confocal data (only the wearpg volume head does), so their results are invariant to it by construction.

Files: stress_campaigns.csv (one row per campaign), stress_summary.csv, stress_metrics.png.

| factor | level | physics_only mae | physics_only within_spearman | physics_only ccc | conditions_only mae | conditions_only within_spearman | physics_conditions mae | physics_conditions within_spearman | physics_conditions ccc |
|---|---|---|---|---|---|---|---|---|---|
| budget_sigma | 0.1 | 0.214 [0.186, 0.265] | 0.900 [0.802, 0.952] | 0.973 [0.960, 0.979] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.217 [0.188, 0.269] | 0.896 [0.798, 0.948] | 0.973 [0.960, 0.979] |
| budget_sigma | 0.3 | 0.215 [0.178, 0.265] | 0.891 [0.751, 0.960] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.272] | 0.886 [0.737, 0.954] | 0.973 [0.961, 0.981] |
| budget_sigma | 0.6 | 0.238 [0.197, 0.275] | 0.834 [0.672, 0.913] | 0.967 [0.957, 0.977] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.242 [0.198, 0.274] | 0.826 [0.655, 0.907] | 0.967 [0.957, 0.977] |
| imaged_fraction_range | [0.005, 0.05] | 0.345 [0.291, 0.422] | 0.741 [0.520, 0.879] | 0.921 [0.891, 0.946] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.348 [0.280, 0.401] | 0.665 [0.347, 0.838] | 0.921 [0.891, 0.946] |
| imaged_fraction_range | [0.01, 0.1] | 0.298 [0.248, 0.340] | 0.829 [0.738, 0.921] | 0.943 [0.914, 0.962] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.307 [0.256, 0.351] | 0.791 [0.597, 0.912] | 0.943 [0.914, 0.962] |
| imaged_fraction_range | [0.02, 0.2] | 0.256 [0.219, 0.282] | 0.849 [0.725, 0.924] | 0.960 [0.946, 0.971] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.264 [0.227, 0.293] | 0.835 [0.709, 0.910] | 0.960 [0.946, 0.971] |
| imaged_fraction_range | [0.05, 0.5] | 0.215 [0.178, 0.265] | 0.891 [0.751, 0.960] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.272] | 0.886 [0.737, 0.954] | 0.973 [0.961, 0.981] |
| material_sigma | 0.0 | 0.218 [0.187, 0.270] | 0.886 [0.804, 0.946] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.221 [0.190, 0.273] | 0.883 [0.787, 0.945] | 0.973 [0.961, 0.981] |
| material_sigma | 0.2 | 0.215 [0.178, 0.265] | 0.891 [0.751, 0.960] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.272] | 0.886 [0.737, 0.954] | 0.973 [0.961, 0.981] |
| material_sigma | 0.5 | 0.216 [0.180, 0.261] | 0.880 [0.714, 0.948] | 0.973 [0.962, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.267] | 0.876 [0.700, 0.947] | 0.973 [0.962, 0.981] |
| min_area_px | 30 | 0.215 [0.178, 0.265] | 0.891 [0.751, 0.960] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.272] | 0.886 [0.737, 0.954] | 0.973 [0.961, 0.981] |
| min_area_px | 80 | 0.214 [0.182, 0.261] | 0.894 [0.748, 0.958] | 0.973 [0.963, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.217 [0.185, 0.267] | 0.888 [0.724, 0.955] | 0.973 [0.963, 0.981] |
| min_area_px | 150 | 0.213 [0.185, 0.256] | 0.894 [0.773, 0.960] | 0.974 [0.964, 0.979] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.217 [0.191, 0.261] | 0.888 [0.745, 0.952] | 0.974 [0.964, 0.979] |
| samples_per_run | 3 | 0.193 [0.165, 0.229] | 0.883 [0.803, 0.951] | 0.979 [0.971, 0.984] | 0.442 [0.383, 0.526] | -0.356 [-0.584, -0.105] | 0.196 [0.166, 0.235] | 0.879 [0.801, 0.948] | 0.979 [0.971, 0.984] |
| samples_per_run | 5 | 0.215 [0.178, 0.265] | 0.891 [0.751, 0.960] | 0.973 [0.961, 0.981] | 0.444 [0.388, 0.534] | -0.352 [-0.560, -0.115] | 0.219 [0.182, 0.272] | 0.886 [0.737, 0.954] | 0.973 [0.961, 0.981] |
| samples_per_run | 8 | 0.235 [0.203, 0.284] | 0.880 [0.731, 0.955] | 0.968 [0.954, 0.976] | 0.445 [0.387, 0.536] | -0.357 [-0.581, -0.102] | 0.240 [0.207, 0.286] | 0.871 [0.713, 0.955] | 0.968 [0.954, 0.976] |

Share of campaigns where physics_conditions has a lower run MAE than the component (Wilson CIs in stress_summary.csv):

| factor | level | share_mae_below_physics_only | share_mae_below_conditions_only |
|---|---|---|---|
| budget_sigma | 0.1 | 0.050 | 1.000 |
| budget_sigma | 0.3 | 0.050 | 1.000 |
| budget_sigma | 0.6 | 0.150 | 1.000 |
| imaged_fraction_range | [0.005, 0.05] | 0.400 | 1.000 |
| imaged_fraction_range | [0.01, 0.1] | 0.100 | 1.000 |
| imaged_fraction_range | [0.02, 0.2] | 0.000 | 1.000 |
| imaged_fraction_range | [0.05, 0.5] | 0.050 | 1.000 |
| material_sigma | 0.0 | 0.100 | 1.000 |
| material_sigma | 0.2 | 0.050 | 1.000 |
| material_sigma | 0.5 | 0.050 | 1.000 |
| min_area_px | 30 | 0.050 | 1.000 |
| min_area_px | 80 | 0.100 | 1.000 |
| min_area_px | 150 | 0.150 | 1.000 |
| samples_per_run | 3 | 0.050 | 1.000 |
| samples_per_run | 5 | 0.050 | 1.000 |
| samples_per_run | 8 | 0.100 | 1.000 |
