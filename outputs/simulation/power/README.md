# Campaign power of the debris-based wear-rate estimators

**SIMULATION - parameter-level synthetic campaigns (no rendering, renderer not calibrated to real ferrograms); design and software verification only, not evidence about real wear**

The generator and the physics_only observation model share their structure (aggregate Archard volume, observation fraction, capture efficiency by magnetic class), so these numbers measure internal consistency, identifiability and campaign power of the debris chain, not its external validity.

Grid: runs per condition x number of conditions (3 materials x {2 loads x 1 speed, 2 x 2, 3 x 2}) x imaged fraction range (LogUniform). Per campaign: pipeline-identical grouped 5-fold cv; paired cluster bootstrap (1000 replicates, clusters = specimens) of the run-MAE difference debris estimator - conditions_only and cluster bootstrap of the within-condition Spearman. Power = share of campaigns with the 95 % CI below 0 (MAE difference) / above 0 (Spearman); Wilson 95 % intervals in power_summary.csv.

Files: power_campaigns.csv, power_summary.csv, power_mae.png, power_within.png, power_both.png.

| model | imaged_fraction_range | n_conditions | repeats | n_runs | n_campaigns | p_mae_ci_below_0 | p_within_ci_above_0 | p_both | mean_mae | mean_mae_conditions_only |
|---|---|---|---|---|---|---|---|---|---|---|
| physics_only | [0.005, 0.05] | 6 | 3 | 18 | 50 | 0.220 | 0.580 | 0.180 | 0.322 | 0.357 |
| physics_conditions | [0.005, 0.05] | 6 | 3 | 18 | 50 | 0.240 | 0.260 | 0.160 | 0.295 | 0.357 |
| physics_only | [0.015, 0.15] | 6 | 3 | 18 | 50 | 0.380 | 0.620 | 0.320 | 0.255 | 0.357 |
| physics_conditions | [0.015, 0.15] | 6 | 3 | 18 | 50 | 0.420 | 0.340 | 0.260 | 0.255 | 0.357 |
| physics_only | [0.05, 0.5] | 6 | 3 | 18 | 50 | 0.480 | 0.740 | 0.440 | 0.201 | 0.357 |
| physics_conditions | [0.05, 0.5] | 6 | 3 | 18 | 50 | 0.540 | 0.600 | 0.480 | 0.210 | 0.357 |
| physics_only | [0.005, 0.05] | 12 | 3 | 36 | 50 | 0.000 | 0.820 | 0.000 | 0.375 | 0.347 |
| physics_conditions | [0.005, 0.05] | 12 | 3 | 36 | 50 | 0.100 | 0.420 | 0.100 | 0.324 | 0.347 |
| physics_only | [0.015, 0.15] | 12 | 3 | 36 | 50 | 0.240 | 0.960 | 0.240 | 0.298 | 0.347 |
| physics_conditions | [0.015, 0.15] | 12 | 3 | 36 | 50 | 0.300 | 0.700 | 0.280 | 0.279 | 0.347 |
| physics_only | [0.05, 0.5] | 12 | 3 | 36 | 50 | 0.520 | 1.000 | 0.520 | 0.233 | 0.347 |
| physics_conditions | [0.05, 0.5] | 12 | 3 | 36 | 50 | 0.600 | 0.920 | 0.580 | 0.229 | 0.347 |
| physics_only | [0.005, 0.05] | 18 | 3 | 54 | 50 | 0.440 | 1.000 | 0.440 | 0.349 | 0.443 |
| physics_conditions | [0.005, 0.05] | 18 | 3 | 54 | 50 | 0.620 | 0.880 | 0.600 | 0.347 | 0.443 |
| physics_only | [0.015, 0.15] | 18 | 3 | 54 | 50 | 0.820 | 1.000 | 0.820 | 0.273 | 0.443 |
| physics_conditions | [0.015, 0.15] | 18 | 3 | 54 | 50 | 0.900 | 1.000 | 0.900 | 0.281 | 0.443 |
| physics_only | [0.05, 0.5] | 18 | 3 | 54 | 50 | 0.980 | 1.000 | 0.980 | 0.213 | 0.443 |
| physics_conditions | [0.05, 0.5] | 18 | 3 | 54 | 50 | 0.980 | 1.000 | 0.980 | 0.217 | 0.443 |
| physics_only | [0.005, 0.05] | 6 | 5 | 30 | 50 | 0.160 | 0.880 | 0.160 | 0.330 | 0.318 |
| physics_conditions | [0.005, 0.05] | 6 | 5 | 30 | 50 | 0.220 | 0.500 | 0.200 | 0.285 | 0.318 |
| physics_only | [0.015, 0.15] | 6 | 5 | 30 | 50 | 0.320 | 0.960 | 0.320 | 0.263 | 0.318 |
| physics_conditions | [0.015, 0.15] | 6 | 5 | 30 | 50 | 0.340 | 0.740 | 0.340 | 0.249 | 0.318 |
| physics_only | [0.05, 0.5] | 6 | 5 | 30 | 50 | 0.500 | 1.000 | 0.500 | 0.210 | 0.318 |
| physics_conditions | [0.05, 0.5] | 6 | 5 | 30 | 50 | 0.500 | 0.880 | 0.500 | 0.211 | 0.318 |
| physics_only | [0.005, 0.05] | 12 | 5 | 60 | 50 | 0.020 | 0.980 | 0.020 | 0.387 | 0.326 |
| physics_conditions | [0.005, 0.05] | 12 | 5 | 60 | 50 | 0.040 | 0.760 | 0.040 | 0.319 | 0.326 |
| physics_only | [0.015, 0.15] | 12 | 5 | 60 | 50 | 0.140 | 0.980 | 0.140 | 0.307 | 0.326 |
| physics_conditions | [0.015, 0.15] | 12 | 5 | 60 | 50 | 0.220 | 0.960 | 0.220 | 0.281 | 0.326 |
| physics_only | [0.05, 0.5] | 12 | 5 | 60 | 50 | 0.520 | 1.000 | 0.520 | 0.233 | 0.326 |
| physics_conditions | [0.05, 0.5] | 12 | 5 | 60 | 50 | 0.580 | 1.000 | 0.580 | 0.227 | 0.326 |
| physics_only | [0.005, 0.05] | 18 | 5 | 90 | 50 | 0.440 | 1.000 | 0.440 | 0.332 | 0.413 |
| physics_conditions | [0.005, 0.05] | 18 | 5 | 90 | 50 | 0.840 | 1.000 | 0.840 | 0.328 | 0.413 |
| physics_only | [0.015, 0.15] | 18 | 5 | 90 | 50 | 0.960 | 1.000 | 0.960 | 0.259 | 0.413 |
| physics_conditions | [0.015, 0.15] | 18 | 5 | 90 | 50 | 1.000 | 1.000 | 1.000 | 0.264 | 0.413 |
| physics_only | [0.05, 0.5] | 18 | 5 | 90 | 50 | 1.000 | 1.000 | 1.000 | 0.200 | 0.413 |
| physics_conditions | [0.05, 0.5] | 18 | 5 | 90 | 50 | 1.000 | 1.000 | 1.000 | 0.202 | 0.413 |
| physics_only | [0.005, 0.05] | 6 | 8 | 48 | 50 | 0.060 | 1.000 | 0.060 | 0.324 | 0.300 |
| physics_conditions | [0.005, 0.05] | 6 | 8 | 48 | 50 | 0.100 | 0.740 | 0.100 | 0.284 | 0.300 |
| physics_only | [0.015, 0.15] | 6 | 8 | 48 | 50 | 0.260 | 1.000 | 0.260 | 0.253 | 0.300 |
| physics_conditions | [0.015, 0.15] | 6 | 8 | 48 | 50 | 0.340 | 0.980 | 0.340 | 0.242 | 0.300 |
| physics_only | [0.05, 0.5] | 6 | 8 | 48 | 50 | 0.500 | 1.000 | 0.500 | 0.202 | 0.300 |
| physics_conditions | [0.05, 0.5] | 6 | 8 | 48 | 50 | 0.580 | 1.000 | 0.580 | 0.201 | 0.300 |
| physics_only | [0.005, 0.05] | 12 | 8 | 96 | 50 | 0.000 | 1.000 | 0.000 | 0.369 | 0.309 |
| physics_conditions | [0.005, 0.05] | 12 | 8 | 96 | 50 | 0.000 | 1.000 | 0.000 | 0.305 | 0.309 |
| physics_only | [0.015, 0.15] | 12 | 8 | 96 | 50 | 0.160 | 1.000 | 0.160 | 0.294 | 0.309 |
| physics_conditions | [0.015, 0.15] | 12 | 8 | 96 | 50 | 0.260 | 1.000 | 0.260 | 0.272 | 0.309 |
| physics_only | [0.05, 0.5] | 12 | 8 | 96 | 50 | 0.660 | 1.000 | 0.660 | 0.222 | 0.309 |
| physics_conditions | [0.05, 0.5] | 12 | 8 | 96 | 50 | 0.840 | 1.000 | 0.840 | 0.221 | 0.309 |
| physics_only | [0.005, 0.05] | 18 | 8 | 144 | 50 | 0.820 | 1.000 | 0.820 | 0.319 | 0.411 |
| physics_conditions | [0.005, 0.05] | 18 | 8 | 144 | 50 | 0.940 | 1.000 | 0.940 | 0.316 | 0.411 |
| physics_only | [0.015, 0.15] | 18 | 8 | 144 | 50 | 1.000 | 1.000 | 1.000 | 0.255 | 0.411 |
| physics_conditions | [0.015, 0.15] | 18 | 8 | 144 | 50 | 1.000 | 1.000 | 1.000 | 0.258 | 0.411 |
| physics_only | [0.05, 0.5] | 18 | 8 | 144 | 50 | 1.000 | 1.000 | 1.000 | 0.198 | 0.411 |
| physics_conditions | [0.05, 0.5] | 18 | 8 | 144 | 50 | 1.000 | 1.000 | 1.000 | 0.199 | 0.411 |
