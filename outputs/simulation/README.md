# Simulation studies of the debris -> wear-rate chain

**SIMULATION - parameter-level synthetic campaigns (no rendering, renderer not calibrated to real ferrograms); design and software verification only, not evidence about real wear**

The generator and the physics_only observation model share their structure (aggregate Archard volume, observation fraction, capture efficiency by magnetic class), so these numbers measure internal consistency, identifiability and campaign power of the debris chain, not its external validity.

Subfolders: stress/ (one factor at a time, 20 campaigns per setting), power/ (design grid, 50 campaigns per cell), stress_wearpg/ (wearpg/full under confocal-volume noise on the rendered v1 pilot data). Values: mean [2.5 %, 97.5 %] over simulated campaigns at the pilot design unless stated.

## Mapping to the Wear claims (paper/FRAMING.md sec. 3) - simulation level only

- C1 (debris volume gives the wear rate): the mechanism-agnostic debris estimator physics_only has run MAE of log10 k 0.215 [0.178, 0.265] vs conditions_only 0.444 [0.388, 0.534]; volume CCC 0.973 [0.961, 0.981] (criterion >= 0.8 met in every setting), but volume slope 0.851 [0.803, 0.888]: the slope-CI-contains-1 criterion fails for a mechanism-agnostic volume (v = 0.3 A^1.5 ignores that severe debris is thicker: in the generator the true metal volume / 0.3 sum A^1.5 is 0.59 in mild and 0.90 in severe intervals, campaign seed 1000), which is the gap mechanism attribution (C2) must close in the pilot.
- C1b (beyond operating conditions): within-condition Spearman of physics_only 0.891 [0.751, 0.960] (lowest setting, imaged fraction 0.005-0.05: 0.741 [0.520, 0.879]); conditions_only -0.352 [-0.560, -0.115] (negative: a run left out of training pulls its own condition estimate away from itself).
- Fusion hypothesis H-F (C1 support): physics_conditions MAE 0.219 [0.182, 0.272] beats conditions_only in every campaign but beats physics_only in only 5% of pilot-design campaigns (at most 40% in any setting): with regime transitions driving the run-to-run variation, the known conditions add nothing the debris volume does not already carry. Against the pre-specified 'beats both components' at simulation level.
- Robustness: imaged fraction is the dominant factor (MAE 0.345 [0.291, 0.422] at 0.005-0.05); budget noise matters little up to sd 0.6; fine-debris loss (min area 30 -> 150 px) and material capture spread (absorbed by the per-material eta offset in cv) leave the estimate unchanged; more samples per run did not help in this generator.

## Design guidance (power)

Smallest simulated campaign (runs = conditions x runs per condition) whose power for BOTH criteria (run-MAE CI below conditions_only and within-condition Spearman CI above 0) is >= 0.8:

- physics_only, imaged fraction [0.005, 0.05]: 18 conditions x 8 runs = 144 runs (power 0.82 [0.69, 0.90])
- physics_only, imaged fraction [0.015, 0.15]: 18 conditions x 3 runs = 54 runs (power 0.82 [0.69, 0.90])
- physics_only, imaged fraction [0.05, 0.5]: 18 conditions x 3 runs = 54 runs (power 0.98 [0.90, 1.00])
- physics_conditions, imaged fraction [0.005, 0.05]: 18 conditions x 5 runs = 90 runs (power 0.84 [0.71, 0.92])
- physics_conditions, imaged fraction [0.015, 0.15]: 18 conditions x 3 runs = 54 runs (power 0.90 [0.79, 0.96])
- physics_conditions, imaged fraction [0.05, 0.5]: 18 conditions x 3 runs = 54 runs (power 0.98 [0.90, 1.00])

Reading the grid (power_summary.csv, power_*.png):
- The run-MAE criterion depends on the condition design more than on the number of runs: with 2 load levels (6 or 12 conditions) the linear conditions_only model reproduces every condition mean (MAE 0.30-0.36) and physics_only beats it with a CI below 0 in at most 66% of campaigns even at the pilot imaged fraction (at low imaged fractions even fewer with more repeats, which sharpen the conditions model); with 3 load levels (18 conditions) the transition-driven, non-linear load response defeats the linear conditions model (MAE 0.42) and the power is 0.98-1.00.
- The within-condition criterion (C1b) has power >= 0.88 for physics_only in every cell with >= 5 runs per condition; 3 runs per condition is marginal when the imaged fraction is low.
- Fusion helps where the debris is noisy and the conditions model is good: at imaged fraction 0.005-0.05 the mean MAE of physics_conditions vs physics_only is 0.29 vs 0.33 (6 conditions), 0.32 vs 0.38 (12 conditions), 0.33 vs 0.33 (18 conditions).
