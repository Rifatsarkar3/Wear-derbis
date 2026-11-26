# Go/no-go gates

**SYNTHETIC DATA - PIPELINE VERIFICATION ONLY - NOT EVIDENCE FOR ANY SCIENTIFIC CLAIM**

**VERIFICATION ONLY - NO GO/NO-GO DECISION** (decision_valid: false)

Overall: FAIL

| gate | status | message |
|---|---|---|
| pilot | PASS | (a) 18/18 conditions have >= 3 runs; (b) 4/4 run aggregates with p_adj < 0.05 and abs(rho) >= 0.3 (best median_log_ecd: rho=0.83, p_adj=0.0008) |
| graph_value | FAIL | particle_gnn macro-F1 beats 0/2 baselines (lower CI bound > 0) |
| mechanism | PASS | agreement PASS (kappa=0.672 [cohen], 2 raters, 10065 multi-rated items); validity PASS (kappa=0.941, n=4191); SEM/EDS coverage PASS (0.416 of 10065 labelled particles) |
| wear | NOT EVALUABLE | repeatability PASS (median SD log10 k = 0.165 over 10 regime-homogeneous conditions, 24 runs in regime-mixed conditions); external NOT EVALUABLE (no external run predictions) |
| wear_rate | FAIL | wearpg/full: run MAE beats 1/2 baselines (upper CI bound < 0) FAIL; volume slope 0.907 [0.862, 0.955], CCC 0.978 [0.966, 0.986] FAIL; within-condition Spearman 0.907 [0.517, 0.946] PASS |
| portfolio | FAIL | runs mae FAIL (wearpg 0.2035 vs best baseline physics_only 0.2); particles macro_f1 FAIL (wearpg 0.8552 vs best baseline fourier_fractal_svm 0.8809) |

## pilot: PASS

(a) 18/18 conditions have >= 3 runs; (b) 4/4 run aggregates with p_adj < 0.05 and abs(rho) >= 0.3 (best median_log_ecd: rho=0.83, p_adj=0.0008)

Criteria:

```json
{
  "min_runs_per_condition": 3,
  "feature_regime_alpha": 0.05,
  "min_abs_spearman": 0.3,
  "detection_min_area_px": 30,
  "frac_large_um": 15,
  "strata": "material_id",
  "multiplicity": "holm",
  "target": "log10 wear_rate_mm3_Nm"
}
```

Evidence:

```json
{
  "runs_per_condition": {
    "AA6061_F120_v0.2": 3,
    "AA6061_F120_v0.6": 3,
    "AA6061_F20_v0.2": 3,
    "AA6061_F20_v0.6": 3,
    "AA6061_F60_v0.2": 3,
    "AA6061_F60_v0.6": 3,
    "AISI52100_F120_v0.2": 3,
    "AISI52100_F120_v0.6": 3,
    "AISI52100_F20_v0.2": 3,
    "AISI52100_F20_v0.6": 3,
    "AISI52100_F60_v0.2": 3,
    "AISI52100_F60_v0.6": 3,
    "CuSn12_F120_v0.2": 3,
    "CuSn12_F120_v0.6": 3,
    "CuSn12_F20_v0.2": 3,
    "CuSn12_F20_v0.6": 3,
    "CuSn12_F60_v0.2": 3,
    "CuSn12_F60_v0.6": 3
  },
  "conditions_below_min": [],
  "ecd_floor_um": 3.0901936161855166,
  "n_particles_used": 9635,
  "features": {
    "median_log_ecd": {
      "rho": 0.8297472517607418,
      "p_value": 0.0001999600079984003,
      "n": 54,
      "p_adj": 0.0007998400319936012,
      "passes": true,
      "pooled_rho": 0.8115886818519834
    },
    "mean_fractal_D": {
      "rho": -0.38801873557442157,
      "p_value": 0.003599280143971206,
      "n": 54,
      "p_adj": 0.003599280143971206,
      "passes": true,
      "pooled_rho": -0.47822357979488644
    },
    "mean_aspect": {
      "rho": 0.8008266006620272,
      "p_value": 0.0001999600079984003,
      "n": 54,
      "p_adj": 0.0007998400319936012,
      "passes": true,
      "pooled_rho": 0.8176879861777852
    },
    "frac_large": {
      "rho": 0.7908420901636614,
      "p_value": 0.0001999600079984003,
      "n": 54,
      "p_adj": 0.0007998400319936012,
      "passes": true,
      "pooled_rho": 0.7966835069058057
    }
  }
}
```

## graph_value: FAIL

particle_gnn macro-F1 beats 0/2 baselines (lower CI bound > 0)

Criteria:

```json
{
  "proposed": [
    "particle_gnn",
    "default"
  ],
  "baselines": [
    "contour_cnn",
    "fourier_fractal_svm"
  ],
  "mode": "ci",
  "metric": "macro_f1",
  "protocol": "cv",
  "cluster": "specimen",
  "n_bootstrap": 2000,
  "ci": 0.95
}
```

Evidence:

```json
{
  "contour_cnn": {
    "available": true,
    "proposed": 0.8745616397071532,
    "baseline": 0.8713274530591185,
    "diff": 0.0032341866480346537,
    "lo": -0.003764004410770324,
    "hi": 0.009102080387571987,
    "p_value": 0.351824087956022,
    "rel_diff": 0.003711792434267784,
    "n_clusters": 54,
    "n_particles": 10065,
    "passes": false
  },
  "fourier_fractal_svm": {
    "available": true,
    "proposed": 0.8745616397071532,
    "baseline": 0.8809343897172137,
    "diff": -0.006372750010060546,
    "lo": -0.013712191436199788,
    "hi": 0.0005067994223952899,
    "p_value": 0.06596701649175413,
    "rel_diff": -0.007234080181733222,
    "n_clusters": 54,
    "n_particles": 10065,
    "passes": false
  }
}
```

## mechanism: PASS

agreement PASS (kappa=0.672 [cohen], 2 raters, 10065 multi-rated items); validity PASS (kappa=0.941, n=4191); SEM/EDS coverage PASS (0.416 of 10065 labelled particles)

Criteria:

```json
{
  "min_kappa": 0.6,
  "min_double_rated": 100,
  "min_validity_kappa": 0.6,
  "min_sem_eds_fraction": 0.1,
  "reference_raters": [
    "SEMREF"
  ]
}
```

Evidence:

```json
{
  "agreement": {
    "n_items": 10065,
    "n_multi_rated": 10065,
    "raters": [
      "A",
      "B"
    ],
    "pairwise_kappa": {
      "A|B": 0.6724801514245482
    },
    "kappa": 0.6724801514245482,
    "kappa_method": "cohen",
    "n_kappa_items": 10065,
    "krippendorff_alpha": 0.6724435252848912,
    "per_class_agreement": {
      "rubbing": 0.8144210001661406,
      "cutting": 0.7938684953610327,
      "spherical": 0.7570900123304563,
      "fatigue_chunk": 0.7179269328802039,
      "laminar": 0.5632058287795992,
      "severe_sliding": 0.7372773536895675,
      "oxide": 0.6516679596586501
    },
    "rater_confusion": [
      [
        2451,
        0,
        25,
        4,
        303,
        1,
        296
      ],
      [
        0,
        984,
        0,
        6,
        112,
        169,
        0
      ],
      [
        30,
        0,
        307,
        0,
        0,
        0,
        68
      ],
      [
        5,
        2,
        1,
        845,
        125,
        107,
        98
      ],
      [
        212,
        70,
        0,
        89,
        773,
        132,
        15
      ],
      [
        3,
        152,
        0,
        138,
        122,
        1159,
        0
      ],
      [
        238,
        0,
        73,
        89,
        19,
        2,
        840
      ]
    ]
  },
  "agreement_passes": true,
  "validity": {
    "kappa": 0.9414277098943096,
    "accuracy": 0.952755905511811,
    "n": 4191
  },
  "validity_passes": true,
  "sem_eds_coverage": 0.4163934426229508,
  "n_labelled": 10065,
  "coverage_passes": true
}
```

## wear: NOT EVALUABLE

repeatability PASS (median SD log10 k = 0.165 over 10 regime-homogeneous conditions, 24 runs in regime-mixed conditions); external NOT EVALUABLE (no external run predictions)

Criteria:

```json
{
  "max_repeat_sd_log10": 0.2,
  "require_external": true,
  "external_metric": "mae of log10 k",
  "external_rule": "proposed <= best run baseline in >= 1 group"
}
```

Evidence:

```json
{
  "repeatability": {
    "median_sd_log10": 0.16472364560461444,
    "sd_log10": {
      "AA6061_F120_v0.2": 0.32318545756148537,
      "AA6061_F120_v0.6": 0.16382026707692127,
      "AA6061_F60_v0.2": 0.059633830615554294,
      "AA6061_F60_v0.6": 0.28657664076955514,
      "AISI52100_F120_v0.6": 0.16562702413230765,
      "AISI52100_F20_v0.2": 0.03865174701461577,
      "AISI52100_F20_v0.6": 0.16853569222835021,
      "CuSn12_F120_v0.6": 0.10732865023604077,
      "CuSn12_F20_v0.2": 0.041904682507709234,
      "CuSn12_F60_v0.6": 0.3313301682741221
    },
    "n_conditions": 10,
    "median_cv": 0.3698998743733846,
    "icc": {
      "icc": 0.9629457237915499,
      "lo": 0.8994553403225775,
      "hi": 0.9897100762242771,
      "k": 3.0,
      "F": 78.96231547267043,
      "p_value": 1.1827939340458523e-13,
      "n_groups": 10,
      "n": 30
    },
    "regime_mixed_conditions": [
      "AA6061_F20_v0.2",
      "AA6061_F20_v0.6",
      "AISI52100_F120_v0.2",
      "AISI52100_F60_v0.2",
      "AISI52100_F60_v0.6",
      "CuSn12_F120_v0.2",
      "CuSn12_F20_v0.6",
      "CuSn12_F60_v0.2"
    ],
    "p_transition": {
      "AA6061_F120_v0.2": 1.0,
      "AA6061_F120_v0.6": 1.0,
      "AA6061_F20_v0.2": 0.3333333333333333,
      "AA6061_F20_v0.6": 0.3333333333333333,
      "AA6061_F60_v0.2": 1.0,
      "AA6061_F60_v0.6": 1.0,
      "AISI52100_F120_v0.2": 0.3333333333333333,
      "AISI52100_F120_v0.6": 1.0,
      "AISI52100_F20_v0.2": 0.0,
      "AISI52100_F20_v0.6": 0.0,
      "AISI52100_F60_v0.2": 0.3333333333333333,
      "AISI52100_F60_v0.6": 0.3333333333333333,
      "CuSn12_F120_v0.2": 0.6666666666666666,
      "CuSn12_F120_v0.6": 1.0,
      "CuSn12_F20_v0.2": 0.0,
      "CuSn12_F20_v0.6": 0.3333333333333333,
      "CuSn12_F60_v0.2": 0.3333333333333333,
      "CuSn12_F60_v0.6": 1.0
    },
    "passes": true
  }
}
```

## wear_rate: FAIL

wearpg/full: run MAE beats 1/2 baselines (upper CI bound < 0) FAIL; volume slope 0.907 [0.862, 0.955], CCC 0.978 [0.966, 0.986] FAIL; within-condition Spearman 0.907 [0.517, 0.946] PASS

Criteria:

```json
{
  "proposed": [
    "wearpg",
    "full"
  ],
  "baselines": [
    "conditions_only",
    "physics_only"
  ],
  "metric": "run mae of log10 k",
  "mae_rule": "upper CI bound of (proposed - baseline) < 0 for every baseline",
  "volume_rule": "slope CI contains 1 and CCC >= min_volume_ccc",
  "min_volume_ccc": 0.8,
  "within_rule": "within-condition Spearman lower CI bound > 0",
  "protocol": "cv",
  "cluster": "specimen",
  "n_bootstrap": 2000,
  "ci": 0.95
}
```

Evidence:

```json
{
  "mae": {
    "conditions_only": {
      "available": true,
      "proposed": 0.20345906526552146,
      "baseline": 0.4231502960510749,
      "diff": -0.21969123078555347,
      "lo": -0.3076756809744017,
      "hi": -0.13518708853916583,
      "p_value": 0.0009995002498750624,
      "rel_diff": -0.5191801419867998,
      "n_runs": 54,
      "passes": true
    },
    "physics_only": {
      "available": true,
      "proposed": 0.20345906526552146,
      "baseline": 0.20004625867178796,
      "diff": 0.0034128065937334695,
      "lo": -0.04823270384984467,
      "hi": 0.05352498687744824,
      "p_value": 0.927536231884058,
      "rel_diff": 0.01706008708382192,
      "n_runs": 54,
      "passes": false
    }
  },
  "volume": {
    "slope": {
      "point": 0.90720917119011,
      "lo": 0.8624169960742785,
      "hi": 0.9550171909428514,
      "se": 0.023427044712097632
    },
    "intercept": {
      "point": -0.00612497024178676,
      "lo": -0.1416946664045619,
      "hi": 0.12961539896476823,
      "se": 0.06811143867027361
    },
    "ccc": {
      "point": 0.9775021848446412,
      "lo": 0.9655129189668599,
      "hi": 0.9857181650048246,
      "se": 0.005198937362157383
    },
    "mean_log_ratio": {
      "point": 0.0035039772771637717,
      "lo": -0.14234375259704934,
      "hi": 0.14270432803781505,
      "se": 0.07614976223993618
    },
    "n_runs": 54,
    "passes": false
  },
  "within_condition_spearman": {
    "point": 0.9070879949807322,
    "lo": 0.5174185756659483,
    "hi": 0.9459608141477033,
    "se": 0.11362796159771929,
    "passes": true
  }
}
```

## portfolio: FAIL

runs mae FAIL (wearpg 0.2035 vs best baseline physics_only 0.2); particles macro_f1 FAIL (wearpg 0.8552 vs best baseline fourier_fractal_svm 0.8809)

Criteria:

```json
{
  "protocol": "cv",
  "claim_metrics": {
    "runs": {
      "metric": "mae",
      "better": "min"
    },
    "particles": {
      "metric": "macro_f1",
      "better": "max"
    }
  },
  "proposed": [
    "wearpg",
    "full"
  ],
  "estimate": "point (seed mean)"
}
```

Evidence:

```json
{
  "runs": {
    "available": true,
    "metric": "mae",
    "comparisons": {
      "conditions_only": {
        "proposed": 0.20345906526552146,
        "baseline": 0.4231502960510749,
        "n": 54
      },
      "descriptor_agg_rf": {
        "proposed": 0.20345906526552146,
        "baseline": 0.3666333170216333,
        "n": 54
      },
      "physics_conditions": {
        "proposed": 0.20345906526552146,
        "baseline": 0.20090684206802634,
        "n": 54
      },
      "physics_only": {
        "proposed": 0.20345906526552146,
        "baseline": 0.20004625867178796,
        "n": 54
      }
    },
    "best_baseline": "physics_only",
    "passes": false
  },
  "particles": {
    "available": true,
    "metric": "macro_f1",
    "comparisons": {
      "contour_cnn": {
        "proposed": 0.8551735999163846,
        "baseline": 0.8713274530591185,
        "n": 10065
      },
      "fourier_fractal_svm": {
        "proposed": 0.8551735999163846,
        "baseline": 0.8809343897172137,
        "n": 10065
      },
      "multiview_svm": {
        "proposed": 0.8551735999163846,
        "baseline": 0.880648591978752,
        "n": 10065
      },
      "particle_gnn": {
        "proposed": 0.8551735999163846,
        "baseline": 0.8745616397071532,
        "n": 10065
      },
      "resnet18": {
        "proposed": 0.8551735999163846,
        "baseline": 0.8590951401935445,
        "n": 10065
      }
    },
    "best_baseline": "fourier_fractal_svm",
    "passes": false
  }
}
```
