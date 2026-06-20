# Domain Shift Report

## Summary

| feature | train mean | test mean | test/train |
| --- | ---: | ---: | ---: |
| rgb_mean | 17.334 | 126.045 | 7.272 |
| rgb_std | 17.867 | 26.905 | 1.506 |
| rgb_b | 18.699 | 124.908 | 6.680 |
| rgb_g | 16.218 | 124.865 | 7.699 |
| rgb_r | 17.085 | 128.363 | 7.513 |
| sat_mean | 106.895 | 20.157 | 0.189 |
| value_mean | 21.778 | 131.992 | 6.061 |
| gray_lap_var | 167.478 | 934.816 | 5.582 |
| tir_mean | 124.033 | 103.435 | 0.834 |
| tir_std | 54.179 | 30.804 | 0.569 |
| tir_lap_var | 296.619 | 47.751 | 0.161 |

## Generated Augmentation Config

```json
{
  "seed": 2026,
  "fusion": "rgbt",
  "probabilities": {
    "brightness_contrast_gamma": 0.75,
    "tir_contrast": 0.6,
    "blur": 0.3,
    "noise": 0.25,
    "jpeg": 0.25
  },
  "brightness_delta_range": [
    45,
    45
  ],
  "contrast_range": [
    1.3,
    1.3
  ],
  "gamma_range": [
    0.85,
    1.2
  ],
  "tir_brightness_delta_range": [
    -30,
    -10
  ],
  "tir_contrast_range": [
    0.7,
    0.7
  ],
  "blur_kernel_choices": [
    3
  ],
  "noise_std_range": [
    1.0,
    6.0
  ],
  "jpeg_quality_range": [
    65,
    92
  ]
}
```

All selected transforms are pixel-only, so bounding boxes remain unchanged.