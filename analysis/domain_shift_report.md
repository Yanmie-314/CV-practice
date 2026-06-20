# Domain Shift Report

## Summary

| feature | train mean | test mean | test/train |
| --- | ---: | ---: | ---: |
| rgb_mean | 85.325 | 98.724 | 1.157 |
| rgb_std | 34.029 | 33.621 | 0.988 |
| rgb_b | 83.493 | 93.809 | 1.124 |
| rgb_g | 85.882 | 98.342 | 1.145 |
| rgb_r | 86.600 | 104.021 | 1.201 |
| sat_mean | 64.368 | 54.802 | 0.851 |
| value_mean | 92.505 | 108.704 | 1.175 |
| gray_lap_var | 605.805 | 1105.055 | 1.824 |
| tir_mean | 133.919 | 111.708 | 0.834 |
| tir_std | 52.278 | 40.154 | 0.768 |
| tir_lap_var | 387.256 | 274.066 | 0.708 |

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
    4,
    28
  ],
  "contrast_range": [
    0.889,
    1.087
  ],
  "gamma_range": [
    0.85,
    1.2
  ],
  "tir_brightness_delta_range": [
    -32,
    -12
  ],
  "tir_contrast_range": [
    0.7,
    0.845
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