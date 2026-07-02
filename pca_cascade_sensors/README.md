# PCA Cascade — Null-Subspace PCA Anomaly Detector

MATLAB implementation of the method in Bilal, M.; Hanif, M.S. "Fast Anomaly
Detection for Vision-Based Industrial Inspection Using Cascades of Null
Subspace PCA Detectors." *Sensors* 2025.

## What this is

For every spatial location in a MobileNetV2 feature map, fits a per-location
PCA and keeps the eigenvectors with the **smallest** eigenvalues (the
approximate null subspace of normal-data statistics). A test image's
anomaly score is the L2 norm of its projection onto that subspace — normal
images project near zero, anomalies don't. Four such detectors (one per
MobileNetV2 layer) are chained into a cascade: each stage confidently
resolves clear-cut cases and only passes ambiguous ones to the next,
deeper layer.

## Quick start

```matlab
addpath(pwd);
run('main_anomaly_pca_cascade.m');
```

Edit `Options.dataDir` inside `getOptions_pca_cascade.m` (or pass
`'dataDir', '...'` as a name-value pair) to point at your local MVTec AD
copy first.

For a fast smoke test instead of the full 15-class sweep:
```matlab
Options = getOptions_pca_cascade('quick_test', true);
```

## Structure

All files are in one flat folder. Grouped here by role (matches each
file's own header comment):

```
Entry point
  main_anomaly_pca_cascade.m   -- run this

Config
  getOptions_pca_cascade.m     -- all configuration

Data
  prepareData.m                -- datastore construction
  resizeAndCropImage.m         -- shared pad/resize/crop/augment

Features
  getEmbeddingsModel.m         -- backbone truncation + multi-output wiring
  getFeatures.m                -- run backbone, collect feature maps

PCA (core method)
  getFeaturesPCA.m             -- per-location null-subspace PCA fit (paper's actual method)
  pca_custom.m                 -- SUPERSEDED: single-global-PCA primitive, not used, kept for reference
  Get_PCA_TrainingFeatures.m   -- SUPERSEDED: single-global-PCA fit, not used, kept for reference
  ApplyPCATransform.m          -- SUPERSEDED: single-global-PCA scoring, not used, kept for reference

Detection
  detectAnomalies.m            -- cascade scoring + gating (paper Eq. 13)

Evaluation
  getROC.m                     -- image-level AUROC
  getAnomalyMaps.m             -- pixel-level AUROC
  plotEigenDiagnostics.m       -- paper Fig. 4/5-style diagnostic plot (opt-in)
  measureInferenceSpeed.m      -- paper Table 3's fps benchmark

Exploratory (not part of the pipeline)
  scratchpad.m                 -- early layer-selection/fusion experiment, not runnable as-is
```

## Bugs found and fixed during this refactor

See each file's header comment for full detail; summarized here since
all four silently affected results, not just code style:

1. **Config wiring** (`main_anomaly_pca_cascade.m`): the entry point used
   to call a leftover debug config (SqueezeNet, single class) instead of
   the paper-reproducing MobileNetV2 4-layer config — meaning running the
   script out of the box did not reproduce the paper, with no indication
   anything was wrong.
2. **Incomplete training augmentation** (`main_anomaly_pca_cascade.m`):
   the multi-pass augmentation loop only concatenated 3 of the cascade's
   4 feature layers (hardcoded `f1,f2,f3`), so the deepest layer's PCA was
   silently fit on fewer augmented samples than the other three.
3. **Unconditional debug breakpoint** (`detectAnomalies.m`): a diagnostic
   plotting function used to run on every call with a hardcoded "Carpet"
   title and end in an unconditional `keyboard`, which would halt a full
   multi-class sweep at every single class.
4. **Unseeded RNG** (`main_anomaly_pca_cascade.m`): `rng_seed = randi(1000);`
   drew a random integer but never called `rng(rng_seed)` to actually
   apply it — it had no effect, so every run's data augmentation (and
   hence the average AUROC) genuinely differed run to run, especially
   within the same MATLAB session (MATLAB doesn't reset its RNG stream
   between script executions). Fixed: the seed is now actually applied
   and printed for later reproduction. It's still drawn freshly each run
   by default — hardcode a fixed integer instead if you want identical
   results every time.

## Ablation alternatives preserved behind flags

Several algorithmic variants that were tried during development are
preserved as explicit options rather than left as inert comments (all
default to the paper's actual method):

- `Options.distance_method` in `detectAnomalies.m` — 11 anomaly-score
  formula variants (eigenvalue-weighted distances, L1/L∞ norms, etc.)
  beyond the paper's plain L2 norm (`'unweighted_l2'`, default).
- `Options.resize_interp`, `Options.denoise` in `resizeAndCropImage.m` —
  interpolation method and optional denoising filters tried before
  feature extraction.
- `getOptions_pca_cascade('backbone', ...)` — ResNet50 and SqueezeNet
  backbone alternatives from the backbone ablation (paper Table 4).
- `getROC`'s `options.threshold_method` — alternative threshold-selection
  strategies beyond Youden's J.

One exploration was found but **not** made into a working flag: an
adaptive, energy-based null-subspace-sizing attempt inside the original
`calculateDistance`, which referenced an undefined `covars` variable
(apparently copy-pasted from the separate PaDiM baseline) and was never
completed. Documented in `detectAnomalies.m` rather than silently dropped,
in case it's worth finishing later.

## Not used by the current pipeline

`pca_custom.m`, `Get_PCA_TrainingFeatures.m`, `ApplyPCATransform.m`
implement an earlier, simpler approach: one PCA fit globally across all
spatial locations pooled together, rather than a separate PCA per
location. Superseded by `getFeaturesPCA.m`, kept for reference.
