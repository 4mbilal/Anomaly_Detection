# Current baseline (stepping stone for further enhancements)

**dw2 + gate** is the locked reference. All new experiments compare against it.

| Item | Value |
|------|-------|
| Backbone | MobileNetV2 (`-b mobilenetv2`) |
| Layers | `features.8.conv.1` (384ch /16) + `features.12.conv.1` (576ch /16) |
| Channels | 960 |
| Gate | `--gate 1` (learnable per-stream weighting; per-class) |
| Streams | `--pca 0 --wavelet 0` |
| downsampling | 16 (first -le is /16) |
| Schedule | 100 epochs (current best); 20ep historical |
| Mean image AUROC | **0.9923** (100ep) / 0.9755 (20ep) |
| Mean pixel AUROC | 0.9852 (100ep) |
| Reference WRN50 | lit. ~0.996 @640ep; our 20ep 0.9841; 100ep run PENDING |

Run: `shell/run-mvtec-mnv2-gateonly.sh`

## Result log (20-epoch, mean image AUROC)
- WRN50 (paper backbone)            0.9841
- MNv2 bottleneck (features.6,13)   0.9290   (thin, rejected)
- MNv2 dw2 (features.8,12)          0.9690
- MNv2 dw3 (features.6,8,12)        0.9670   (un-gated; /8 layer a wash)
- MNv2 dw2b (features.4,8)          0.9350   (too lean, rejected)
- MNv2 dw2 + PCA stream             0.9000   (rejected: structural collapse)
- MNv2 dw2 + gate + wavelet         0.9174   (rejected: structural collapse)
- MNv2 dw3 + gate (features.6,8,12) 0.9721  (rejected: below dw2+gate)
- MNv2 dw2 + gate  20ep  0.9755
- **MNv2 dw2 + gate 100ep  0.9923  (CURRENT BASELINE)**

## Rejected, with reason
- Handcrafted PCA-residual and wavelet streams both reduce accuracy via
  structural-class collapse (metal_nut/transistor/screw/capsule), even with the
  gate. MobileNetV2's expanded depthwise features already carry that info; extra
  handcrafted streams inject class-dependent noise a scalar gate can't suppress.

## Keeper
- The learnable per-stream **gate** helps on its own (+0.0065 over dw2),
  improving 10/15 classes with ~no losers. Folded into the baseline.

## Rejected feature-side additions (all below dw2+gate)
- PCA stream, wavelet stream, AND a 3rd backbone /8 layer (features.6),
  even with the gate. Three independent feature-side attempts all fail to
  beat dw2+gate -> feature augmentation has hit its ceiling on this
  backbone/schedule. Laggards (screw 0.85, toothbrush 0.92) are likely a
  synthesis-fit or undertraining issue, not a feature issue.

## Next: schedule confirmation
- Matched 100-epoch pair (same schedule both sides):
  shell/run-dw2gate-100ep.sh  and  shell/run-wrn50-100ep.sh
  Then 640-epoch final pair if the gap/laggards justify it.

## (old) Open experiment
- Gated 3-layer (`features.6,8,12` + gate, downsampling 8):
  `shell/run-mvtec-mnv2-dw3-gate.sh`. Hypothesis: gate harvests the /8 layer's
  selective benefit on screw/carpet/capsule that force-averaging wasted in dw3.


## 100-epoch result (the breakthrough)
dw2+gate at 100 epochs = 0.9923 mean image AUROC (from 0.9755 at 20ep, +1.68).
The gain was almost entirely the former "laggards", confirming they were
UNDERTRAINED, not feature-limited:
  screw 0.849->0.941 (+0.092, best_epoch 94)
  toothbrush 0.936->1.000 (+0.064)
  cable 0.959->0.997 (+0.038)
  carpet 0.958->0.982 (+0.024)
  capsule 0.981->0.999 (+0.018)
  pill 0.968->0.983 (+0.015)
pixel: AUROC 0.9795->0.9852, AP 0.594->0.626, PRO 0.914->0.926.
=> This is why no feature-side addition helped: the gap was training time.

## Headline framing
Literature WRN50-GLASS on MVTec ~ 0.996 (image AUROC, 640 epochs).
Our MobileNetV2-GLASS + gate hits 0.9923 at only 100 epochs, at a large
backbone-cost saving. Strong, journal-worthy result.
CAVEAT (must close before claiming the gap): our WRN50 number is 20ep (0.9841).
For an apples-to-apples claim, report the matched WRN50 run
(shell/run-wrn50-100ep.sh) and/or the 640ep pair. Some classes (screw bp94,
tile bp97) still climbing at 100 -> 640 may add a little more.

## Pending
- WRN50 @100ep (matched): shell/run-wrn50-100ep.sh   [REQUIRED for fair gap]
- Optional 640ep final pair for paper figures.
