#!/usr/bin/env bash
# ============================================================
#  GATED 3-layer: dw2 (features.8+12, /16) + features.6 (/8) + per-stream GATE.
#    -le features.6.conv.1   192ch /8   (reference grid -> localization)
#    -le features.8.conv.1   384ch /16
#    -le features.12.conv.1  576ch /16
#  Rationale: the /8 layer helped laggards screw/carpet/capsule but hurt
#  pill/toothbrush when force-averaged (un-gated dw3 = wash). The GATE lets
#  each class keep or suppress it -> should harvest the selective benefit.
#  Compare against: dw2+gate (0.9755, current baseline) and un-gated dw3 (0.967).
#  RULE: first -le is /8 -> --downsampling 8.
# ============================================================
datapath=/content/mvtec_anomaly_detection
augpath=/content/dtd/images

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --run_name mnv2_dw3_gate_20ep \
  net \
    -b mobilenetv2 \
    -le features.6.conv.1 \
    -le features.8.conv.1 \
    -le features.12.conv.1 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 20 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
    --gate 1 \
    --wavelet 0 \
    --pca 0 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 1 \
    --rand_aug 1 \
    --downsampling 8 \
    --batch_size 8 \
    --num_workers 4 \
    --resize 288 \
    --imagesize 288 \
    -d carpet -d grid -d leather -d tile -d wood -d bottle -d cable -d capsule \
    -d hazelnut -d metal_nut -d pill -d screw -d toothbrush -d transistor -d zipper \
    mvtec $datapath $augpath
