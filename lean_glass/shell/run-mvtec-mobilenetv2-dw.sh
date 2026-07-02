#!/usr/bin/env bash
# ============================================================
#  MobileNetV2 with EXPANDED DEPTHWISE features (replicates the
#  MATLAB SimpleNet setup that worked well):
#    -le features.8.conv.1   -> 384ch @ /16  (== block7_depthwise)
#    -le features.12.conv.1  -> 576ch @ /16  (== block11_depthwise)
#  960 concat channels vs the 128 of the bottleneck-output version,
#  which is why the first MobileNetV2 run scored low on structural classes.
#  GLASS hooks module[-1] of each Conv2dNormActivation = the depthwise
#  ReLU6 output. For the pre-ReLU BN (exact MATLAB *_BN), use
#  features.8.conv.1.1 / features.12.conv.1.1 instead.
#  IMPORTANT: --downsampling MUST equal the stride of the FIRST -le layer
#  (the reference grid). Both layers here are /16, so --downsampling 16.
#  (The default 8 builds the synthesis mask at /8 and crashes with a
#   mask/feature shape mismatch.)
#  All other settings held identical to the 20-epoch baseline.
#  EDIT the two paths below.
# ============================================================
datapath=/content/mvtec_anomaly_detection
augpath=/content/dtd/images

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --run_name mnv2_dw_20ep \
  net \
    -b mobilenetv2 \
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
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 1 \
    --rand_aug 1 \
    --downsampling 16 \
    --batch_size 8 \
    --num_workers 4 \
    --resize 288 \
    --imagesize 288 \
    -d carpet -d grid -d leather -d tile -d wood -d bottle -d cable -d capsule \
    -d hazelnut -d metal_nut -d pill -d screw -d toothbrush -d transistor -d zipper \
    mvtec $datapath $augpath
