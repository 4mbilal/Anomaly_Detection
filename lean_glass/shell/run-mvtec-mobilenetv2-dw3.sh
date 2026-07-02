#!/usr/bin/env bash
# ============================================================
#  MobileNetV2, THREE depthwise layers: rich features + finer localization.
#    -le features.6.conv.1   -> 192ch @ /8  (reference grid, 36x36 -> localization)
#    -le features.8.conv.1   -> 384ch @ /16 (== block7_depthwise)
#    -le features.12.conv.1  -> 576ch @ /16 (== block11_depthwise)
#  1152 concat channels. Adds the /8 layer back to fix small-defect classes
#  (screw, toothbrush) and pixel localization, while keeping the structural
#  recovery from the deep depthwise features.
#  RULE: --downsampling = stride of the FIRST -le layer. First is /8 -> 8.
#  Fresh --run_name so it does not reuse the 2-layer checkpoints.
#  All other settings identical to the 20-epoch baseline.
#  EDIT the two paths below.
# ============================================================
datapath=/content/mvtec_anomaly_detection
augpath=/content/dtd/images

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --run_name mnv2_dw3_20ep \
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
