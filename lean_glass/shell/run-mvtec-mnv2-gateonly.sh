#!/usr/bin/env bash
# ============================================================
#  CONTROL: dw2 (features.8+12, /16) + per-stream GATE only. NO wavelet.
#  Isolates the gate's effect on the two backbone streams, so:
#    (this) - dw2_plain     = what the gate alone does
#    dw2_wav_gate - (this)  = the wavelet stream's true contribution
#  Identical to the wavelet+gate run EXCEPT --wavelet 0.
#  --downsampling 16 (first -le is /16). Fresh run_name.
#  (# Windows full sweep needs --distribution 0 xlsx readable; else run Colab.)
# ============================================================
datapath=/content/mvtec_anomaly_detection
augpath=/content/dtd/images

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --run_name mnv2_dw2_gateonly_20ep \
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
    --gate 1 \
    --wavelet 0 \
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
