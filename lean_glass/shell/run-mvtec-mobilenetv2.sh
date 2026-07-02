#!/usr/bin/env bash
# ============================================================
#  STEP 2 experiment: MobileNetV2 backbone, all 15 MVTec classes.
#  Identical to the 20-epoch WRN50 baseline EXCEPT:
#    -b mobilenetv2                       (was wideresnet50)
#    -le features.6 -le features.13       (stride /8 and /16, mirror layer2/3)
#    --run_name mnv2_20ep                 (fresh dir; avoids ckpt-skip collision)
#  Everything else (epochs, noise, radius, sizes, fg, distribution) is held
#  fixed so the ONLY variable vs baseline is the backbone.
#  --distribution 0 reads datasets/excel/mvtec_distribution.xlsx (must be present).
#  EDIT the two paths below.
# ============================================================
datapath=/content/mvtec_anomaly_detection
augpath=/content/dtd/images

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --run_name mnv2_20ep \
  net \
    -b mobilenetv2 \
    -le features.6 \
    -le features.13 \
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
    --batch_size 8 \
    --num_workers 4 \
    --resize 288 \
    --imagesize 288 \
    -d carpet -d grid -d leather -d tile -d wood -d bottle -d cable -d capsule \
    -d hazelnut -d metal_nut -d pill -d screw -d toothbrush -d transistor -d zipper \
    mvtec $datapath $augpath
