#!/usr/bin/env bash
# Matched ResNet18-GLASS baseline on additional dataset: pill.
# Same split/schedule/config as the proposed model; ONLY the backbone differs.
# Faithful GLASS light backbone: layer2 (/8) + layer3, NO gate.
# Maskless dataset: fg=0, forced distribution 3, image-level metrics only.
# NOTE: layer2 is /8  ->  --downsampling 8 (differs from the MobileNetV2 scripts).
datapath=/content/anomaly_extra
augpath=/content/dtd/images

python main.py \
    --gpu 0 --seed 0 --test ckpt --run_name pill_resnet18_100ep \
  net \
    -b resnet18 -le layer2 -le layer3 \
    --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
    --patchsize 3 --meta_epochs 100 --eval_epochs 1 \
    --dsc_layers 2 --dsc_hidden 1024 --pre_proj 1 \
    --mining 1 --noise 0.015 --radius 0.75 --p 0.5 --step 20 --limit 392 --gate 0 \
  dataset \
    --distribution 3 --mean 0.5 --std 0.1 --fg 0 --rand_aug 1 \
    --downsampling 8 --batch_size 8 --num_workers 4 --resize 288 --imagesize 288 \
    -d pill \
    mvtec $datapath $augpath
