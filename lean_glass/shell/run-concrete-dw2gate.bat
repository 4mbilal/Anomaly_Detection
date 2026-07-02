@echo off
REM Additional dataset: concrete. Image-level metrics only (no GT masks).
REM Locked model: MobileNetV2 dw2 + gate, 100 epochs, fg=0, forced distribution 2.
set datapath=C:\data\anomaly_extra
set augpath=C:\data\dtd\images

python main.py ^
    --gpu 0 --seed 0 --test ckpt --run_name concrete_dw2gate_100ep ^
  net ^
    -b mobilenetv2 -le features.8.conv.1 -le features.12.conv.1 ^
    --pretrain_embed_dimension 1536 --target_embed_dimension 1536 ^
    --patchsize 3 --meta_epochs 100 --eval_epochs 1 ^
    --dsc_layers 2 --dsc_hidden 1024 --pre_proj 1 ^
    --mining 1 --noise 0.015 --radius 0.75 --p 0.5 --step 20 --limit 392 --gate 1 ^
  dataset ^
    --distribution 2 --mean 0.5 --std 0.1 --fg 0 --rand_aug 1 ^
    --downsampling 16 --batch_size 8 --num_workers 4 --resize 288 --imagesize 288 ^
    -d concrete ^
    mvtec %datapath% %augpath%
