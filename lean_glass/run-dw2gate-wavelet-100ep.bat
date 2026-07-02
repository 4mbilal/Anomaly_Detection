@echo off
REM 100-epoch: dw2 + GATE + WAVELET. Matched partner to run-dw2gate-100ep.
REM ONLY difference vs dw2+gate@100 (0.9923) is --wavelet 1. Compare pair.
REM Read pixel_ap / pixel_pro on textural classes (carpet, wood, grid):
REM that is where the wavelet showed durable-looking signal at 20ep.
set DATAPATH=D:\RnD\Frameworks\Datasetsnomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasetsnomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name mnv2_dw2_gate_wav_100ep ^
  net ^
    -b mobilenetv2 ^
    -le features.8.conv.1 ^
    -le features.12.conv.1 ^
    --pretrain_embed_dimension 1536 ^
    --target_embed_dimension 1536 ^
    --patchsize 3 ^
    --meta_epochs 100 ^
    --eval_epochs 1 ^
    --dsc_layers 2 ^
    --dsc_hidden 1024 ^
    --pre_proj 1 ^
    --mining 1 ^
    --noise 0.015 ^
    --radius 0.75 ^
    --p 0.5 ^
    --step 20 ^
    --limit 392 ^
    --gate 1 ^
    --wavelet 1 ^
    --wavelet_levels 2 ^
  dataset ^
    --distribution 0 ^
    --mean 0.5 ^
    --std 0.1 ^
    --fg 1 ^
    --rand_aug 1 ^
    --downsampling 16 ^
    --batch_size 8 ^
    --num_workers 4 ^
    --resize 288 ^
    --imagesize 288 ^
    -d carpet -d grid -d leather -d tile -d wood -d bottle -d cable -d capsule ^
    -d hazelnut -d metal_nut -d pill -d screw -d toothbrush -d transistor -d zipper ^
    mvtec %DATAPATH% %AUGPATH%
