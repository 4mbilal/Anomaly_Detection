@echo off
REM ============================================================
REM  CONTROL: dw2 (features.8+12, /16) + per-stream GATE only. NO wavelet.
REM  Isolates the gate's effect on the two backbone streams, so:
REM    (this) - dw2_plain     = what the gate alone does
REM    dw2_wav_gate - (this)  = the wavelet stream's true contribution
REM  Identical to the wavelet+gate run EXCEPT --wavelet 0.
REM  --downsampling 16 (first -le is /16). Fresh run_name.
REM  (REM Windows full sweep needs --distribution 0 xlsx readable; else run Colab.)
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name mnv2_dw2_gateonly_20ep ^
  net ^
    -b mobilenetv2 ^
    -le features.8.conv.1 ^
    -le features.12.conv.1 ^
    --pretrain_embed_dimension 1536 ^
    --target_embed_dimension 1536 ^
    --patchsize 3 ^
    --meta_epochs 20 ^
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
    --wavelet 0 ^
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
