@echo off
REM ============================================================
REM  TOOTHBRUSH only: dw2 + WAVELET + per-stream GATE.
REM    -le features.8.conv.1   384ch /16
REM    -le features.12.conv.1  576ch /16
REM    --wavelet 1   2-level Haar detail subbands (6ch), fine edge/texture cue
REM    --gate 1      learnable per-stream weights (init=equal; trains per class)
REM                  so the wavelet stream isn't force-averaged at equal weight.
REM  --distribution 2 = MANIFOLD (toothbrush's shipped value) -> no xlsx lookup.
REM  --downsampling 16 (first -le layer is /16). Fresh run_name.
REM  Controls to compare against: dw2 (your locked result) and, ideally,
REM  dw2+gate with --wavelet 0 to isolate the gate from the wavelet.
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name tb_dw2_wav_gate_20ep ^
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
    --wavelet 1 ^
    --wavelet_levels 2 ^
  dataset ^
    --distribution 2 ^
    --mean 0.5 ^
    --std 0.1 ^
    --fg 1 ^
    --rand_aug 1 ^
    --downsampling 16 ^
    --batch_size 8 ^
    --num_workers 4 ^
    --resize 288 ^
    --imagesize 288 ^
    -d toothbrush ^
    mvtec %DATAPATH% %AUGPATH%
