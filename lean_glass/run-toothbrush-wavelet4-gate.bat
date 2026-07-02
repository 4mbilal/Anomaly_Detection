@echo off
REM ============================================================
REM  TOOTHBRUSH only: dw2 + GATE + 4-LEVEL wavelet (quick test, 20 epochs).
REM    -le features.8.conv.1   384ch /16
REM    -le features.12.conv.1  576ch /16
REM    --wavelet 1 --wavelet_levels 4   (12 detail channels; adds coarser bands)
REM    --gate 1
REM  NOTE: more levels add COARSER bands, not finer resolution -- the stream is
REM  pooled to the 18x18 reference grid regardless. Compare image_auroc and
REM  pixel_ap/pro against the 2-level toothbrush run (tb_dw2_wav_gate).
REM  --distribution 2 = MANIFOLD (toothbrush's shipped value) -> no xlsx lookup.
REM  --downsampling 16 (first -le is /16). Fresh run_name.
REM  EDIT the two paths below.
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name tb_dw2_wav4_gate_20ep ^
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
    --wavelet_levels 4 ^
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
