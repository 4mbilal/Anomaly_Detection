@echo off
REM ============================================================
REM  3-layer MobileNetV2 (depthwise) + PCA-residual stream (Windows).
REM  Backbone layers (unchanged from dw3):
REM    -le features.6.conv.1   192ch /8   (reference grid)
REM    -le features.8.conv.1   384ch /16
REM    -le features.12.conv.1  576ch /16
REM  Added stream:
REM    --pca 1   patch-level PCA reconstruction-residual map, fit on normal
REM              training images, injected as a 4th feature stream.
REM  --downsampling 8 (first -le layer is /8). Fresh run_name. --pca 0 == dw3.
REM  Requirements: fg_mask present (--fg 1); AUGPATH = dtd\images (texture
REM  subfolders); do NOT change --resize/--imagesize.
REM  EDIT the two paths below.
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name mnv2_dw3_pca_20ep ^
  net ^
    -b mobilenetv2 ^
    -le features.6.conv.1 ^
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
    --pca 1 ^
    --pca_components 16 ^
    --pca_patch 8 ^
    --pca_stride 8 ^
  dataset ^
    --distribution 0 ^
    --mean 0.5 ^
    --std 0.1 ^
    --fg 1 ^
    --rand_aug 1 ^
    --downsampling 8 ^
    --batch_size 8 ^
    --num_workers 4 ^
    --resize 288 ^
    --imagesize 288 ^
    -d carpet -d grid -d leather -d tile -d wood -d bottle -d cable -d capsule ^
    -d hazelnut -d metal_nut -d pill -d screw -d toothbrush -d transistor -d zipper ^
    mvtec %DATAPATH% %AUGPATH%
