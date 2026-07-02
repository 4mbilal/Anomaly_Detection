@echo off
REM ============================================================
REM  MobileNetV2, leaner 2-layer depthwise (/8 + /16) (Windows):
REM    -le features.4.conv.1   144ch /8   (== block_3_depthwise 28x28x144)
REM    -le features.8.conv.1   384ch /16  (== block_7_depthwise 14x14x384)
REM  528 concat channels (vs dw3's 1152). --downsampling 8. --pca 0.
REM  NOTE: full 15-class sweep needs --distribution 0 -> the xlsx
REM  datasets\excel\mvtec_distribution.xlsx must be readable from the
REM  launch dir. If it falls into "Distribution: HyperSphere" and exits,
REM  run on Colab, or run classes individually with --distribution 2
REM  (manifold) / 3 (hypersphere).
REM  EDIT the two paths below.
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name mnv2_dw2b_20ep ^
  net ^
    -b mobilenetv2 ^
    -le features.4.conv.1 ^
    -le features.8.conv.1 ^
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
    --pca 0 ^
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
