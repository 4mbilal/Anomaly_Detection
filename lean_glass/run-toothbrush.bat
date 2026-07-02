@echo off
REM ============================================================
REM  GLASS baseline: toothbrush only, full paper settings.
REM  Train + test for a faithful comparison against published GLASS.
REM
REM  Requirements before running:
REM   - venv active, numpy 1.26.x
REM   - fg_mask downloaded: %DATAPATH%\fg_mask\toothbrush\*.png  (because --fg 1)
REM   - AUGPATH must contain DTD texture subfolders, each full of .jpg
REM   - DO NOT change --resize / --imagesize: toothbrush uses a special
REM     329/288 resize ratio that only matches the paper at imagesize 288.
REM  --distribution 2 forces the MANIFOLD hypothesis (toothbrush's shipped
REM  baseline value), so it does not need the distribution xlsx lookup.
REM  EDIT the two paths below before running.
REM ============================================================
set DATAPATH=D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasets\anomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
  net ^
    -b wideresnet50 ^
    -le layer2 ^
    -le layer3 ^
    --pretrain_embed_dimension 1536 ^
    --target_embed_dimension 1536 ^
    --patchsize 3 ^
    --meta_epochs 640 ^
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
  dataset ^
    --distribution 2 ^
    --mean 0.5 ^
    --std 0.1 ^
    --fg 1 ^
    --rand_aug 1 ^
    --batch_size 8 ^
    --num_workers 4 ^
    --resize 288 ^
    --imagesize 288 ^
    -d toothbrush ^
    mvtec %DATAPATH% %AUGPATH%
