@echo off
REM ============================================================
REM  GLASS smoke test (Windows): 1 class, 4 epochs, small images.
REM  Purpose: confirm the pipeline runs end to end, NOT to get
REM  meaningful accuracy. Runs on CPU (slowly) or GPU.
REM  EDIT the two paths below before running.
REM  Uses --fg 0 (no foreground masks) so it runs without the fg_mask download.
REM ============================================================
set DATAPATH=C:\path\to\MVTec
REM AUGPATH must contain the texture subfolders (banded, blotchy, ...), each full of .jpg
set AUGPATH=C:\path\to\dtd\images

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
    --meta_epochs 4 ^
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
    --distribution 0 ^
    --fg 0 ^
    --rand_aug 1 ^
    --batch_size 2 ^
    --num_workers 4 ^
    --resize 256 ^
    --imagesize 256 ^
    -d bottle ^
    mvtec %DATAPATH% %AUGPATH%
