@echo off
REM 100-epoch matched run: run-wrn50-100ep
REM Compare ONLY against the other 100-epoch run (same schedule).
set DATAPATH=D:\RnD\Frameworks\Datasetsnomaly\mvtec_anomaly_detection
set AUGPATH=D:\RnD\Frameworks\Datasetsnomaly\dtd\images

python main.py ^
    --gpu 0 ^
    --seed 0 ^
    --test ckpt ^
    --run_name wrn50_100ep ^
  net ^
    -b wideresnet50 ^
    -le layer2 ^
    -le layer3 ^
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
    --gate 0 ^
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
