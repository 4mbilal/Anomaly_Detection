# GLASS (minimal) — stripped baseline

This is the official GLASS repo (https://github.com/cqylunlun/glass) reduced to
the minimum needed to reproduce an MVTec AD baseline. **The algorithm is
unchanged.** Only peripheral files were removed and `main.py`'s dataset table was
narrowed to MVTec. See the original `LICENSE` and `README.md` (kept) for full
detail and citation; this is their work.

## What was removed (and why it's safe)

- `onnx/` — ONNX export / runtime. Deployment only, not used in train/eval.
- `figures/` — README images.
- `datasets/visa.py` and the `visa`/`mpdd`/`wfdd` entries in `main.py` — other
  datasets. (VisA is one file to restore later for cross-dataset evaluation.)
- `datasets/excel/mvtec_distribution.xlsx` — KEPT. Required: with `--distribution 0`
  ("choose by file") the trainer reads each class's manifold-vs-hypersphere
  decision from this file. Without it the run silently falls into the
  distribution-judgment branch and exits without training. (Only the MVTec file
  is kept; visa/mpdd/wfdd ones were removed with their datasets.)
- `shell/run-{visa,mpdd,wfdd,mad-man,mad-sys,custom}.sh` — non-MVTec launchers.
- ONNX / CUDA-python / matplotlib lines in `requirements.txt` (not imported by
  any kept file).

## What was NOT touched (the algorithm core)

`glass.py` (training + tester + global/local synthesis), `loss.py`, `model.py`
(Projection, Discriminator, PatchMaker), `common.py` (feature aggregation),
`backbones.py`, `perlin.py`, `metrics.py`, `utils.py`, `datasets/mvtec.py`.

## Requirements / data

```bash
pip install -r requirements.txt    # pin numpy 1.26.x (imgaug is not numpy-2 ready)
```

Two datasets are needed, because GLASS's local synthesis blends in DTD textures:

- MVTec AD  -> `datapath`
- DTD (Describable Textures) images -> `augpath` (the `aug_path` argument)

## Run the baseline (step 1)

Edit `shell/run-mvtec.sh` so `datapath` and `augpath` point at your copies, then:

```bash
cd shell && bash run-mvtec.sh
```

Defaults reproduce the paper setup: `wideresnet50`, layers `layer2`+`layer3`,
resize/imagesize 288, noise 0.015, radius 0.75, mining on. Results (image/pixel
AUROC, AP, PRO) are written under `results/`. **Lock these numbers before
changing anything** — they are your baseline.

## Windows

Two batch files in the repo root wrap the long command lines. Edit the
`DATAPATH` and `AUGPATH` variables at the top of each, activate your venv, and
run from the repo root:

```bat
smoke-test.bat   :: 1 class, 4 epochs, 256px -- confirms the pipeline runs
run-mvtec.bat    :: full 15-class baseline (paper settings)
```

Windows notes:
- Use Python 3.10/3.11 and install torch from the matching index
  (`--index-url .../cu118` for GPU, `.../cpu` for CPU-only). Keep numpy 1.26.x.
- `num_workers` is set to 4, not 0: the DataLoaders hardcode `prefetch_factor=2`,
  which errors when workers is 0 on torch 2.1.
- CPU fallback: `utils.set_torch_device` now also checks `torch.cuda.is_available()`,
  so `--gpu 0` resolves to CPU on a machine without CUDA. (One-line portability
  edit; the algorithm is untouched.)
- The full baseline on wideresnet50 is a long GPU job and impractical on CPU; use
  the laptop for `smoke-test.bat` and run the real baseline on a GPU / Colab.

## Where step 2 (MobileNetV2) plugs in

Backbone selection is the `-b` flag, resolved in `backbones.py` via a name ->
loader table. MobileNetV2 is not in that table yet; adding it is the step-2 edit,
and it also requires choosing which feature stages to pass to `-le`
(MobileNetV2's stage names differ from ResNet's `layer2`/`layer3`). That is a
deliberate one-variable change to make and measure against the locked baseline,
so it is intentionally left undone here.

## ROC curves

On the eval path the run now also saves image- and pixel-level ROC curves to
`results/curves/<class>_roc.png` (additive plotting via `metrics.save_roc_curves`;
it does not change any reported metric). Requires matplotlib.
