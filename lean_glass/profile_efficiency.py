"""End-to-end efficiency profiler for GLASS configs.

Measures the DEPLOYMENT inference path only (no synthesis / loss / optimizer):
    image -> backbone(truncated at last extracted layer) -> preprocessing
          -> gate -> aggregator -> pre_projection -> discriminator -> scores

Reports, per config:
  - parameter count, split BACKBONE vs HEAD (preprocessing/aggregator are
    parameter-free; head = gate + pre_projection + discriminator)
  - FLOPs of the full inference path (fvcore; MACs reported, x2 for FLOPs)
  - inference latency (mean +/- std) at batch 1 and a larger batch, with
    warmup and CUDA synchronization

The backbone is truncated automatically: GLASS's aggregator raises
LastLayerToExtractReachedException at the last extracted layer, so layers
beyond it never execute and are excluded from FLOPs and latency.

Run on EACH device (laptop CPU/GPU, then Jetson):
    python profile_efficiency.py --device cuda
    python profile_efficiency.py --device cpu
FLOPs and params are device-independent (consistency check across machines);
latency and are what differ per device.

Weights are random (weights=None): params/FLOPs/latency depend on architecture,
not weight values, so no download is needed.
"""
import argparse
import time
import numpy as np
import torch
import torchvision.models as models

import glass as glass_mod


CONFIGS = {
    "glass_resnet18": dict(
        backbone="resnet18", layers=["layer2", "layer3"], gate=0, wavelet=0,
    ),
    "glass_mobilenetv2_dw2_gate": dict(
        backbone="mobilenetv2", layers=["features.8.conv.1", "features.12.conv.1"],
        gate=1, wavelet=0,
    ),
    # optional extra rows:
    "glass_mobilenetv2_dw2_gate_wavelet": dict(
        backbone="mobilenetv2", layers=["features.8.conv.1", "features.12.conv.1"],
        gate=1, wavelet=1,
    ),
    "glass_wideresnet50": dict(
        backbone="wideresnet50", layers=["layer2", "layer3"], gate=0, wavelet=0,
    ),
}

_BUILDERS = {
    "resnet18": lambda: models.resnet18(weights=None),
    "mobilenetv2": lambda: models.mobilenet_v2(weights=None),
    "wideresnet50": lambda: models.wide_resnet50_2(weights=None),
}


def build_glass(cfg, device, imagesize=288):
    bb = _BUILDERS[cfg["backbone"]]()
    bb.name = cfg["backbone"]
    bb.seed = None
    g = glass_mod.GLASS(device)
    g.load(
        backbone=bb,
        layers_to_extract_from=cfg["layers"],
        device=device,
        input_shape=[3, imagesize, imagesize],
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        meta_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        pre_proj=1,
        gate=cfg["gate"],
        wavelet=cfg["wavelet"],
    )
    for m in g.forward_modules.values():
        m.eval()
    if g.pre_proj > 0:
        g.pre_projection.eval()
    g.discriminator.eval()
    return g


class InferenceForward(torch.nn.Module):
    """The deployed forward: embed -> projection -> discriminator -> scores."""
    def __init__(self, g):
        super().__init__()
        self.g = g

    @torch.no_grad()
    def forward(self, images):
        feats, _ = self.g._embed(images, evaluation=True)
        if self.g.pre_proj > 0:
            feats = self.g.pre_projection(feats)
        scores = self.g.discriminator(feats)
        return scores


def count_params(g):
    """Split parameters: backbone (executed part) vs head."""
    bb = g.forward_modules["feature_aggregator"].backbone
    backbone_p = sum(p.numel() for p in bb.parameters())
    head_p = 0
    if g.pre_proj > 0:
        head_p += sum(p.numel() for p in g.pre_projection.parameters())
    head_p += sum(p.numel() for p in g.discriminator.parameters())
    if g.stream_gate is not None:
        head_p += sum(p.numel() for p in g.stream_gate.parameters())
    return backbone_p, head_p


def measure_flops(model, device, imagesize=288):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    x = torch.randn(1, 3, imagesize, imagesize, device=device)
    fca = FlopCountAnalysis(model, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total()  # MACs


def measure_latency(model, device, imagesize=288, batch=1, warmup=10, iters=50):
    x = torch.randn(batch, 3, imagesize, imagesize, device=device)
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return ts.mean(), ts.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--imagesize", type=int, default=288)
    ap.add_argument("--batch", type=int, default=8, help="second (throughput) batch size")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--configs", nargs="*", default=["glass_resnet18", "glass_mobilenetv2_dw2_gate"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"device: {device}  | imagesize: {args.imagesize}\n")
    header = f"{'config':38} {'backbone_P':>11} {'head_P':>9} {'total_P':>11} {'GMACs':>8} {'lat_b1_ms':>14} {'lat_b'+str(args.batch)+'_ms':>14}"
    print(header); print("-" * len(header))

    for name in args.configs:
        cfg = CONFIGS[name]
        g = build_glass(cfg, device, args.imagesize)
        fwd = InferenceForward(g).to(device).eval()

        bb_p, head_p = count_params(g)
        macs = measure_flops(fwd, device, args.imagesize)
        m1, s1 = measure_latency(fwd, device, args.imagesize, batch=1, iters=args.iters)
        mb, sb = measure_latency(fwd, device, args.imagesize, batch=args.batch, iters=args.iters)

        gmacs = f"{macs/1e9:.2f}" if macs is not None else "n/a"
        print(f"{name:38} {bb_p/1e6:>9.3f}M {head_p/1e6:>7.3f}M {(bb_p+head_p)/1e6:>9.3f}M "
              f"{gmacs:>8} {m1:>8.2f}+-{s1:<4.2f} {mb:>8.2f}+-{sb:<4.2f}")

    print("\nNotes: P=parameters; GMACs=multiply-accumulates (FLOPs ~ 2x MACs);")
    print("backbone is truncated at the last extracted layer; latency is warmed + synced.")


if __name__ == "__main__":
    main()
