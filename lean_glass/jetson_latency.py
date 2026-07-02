"""Standalone inference-latency probe for the Jetson Nano.

Self-contained: depends ONLY on torch + torchvision, NOT on glass_min, so it
runs on the old Jetson stack (PyTorch 1.8 / torchvision 0.9 / Python 3.6) where
the full framework's modern-API code would not import.

It reproduces the DEPLOYED forward path's compute structure:
  backbone (truncated at the extraction layer, via an early-stop hook)
  -> per-patch features pooled to embed_dim (parameter-free, like GLASS Preprocessing)
  -> pre-projection (Linear embed->embed)
  -> discriminator MLP (embed->hidden->1)
The head is the SAME for every backbone, so the comparison isolates the backbone.

Measures batch-1 (per-frame deployment latency) and a throughput batch, with
warmup and CUDA synchronization. Reports mean +/- std in ms.

Usage on Jetson:
    sudo nvpmodel -m 0 && sudo jetson_clocks
    python3 jetson_latency.py --device cuda
    python3 jetson_latency.py --device cuda --backbones resnet18 mobilenetv2
"""
from __future__ import print_function
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


EMBED = 1536
HIDDEN = 1024

# (builder, list-of-extraction-layer-attribute-paths, downsampling-stride)
# torchvision 0.9 API: use pretrained=False (no `weights=`).
CONFIGS = {
    "resnet18":    dict(build=lambda: models.resnet18(pretrained=False),
                        layers=["layer2", "layer3"]),
    "mobilenetv2": dict(build=lambda: models.mobilenet_v2(pretrained=False),
                        layers=["features.8.conv.1", "features.12.conv.1"]),
    "wideresnet50": dict(build=lambda: models.wide_resnet50_2(pretrained=False),
                         layers=["layer2", "layer3"]),
}


class _Stop(Exception):
    pass


def get_module(model, dotted):
    m = model
    for p in dotted.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


class TruncatedBackbone(nn.Module):
    """Runs the backbone only up to the last extraction layer (early-stop)."""
    def __init__(self, model, layers):
        super(TruncatedBackbone, self).__init__()
        self.model = model
        self.layers = layers
        self.outputs = {}
        self._last = layers[-1]
        for name in layers:
            mod = get_module(model, name)
            target = mod[-1] if isinstance(mod, nn.Sequential) else mod
            target.register_forward_hook(self._mk(name))

    def _mk(self, name):
        def hook(mod, inp, out):
            self.outputs[name] = out
            if name == self._last:
                raise _Stop()
        return hook

    def forward(self, x):
        self.outputs = {}
        try:
            self.model(x)
        except _Stop:
            pass
        return [self.outputs[n] for n in self.layers]


class GlassLikeHead(nn.Module):
    """Parameter-free pooling to EMBED per stream + projection + discriminator."""
    def __init__(self, embed=EMBED, hidden=HIDDEN):
        super(GlassLikeHead, self).__init__()
        self.projection = nn.Linear(embed, embed)
        self.discriminator = nn.Sequential(
            nn.Linear(embed, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        self.embed = embed

    def forward(self, feats):
        # resize each stream to the first stream's grid, pool channels->embed
        ref = feats[0].shape[-2:]
        pooled = []
        for f in feats:
            if f.shape[-2:] != ref:
                f = nn.functional.interpolate(f, size=ref, mode="bilinear", align_corners=False)
            b, c, h, w = f.shape
            # per-location vector, adaptive-pooled to embed (parameter-free, like Preprocessing)
            v = f.permute(0, 2, 3, 1).reshape(b * h * w, c)
            v = nn.functional.adaptive_avg_pool1d(v.unsqueeze(1), self.embed).squeeze(1)
            pooled.append(v)
        x = torch.stack(pooled, dim=1).mean(dim=1)   # aggregate streams
        x = self.projection(x)
        return self.discriminator(x)


class FullInference(nn.Module):
    def __init__(self, backbone, layers):
        super(FullInference, self).__init__()
        self.bb = TruncatedBackbone(backbone, layers)
        self.head = GlassLikeHead()

    def forward(self, x):
        return self.head(self.bb(x))


def count_params(model):
    bb = sum(p.numel() for p in model.bb.model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return bb, head


def latency(model, device, imagesize, batch, warmup, iters):
    x = torch.randn(batch, 3, imagesize, imagesize, device=device)
    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.time()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1000.0)
    ts = np.array(ts)
    return ts.mean(), ts.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagesize", type=int, default=288)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--backbones", nargs="*", default=["resnet18", "mobilenetv2"])
    args = ap.parse_args()

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: {}  | torch {} | imagesize {}".format(device, torch.__version__, args.imagesize))
    print("cuda available: {}".format(torch.cuda.is_available()))
    print("")
    print("{:14} {:>11} {:>10} {:>11} {:>16} {:>16}".format(
        "backbone", "backbone_P", "head_P", "total_P", "lat_b1_ms", "lat_b%d_ms" % args.batch))
    print("-" * 82)

    for name in args.backbones:
        cfg = CONFIGS[name]
        model = FullInference(cfg["build"](), cfg["layers"]).to(device).eval()
        bb, head = count_params(model)
        m1, s1 = latency(model, device, args.imagesize, 1, args.warmup, args.iters)
        mb, sb = latency(model, device, args.imagesize, args.batch, args.warmup, args.iters)
        print("{:14} {:>9.3f}M {:>8.3f}M {:>9.3f}M {:>9.2f}+-{:<4.2f} {:>9.2f}+-{:<4.2f}".format(
            name, bb/1e6, head/1e6, (bb+head)/1e6, m1, s1, mb, sb))

    print("\nReport the Jetson software stack (JetPack/L4T, torch, torchvision) with these numbers.")
    print("batch-1 = per-frame deployment latency; larger batch = throughput context.")


if __name__ == "__main__":
    main()
