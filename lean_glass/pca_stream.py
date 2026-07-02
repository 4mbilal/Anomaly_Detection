"""Patch-level PCA reconstruction-residual stream (additive to GLASS).

Fits a PCA basis on patches of the NORMAL training images, then at inference
returns a per-location reconstruction-*residual* map: the energy each patch
leaves unexplained by the normal subspace. The residual (not the projection)
is the anomaly cue -- it is large exactly where the image departs from normal.

This is injected as one extra "layer" in GLASS._embed and flows through the
existing (parameter-free) preprocessing/aggregator path. It is gated by a flag;
when disabled, GLASS behaves exactly as before.
"""
import torch
import torch.nn.functional as F


class PCAPatchResidual:
    def __init__(self, patch_size=8, stride=8, n_components=16, max_patches=100000):
        self.patch_size = patch_size
        self.stride = stride
        self.n_components = n_components
        self.max_patches = max_patches
        self.out_channels = 1          # residual-energy map
        self.mean = None
        self.components = None
        self.fitted = False
        self.device = None

    def _patch_dim(self, img):
        return img.shape[1] * self.patch_size * self.patch_size

    def _unfold(self, x):
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)  # (B, C*p*p, L)
        return patches.transpose(1, 2)  # (B, L, D)

    @torch.no_grad()
    def fit(self, dataloader, device):
        self.device = device
        collected, total = [], 0
        for data in dataloader:
            img = data["image"].to(device)
            P = self._unfold(img).reshape(-1, self._patch_dim(img))
            if P.shape[0] > 2048:  # subsample per batch to bound memory
                idx = torch.randperm(P.shape[0], device=device)[:2048]
                P = P[idx]
            collected.append(P.cpu())
            total += P.shape[0]
            if total >= self.max_patches:
                break
        X = torch.cat(collected, 0)[: self.max_patches]
        mean = X.mean(0, keepdim=True)
        Xc = X - mean
        _, _, V = torch.linalg.svd(Xc, full_matrices=False)
        self.mean = mean.to(device)
        self.components = V[: self.n_components].to(device)
        self.fitted = True

    @torch.no_grad()
    def residual_map(self, images):
        b, _, h, w = images.shape
        P = self._unfold(images)                  # (B, L, D)
        flat = P.reshape(-1, P.shape[-1])
        c = flat - self.mean
        proj = c @ self.components.t()            # project onto normal subspace
        recon = proj @ self.components            # reconstruct
        res = (c - recon).pow(2).mean(1)          # per-patch residual energy
        oh = (h - self.patch_size) // self.stride + 1
        ow = (w - self.patch_size) // self.stride + 1
        return res.reshape(b, 1, oh, ow).to(images.dtype)

    def state(self):
        if not self.fitted:
            return None
        return {"mean": self.mean.cpu(), "components": self.components.cpu()}

    def load_state(self, s):
        dev = self.device or "cpu"
        self.mean = s["mean"].to(dev)
        self.components = s["components"].to(dev)
        self.fitted = True
