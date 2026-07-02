"""Discrete wavelet high-frequency stream (additive to GLASS).

Returns multi-level detail-subband magnitudes (LH, HL, HH per level) of the
input luminance, upsampled to a common grid and concatenated. This is a fixed
transform (no parameters, no fitting): it carries fine, local edge/texture
information -- the high-frequency content most likely to be missing from a
lean backbone, and the kind of cue fine-defect classes need.

Injected as an extra "layer" in GLASS._embed, like the PCA stream, and gated by
a flag; disabled by default. Pair with --gate 1 so it isn't force-averaged at
equal weight.
"""
import torch
import torch.nn.functional as F

_FILTERS = {
    "haar": {"lo": [0.70710678, 0.70710678], "hi": [0.70710678, -0.70710678]},
    "db2":  {"lo": [-0.12940952, 0.22414387, 0.83651630, 0.48296291],
             "hi": [-0.48296291, 0.83651630, -0.22414387, -0.12940952]},
}


def _detail_bank(wavelet):
    f = _FILTERS[wavelet]
    lo = torch.tensor(f["lo"], dtype=torch.float32)
    hi = torch.tensor(f["hi"], dtype=torch.float32)
    bank = torch.stack([torch.outer(lo, hi),   # LH
                        torch.outer(hi, lo),   # HL
                        torch.outer(hi, hi)])  # HH
    return bank.unsqueeze(1)  # (3,1,k,k)


class WaveletStream:
    def __init__(self, wavelet="haar", levels=2):
        self.levels = int(levels)
        self.bank = _detail_bank(wavelet)
        lo = _FILTERS[wavelet]["lo"]
        self.lo = torch.tensor(lo).view(1, 1, 1, -1)
        self.k = self.bank.shape[-1]
        self.out_channels = 3 * self.levels
        self._device = None

    def _to(self, device):
        if self._device != device:
            self.bank = self.bank.to(device)
            self.lo = self.lo.to(device)
            self._device = device

    def _luma(self, x):
        if x.shape[1] == 3:
            w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            return (x * w).sum(1, keepdim=True)
        return x

    @torch.no_grad()
    def _detail(self, x):
        pad = self.k // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(xp, self.bank, stride=2)  # (B,3,h/2,w/2)

    @torch.no_grad()
    def _approx(self, x):
        lo_row = self.lo
        lo_col = lo_row.transpose(-1, -2)
        pad = self.lo.shape[-1] // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        a = F.conv2d(xp, lo_row, stride=(1, 2))
        a = F.conv2d(a, lo_col, stride=(2, 1))
        return a

    @torch.no_grad()
    def feature_map(self, images):
        self._to(images.device)
        x = self._luma(images)
        bands, approx, ref = [], x, None
        for _ in range(self.levels):
            d = self._detail(approx).abs()
            if ref is None:
                ref = d.shape[-2:]
            d = F.interpolate(d, size=ref, mode="bilinear", align_corners=False)
            bands.append(d)
            approx = self._approx(approx)
        return torch.cat(bands, dim=1).to(images.dtype)
