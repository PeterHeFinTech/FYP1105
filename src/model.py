"""Time-conditioned MLP; official TabDDPM mixed diffusion objective/sampler."""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


class Denoiser(nn.Module):
    def __init__(self, width, hidden=256):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(width, hidden)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, width))

    def forward(self, x, t, y=None):
        # y from the upstream sampler is deliberately unused: approval is a
        # categorical coordinate of x, not a separately predicted target.
        freq = torch.exp(-math.log(10000) * torch.arange(64, device=x.device) / 64)
        phase = t.float()[:, None] * freq[None, :]
        emb = torch.cat([phase.cos(), phase.sin()], dim=1)
        return self.net(self.input(x) + self.time(emb))


def make_model(num_features, cardinalities, steps, hidden, device):
    denoiser = Denoiser(num_features + sum(cardinalities), hidden)
    return GaussianMultinomialDiffusion(
        np.array(cardinalities), num_features, denoiser,
        num_timesteps=steps, scheduler='cosine', device=torch.device(device)
    ).to(device)
