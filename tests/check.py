"""Small numerical checks only; no optimizer or HMDA training."""
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
import numpy as np
import torch
from model import make_model
from hmda import device_for


def main():
    device=device_for('auto')
    torch.manual_seed(7)
    model=make_model(2,[2,3],8,32,device)
    x=torch.cat([torch.randn(16,2),torch.randint(2,(16,1)),torch.randint(3,(16,1))],dim=1).to(device)
    cat,num=model.mixed_loss(x,{})
    loss=cat+num
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        sample,_=model.sample(4,torch.ones(1,device=device))
    assert sample.shape==(4,4) and torch.isfinite(sample).all()
    assert ((sample[:,2]>=0)&(sample[:,2]<2)).all()
    assert ((sample[:,3]>=0)&(sample[:,3]<3)).all()
    # The single dummy conditioning class must not affect the denoiser output.
    t=torch.zeros(4,dtype=torch.long,device=device)
    z=torch.randn(4,7,device=device)
    assert torch.equal(model._denoise_fn(z,t,y=torch.zeros(4,device=device)),model._denoise_fn(z,t,y=torch.ones(4,device=device)))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'state.pt';torch.save(model.state_dict(),p)
        copy=make_model(2,[2,3],8,32,device)
        copy.load_state_dict(torch.load(p,map_location=device,weights_only=True))
        assert torch.equal(model._denoise_fn(z,t),copy._denoise_fn(z,t))
    print(f'PASS on {device}: mixed loss, gradients, reverse sampling, category ranges, checkpoint round-trip. No training run.')


if __name__=='__main__':main()
