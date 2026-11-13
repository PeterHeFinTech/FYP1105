"""Vertical split GAN experiment using independent local worker processes."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from two_party_quality_candidate import Codec, LABEL, net


def normalize(module):
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.utils.parametrizations.spectral_norm(layer)
    return module


def worker(pipe, path, groups, numeric, seed):
    import two_party_quality_candidate as candidate

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).set_index('alignment_id')
    candidate.NUM = [column for column in numeric if column in frame
                     and pd.to_numeric(frame[column], errors='coerce').notna().sum() > 1]
    codecs = [Codec(frame, group) for group in groups]
    real = [codec.encode(frame) for codec in codecs]
    encoder = normalize(net(sum(codec.width for codec in codecs), 64))
    optimizer = torch.optim.Adam(encoder.parameters(), lr=2e-4, betas=(.5, .9))
    import hashlib
    alignment = hashlib.sha256(pd.util.hash_pandas_object(
        frame[LABEL], index=True).to_numpy().tobytes()).hexdigest()
    pipe.send(([codec.width for codec in codecs], alignment))
    batch_number = 0
    while True:
        try:
            command, payload = pipe.recv()
        except EOFError:
            return
        if command == 'stop':
            return
        if command == 'labels':
            pipe.send(frame[LABEL].to_numpy())
            continue
        batch_number += 1
        # Both workers apply the same categorical draws to the shared logits.
        torch.manual_seed(10000 + batch_number)
        if command == 'decode':
            with torch.no_grad():
                pipe.send(pd.concat([
                    codec.decode(codec.activate(torch.from_numpy(x)).numpy())
                    for codec, x in zip(codecs, payload)
                ], axis=1))
            continue
        indices, arrays, classes = payload
        logits = [torch.from_numpy(x).requires_grad_(command == 'generator') for x in arrays]
        fake = [codec.activate(x) for codec, x in zip(codecs, logits)]
        observed = [x[indices] for x in real]
        encoder.requires_grad_(command == 'discriminator')
        fake_embedding = encoder(torch.cat(fake, dim=1))
        if command == 'discriminator':
            real_embedding = encoder(torch.cat(observed, dim=1))
            pipe.send((real_embedding.detach().numpy(), fake_embedding.detach().numpy()))
            real_grad, fake_grad = pipe.recv()
            optimizer.zero_grad()
            torch.autograd.backward(
                [real_embedding, fake_embedding],
                [torch.from_numpy(real_grad), torch.from_numpy(fake_grad)],
            )
        else:
            pipe.send(fake_embedding.detach().numpy())
            gradient = torch.from_numpy(pipe.recv())
            losses = []
            for codec, target, generated in zip(codecs, observed, fake):
                for _, start, numeric_field, _, _, states in codec.spec:
                    stop = start + len(states) + int(numeric_field)
                    actual = target[:, start:stop].reshape(classes, -1, stop-start)
                    sample = generated[:, start:stop].reshape(classes, -1, stop-start)
                    losses.append((sample.mean(1)-actual.mean(1)).abs().sum(1).mean())
                    if numeric_field:
                        losses.append((sample[:,:,0].std(1)-actual[:,:,0].std(1)).abs().mean())
            # Shared fields contribute on both workers in this prototype.
            loss = (fake_embedding * gradient).sum() + 2.5 * torch.stack(losses).mean()
            loss.backward()
            pipe.send([x.grad.numpy() for x in logits])
            continue
        optimizer.step()
        pipe.send(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/two_party_full'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--rows', type=int, default=10000)
    args = parser.parse_args()
    if min(args.epochs, args.rows, args.steps or 1) < 1:
        parser.error('epochs, rows and steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    schema = json.loads((args.source/'report.json').read_text())
    shared = schema['shared']
    private = [schema['party_a_private'], schema['party_b_private']]
    from two_party_quality_candidate import NUM
    numeric = NUM + ['loan_to_value_ratio','rate_spread','lender_credits','loan_term',
                     'prepayment_penalty_term','intro_rate_period','multifamily_affordable_units',
                     'ffiec_msa_md_median_family_income'] + [c for c in private[1] if c.startswith('tract_')]
    args.out.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    pipes, workers = [], []
    try:
        for index, name in enumerate(('a', 'b')):
            pipe, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(
                child, args.source/f'party_{name}_train.csv', [shared, private[index]], numeric, 43+index,
            ))
            process.start()
            child.close()
            pipes.append(pipe)
            workers.append(process)
        (wa, a_hash), (wb, b_hash) = [pipe.recv() for pipe in pipes]
        if a_hash != b_hash or wa[0] != wb[0]:
            raise ValueError('Party alignment, labels or shared width disagree')
        widths = [wa[0], wa[1], wb[1]]
        pipes[0].send(('labels', None))
        labels, y = np.unique(pipes[0].recv(), return_inverse=True)
        classes = len(labels)
        pools = [np.flatnonzero(y == i) for i in range(classes)]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = nn.Sequential(nn.Linear(64+classes,256),nn.ReLU(),nn.Linear(256,256),
                                  nn.ReLU(),nn.Linear(256,sum(widths))).to(device)
        discriminator = normalize(net(128+classes,1)).to(device)
        gen_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(.5,.9))
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5,.9))

        def inputs(codes):
            condition = F.one_hot(torch.as_tensor(codes, device=device), classes).float()
            noise = torch.randn(len(codes), 64, device=device)
            return condition, generator(torch.cat([noise, condition], 1)).split(widths, 1)

        def send(command, parts, indices=None):
            arrays = [x.detach().cpu().numpy() for x in parts]
            for pipe, unique in zip(pipes, arrays[1:]):
                view = [arrays[0], unique]
                pipe.send((command, view if indices is None else (indices, view, classes)))

        def embedding(array):
            return torch.tensor(array, device=device, requires_grad=True)

        for epoch in range(args.epochs):
            for _ in range(args.steps or (len(y)+255)//256):
                indices = np.concatenate([rng.choice(pool, 256//classes) for pool in pools])
                condition, fake = inputs(y[indices])
                send('discriminator', fake, indices)
                actual, generated = zip(*[pipe.recv() for pipe in pipes])
                actual, generated = list(map(embedding, actual)), list(map(embedding, generated))
                disc_opt.zero_grad()
                dl = (F.softplus(-discriminator(torch.cat([*actual, condition],1))).mean()
                      + F.softplus(discriminator(torch.cat([*generated, condition],1))).mean())
                dl.backward()
                disc_opt.step()
                for pipe, real, syn in zip(pipes, actual, generated):
                    pipe.send((real.grad.cpu().numpy(), syn.grad.cpu().numpy()))
                for pipe in pipes:
                    pipe.recv()
                condition, fake = inputs(y[indices])
                send('generator', fake, indices)
                generated = [embedding(pipe.recv()) for pipe in pipes]
                loss = F.softplus(-discriminator(torch.cat([*generated, condition],1))).mean()
                gradients = torch.autograd.grad(loss, generated)
                for pipe, gradient in zip(pipes, gradients):
                    pipe.send(gradient.cpu().numpy())
                (shared_a, grad_a), (shared_b, grad_b) = [pipe.recv() for pipe in pipes]
                gen_opt.zero_grad()
                torch.autograd.backward(fake, [torch.as_tensor(g, device=device)
                                               for g in (shared_a+shared_b, grad_a, grad_b)])
                gen_opt.step()
            print(f'epoch={epoch+1} discriminator_loss={dl.item():.4f}', flush=True)

        probabilities = np.bincount(y)/len(y)
        codes = rng.choice(classes, args.rows, p=probabilities)
        samples = []
        with torch.no_grad():
            for start in range(0, len(codes), 256):
                batch = codes[start:start+256]
                _, parts = inputs(batch)
                send('decode', parts)
                a, b = [pipe.recv() for pipe in pipes]
                # Decode shared fields only once in the output.
                sample = pd.concat([a, b[private[1]]], axis=1)
                sample[LABEL] = labels[batch]
                samples.append(sample)
        pd.concat(samples, ignore_index=True).to_csv(args.out/'synthetic.csv', index=False)
        torch.save({'generator':generator.state_dict(),'widths':widths,'labels':labels.tolist()},
                   args.out/'generator.pt')
        (args.out/'report.json').write_text(json.dumps({
            'shared':shared,'party_a_private':private[0],'party_b_private':private[1],
            'epochs':args.epochs,'rows':args.rows,'device':device,
            'exchange':['alignment digest','decision labels from A','batch indices',
                        'embeddings','gradients','synthetic logits and decoded synthetic rows'],
            'deployment':'Local process simulation. No formal privacy guarantee.',
        }, indent=2))
    finally:
        for pipe in pipes:
            pipe.close()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()


if __name__ == '__main__':
    main()
