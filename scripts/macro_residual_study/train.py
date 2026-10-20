"""Train low-capacity neural count bases and state/sequence residuals."""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from dsg2former.experiment import set_seed
from dsg2former.paper_optimizer import TensorFlowAdam
from dsg2former.residual_popularity import NumericCountBase, ResidualPopularity
from dsg2former.pooled_metrics import pooled_popularity_metrics
from scripts.baseline_comparison.prepare_features import sha
from scripts.macro_residual_study.common import R, CACHE, META, read, write, partition, micro_state


def numeric(a, resources):
    x = a['ics'] if resources['group'] == 'ics' else np.c_[a['ics'], a['profile']]
    z = (x-resources['mean'])/resources['scale']
    return torch.as_tensor(z if resources['group'] == 'ics' else np.clip(z, -12, 12), dtype=torch.float32)


def resources(a, group):
    if group == 'ics':
        return dict(group=group, mean=np.zeros(3), scale=np.ones(3))
    scaler = StandardScaler().fit(np.c_[a['ics'], a['profile']])
    return dict(group=group, mean=scaler.mean_, scale=scaler.scale_)


@torch.no_grad()
def base_prediction(model, a, normalizer):
    model.eval()
    return model(numeric(a, normalizer), torch.as_tensor(a['observed'], dtype=torch.float32)).numpy()


def save_validation(out, a, p):
    np.savez_compressed(out/'validation_predictions.npz', prediction=p, **{k: a[k] for k in META})


def fit_base(out, tr, va, group, seed):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completion.json').exists():
        return
    set_seed(seed+71); normalizer = resources(tr, group); x = numeric(tr, normalizer)
    observed = torch.as_tensor(tr['observed'], dtype=torch.float32); y = torch.as_tensor(tr['target'], dtype=torch.float32)
    model = NumericCountBase(x.shape[1]); optimizer = TensorFlowAdam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed+71); best = None; stale = 0; history = []
    for epoch in range(1, 101):
        model.train(); losses = []
        for _ in range(int(np.ceil(len(y)/256))):
            ix = torch.as_tensor(rng.integers(len(y), size=256)); optimizer.zero_grad(set_to_none=True)
            p = model(x[ix], observed[ix]); loss = (torch.log1p(p)-torch.log1p(y[ix])).square().mean()
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        p = base_prediction(model, va, normalizer); m = pooled_popularity_metrics(va['target'], p)
        rank = (m['msle'], m['mse']); improved = best is None or rank < best
        if improved:
            best = rank; state = copy.deepcopy(model.state_dict()); best_epoch = epoch; stale = 0
        else:
            stale += 1
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=m, improved=improved))
        if epoch >= 20 and stale >= 10:
            break
    model.load_state_dict(state); p = base_prediction(model, va, normalizer)
    torch.save(dict(architecture='NumericCountBase', options=dict(dim=x.shape[1]), state_dict=state,
        resources=normalizer, best_epoch=best_epoch, seed=seed), out/'model.pt')
    save_validation(out, va, p); write(out/'training_history.json', history)
    write(out/'completion.json', dict(complete=True, validation=pooled_popularity_metrics(va['target'], p), test_used=False, best_epoch=best_epoch))
    print(dict(fitted_base=str(out), epoch=best_epoch), flush=True)


def make_inputs(a, state, base_resources, state_resources):
    return (numeric(a, base_resources), torch.as_tensor(a['observed'], dtype=torch.float32),
        torch.as_tensor(a['sequence'], dtype=torch.float32),
        torch.as_tensor(np.clip((state-state_resources['mean'])/state_resources['scale'], -8, 8), dtype=torch.float32),
        torch.as_tensor(a['window'] == 0))


@torch.no_grad()
def predict(model, inputs):
    model.eval()
    return np.concatenate([model(*[x[i:i+1024] for x in inputs])[0].numpy() for i in range(0, len(inputs[0]), 1024)])


def normalized_risk(metrics, reference):
    return max(metrics['mse']/reference['mse'], metrics['msle']/reference['msle'])


def fit_residual(out, tr, va, ts, vs, cfg, seed, base_path, reference):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completion.json').exists():
        return
    base = torch.load(base_path/'model.pt', weights_only=False, map_location='cpu')
    set_seed(seed+883); options = dict(base_dim=base['options']['dim'], state_dim=ts.shape[1],
        first_cap=cfg['first_cap'], continuation_cap=cfg['continuation_cap'])
    model = ResidualPopularity(**options); model.base.load_state_dict(base['state_dict'])
    for p in model.base.parameters():
        p.requires_grad_(cfg['fine_tune'])
    scaler = StandardScaler().fit(ts); sr = dict(mean=scaler.mean_, scale=scaler.scale_)
    inputs = make_inputs(tr, ts, base['resources'], sr); valid = make_inputs(va, vs, base['resources'], sr)
    parameters = [p for name, p in model.named_parameters() if not name.startswith('base.')]
    groups = [dict(params=parameters, lr=.001, weight_decay=.01)]
    if cfg['fine_tune']:
        groups.append(dict(params=model.base.parameters(), lr=.0001, weight_decay=.001))
    optimizer = torch.optim.AdamW(groups); y = torch.as_tensor(tr['target'], dtype=torch.float32)
    raw_scale = max(1., float(y.square().mean())); rng = np.random.default_rng(seed+884)
    best = None; stale = 0; history = []
    for epoch in range(1, 81):
        model.train(); losses = []; order = rng.permutation(len(y))
        for start in range(0, len(order), 256):
            ix = torch.as_tensor(order[start:start+256]); optimizer.zero_grad(set_to_none=True)
            p, delta = model(*[x[ix] for x in inputs])
            loss = (torch.log1p(p)-torch.log1p(y[ix])).square().mean()
            loss = loss + cfg['raw_weight']*(p-y[ix]).square().mean()/raw_scale + cfg['penalty']*delta.square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); losses.append(float(loss.detach()))
        p = predict(model, valid); m = pooled_popularity_metrics(va['target'], p)
        rank = (normalized_risk(m, reference), m['msle']/reference['msle']+m['mse']/reference['mse'])
        improved = best is None or rank < best
        if improved:
            best = rank; state = copy.deepcopy(model.state_dict()); best_epoch = epoch; stale = 0
        else:
            stale += 1
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=m, risk=rank[0], improved=improved))
        if epoch >= 15 and stale >= 10:
            break
    model.load_state_dict(state); p = predict(model, valid)
    torch.save(dict(architecture='ResidualPopularity', options=options, state_dict=state,
        base_resources=base['resources'], state_resources=sr, best_epoch=best_epoch,
        base_checkpoint_sha256=sha(base_path/'model.pt'), seed=seed, config=cfg, reference=reference), out/'model.pt')
    save_validation(out, va, p); write(out/'training_history.json', history)
    write(out/'completion.json', dict(complete=True, best_epoch=best_epoch, validation=pooled_popularity_metrics(va['target'], p), test_used=False))
    print(dict(fitted_residual=str(out), epoch=best_epoch), flush=True)


def load_predict(out, a, state):
    cp = torch.load(out/'model.pt', weights_only=False, map_location='cpu')
    model = ResidualPopularity(**cp['options']); model.load_state_dict(cp['state_dict'])
    return predict(model, make_inputs(a, state, cp['base_resources'], cp['state_resources']))


def run(args):
    torch.set_num_threads(1); plan = read(R/'plan.json')
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    common = torch.load(R/'development_common.pt', weights_only=False, map_location='cpu')
    tr, va, _, _ = partition(common, args.fold)
    if args.mode == 'base':
        fit_base(R/'base'/args.group/f'fold_{args.fold}'/f'seed_{args.seed}', tr, va, args.group, args.seed)
        return
    cfg = plan['candidates'][args.config]
    states = torch.load(CACHE/args.backbone/f'seed_{args.seed}/development.pt', weights_only=False, map_location='cpu')
    ts, vs = micro_state(states, args.variant, args.fold)
    base_path = R/'base'/cfg['group']/f'fold_{args.fold}'/f'seed_{args.seed}'
    reference = read(R/'base'/'ics'/f'fold_{args.fold}'/f'seed_{args.seed}/completion.json')['validation']
    out = (R/'screen'/args.config/f'fold_{args.fold}'/args.variant if args.mode == 'screen'
           else R/'confirmed'/args.config/args.variant/args.backbone/f'seed_{args.seed}')
    fit_residual(out, tr, va, ts, vs, cfg, args.seed, base_path, reference)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['base', 'screen', 'confirm'])
    p.add_argument('--config'); p.add_argument('--group', default='ics'); p.add_argument('--fold', type=int, default=2)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--backbone', default='d48_strong_reg')
    p.add_argument('--variant', default='temporal'); run(p.parse_args())
