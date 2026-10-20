"""Validation-only temporal adapters and peak-aware neural count heads."""
import argparse
import copy
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from dsg2former.experiment import set_seed
from dsg2former.sds_aligned import PaperAlignedDSG2former, align_graph_positions
from dsg2former.peak_popularity import PeakAwarePopularity
from dsg2former.memory_count_gate import MemoryCountGate
from dsg2former.pooled_metrics import pooled_popularity_metrics
from scripts.baseline_comparison.train_method import rank, action_summary
from scripts.weibo_comparison import neural
from scripts.macro_residual_study import train as count
from scripts.np50_temporal_peak.common import *
from scripts.np50_temporal_peak.encode import encode, effective, probability, pool, META


def load_model(path):
    cp = torch.load(path, weights_only=False, map_location='cpu')
    model = PaperAlignedDSG2former(**cp['options'])
    model.load_state_dict(cp['state_dict'], strict=True)
    return model, cp


def fit_temporal(model, dev, policy, seed, out, cp, device):
    if (out / 'temporal_model.pt').exists():
        return load_model(out / 'temporal_model.pt')[0]
    out.mkdir(parents=True, exist_ok=True); set_seed(seed+211)
    model = model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('gate.', 'temporal_head.')))
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    tr, va = dev['train']['action'], dev['validation']['action']
    inputs = effective(tr, policy, device); target = tr['target'].to(device)
    vinputs = effective(va, policy, device); vy = va['target'].numpy()

    @torch.no_grad()
    def evaluate():
        model.eval()
        p = torch.cat([model.action_cached(*[x[i:i+8192] for x in vinputs]).softmax(-1).cpu()
                       for i in range(0, len(vy), 8192)]).numpy()
        return action_summary(vy, p), p

    metric, pred = evaluate(); best = rank(metric); state = copy.deepcopy(model.cpu().state_dict())
    model = model.to(device); best_epoch = 0; stale = 0
    history = [dict(epoch=0, validation=metric, improved=True)]
    rng = np.random.default_rng(seed+211); started = time.monotonic()
    for epoch in range(1, 101):
        model.train(); order = rng.permutation(len(target)); losses = []
        for start in range(0, len(order), 4096):
            ix = torch.as_tensor(order[start:start+4096], device=device)
            opt.zero_grad(set_to_none=True)
            logits = model.action_cached(*[x[ix] for x in inputs])
            loss = F.cross_entropy(logits, target[ix], label_smoothing=.02)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.); opt.step()
            losses.append(float(loss.detach()))
        metric, pred = evaluate(); q = rank(metric); improved = q > best
        if improved:
            best = q; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; stale = 0
        else:
            stale += 1
        row = dict(epoch=epoch, validation=metric, improved=improved, loss=float(np.mean(losses)))
        history.append(row)
        print(dict(stage='temporal', policy=policy, seed=seed, seconds=time.monotonic()-started, **row), flush=True)
        if epoch >= 20 and stale >= 15:
            break
    model.load_state_dict(state)
    for k, value in cp['state_dict'].items():
        if not k.startswith(('gate.', 'temporal_head.')):
            torch.testing.assert_close(value, state[k], atol=0, rtol=0)
    torch.save(dict(architecture='PaperAlignedDSG2former', options=cp['options'], state_dict=state,
                    policy=policy, best_epoch=best_epoch, seed=seed,
                    spatial_weights_frozen=True), out / 'temporal_model.pt')
    write(out / 'temporal_history.json', history)
    del inputs, vinputs, target; gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return model


def micro(args):
    work, root = paths(args.dataset); out = root / args.backbone / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'micro_completion.json').exists(): return
    parent = work / 'results/neural' / args.backbone / f'seed_{args.seed}'
    model, cp = load_model(parent / 'spatial_model.pt')
    cache = out / 'encoding_development.pt'
    if cache.exists():
        dev = torch.load(cache, weights_only=False, map_location='cpu')
    else:
        data = torch.load(work / 'data/pheme_model_improvement/graphs.pt', weights_only=False, map_location='cpu')
        align_graph_positions(data)
        dev = encode(model, data, ['train', 'validation'], args.device)
        torch.save(dev, cache); del data; gc.collect()
    # The parent temporal/spatial probabilities remain an explicit control.
    old = torch.load(parent / 'development_final.pt', weights_only=False, map_location='cpu')
    prior, _ = load_model(parent / 'temporal_model.pt')
    oldprob = neural.action_probability(prior, old['validation']['action'], 'temporal')
    np.savez_compressed(out / 'chain_validation.npz', probability=oldprob,
                        **{k: (old['validation']['action'][k].numpy() if k == 'target' else old['validation']['action'][k]) for k in META})
    for split in ['train', 'validation']:
        for k in META:
            np.testing.assert_array_equal(dev[split]['action'][k], old[split]['action'][k])
    del old, prior; gc.collect()
    for policy in POLICIES:
        model, _ = load_model(parent / 'spatial_model.pt')
        folder = out / policy
        model = fit_temporal(model, dev, policy, args.seed, folder, cp, args.device)
        state_path = folder / 'development_states.pt'
        if not state_path.exists():
            states = {}
            for split in ['train', 'validation']:
                state = pool(model, dev[split]['macro'], policy, True, args.device)
                states[split] = dict(state=state, **{k: dev[split]['macro'][k] for k in META})
            torch.save(states, state_path)
        p = probability(model, dev['validation']['action'], policy, args.device)
        a = dev['validation']['action']
        np.savez_compressed(folder / 'validation.npz', probability=p,
                            **{k: (a[k].numpy() if k == 'target' else a[k]) for k in META})
        write(folder / 'validation.json', action_summary(a['target'].numpy(), p))
        del model; gc.collect()
        if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
    coverage = {}
    for split in ['train', 'validation']:
        a = dev[split]['action']
        coverage[split] = dict(rows=len(a['target']), global_cache=float(a['mask'].float().mean()),
            global_bootstrap=float((a['mask'] | a['has_history']).float().mean()),
            historical_profile=float(a['has_history'].float().mean()))
    write(out / 'micro_completion.json', dict(complete=True, test_used=False, coverage=coverage,
          causal_audit=dev['causal_audit'], spatial_checkpoint_sha256=sha(parent / 'spatial_model.pt')))


def select_micro(name):
    _, root = paths(name); candidates = {}
    for policy in ['chain', *POLICIES]:
        runs = []
        for seed in SEEDS:
            pieces = []
            for backbone in BACKBONES:
                folder = root / backbone / f'seed_{seed}'
                path = folder / 'chain_validation.npz' if policy == 'chain' else folder / policy / 'validation.npz'
                with np.load(path) as a:
                    pieces.append(a['probability'].astype(float)); y = a['target']
            runs.append(action_summary(y, sum(pieces)/3))
        candidates[policy] = dict(per_seed=runs,
            mean={k: float(np.mean([r[k] for r in runs])) for k in ['accuracy', 'cross_entropy', 'auc']})
    selected = max(candidates, key=lambda k: rank(candidates[k]['mean']))
    write(root / 'micro_selection.json', dict(policy=selected, candidates=candidates,
        selection='Three-seed mean validation Accuracy, then CE, then AUC; equal three-backbone probabilities',
        test_used=False))
    print(dict(dataset=name, selected_micro_policy=selected, validation=candidates[selected]['mean']), flush=True)


def state_data(name, backbone, seed, variant):
    work, root = paths(name); parent = work / 'results/neural' / backbone / f'seed_{seed}'
    policy = read(root / 'micro_selection.json')['policy']
    if variant == 'spatial' or policy == 'chain':
        d = torch.load(parent / 'development_final.pt', weights_only=False, map_location='cpu')
        states = {s: d[s]['state'][variant] for s in ['train', 'validation']}
    else:
        d = torch.load(root / backbone / f'seed_{seed}' / policy / 'development_states.pt', weights_only=False, map_location='cpu')
        states = {s: d[s]['state'] for s in ['train', 'validation']}
    return states


def memory_predict(folder, common, state, peak=True):
    # Explicit classes prevent a process-local monkeypatch from loading a legacy
    # checkpoint as the new architecture (or the reverse).
    from dsg2former.residual_popularity import ResidualPopularity
    cp = torch.load(folder / 'model.pt', weights_only=False, map_location='cpu')
    cls = PeakAwarePopularity if peak else ResidualPopularity
    model = cls(**cp['options']); model.load_state_dict(cp['state_dict'], strict=True)
    p = count.predict(model, count.make_inputs(common, state, cp['base_resources'], cp['state_resources']))
    c = torch.load(folder / 'memory_gate.pt', weights_only=False, map_location='cpu')
    gate = MemoryCountGate(**c['options']); gate.load_state_dict(c['state_dict'])
    with torch.no_grad(): return gate(*neural.gate_inputs(common, p)).numpy()


def legacy_validation(name, backbone, seed, variant, common, state):
    work, _ = paths(name)
    folder = work / 'results/neural' / backbone / f'seed_{seed}' / variant
    return memory_predict(folder, common, state, peak=False)


def count_job(args):
    work, root = paths(args.dataset); parent = work / 'results/neural'
    config = args.config
    if config is None:
        config = read(root / 'macro_selection.json')[args.variant]['config']
    if config == 'legacy': return
    folder = root / 'counts' / config / args.variant / args.backbone / f'seed_{args.seed}'
    if (folder / 'completion_peak.json').exists(): return
    states = state_data(args.dataset, args.backbone, args.seed, args.variant)
    common = torch.load(parent / 'common.pt', weights_only=False, map_location='cpu')
    tr, va = common['train'], common['validation']; ts, vs = states['train'], states['validation']
    reference_p = legacy_validation(args.dataset, args.backbone, args.seed, args.variant, va, vs)
    reference = pooled_popularity_metrics(va['target'], reference_p)
    count.ResidualPopularity = PeakAwarePopularity
    count.fit_residual(folder, tr, va, ts, vs, MACRO_CONFIGS[config], args.seed,
                       parent / 'numeric_base' / f'seed_{args.seed}', reference)
    # Preserve the correct architecture identity, even though the generic trainer
    # writes the legacy architecture label before this explicit version wrapper.
    path = folder / 'model.pt'; cp = torch.load(path, weights_only=False, map_location='cpu')
    if cp['architecture'] != 'PeakAwarePopularity':
        cp['architecture'] = 'PeakAwarePopularity'; torch.save(cp, path)
    tp = count.load_predict(folder, tr, ts); vp = count.load_predict(folder, va, vs)
    neural.fit_gate(folder, tr, va, tp, vp)
    p = memory_predict(folder, va, vs)
    np.savez_compressed(folder / 'validation_final.npz', prediction=p, **{k: va[k] for k in META})
    write(folder / 'completion_peak.json', dict(complete=True, test_used=False,
        validation=pooled_popularity_metrics(va['target'], p), legacy_reference=reference,
        configuration=MACRO_CONFIGS[config], micro_policy=read(root / 'micro_selection.json')['policy']))


def select_macro(name):
    work, root = paths(name); selection = {}
    common = torch.load(work / 'results/neural/common.pt', weights_only=False, map_location='cpu')['validation']
    for variant in ['spatial', 'temporal']:
        state = state_data(name, 'd48_strong_reg', 42, variant)['validation']
        p = legacy_validation(name, 'd48_strong_reg', 42, variant, common, state)
        reference = pooled_popularity_metrics(common['target'], p)
        candidates = dict(legacy=reference)
        for config in MACRO_CONFIGS:
            candidates[config] = read(root / 'counts' / config / variant / 'd48_strong_reg/seed_42/completion_peak.json')['validation']
        def quality(k):
            m = candidates[k]
            return (count.normalized_risk(m, reference), m['mse']/reference['mse']+m['msle']/reference['msle'])
        selected = min(candidates, key=quality)
        selection[variant] = dict(config=selected, candidates=candidates,
            screening_backbone='d48_strong_reg', screening_seed=42,
            selection='Minimize worst normalized validation MSE/MSLE, then sum; frozen before all-seed confirmation and test', test_used=False)
    write(root / 'macro_selection.json', selection)
    print(dict(dataset=name, selected_macro={k: v['config'] for k, v in selection.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['micro', 'select_micro', 'count', 'select_macro'])
    p.add_argument('--dataset', choices=list(SPEC), required=True); p.add_argument('--backbone', choices=BACKBONES)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cpu')
    p.add_argument('--variant', choices=['spatial', 'temporal']); p.add_argument('--config', choices=list(MACRO_CONFIGS))
    args = p.parse_args(); torch.set_num_threads(1); verify_plan()
    if args.mode == 'micro': micro(args)
    elif args.mode == 'select_micro': select_micro(args.dataset)
    elif args.mode == 'count': count_job(args)
    else: select_macro(args.dataset)
