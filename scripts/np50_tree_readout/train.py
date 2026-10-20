"""Fit matched raw-feature trees, and forecast-calibration trees."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scripts.np50_tree_readout.common import *


def count_prediction(model, x, loss):
    z = model.predict(x)
    return np.maximum(np.expm1(z) if loss == 'log' else z, 0.)


def train(args):
    _, _, out = paths(args.dataset); root = out / f'seed_{args.seed}'; root.mkdir(exist_ok=True)
    features = torch.load(out / 'features.pt', weights_only=False, map_location='cpu')
    for key, config in ACTION_CONFIGS.items():
        folder = root / 'action' / key; folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'validation.npz').exists(): continue
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=.05,
            max_leaf_nodes=config['leaves'], l2_regularization=config['regularization'],
            early_stopping=False, random_state=args.seed)
        model.fit(features['train']['x'], features['train']['target'])
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
        joblib.dump(model, folder / 'model.joblib')
        np.savez_compressed(folder / 'validation.npz', target=features['validation']['target'],
                            probability=model.predict_proba(features['validation']['x']))
        print(dict(completed='action', dataset=args.dataset, seed=args.seed, config=key), flush=True)
    del features
    forecasts = torch.load(root / 'forecast_inputs.pt', weights_only=False, map_location='cpu')
    for name in MODELS:
        tr, va = forecasts[name]['train'], forecasts[name]['validation']
        for key, config in COUNT_CONFIGS.items():
            folder = root / name / key; folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'validation.npz').exists(): continue
            model = HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
                max_leaf_nodes=config['leaves'], l2_regularization=100., early_stopping=False,
                loss='squared_error' if config['loss'] == 'log' else 'poisson', random_state=args.seed)
            model.fit(tr['x'], np.log1p(tr['target']) if config['loss'] == 'log' else tr['target'])
            joblib.dump(model, folder / 'model.joblib')
            np.savez_compressed(folder / 'validation.npz', prediction=count_prediction(model, va['x'], config['loss']))
            print(dict(completed='count', dataset=args.dataset, seed=args.seed, model=name, config=key), flush=True)
    write(root / 'training_completion.json', dict(complete=True, test_used=False,
          action_trees=len(ACTION_CONFIGS), count_trees=len(COUNT_CONFIGS)*2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dataset', choices=list(SPEC), required=True)
    p.add_argument('--seed', type=int, required=True); args = p.parse_args()
    torch.set_num_threads(1); verify_plan(); train(args)
