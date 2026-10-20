# DSG²former

Research code for **Modeling Rumor Propagation via Dynamic Sentiment Game Driven Graph Transformer**.

DSG²former combines sentiment-game features, graph-relative multi-head attention, and chronological user-state fusion for next-hour action prediction. A separate neural count head forecasts first-time participants in the next hour.

This release contains the dense model, an explicitly separate sparse-attention variant, the retained staged training modules, and six dataset configurations. The repository layout follows [RUDG](https://github.com/yzhouli/RUDG).

**Release scope:** training starts from the audited 184-feature graph interface. Original social-media data, prepared graph caches, text-model weights, and trained checkpoints are **not bundled**. `preprocess.py` imports and validates prepared graphs; it does not download or reconstruct the raw datasets. See [data preparation](docs/DATA.md) for the required files and upstream modules. This is a prepared-data code release, not a claim of one-command reproduction from public raw archives.

## Requirements

```bash
git clone https://github.com/yzhouli/DSG2former.git
cd DSG2former
conda create -n dsg2former python=3.10 -y
conda activate dsg2former
pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/verify_release.py
```

Install a PyTorch build appropriate for your hardware before the other dependencies when using CUDA. CPU execution is supported. The optional fused sparse kernel requires CUDA and Triton; the PyTorch sparse reference path works without Triton. Text-resource preparation additionally requires `pip install -r requirements-text.txt`.

## Datasets and preprocessing

| Dataset | Configuration | Source items | Train / validation / test |
|---|---|---:|---:|
| PHEME | `configs/pheme.json` | 1,972 | 1,577 / 197 / 198 |
| SocialNet (previously Weibo-V1) | `configs/socialnet.json` | 1,000 | 800 / 100 / 100 |
| Weibo (previously Weibo-public) | `configs/weibo.json` | 2,313 | 1,850 / 231 / 232 |
| Twitter15 | `configs/twitter15.json` | 369 | 295 / 37 / 37 |
| Twitter16 | `configs/twitter16.json` | 205 | 164 / 20 / 21 |
| TikTok | `configs/tiktok.json` | 237 | 189 / 24 / 24 |

The split is **chronological at the source-information level**. All snapshots and supervision rows belonging to one source item stay in the same partition. Users may occur in more than one partition. Historical inputs must precede their prediction cutoff; later partitions cannot supply state to earlier ones.

Obtain the datasets under their original terms, generate the audited graph interface described in [DATA.md](docs/DATA.md), then import it:

```bash
python preprocess.py --config configs/pheme.json \
  --graphs /path/to/prepared/graphs.pt \
  --threads /path/to/source_metadata/threads.jsonl
```

Repeat with the configuration for each dataset. Existing imports with a different hash are rejected. Only load trusted local PyTorch files. The importer checks identities, shapes, chronological source splits, label alignment, and candidate-independent macro graphs. Feature causality and weak-label provenance require the upstream preparation audit; shape checks alone cannot establish them.

Labels are `0 = rumor spreading`, `1 = anti-rumor`, `2 = non-participation`. They are **weak action labels**, not independently verified stance annotations. Non-participation is sampled absence, not neutral speech. Twitter16's recorded balanced variant additionally reassigns some source retweets to anti-rumor; these are artificial labels. See [PROTOCOL.md](docs/PROTOCOL.md).

## Training

```bash
python train.py --config configs/pheme.json --device cuda:0
```

The default `--stage all` runs the original staged pipeline: spatial and chain-temporal fits; count bases, residuals and memory gates; validation selection of cross-information temporal policies and peak-aware count heads. It trains three backbones for each seed `42`, `2025`, `2026`. It does not fit on test labels. This is a substantial experiment, not a single-epoch efficiency run.

Stages can be resumed independently:

```bash
python train.py --config configs/pheme.json --stage micro --device cuda:0
python train.py --config configs/pheme.json --stage macro --device cuda:0
python train.py --config configs/pheme.json --stage optimized --device cuda:0
```

Run other datasets by replacing the configuration. Independent dataset processes can use different GPUs and output directories. Training histories, checkpoints, configuration/input hashes, and the frozen validation choices are saved under `outputs/<dataset>/`. Changing inputs, training code, or configuration requires a new output directory.

## Testing

```bash
python test.py --config configs/pheme.json --pipeline optimized --device cuda:0
python test.py --config configs/pheme.json --pipeline optimized --device cuda:0 --replay
```

Testing freezes checkpoint hashes before inference. `--replay` recomputes predictions and requires exact equality to the saved test arrays in the same execution environment. Use `--pipeline base` to evaluate the original chain-temporal control before the optimized stage. `metrics.json` contains each seed's pooled metrics and the mean/sample standard deviation across seeds. A single seed has no estimated standard deviation.

Accuracy and classwise precision/recall/F1 are computed from all action rows; AUC is one-vs-rest. Macro MSE and RMSE pool all count windows. MAPE uses only positive ground-truth windows and reports coverage. Undefined AUC/MAPE is reported as `null`. See [reproduction notes](docs/REPRODUCIBILITY.md).

## Sparse attention

```bash
python train.py --config configs/pheme_sparse.json --device cuda:0
python test.py --config configs/pheme_sparse.json --device cuda:0
```

Each dataset has a `*_sparse.json` configuration with a separate output directory. Sparse attention retains observed graph edges, self connections, and source-token connections; it changes the attention neighborhood. Loading dense weights into a sparse topology is not proof of unchanged performance. The recorded dense benchmark is not relabeled as sparse. Dense graph caches are still used on the CPU; packing transfers edge-based inputs to the sparse GPU encoder.

## Installation smoke test

This small **synthetic** fixture checks the executable path and is not experimental data:

```bash
python tools/create_demo.py
python preprocess.py --config configs/demo.json \
  --graphs data/demo_source/graphs.pt --threads data/demo_source/threads.jsonl
python train.py --config configs/demo.json --stage micro --device cpu
python train.py --config configs/demo.json --stage macro --device cpu
python test.py --config configs/demo.json --pipeline base --device cpu
python test.py --config configs/demo.json --pipeline base --device cpu --replay
```

`demo_sparse.json` exercises the sparse path using the same imported fixture. The demo reduces only micro training to two epochs and one seed/backbone; count-head fitting retains its original settings. The optimized three-seed selector deliberately rejects demo configurations.

## Baselines

The baseline mechanisms and task adapters are included under `dsg2former/baselines/` and `scripts/baseline_comparison/`. Their task/resource substitutions are described in [BASELINES.md](docs/BASELINES.md); they are not the original papers' reported detection scores. The supported root CLI currently trains DSG²former and its spatial control. Baseline resource generation still uses the research modules and the additional resources documented there.

## Repository layout

```text
configs/          Dense, sparse and synthetic-demo configurations
dsg2former/       Model, causal state, sparse attention and count heads
scripts/          Retained experiment and feature/resource modules
preprocess.py     Prepared-graph validation and import
train.py          Staged training entry point
test.py           Frozen inference, pooled metrics and replay
tests/            Equation, causality, padding and sparse-equivalence checks
docs/             Data contracts, protocol, baseline provenance and references
data/             Local inputs, ignored by Git
outputs/          Local checkpoints and predictions, ignored by Git
```

## Citation and license

Dataset and TRGCN BibTeX entries are in [docs/](docs/). `CITATION.cff` uses the supplied manuscript's author list. No publication DOI or venue is invented; add final publication metadata when available.

Original code in this package is provided under the [MIT License](LICENSE). Datasets, pretrained models, and external resources retain their own terms and are not covered by this license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
