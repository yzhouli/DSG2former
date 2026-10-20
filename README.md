# DSG²former

Code and prepared datasets for **Modeling Rumor Propagation via Dynamic Sentiment Game Driven Graph Transformer**.

DSG²former combines sentiment-game features, graph-relative attention and temporal user states to predict next-hour user actions and propagation counts.

## Installation

Install [Git LFS](https://git-lfs.com/) before cloning the repository, then run:

```bash
git lfs install
git clone https://github.com/yzhouli/DSG2former.git
cd DSG2former
git lfs pull
conda create -n dsg2former python=3.10 -y
conda activate dsg2former
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA environment. The fused sparse-attention kernel uses Triton on supported CUDA devices.

## Datasets

The `datasets/` directory contains the complete prepared graph inputs, action labels, propagation targets and source metadata. Extract a dataset before training:

```bash
python preprocess.py --config configs/pheme.json
```

| Dataset | Configuration | Items | Train / validation / test |
|---|---|---:|---:|
| PHEME | `pheme.json` | 1,972 | 1,577 / 197 / 198 |
| SocialNet | `socialnet.json` | 1,000 | 800 / 100 / 100 |
| Weibo | `weibo.json` | 2,313 | 1,850 / 231 / 232 |
| Twitter15 | `twitter15.json` | 369 | 295 / 37 / 37 |
| Twitter16 | `twitter16.json` | 205 | 164 / 20 / 21 |
| TikTok | `tiktok.json` | 237 | 189 / 24 / 24 |

Each archive extracts to `data/<dataset>/neural/graphs.pt` and `data/<dataset>/hourly/threads.jsonl`. The split is chronological by source information item; its snapshots and action samples remain together. Labels are `0 = rumor`, `1 = anti-rumor`, and `2 = non-participation`. Action labels use the weak-label protocol; non-participation is sampled, and Twitter16 uses the balanced variant with reassigned retweets.

## Training and evaluation

```bash
python train.py --config configs/pheme.json --device cuda:0
python test.py --config configs/pheme.json --device cuda:0
```

Replace `pheme.json` with another dataset configuration as needed. Training runs the spatial, temporal and propagation-count stages with seeds `42`, `2025` and `2026`. Use `--stage micro`, `--stage macro` or `--stage optimized` to run an individual stage. Checkpoints and predictions are saved under `outputs/<dataset>/`.

Evaluation reports Accuracy, classwise Precision/Recall/F1, AUC, MSE, RMSE and MAPE. For CPU execution, use `--device cpu`.

## Sparse attention

```bash
python train.py --config configs/pheme_sparse.json --device cuda:0
python test.py --config configs/pheme_sparse.json --device cuda:0
```

The `*_sparse.json` configurations use the same prepared data and save outputs separately. Sparse attention connects observed edges, self connections and source tokens.

## License

MIT License

Copyright (c) 2026 DSG2former contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The software license applies to the code. Datasets retain their original terms of use.
