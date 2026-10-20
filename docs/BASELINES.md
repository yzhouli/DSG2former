# Baseline implementations and adaptations

The package includes project reconstructions of baseline mathematical modules, not copies of every upstream repository or claims to reproduce the papers' original dataset scores. The shared task is next-hour three-class user action plus first-time-participant count prediction.

| Method | Core module | Original mechanism / task adaptation |
|---|---|---|
| ICS-SVM | `baselines/native.py` | Cuckoo-search SVM for retweet prediction; three-action classifier |
| K-GraphSAGE | `baselines/native.py`, `graph_models.py` | Knowledge/neighbor aggregation; node-level three-action output |
| RAM-GAT | `baselines/native.py`, `graph_models.py` | Sparse representation, transfer and graph attention; adapted action classifier |
| TRS-GCN | `baselines/native.py`, `text_models.py`, `graph_models.py` | Topic relevance and sentiment-aware propagation; adapted action classifier |
| Rumor-DPM | `baselines/native.py`, `graph_models.py` | Image/restoration-inspired propagation mechanisms; action/graph representations |
| TRGCN | `baselines/trgcn.py` | GCN local structure plus Transformer multi-head long-range dependencies for rumor detection; node-action output and shared count adapter |

Paths above are relative to `dsg2former/`. The training adapters are in `scripts/baseline_comparison/train_method.py`, `scripts/weibo_comparison/baselines.py`, and `scripts/trgcn_baseline/train.py`. Resource generation uses `scripts/baseline_comparison/prepare_features.py` and `prepare_method_resources.py`; it requires aligned local text/teacher caches and the lexicon. These adapters preserve repeated action rows and add a candidate-independent count head. The root training CLI does not currently orchestrate the full baseline resource workflow.

Missing demographics are explicit zeros. Recoverable interactions substitute unavailable complete follower networks. Cached BERT/lexical resources replace unavailable original text/keyword components where documented. Do not equate these adapted results with original paper scores.

Original references:

- ICS-SVM: *ICS-SVM: A user retweet prediction method for hot topics based on improved SVM*, DOI `10.1016/j.dcan.2021.07.003`.
- K-GraphSAGE: *A rumor heat prediction model based on rumor and anti-rumor multiple messages and knowledge representation*, DOI `10.1016/j.ipm.2023.103337`.
- RAM-GAT: *A prediction model for rumor user propagation behavior based on sparse representation and transfer learning*, DOI `10.1016/j.ins.2024.120590`.
- TRS-GCN: *Derived Topic Propagation Model Based on Topic Relevance and User Sentiment*, DOI `10.1109/TAFFC.2026.3657357`.
- Rumor-DPM: *Spreading Mosaic: An Image Restoration-Inspired Social Rumor Propagation Model*, DOI `10.1109/TMM.2023.3305095`.
- TRGCN: *TRGCN: a hybrid framework for social network rumor detection*, DOI `10.1057/s41599-026-06946-1`; BibTeX in `TRGCN_REFERENCE.bib`.

Using Transformer attention to capture long-range dependencies is shared prior art, including TRGCN. This release does not claim that generic GCN–Transformer combination as a first invention. DSG²former's particular sentiment-game inputs, graph-relative masking and chronological state/count protocol should be described separately from shared architectural ingredients.
