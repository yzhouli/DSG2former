# Third-party materials

The MIT license covers original source distributed in this package. It does not grant rights to datasets, pretrained weights, lexicons, papers, or downloaded author code.

No raw social-media archives, full user histories, local pretrained weights, vendored Python packages, VADER lexicon, or external author repositories are bundled. Obtain them independently and retain their upstream notices/terms. In particular, the original workspace's `external/CG_LDGR`, `external/CAGE`, `external/TRGCN`, and `vendor/` trees are excluded.

Core dependencies (PyTorch, NumPy, SciPy, scikit-learn, joblib and Matplotlib) are installed separately. Optional text preparation uses Transformers, tokenizers, NLTK and the cited teachers; their licenses and each model card govern use. The fused sparse implementation optionally imports separately installed Triton.

Attention block initialization and equation comparisons acknowledge the available ancestor spatial implementation and the cited research, as noted in `dsg2former/paper_model.py`. Baseline modules are project mathematical reconstructions/adapters; [docs/BASELINES.md](docs/BASELINES.md) records the sources and substitutions. Upstream author code used for development-time oracle comparisons is not redistributed as this project's code.

Dataset citations are provided in `docs/DATASET_REFERENCES.bib`. A citation is not permission to redistribute data. SocialNet/TikTok need an explicit distribution route before raw data or prepared user/text features can be added to a public release.
