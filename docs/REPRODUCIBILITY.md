# Reproduction

The package retains the model and staged training implementation. Portable wrappers expose configuration, prepared-data import, dense/sparse model selection, training and evaluation. The original research workspace remains independent.

## Tests

`tests/` includes independent attention-equation comparisons, causality checks, padding isolation, temporal freezing, pooled-metric definitions and sparse output/gradient checks. The README provides a synthetic installation smoke test. CUDA-only tests require a GPU.

## Required inputs and scope

1. Raw/prepared datasets, teacher/resource caches and trained checkpoints are not shipped. The importer requires independently obtained, audited prepared inputs. SocialNet/TikTok still need a distribution route for complete public raw-data reproduction.
2. The supported CLI reuses the neural stages. Baseline modules are included; full baseline resource preparation uses the retained research modules.
3. Dense and sparse attention require separate evaluation under the same protocol.
4. The optimized selector requires three backbones and seeds 42/2025/2026. For an installation fixture, use the demo base pipeline.
5. This package contains no recorded performance tables, source-hash inventories or server-validation reports. Training and evaluation generate their own local outputs.

Use a distinct output directory for each dataset version, seed set, attention type and protocol. Inspect configuration and per-stage validation histories before comparing runs.
