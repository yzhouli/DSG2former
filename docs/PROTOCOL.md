# Evaluation protocol

- Split unit: source information item, ordered by source publication time, 80/10/10. All its windows/action rows stay together. Users can recur across partitions. Rounded partition sizes are listed in the README.
- Window: one hour. PHEME retains the observed source horizon; the other five collections use the first 72 hours.
- Supervision: raw action records, including repeated observations. Stance can vary across time for the same user/information pair. Weak-label thresholds are fitted on training records.
- Non-participation: sampled from other information items in the same partition; users appearing anywhere in the current item's complete chain are excluded. The retained count is 50% of positive action rows. This sampling uses complete-chain exclusion to define evaluation negatives; it is not a model-accessible future feature or a deployment candidate-discovery rule.
- Graph: observable diffusion/user interactions plus user–user and user–information text-similarity relations at cosine threshold 0.98. These relations are not a full follower network. Historical content must precede the cutoff; the current source is available at publication.
- Parameters: original-action discount `gamma=0.23`; propagation half-life `w=900 seconds`; retrieved history `K=5`; relative-position table size 50; mask thresholds `beta=(1.0,0.5)`. These mask values are distance-index thresholds, not independent random edge-deletion rates.
- Model: 184 input features, two blocks with two attention/MLP residual units per block, three attention heads; three backbones of hidden width 24/96/48 with dropout 0.10/0.20/0.35. Their AdamW learning rates are 0.001/0.0005/0.001 and weight decay 0.001/0.001/0.01.
- Training: spatial CE with label smoothing 0.02; frozen-spatial temporal training; separate numeric/residual/peak count stages. There is no single jointly optimized loss weight `lambda` in this staged benchmark. The micro and macro selection candidates/rules remain in the retained training modules.
- Temporal state: chain control or strictly earlier cross-information observed states. Same-time queries occur before same-time writes. Later source partitions never provide state to earlier partitions. Global-bootstrap is an explicitly declared historical-feature initializer, not an actual earlier hidden state.
- Classification: all test action rows are pooled within each seed before scoring. Report Accuracy, three classwise precision/recall/F1, and OvR AUC. Optional macro class averages weight the three classes equally. Missing-class AUC is undefined.
- Count target: next-hour distinct first-time real participants, excluding source author and sampled nonparticipants. This differs from action-record count, cumulative popularity, and summing probabilities over sampled candidates.
- Count metrics: pooled MSE, square root of pooled MSE, and MAPE only on positive true-count windows; report positive-window coverage. Original validation selection also uses MSLE internally, even when publication tables omit MSLE.
- Uncertainty: seedwise metrics, then mean and sample SD (`ddof=1`) over seeds 42/2025/2026. Do not attach an old run's SD to a new configuration or replace true/rounded zero SD by an invented nonzero value.

## Weak-label limits

PHEME `rumours` are not all verified false. Frozen stance teachers can have training-domain overlap; labels are not human-verified future actions. Twitter16's recorded balanced weak variant reassigns some source retweets to anti-rumor and cannot be described as verified refutation. Prepared input data must state whether that balanced variant or the original stance-only variant was used. The publication comparison was previously inspected during development and is not an independent blind-test/SOTA certification.
