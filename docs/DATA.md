# Data preparation and interface

The released training entry points consume the existing neural feature cache, not the earlier minimal equation-only cache. The expected files after import are:

```text
data/<dataset>/
  import_manifest.json
  neural/graphs.pt
  hourly/threads.jsonl
```

`graphs.pt` is a trusted PyTorch-serialized Python list. Each item has `thread_id`, `split` (`train`, `validation`, `test`), `snapshots`, and `macro_snapshots`. All representations, normalizers, and supervision must be generated under the same protocol/version. Do not substitute a random row split.

Each snapshot contains:

| Field | Shape/type | Meaning |
|---|---|---|
| `x`, `raw_x` | float32 `[N,184]` | Train-normalized inputs and original feature coordinates |
| `drivers` | float32 `[N,2,3]` | Internal/external game drivers |
| `adjacency` | numeric `[N,N]` | Observed/semantic graph relations |
| `positions` | integer `[N,N]`, values 0–49 | Shifted directed shortest-path lookup indices |
| `role` | integer `[N]`, 0/1 | User and information-token roles |
| `user_count` | integer `U`, `0 < U < N` | Users occupy `[0,U)`, source token is at `U` |
| `node_ids` | list of `U` strings | User identities; stable across snapshots |
| `sample_nodes`, `target` | integer `[R]` | Supervised user-node indices and labels 0/1/2; repeated rows retained |
| `state_update_users` | collection of user IDs | Visible historical users for chain-cache updates |
| `window`, `cutoff`, `end` | number | Hour index and UTC-second prediction boundaries; `end=cutoff+3600` |
| `history_summary` | float32 `[15]` in the benchmark | Four summary fields, eight earlier count windows, then auxiliary features; at least 13 coordinates accepted |
| `popularity_target` | nonnegative number | First-time distinct participants in the following hour |

History summary coordinates 0–3 are known participants, previous-hour new users, previous-hour active users, and elapsed hours. Coordinates 4–11 are earlier count windows with left padding. The importer preserves values; it does not fit a normalizer. `raw_x` is required by numeric count features and historical-profile initialization.

Macro graphs occur exactly once per source/hour, contain only visible historical users, and have empty `sample_nodes` and `target`. Action-candidate membership must not determine their nodes or edges. All source items require macro graphs even when they have no eligible action rows in an hour.

The minimal source-metadata file has one physical JSONL line per item:

```json
{"thread_id":"123","split":"train","source":{"uid":"stable-author-id","timestamp":1600000000}}
```

The count-memory adapter needs stable author IDs and source times. No source/reply text is required in this minimal import file. Full extraction-stage metadata can also be supplied locally. Do not publish raw user IDs or text merely to demonstrate the schema; use `tools/create_demo.py` for that purpose.

## Upstream resources

- PHEME: [archived dataset](https://doi.org/10.6084/m9.figshare.4010619). The benchmark retains 1,972 `rumours` items; this does not imply all were verified false.
- Weibo: the collection associated with [Ma et al., IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/537.pdf); only the retained rumor subset enters this benchmark.
- Twitter15/Twitter16: the propagation-structure corpus associated with [Ma et al., ACL 2017](https://aclanthology.org/P17-1066/). Consult the original distribution terms for text/IDs.
- SocialNet and TikTok: the project uses locally supplied collections. This package does not invent a public download URL, redistribution permission, or standalone dataset DOI for them.

The retained upstream graph builder is `scripts/unified_action_comparison/build.py`: `targets()` constructs NP50 supervision and `graphs()` constructs causal semantic snapshots. The feature stages are `scripts/unified_action_comparison/prepare.py` and `scripts/unified_action_execution/public_prepare.py`. They expect extraction/teacher caches: source/reaction metadata, aligned embeddings, sentiment/stance probabilities, native feature resources, and locally installed text encoders/lexicon. `scripts/weibo_comparison/extract.py` and `scripts/twitter_comparison/extract.py` retain platform adapters. These research modules are included for transparency; they are not a complete portable raw-to-cache CLI for all six datasets.

Frozen teacher outputs must include their model/revision and file hashes, label mapping, missing-text treatment, and record alignment. Text teacher inference may use future reaction text **only for target generation**, never as a forecast input. Fit calibrators, vocabularies, PCA, and normalizers on training observations only.

Current `configs/*.json` document the graph-generation parameters. Their `protocol` fields are metadata for already prepared inputs: changing them does not rebuild a cache. After changing any graph-generation rule, regenerate and independently audit the upstream cache, import it to a new data directory, and use a new output directory.
