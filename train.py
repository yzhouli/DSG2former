"""Run the retained spatial, temporal and count-head training stages."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from cli_utils import config, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("micro", "macro", "optimized", "all"), default="all")
    args = parser.parse_args()
    value = config(args.config)
    if args.stage in ("optimized", "all") and (value["seeds"] != [42, 2025, 2026] or value["backbones"] != ["d24_regular", "d96_regular", "d48_strong_reg"]):
        raise ValueError("Published optimized selection requires all three backbones and seeds 42, 2025, 2026")
    neural, peak, _, name = configure(value)
    runs = [(b, s) for b in value["backbones"] for s in value["seeds"]]
    if args.stage in ("micro", "all"):
        for backbone, seed in runs:
            neural.fit(SimpleNamespace(backbone=backbone, seed=seed, device=args.device))
    if args.stage in ("macro", "all"):
        common = neural.R / "common.pt"
        if not common.exists():
            data = torch.load("data/pheme_model_improvement/graphs.pt", weights_only=False, map_location="cpu")
            torch.save(neural.numeric_common(data), common)
        c = torch.load(common, weights_only=False, map_location="cpu")
        for seed in value["seeds"]:
            neural.fit_base(neural.R / "numeric_base" / f"seed_{seed}", c["train"], c["validation"], "ics", seed)
        for backbone, seed in runs:
            neural.count_fit(SimpleNamespace(backbone=backbone, seed=seed))
    if args.stage in ("optimized", "all"):
        for backbone, seed in runs:
            peak.micro(SimpleNamespace(dataset=name, backbone=backbone, seed=seed, device=args.device))
        peak.select_micro(name)
        for variant in ("spatial", "temporal"):
            for candidate in peak.MACRO_CONFIGS:
                peak.count_job(SimpleNamespace(dataset=name, backbone="d48_strong_reg", seed=42,
                                                variant=variant, config=candidate))
        peak.select_macro(name)
        for variant in ("spatial", "temporal"):
            for backbone, seed in runs:
                peak.count_job(SimpleNamespace(dataset=name, backbone=backbone, seed=seed,
                                                variant=variant, config=None))
    print(f"Completed {args.stage}; artifacts: {Path(value['output_dir'])}")


if __name__ == "__main__":
    main()
