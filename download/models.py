"""Download Astronex-World and prepare a Dynamics Forcing generator checkpoint."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

ASTRONEX_REPO = "Astronex-Lab/Astronex-World"
ASSETS_REPO = "Wan-AI/Wan2.2-TI2V-5B"
ASSET_PATTERNS = ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/**"]
MODEL_PATTERNS = ["model-*.safetensors", "model.safetensors.index.json", "transformer/config.json",
                  "config.json", "README.md", "LICENSE"]


def convert(model_dir, output):
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "engine"))
    from wan_5b.modules.causal_model import CausalWanModel

    config = json.loads((model_dir / "transformer/config.json").read_text())
    params = inspect.signature(CausalWanModel).parameters
    with torch.device("meta"):
        model = CausalWanModel(**{k: v for k, v in config.items() if k in params})
    expected = model.state_dict()
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    indexed_keys = {key.removeprefix("model."): key for key in index}
    missing = sorted(expected.keys() - indexed_keys.keys())
    if missing:
        raise ValueError(f"The source export is missing generator tensors: {missing[:5]}")
    state = {}
    for shard in sorted({index[indexed_keys[key]] for key in expected}):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for stored_key in handle.keys():
                key = stored_key.removeprefix("model.")
                if key in expected and indexed_keys[key] == stored_key and index[stored_key] == shard:
                    tensor = handle.get_tensor(stored_key)
                    if tensor.shape != expected[key].shape:
                        raise ValueError(f"Incompatible generator tensor: {stored_key}")
                    state["model." + key] = tensor
    if len(state) != len(expected):
        raise ValueError(f"Expected {len(expected)} generator tensors, loaded {len(state)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".generator-") as staged:
        torch.save({"generator": state}, staged)
        staged.flush()
        os.link(staged.name, output)
    print(f"Generator checkpoint: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--repo", help="Override the model repository ID. Required for a ModelScope mirror.")
    parser.add_argument("--output", type=Path, default=Path("weights/Astronex-World"))
    parser.add_argument("--assets-root", type=Path, default=Path("assets"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--raw-only", action="store_true", help="Download the complete model snapshot only.")
    group.add_argument("--from-local", type=Path, metavar="DIR", help="Convert an existing model snapshot.")
    parser.add_argument("--skip-assets", action="store_true", help="Reuse existing VAE and text encoder assets.")
    parser.add_argument("--dry-run", action="store_true", help="Show destinations without downloading.")
    args = parser.parse_args()
    if args.source == "modelscope" and not args.repo and not args.from_local:
        parser.error("--repo is required with --source modelscope")
    repo = args.repo or ASTRONEX_REPO
    checkpoint = args.output / "causal-generator.pt"
    assets = args.assets_root / "wan_models" / "Wan2.2-TI2V-5B"
    if args.dry_run:
        print(f"Model: {args.from_local or repo}")
        print(f"Output: {args.output if args.raw_only else checkpoint}")
        if not args.raw_only and not args.skip_assets:
            print(f"Video and text encoder assets: {assets}")
        return

    from huggingface_hub import snapshot_download

    if not args.from_local and (args.raw_only or not checkpoint.is_file()):
        if args.source == "huggingface":
            snapshot_download(repo, local_dir=args.output, allow_patterns=None if args.raw_only else MODEL_PATTERNS)
        else:
            try:
                from modelscope import snapshot_download as modelscope_download
            except ImportError:
                parser.error("Install modelscope to use this source.")
            modelscope_download(repo, local_dir=str(args.output),
                                allow_patterns=None if args.raw_only else MODEL_PATTERNS)
    if args.raw_only:
        print(f"Model snapshot: {args.output}")
        return
    if checkpoint.is_file():
        if args.from_local:
            parser.error(f"Checkpoint already exists: {checkpoint}")
        print(f"Using existing generator checkpoint: {checkpoint}")
    else:
        convert(args.from_local or args.output, checkpoint)
    if not args.skip_assets:
        snapshot_download(ASSETS_REPO, local_dir=assets, allow_patterns=ASSET_PATTERNS)
        print(f"Video and text encoder assets: {assets}")


if __name__ == "__main__":
    main()
