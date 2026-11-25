#!/usr/bin/env python3
"""Dynamics Forcing command-line entry point."""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s {--train | --infer | --download} [arguments]")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", "--trian", action="store_const", const="train", dest="mode", help="train or prepare video data")
    modes.add_argument("--infer", action="store_const", const="infer", dest="mode", help="generate a video")
    modes.add_argument("--download", action="store_const", const="download", dest="mode", help="download model weights")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    mode = parser.parse_args(sys.argv[1:2]).mode
    args = sys.argv[2:]
    if mode == "train":
        tools = {"--prepare-data": "prepare_data.py", "--download-data": "download_robodojo.py",
                 "--merge": "merge_lora.py", "--grpo": "grpo.py"}
        if args and args[0] in tools:
            script = ROOT / "train" / tools[args.pop(0)]
        else:
            script = ROOT / "train" / "supervised.py"
            args = json.loads((ROOT / "config" / "train_defaults.json").read_text()) + args
        sys.path.insert(0, str(ROOT / "train"))
    elif mode == "infer":
        script = ROOT / "infer" / "generate.py"
    else:
        script = ROOT / "download" / "models.py"
    sys.argv = [f"df.py --{mode}", *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
