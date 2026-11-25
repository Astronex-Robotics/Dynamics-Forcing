"""Convert captioned video folders into a training-ready 24 fps dataset."""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path


BUCKETS = {
    "S": list(range(24, 65, 8)),
    "M": list(range(72, 121, 8)),
    "L": list(range(128, 193, 8)),
}
CONDITIONING = {"frame_index": 0, "mode": "i2v"}
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
FRAMES = 285


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def convert(source, destination):
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", "fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-frames:v", str(FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(destination),
    ], check=True)
    count = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(destination),
    ], text=True).strip()
    return count == str(FRAMES)


def build(inputs, output, staging, seed):
    rows = []
    missing = short = 0
    (staging / "videos").mkdir()
    for folder in inputs:
        for video in sorted(folder.rglob("*")):
            if not video.is_file() or video.suffix.lower() not in EXTENSIONS:
                continue
            caption = video.with_suffix(".txt")
            instruction = caption.read_text(encoding="utf-8").strip() if caption.is_file() else ""
            if not instruction:
                missing += 1
                continue
            relative = video.relative_to(folder).as_posix()
            sample_id = sha256(f"{folder.name}/{relative}".encode())[:32]
            filename = f"{sample_id}.mp4"
            destination = staging / "videos" / filename
            if not convert(video, destination):
                destination.unlink()
                short += 1
                continue
            rows.append(dict(
                bucket="F72", conditioning=CONDITIONING,
                contract_version="robot-ti2v-length-mixed-index-v1",
                first_frame_source="decoded frame 0 of video", id=sample_id,
                instruction=instruction, source_dataset=folder.name,
                source_id=sample_id, source_manifest_id=folder.name,
                source_name=relative, stage="M", temporal_padding=False,
                temporal_stretching=False, tier="prepared",
                valid_duration_seconds=FRAMES / 24, valid_latent_frames=72,
                valid_raw_frames=FRAMES, video=str(output / "videos" / filename),
            ))
    if len(rows) < 2:
        raise ValueError(f"Need two captioned clips with at least {FRAMES} frames at 24 fps. "
                         f"Found {len(rows)}. Skipped {missing} missing captions and {short} short clips.")
    random.Random(seed).shuffle(rows)
    val_count = max(1, len(rows) // 10)
    manifest_dir = staging / "manifests" / "M"
    manifest_dir.mkdir(parents=True)
    outputs = []
    for split, samples in (("train", rows[val_count:]), ("val", rows[:val_count])):
        name = f"{split}.jsonl"
        payload = "".join(json.dumps(dict(row, split=split), ensure_ascii=False) + "\n"
                          for row in samples).encode("utf-8")
        (manifest_dir / name).write_bytes(payload)
        outputs.append(dict(path=str(output / "manifests" / "M" / name), stage="M",
                            split=split, record_count=len(samples), size_bytes=len(payload),
                            sha256=sha256(payload)))
    receipt = dict(contract_version="robot-ti2v-length-mixed-receipt-v1",
                   status="accepted_metadata_index_no_media_copy", conditioning=CONDITIONING,
                   media_policy=dict(target_fps=24, temporal_compression_ratio=4),
                   stage_policy=BUCKETS, outputs=outputs)
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = sha256(canonical.encode("utf-8"))
    (staging / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows) - val_count} train and {val_count} validation clips. "
          f"Skipped {missing} missing captions and {short} short clips.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    inputs = [folder.expanduser().resolve() for folder in args.input]
    output = args.output.expanduser().resolve()
    if any(not folder.is_dir() for folder in inputs):
        parser.error("Every input must be an existing directory.")
    if len({folder.name for folder in inputs}) != len(inputs):
        parser.error("Give each input directory a distinct name.")
    if output.exists() or any(output.is_relative_to(folder) for folder in inputs):
        parser.error("Output must be a new directory outside the input directories.")
    if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        parser.error("Install FFmpeg and ffprobe before preparing the dataset.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        try:
            build(inputs, output, staging, args.seed)
            staging.rename(output)
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f"Data preparation failed: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
