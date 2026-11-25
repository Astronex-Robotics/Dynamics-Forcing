"""Download complete LeRobot shards for a task-balanced RoboDojo subset."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow.parquet as pq


ROOT = "data/RoboDojo_lerobot_v30_video"
API = "https://modelscope.cn/api/v1/datasets/RoboDojo-Benchmark/RoboDojo/repo"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
META = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet")


def repo_files():
    files = {}
    page = 1
    while True:
        params = urlencode(dict(Revision="master", Root=ROOT, Recursive="true",
                                PageNumber=page, PageSize=100))
        with urlopen(f"{API}/tree?{params}", timeout=45) as response:
            batch = json.load(response)["Data"]["Files"]
        if not batch:
            return files
        for item in batch:
            if item["Type"] == "blob":
                files[item["Path"].removeprefix(ROOT + "/")] = item
        if len(batch) < 100:
            return files
        page += 1


def shards(row):
    paths = {f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"}
    for camera in CAMERAS:
        key = f"videos/observation.images.{camera}"
        paths.add(f"{key}/chunk-{row[key + '/chunk_index']:03d}/file-{row[key + '/file_index']:03d}.mp4")
    return paths


def plan(episodes, files, budget, per_task):
    groups = defaultdict(list)
    for row in episodes:
        groups[tuple(row["tasks"])].append((row["episode_index"], shards(row)))
    selected = {task: [] for task in sorted(groups)}
    known = set(META)
    used = sum(files[path]["Size"] for path in known)
    order = sorted(known)
    for _ in range(per_task):
        made_progress = False
        for task, episodes_for_task in sorted(groups.items()):
            candidates = []
            for index, paths in episodes_for_task:
                if index in selected[task]:
                    continue
                extra = paths - known
                cost = sum(files[path]["Size"] for path in extra)
                candidates.append((cost, index, extra))
            if not candidates:
                continue
            cost, index, extra = min(candidates, key=lambda item: (item[0], item[1]))
            if used + cost > budget:
                continue
            known.update(extra)
            order.extend(sorted(extra))
            used += cost
            selected[task].append(index)
            made_progress = True
        if not made_progress:
            break
    if any(not values for values in selected.values()):
        raise ValueError("Budget is too small to cover every task with the required shards.")
    return selected, order, used


def download(output, relative, info):
    target = output / relative
    if not target.is_file() or target.stat().st_size != info["Size"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        url = API + "?" + urlencode(dict(Revision="master", FilePath=f"{ROOT}/{relative}"))
        subprocess.run(["curl", "-fLsS", "--retry", "5", "--connect-timeout", "20",
                        "--continue-at", "-", "--output", str(temporary), url], check=True)
        if temporary.stat().st_size != info["Size"]:
            raise ValueError(f"Size differs from repository metadata: {relative}")
        with temporary.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if info.get("Sha256") and digest.hexdigest() != info["Sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        temporary.replace(target)
    print(f"Ready {relative}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-gb", default=300, type=int)
    parser.add_argument("--episodes-per-task", default=20, type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.budget_gb < 1 or args.episodes_per_task < 1:
        parser.error("Budget and episodes per task must be positive.")
    files = repo_files()
    for path in META:
        download(args.output, path, files[path])
    episodes = pq.read_table(args.output / META[-1]).to_pylist()
    selected, paths, size = plan(episodes, files, args.budget_gb * 10**9, args.episodes_per_task)
    record = dict(tasks=[dict(instructions=list(task), episode_indices=indices)
                         for task, indices in selected.items()],
                  files=paths, bytes=size)
    (args.output / "subset-plan.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Plan: {len(selected)} tasks, {sum(map(len, selected.values()))} episodes, "
          f"{len(paths)} files, {size / 1e9:.2f} GB", flush=True)
    if not args.plan_only:
        for path in paths:
            download(args.output, path, files[path])
        print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
