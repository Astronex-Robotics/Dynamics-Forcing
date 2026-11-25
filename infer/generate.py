"""Rolling joint-denoising inference for the block-causal generator, scored like ws_probe.py.

Window of W blocks: block b enters at round b and leaves after W rounds; every round advances each block
in the window by S/W Euler steps, oldest first, so a block is denoised while attending (through the KV
cache) to its older, cleaner-but-unfinished predecessors at their current noise level -- the state the
joint-denoising training shows it. A block that has finished is committed to the cache with a clean
(t = 0) pass, as in the stock pipeline. W = 1 reduces to ordinary block-by-block Euler sampling.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageOps

from infer import pipeline as inf
from infer.video import read_video, shifted_sigmas

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", "--base", dest="base", type=Path, required=True)
p.add_argument("--windows", default="1")
p.add_argument("--assets-root", type=Path, default=Path("assets"))
source = p.add_mutually_exclusive_group(required=True)
source.add_argument("--video", type=Path)
source.add_argument("--image", type=Path)
source.add_argument("--jobs", type=Path, help="JSON array of video or image jobs")
p.add_argument("--prompt")
p.add_argument("--seed", type=int, default=1000)
p.add_argument("--steps", type=int, help="Euler steps per block (multiple of every window)")
p.add_argument("--blocks", type=int, default=6)
p.add_argument("--anchor", type=float, help="first-frame noise in the negative branch")
p.add_argument("--guidance", type=float, help="text CFG scale override (0 = config value)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--dry-run", action="store_true", help="Validate the inference configuration without loading weights.")
a = p.parse_args()
if a.jobs is None and not a.prompt:
    p.error("--prompt is required with --video or --image")

cfg = inf.load_config(inf.CONFIG)
a.steps = int(cfg.inference.sampling_steps) if a.steps is None else a.steps
a.anchor = float(cfg.inference.anchor_noise) if a.anchor is None else a.anchor
a.guidance = float(cfg.inference.guidance_scale) if not a.guidance else a.guidance
if not 0 <= a.anchor <= 1:
    p.error("--anchor must be between 0 and 1")
window = int(a.windows)
if window < 1 or a.steps < 1 or a.steps % window or a.blocks < 1:
    p.error("Use positive blocks and steps, with steps divisible by the window.")
if a.dry_run:
    print(f"Valid inference configuration: {a.steps} steps, {a.blocks} blocks, guidance={a.guidance}, anchor={a.anchor}")
    raise SystemExit(0)

device = "cuda:0"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pipe = inf.build_pipeline(cfg, a.base.resolve(), a.assets_root.resolve(), device)

sig = shifted_sigmas(a.steps, float(cfg.model_kwargs.timestep_shift)).tolist()
guidance = a.guidance
nb = a.blocks
n_raw = 4 * (8 * nb - 1) + 1
fsl = pipe.frame_seq_length


def encode(video):
    pipe.generator.to("cpu"); pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            pix = (video * 2 - 1).permute(1, 0, 2, 3)[None].to(device, torch.bfloat16)
            return pipe.vae.encode_to_latent(pix).to(device, torch.bfloat16)
    finally:
        pipe.generator.to(device); pipe.text_encoder.to(device)


@torch.no_grad()
def rolling(noise, first, instruction, W):
    dit = pipe._dit_model
    dit.local_attn_size = pipe.local_attn_size
    pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
    if pipe.sink_size is not None:
        pipe._set_all_modules_sink_size(pipe.sink_size)
    pipe._set_all_modules_global_sink_size(pipe.global_sink_size)
    pipe.clear_cache()
    pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    cond = pipe.text_encoder(text_prompts=[instruction])
    unc = pipe.text_encoder(text_prompts=[pipe.negative_prompt])

    def model(b, lat, t):
        ts = torch.full((1, 8), t, device=device, dtype=torch.float32)
        if b == 0:
            lat = lat.clone(); lat[:, 0] = first[:, 0]; ts[:, 0] = 0
        kw = dict(noisy_image_or_video=lat, timestep=ts, current_start=b * 8 * fsl, cache_start=b * 8 * fsl)
        fc, _ = pipe.generator(conditional_dict=cond, kv_cache=pipe.kv_cache_pos,
                               crossattn_cache=pipe.crossattn_cache_pos, **kw)
        kw_n = kw
        if b == 0 and a.anchor > 0:                       # weak anchor for the negative branch only
            lat_n, ts_n = lat.clone(), ts.clone()
            lat_n[:, 0] = ((1 - a.anchor) * first[:, 0].float() + a.anchor * anchor_eps[:, 0].float()).to(lat.dtype)
            ts_n[:, 0] = a.anchor * 1000.0
            kw_n = dict(kw, noisy_image_or_video=lat_n, timestep=ts_n)
        fu, _ = pipe.generator(conditional_dict=unc, kv_cache=pipe.kv_cache_neg,
                               crossattn_cache=pipe.crossattn_cache_neg, **kw_n)
        return fu + guidance * (fc - fu)

    anchor_eps = torch.randn(first.shape, device=device, dtype=first.dtype,
                             generator=torch.Generator(device=device).manual_seed(7))
    x = [noise[:, b * 8:(b + 1) * 8].clone() for b in range(nb)]
    k = [0] * nb
    done = [False] * nb
    per_round = a.steps // W
    for r in range(nb + W - 1):
        active = [b for b in range(nb) if b <= r and not done[b]]
        for _ in range(per_round):
            for b in active:                                   # oldest (cleanest) first
                if k[b] >= a.steps:
                    continue
                v = model(b, x[b], sig[k[b]] * 1000.0)
                x[b] = (x[b].float() + (sig[k[b] + 1] - sig[k[b]]) * v.float()).to(x[b].dtype)
                if b == 0:
                    x[0][:, 0] = first[:, 0]
                k[b] += 1
        for b in active:
            if k[b] >= a.steps and not done[b]:
                model(b, x[b], 0.0)                            # commit the clean block to the KV cache
                done[b] = True
    pipe.clear_cache()
    return torch.cat(x, dim=1)


def run(clip, W, output):
    if "video" in clip:
        frames = read_video(clip["video"], n_raw, inf.HEIGHT, inf.WIDTH)
    else:
        with Image.open(clip["image"]) as image:
            image = ImageOps.fit(image.convert("RGB"), (inf.WIDTH, inf.HEIGHT), Image.Resampling.BICUBIC)
            frames = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().div(255)
    first = encode(frames)[:, :1]
    eps = torch.randn((1, 8 * nb, *first.shape[2:]), device=device, dtype=torch.bfloat16,
                      generator=torch.Generator(device=device).manual_seed(a.seed))
    lat = rolling(eps, first, clip["instruction"], W)
    pipe.generator.to("cpu"); pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            vid = ((pipe.vae.decode_to_pixel(lat) * 0.5 + 0.5).clamp(0, 1))[0].cpu()
    finally:
        pipe.generator.to(device); pipe.text_encoder.to(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, (vid.permute(0, 2, 3, 1) * 255).byte().numpy(), fps=24,
                     codec="libx264", quality=8, macro_block_size=16)
    print(f"Saved {output}", flush=True)


if a.jobs:
    for job in json.loads(a.jobs.read_text()):
        name = job["name"]
        if Path(name).name != name or not name:
            p.error("Each job name must be a filename stem.")
        print(f"Generating {name}", flush=True)
        run(job, window, a.out / f"{name}.mp4")
else:
    key, path = ("video", a.video) if a.video else ("image", a.image)
    run({key: str(path.resolve()), "instruction": a.prompt}, window, a.out)
