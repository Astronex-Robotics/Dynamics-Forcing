"""Flow-GRPO for the block-causal Astronex-World generator, rewarded against the real training video.

Each iteration takes one robot clip (first frame + instruction + real video), samples a group of G
videos with an SDE version of the flow sampler (Flow-GRPO), scores every video against the real one,
and pushes the LoRA towards the better members of the group.

Rollout:  the stock KV-cache pipeline, with its scheduler swapped for an Euler-SDE step that records
          (x_t, x_prev, mean, std) for every SDE step of every 8-frame block.
Update:   the same pipeline replayed on the recorded trajectory. Only K selected SDE steps per block
          run the generator with gradients (history comes from the clean KV cache, as at inference);
          the other noisy calls are skipped, the clean t=0 cache refreshes always run.
          loss = -A * clip(ratio) + beta * ||mu_theta - mu_ref||^2 / (2 std^2), backpropagated per step.
Reward:   DINOv2 features of the generated and the real video (no teacher model):
          align   time-aligned CLS similarity, later frames weighted up (task progress, causal order:
                  a door that opens before the gripper arrives does not match the real frame then)
          patch   per generated patch, best match among real patches within +-2 sampled frames; mean of
                  the worst 20% per frame (glassy / deformed arms and invented objects have no match)
          motion  -|log(motion_gen / motion_real)| (penalises both the frozen and the chaotic video)
          Each term is z-scored inside the group, then A = z_align + z_patch + 0.5 z_motion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from infer import pipeline as inf
from infer.video import read_video, shifted_sigmas

TEST_IDS = {"000d10184ace537e6ac7bf1f906f6faa", "00070f914d20c26005542ae68933f734"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="Merged generator .pt the LoRA starts from.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, default=Path("/data/assets"))
    p.add_argument("--manifest", type=Path,
                   default=Path("datasets/robot-video/manifests/M/train.jsonl"))
    p.add_argument("--dino", type=Path, default=Path("/data/models/dinov2-base"))
    p.add_argument("--latent-frames", type=int, default=72)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=16)
    p.add_argument("--ode-tail", type=int, default=2, help="Last N sampling steps stay deterministic.")
    p.add_argument("--train-steps", type=int, default=2, help="SDE steps per block trained per rollout.")
    p.add_argument("--shared-blocks", type=int, default=2,
                   help="Leading 8-frame blocks sampled once and shared by the group (tree rollout).")
    p.add_argument("--eta", type=float, default=0.7)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--beta", type=float, default=0.01, help="KL to the frozen start model; 0 skips the ref pass.")
    p.add_argument("--fp8", action="store_true", help="FP8 matmuls for frozen base layers in no-grad calls.")
    p.add_argument("--clip", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261104)
    return p.parse_args()


# ----------------------------------------------------------------------------- sampler


def sde_mean_std(x, v, sigma, sigma_next, sigma_max, eta):
    """Flow-GRPO Euler-Maruyama step of the reverse SDE sharing marginals with the flow ODE."""
    dt = sigma_next - sigma  # negative
    denom = 1 - (sigma_max if sigma >= 1.0 else sigma)
    std_t = math.sqrt(sigma / denom) * eta
    mean = x * (1 + std_t ** 2 / (2 * sigma) * dt) + v * (1 + std_t ** 2 * (1 - sigma) / (2 * sigma)) * dt
    return mean, std_t * math.sqrt(-dt)


class Trajectory:
    """Shared state between the swapped scheduler and the generator wrapper."""

    def __init__(self, sigmas, n_sde, eta):
        self.sigmas, self.n_sde, self.eta = sigmas, n_sde, eta
        self.mode = "sample"   # or "replay"
        self.chunk = -1
        self.records = {}       # (chunk, step) -> dict
        self.selected = set()   # (chunk, step) trained during replay
        self.step = 0
        self.on_step = None     # replay callback(chunk, step, flow_pred, record)
        self.rng = None
        self.frozen = {}        # chunk -> latents shared by the whole group (prefix); not denoised again


class SwapScheduler:
    def __init__(self, traj):
        self.traj = traj
        traj.chunk += 1
        traj.step = 0
        self.timesteps = (traj.sigmas[:-1] * 1000.0).float()

    def step(self, flow_pred, t, latents, return_dict=False):
        tr = self.traj
        i, c = tr.step, tr.chunk
        sigma, sigma_next = float(tr.sigmas[i]), float(tr.sigmas[i + 1])
        if c in tr.frozen:
            out = tr.frozen[c].to(latents.device, latents.dtype)
        elif tr.mode == "sample":
            v = flow_pred.float()
            x = latents.float()
            if i < tr.n_sde:
                mean, std = sde_mean_std(x, v, sigma, sigma_next, float(tr.sigmas[1]), tr.eta)
                noise = torch.randn(mean.shape, device=mean.device, generator=tr.rng)
                prev = mean + std * noise
                tr.records[(c, i)] = {"x": latents.detach().to(torch.bfloat16).cpu(),
                                      "prev": prev.to(torch.bfloat16).cpu(),
                                      "logp": gaussian_logp(prev, mean, std, c).item(), "std": std}
            else:
                prev = x + (sigma_next - sigma) * v
            out = prev.to(latents.dtype)
        else:
            rec = tr.records.get((c, i))
            if (c, i) in tr.selected:
                tr.on_step(c, i, flow_pred, rec)
            out = (rec["prev"] if rec is not None else tr.final[c][i]).to(latents.device, latents.dtype)
        tr.step += 1
        return (out,)


def gaussian_logp(prev, mean, std, chunk):
    lp = -((prev.float() - mean.float()) ** 2) / (2 * std ** 2) - math.log(std) - 0.5 * math.log(2 * math.pi)
    if chunk == 0:
        lp = lp[:, 1:]  # frame 0 is the clamped conditioning frame
    return lp.mean()


# ----------------------------------------------------------------------------- reward

class Reward:
    def __init__(self, dino_dir, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(dino_dir, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, frames):
        """frames: (T, 3, H, W) in [0, 1] -> cls (T, D), patches (T, N, D), both L2-normalised."""
        x = F.interpolate(frames.to(self.device).float(), size=(224, 392), mode="bilinear", antialias=True)
        x = ((x - self.mean) / self.std).half()
        cls, patches = [], []
        for i in range(0, x.shape[0], 16):
            h = self.model(pixel_values=x[i:i + 16]).last_hidden_state.float()
            cls.append(F.normalize(h[:, 0], dim=-1))
            patches.append(F.normalize(h[:, 1:], dim=-1))
        return torch.cat(cls), torch.cat(patches)

    @torch.no_grad()
    def score(self, gen, real):
        """gen, real: (T, 3, H, W) in [0, 1], same sampled time stamps."""
        T = min(gen.shape[0], real.shape[0])
        gen, real = gen[:T], real[:T]
        gc, gp = self.features(gen)
        rc, rp = self.features(real)
        w = torch.linspace(1.0, 2.0, T, device=self.device)
        frame_align = (gc * rc).sum(-1)
        align = (frame_align * w).sum() / w.sum()
        worst, best_maps = [], []
        for t in range(1, T):
            lo, hi = max(0, t - 2), min(T, t + 3)
            sim = gp[t] @ rp[lo:hi].reshape(-1, rp.shape[-1]).T  # (N, M)
            best = sim.max(dim=1).values
            best_maps.append(best)
            k = max(1, best.numel() // 5)
            worst.append(best.topk(k, largest=False).values.mean())
        patch = torch.stack(worst).mean()
        # where / when the video fails: kept for locating the responsible weights later
        self.last_maps = {"frame_align": frame_align.half().cpu(), "frame_patch": torch.stack(worst).half().cpu(),
                          "patch_map": torch.stack(best_maps).half().cpu()}  # (T-1, 16*28) best match per patch
        small = lambda v: F.interpolate(v.float(), size=(60, 104), mode="area")
        mg = (small(gen[1:]) - small(gen[:-1])).abs().mean() + 1e-4
        mr = (small(real[1:]) - small(real[:-1])).abs().mean() + 1e-4
        motion = -torch.log(mg / mr).abs()
        return {"align": align.item(), "patch": patch.item(), "motion": motion.item(),
                "motion_ratio": (mg / mr).item()}


def advantages(scores):
    def z(key):
        v = torch.tensor([s[key] for s in scores], dtype=torch.float64)
        return (v - v.mean()) / (v.std(unbiased=False) + 1e-6)
    a = z("align") + z("patch") + 0.5 * z("motion")
    return ((a - a.mean()) / (a.std(unbiased=False) + 1e-6)).tolist()


# ----------------------------------------------------------------------------- data

def load_clips(manifest, latent_frames):
    clips = []
    for line in manifest.open():
        r = json.loads(line)
        if r["source_dataset"] != "RoVid-X" or r["id"] in TEST_IDS:
            continue
        if r["valid_latent_frames"] >= latent_frames and Path(r["video"]).is_file():
            clips.append(r)
    return clips



# ----------------------------------------------------------------------------- main

def main():
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    from omegaconf import OmegaConf

    cfg = inf.load_config(inf.CONFIG)
    cfg.inference.guidance_scale = a.guidance
    pipe = inf.build_pipeline(cfg, a.base.resolve(), a.assets_root.resolve(), device)
    shift = float(cfg.model_kwargs.timestep_shift)

    # LoRA on the causal DiT, same targets/rank as SFT so merge_lora.py folds it the same way
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    import safetensors.torch as st
    from supervised import LORA_TARGETS
    wrapper = pipe.generator
    wrapper.model = inject_adapter_in_model(
        LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=LORA_TARGETS), wrapper.model)
    wrapper.model.to(device=device, dtype=torch.bfloat16)
    lora_params = {n: p for n, p in wrapper.named_parameters() if "lora_" in n}
    for n, p in wrapper.named_parameters():
        p.requires_grad_(n in lora_params)
    tuner_layers = [m for m in wrapper.modules() if isinstance(m, BaseTunerLayer)]

    # FP8 (sm_89) matmuls for the frozen base Linear layers, no-grad calls only (rollouts, uncond,
    # cache refreshes); the few gradient calls keep bf16, so no transposed FP8 copy is needed
    fp8_layers = []
    if a.fp8:
        from fp8_lora import E4, M4, _q

        def fp8_linear(lin, x, _bf16=None):
            if torch.is_grad_enabled() or x.shape[-1] % 16 or lin.weight.shape[0] % 16:
                return _bf16(x)
            if getattr(lin, "_f8", None) is None:
                wq, w_inv = _q(lin.weight.detach(), E4, M4)
                lin._f8 = (wq.contiguous(), w_inv)
            wq, w_inv = lin._f8
            shp = x.shape
            xq, x_inv = _q(x.reshape(-1, shp[-1]), E4, M4)
            b = None if lin.bias is None else lin.bias.to(torch.bfloat16)
            y = torch._scaled_mm(xq, wq.t(), scale_a=x_inv, scale_b=w_inv, out_dtype=torch.bfloat16,
                                 bias=b, use_fast_accum=True)
            return y.reshape(*shp[:-1], -1).to(x.dtype)

        for m in tuner_layers:
            base = getattr(m, "base_layer", None)
            if isinstance(base, torch.nn.Linear):
                base.forward = (lambda lin, f: (lambda x: fp8_linear(lin, x, f)))(base, base.forward)
                fp8_layers.append(base)
        print(f"[grpo] FP8 no-grad matmuls on {len(fp8_layers)} frozen base Linear layers", flush=True)

    def drop_fp8():
        for lin in fp8_layers:
            lin._f8 = None
    master = {n: p.detach().float().clone().requires_grad_(True) for n, p in lora_params.items()}
    opt = torch.optim.AdamW(master.values(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    print(f"[grpo] LoRA tensors: {len(lora_params)}  params: {sum(p.numel() for p in master.values())/1e6:.1f}M",
          flush=True)

    # cache write-backs must not keep autograd history across steps
    from wan_5b.modules.causal_model import CausalWanModel
    original_apply = CausalWanModel._apply_cache_updates

    def apply_detached(self, kv_cache, infos):
        for _, (_, _, info) in infos:
            if info is not None:
                for k in ("new_k", "new_v"):
                    if torch.is_tensor(info.get(k)):
                        info[k] = info[k].detach()
        return original_apply(self, kv_cache, infos)

    CausalWanModel._apply_cache_updates = apply_detached

    # T5-xxl (~11 GB) only encodes two prompts per iteration: cache its outputs, keep it on the CPU
    text_encoder = pipe.text_encoder
    t5_forward = text_encoder.forward
    t5_cache = {}

    def cached_t5(text_prompts):
        key = tuple(text_prompts)
        if key not in t5_cache:
            text_encoder.to(device)
            try:
                with torch.no_grad():
                    out = t5_forward(text_prompts)
                t5_cache[key] = {k: v.detach() for k, v in out.items()}
            finally:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()
        return t5_cache[key]

    text_encoder.forward = cached_t5
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # the replayed (grad) forward keeps only each block's input; the block, incl. its clone of the
    # KV cache, is recomputed in backward
    import torch.utils.checkpoint as cp
    for blk in wrapper.model.blocks:
        def ckpt_forward(*args, _f=blk.forward, **kw):
            if torch.is_grad_enabled():
                return cp.checkpoint(_f, *args, use_reentrant=False, **kw)
            return _f(*args, **kw)
        blk.forward = ckpt_forward

    sigmas = shifted_sigmas(a.sample_steps, shift)
    traj = Trajectory(sigmas, a.sample_steps - a.ode_tail, a.eta)
    pipe._initialize_sample_scheduler = lambda noise: SwapScheduler(traj)

    # Activation recorder: during sampling, token-mean of every DiT block output (residual stream) and
    # of every FFN hidden layer (post-GELU input of ffn.2), conditional branch only, at a few denoising
    # steps per block. Paired with the per-rollout reward this gives difference-of-means directions
    # (good vs. bad videos) per layer for later weight-space edits.
    record_steps = sorted({1, a.sample_steps // 2, a.sample_steps - 1})
    acts = {"key": None, "store": {}}
    n_layers = len(wrapper.model.blocks)

    def slot(kind, i, vec):
        key = acts["key"]
        if key is None:
            return
        buf = acts["store"].setdefault(key, {})
        if kind not in buf:
            buf[kind] = torch.zeros(n_layers, vec.shape[-1], dtype=torch.float16)
        buf[kind][i] = vec.detach().float().mean(dim=tuple(range(vec.dim() - 1))).half().cpu()

    for i, blk in enumerate(wrapper.model.blocks):
        blk.register_forward_hook(lambda m, inp, out, i=i: slot("block", i, out[0] if isinstance(out, tuple) else out))
        blk.ffn[2].register_forward_pre_hook(lambda m, inp, i=i: slot("ffn", i, inp[0]))

    original_forward = wrapper.forward
    grad_state = {}

    def forward(*args, **kw):
        ts = kw.get("timestep")
        noisy = torch.is_tensor(ts) and bool((ts > 0).any())
        if (traj.mode == "sample" and noisy and traj.chunk not in traj.frozen and traj.step in record_steps
                and kw.get("kv_cache") is pipe.kv_cache_pos):
            acts["key"] = (traj.chunk, traj.step)
            try:
                with torch.no_grad():
                    return original_forward(*args, **kw)
            finally:
                acts["key"] = None
        if noisy and traj.chunk in traj.frozen:
            return torch.zeros_like(kw["noisy_image_or_video"]), None
        if traj.mode == "replay" and noisy:
            key = (traj.chunk, traj.step)
            if key not in traj.selected:
                x = kw["noisy_image_or_video"]
                return torch.zeros_like(x), None
            if kw.get("kv_cache") is pipe.kv_cache_neg:
                with torch.no_grad():
                    return original_forward(*args, **kw)
            if a.beta > 0:
                with torch.no_grad():
                    for m in tuner_layers:
                        m.enable_adapters(False)
                    try:
                        grad_state["ref"] = original_forward(*args, **kw)[0].detach()
                    finally:
                        for m in tuner_layers:
                            m.enable_adapters(True)
            with torch.enable_grad():
                out = original_forward(*args, **kw)
            grad_state["cond"] = out[0]
            return out
        with torch.no_grad():
            return original_forward(*args, **kw)

    wrapper.forward = forward

    reward = Reward(a.dino, device)
    clips = load_clips(a.manifest, a.latent_frames)
    print(f"[grpo] {len(clips)} RoVid-X clips with >= {a.latent_frames} latent frames (test clips excluded)",
          flush=True)

    start = 0
    state_path = a.output / "state.pt"
    if state_path.exists():
        s = torch.load(state_path, map_location="cpu", weights_only=False)
        for n, t in s["master"].items():
            master[n].data.copy_(t)
            lora_params[n].data.copy_(t.to(torch.bfloat16))
        opt.load_state_dict(s["opt"])
        start = s["iter"]
        random.setstate(s["py_rng"])
        print(f"[grpo] resumed at iteration {start}", flush=True)
    else:
        random.seed(a.seed)

    n_raw = 4 * (a.latent_frames - 1) + 1
    frames_per_block = 8
    n_blocks = a.latent_frames // frames_per_block
    log = (a.output / "log.jsonl").open("a")

    def decode(latents):
        pipe.clear_cache()
        drop_fp8()
        wrapper.to("cpu"); reward.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                vids = [((pipe.vae.decode_to_pixel(l.to(device)) * 0.5 + 0.5).clamp(0, 1))[0].cpu()
                        for l in latents]
        finally:
            wrapper.to(device); reward.model.to(device)
        return vids

    for it in range(start, a.iters):
        t0 = time.time()
        for k in [k for k in t5_cache if len(k) > 1 or k[0] != pipe.negative_prompt]:
            del t5_cache[k]
        clip = random.choice(clips)
        real = read_video(clip["video"], n_raw, inf.HEIGHT, inf.WIDTH)
        if real.shape[0] < n_raw:
            continue
        prompts = [[clip["instruction"]] * n_blocks]
        with torch.no_grad():
            pixels = (real[:1] * 2 - 1).permute(1, 0, 2, 3)[None].to(device, torch.bfloat16)
            first = pipe.vae.encode_to_latent(pixels).to(device, torch.bfloat16)

        # ---- rollouts
        # The group shares its first --shared-blocks blocks (sampled once by member 0) and branches after
        # them; the shared part has zero group advantage, so it is neither re-sampled nor trained.
        groups = []
        traj.frozen = {}
        for g in range(a.group):
            seed = random.randrange(1 << 31)
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, a.latent_frames, 48, inf.HEIGHT // 16, inf.WIDTH // 16), device=device,
                                dtype=torch.bfloat16, generator=gen)
            traj.mode, traj.chunk, traj.records, traj.rng = "sample", -1, {}, gen
            acts["store"] = {}
            pipe.clear_cache()
            with torch.no_grad():
                lat = pipe.inference(noise=noise, text_prompts=prompts, initial_latent=first, return_latents=True)
            groups.append({"latents": lat.detach().cpu(), "records": traj.records, "noise": noise.cpu(),
                           "seed": seed, "acts": acts["store"]})
            if g == 0:
                traj.frozen = {c: lat[:, c * 8:(c + 1) * 8].detach() for c in range(a.shared_blocks)}
        t_sample = time.time() - t0

        # ---- rewards (24 fps video, every 8th frame)
        vids = decode([g["latents"] for g in groups])
        idx = torch.arange(0, n_raw, 8)
        real_s = real[idx]
        scores, maps = [], []
        for v in vids:
            scores.append(reward.score(v[idx.clamp(max=v.shape[0] - 1)], real_s))
            maps.append(reward.last_maps)
        adv = advantages(scores)
        del vids
        t_reward = time.time() - t0 - t_sample

        # ---- policy update: replay every rollout, train K SDE steps per block
        for p in master.values():
            p.grad = None
        stats = {"ratio": [], "kl": []}
        n_terms = a.group * (n_blocks - a.shared_blocks) * a.train_steps
        for g, grp in enumerate(groups):
            A = adv[g]
            traj.mode, traj.chunk, traj.records = "replay", -1, grp["records"]
            sde_steps = list(range(traj.n_sde))
            traj.selected = {(c, s) for c in range(a.shared_blocks, n_blocks)
                             for s in random.sample(sde_steps, a.train_steps)}
            # ODE-tail outputs are needed as replay targets too: rebuild them from the stored latents
            lat = grp["latents"]
            traj.final = {c: {s: lat[:, c * 8:(c + 1) * 8] for s in range(a.sample_steps)} for c in range(n_blocks)}

            def on_step(c, s, flow_uncond_mixed, rec, A=A):
                cond = grad_state.pop("cond")
                ref = grad_state.pop("ref", None)
                # flow_pred handed to the scheduler is uncond + w (cond - uncond); rebuild it with the
                # gradient flowing through the conditional branch only
                uncond = ((flow_uncond_mixed.detach() - a.guidance * cond.detach()) / (1 - a.guidance))
                v = uncond + a.guidance * (cond - uncond)
                x = rec["x"].to(device).float()
                sig, sig_n = float(sigmas[s]), float(sigmas[s + 1])
                mean, std = sde_mean_std(x, v.float(), sig, sig_n, float(sigmas[1]), a.eta)
                prev = rec["prev"].to(device)
                logp = gaussian_logp(prev, mean, std, c)
                # one update per rollout batch: the policy that sampled is the current one (up to FP8
                # rounding in the rollout), so the ratio is taken against the detached replay log-prob
                ratio = torch.exp(logp - logp.detach())
                loss = -(ratio * A)
                kl = torch.zeros((), device=device)
                if ref is not None:
                    v_ref = uncond + a.guidance * (ref - uncond)
                    mean_ref, _ = sde_mean_std(x, v_ref.float(), sig, sig_n, float(sigmas[1]), a.eta)
                    diff = (mean - mean_ref.detach()) ** 2
                    if c == 0:
                        diff = diff[:, 1:]
                    kl = diff.mean() / (2 * std ** 2)
                    loss = loss + a.beta * kl
                stats["ratio"].append(torch.exp(logp.detach() - rec["logp"]).item())
                loss = loss / n_terms
                loss.backward()
                stats["kl"].append(kl.item())

            traj.on_step = on_step
            pipe.clear_cache()
            pipe.inference(noise=grp["noise"].to(device), text_prompts=prompts, initial_latent=first,
                           return_latents=True)
            pipe.clear_cache()

        for n, p in lora_params.items():
            master[n].grad = p.grad.float() if p.grad is not None else torch.zeros_like(master[n])
            p.grad = None
        # which adapters the reward pushes, before clipping (per LoRA tensor)
        module_grad = {n: master[n].grad.norm().item() for n in master}
        gnorm = torch.nn.utils.clip_grad_norm_(master.values(), a.max_grad_norm).item()
        opt.step()
        with torch.no_grad():
            for n, p in lora_params.items():
                p.copy_(master[n].to(torch.bfloat16))

        rec = {"iter": it + 1, "clip": clip["id"], "instruction": clip["instruction"],
               "reward": scores, "adv": adv, "grad_norm": gnorm,
               "ratio_mean": float(np.mean(stats["ratio"])), "kl_mean": float(np.mean(stats["kl"])),
               "sec_sample": round(t_sample), "sec_reward": round(t_reward), "sec_total": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        rdir = a.output / "records"
        rdir.mkdir(exist_ok=True)
        torch.save({"iter": it + 1, "clip": clip["id"], "instruction": clip["instruction"],
                    "shared_blocks": a.shared_blocks, "record_steps": record_steps,
                    "sigmas": sigmas[record_steps].tolist(), "seeds": [g["seed"] for g in groups],
                    "reward": scores, "adv": adv, "maps": maps, "acts": [g["acts"] for g in groups],
                    "module_grad": module_grad}, rdir / f"iter_{it + 1:05d}.pt")
        mean_r = {k: float(np.mean([s[k] for s in scores])) for k in ("align", "patch", "motion")}
        print(f"[iter {it + 1:05d}] align={mean_r['align']:.4f} patch={mean_r['patch']:.4f} "
              f"motion={mean_r['motion']:.3f} kl={rec['kl_mean']:.2e} grad={gnorm:.2e} "
              f"time={rec['sec_total']}s (sample {rec['sec_sample']}s)", flush=True)

        if (it + 1) % a.save_every == 0:
            hist = a.output / "lora_history"
            hist.mkdir(exist_ok=True)
            st.save_file({n: master[n].detach().to(torch.bfloat16).contiguous() for n in master},
                         str(hist / f"iter_{it + 1:05d}.safetensors"))
            tmp = state_path.with_suffix(".tmp")
            torch.save({"master": {n: t.detach() for n, t in master.items()}, "opt": opt.state_dict(),
                        "iter": it + 1, "py_rng": random.getstate()}, tmp)
            os.replace(tmp, state_path)
            print(f"[grpo] saved iteration {it + 1}", flush=True)


if __name__ == "__main__":
    main()
