"""Dynamics Forcing LoRA launcher for causal video training.

Wraps the official stage1/run.py flow. Differences:
  * Replicated data parallelism with synchronized LoRA gradients
  * LoRA adapters on every block's attention and FFN linears; base frozen
  * frozen generator base and T5 stored in bf16; LoRA kept fp32 outside FSDP
  * EMA off (it would shadow all 5B params on CPU every step)
  * our own RoVid-X/EgoVerse/EgoDex manifests and receipt
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config"
CORE = Path(__file__).resolve().parent / "engine"
DATA_ROOT = Path("datasets/robot-video")
STEPS = {"S": 28602, "M": 11155}
LORA_TARGETS = r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)|blocks\.\d+\.ffn\.(0|2)"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STEPS, default="S")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, default=Path("assets"))
    p.add_argument("--checkpoint", "--generator", dest="generator", type=Path, default=Path("weights/Astronex-World/causal-generator.pt"))
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help="Manifest root (manifests/<stage>/<split>.jsonl + receipt.json).")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Micro-steps (one clip each) per optimizer update. Official runs average "
                        "64 clips per update; with batch 1 the diffusion gradient is very noisy.")
    p.add_argument("--height", type=int, default=480, help="Train resolution; official is 704x1280.")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Official config uses 10; LoRA at 2e-4 diverged at step 650 under that.")
    p.add_argument("--warmup", type=int, default=100, help="Linear LR warmup steps.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="Cosine floor as a fraction of --lr.")
    p.add_argument("--save-every", "--checkpoint-every", dest="save_every", type=int, default=500,
                   help="Save every N optimizer updates.")
    p.add_argument("--batch-size", type=int, default=1, help="Video clips per microbatch on each GPU.")
    p.add_argument("--epochs", type=int, help="Set the training length in data epochs.")
    p.add_argument("--gpus", type=int, help="Launch this many local GPU workers.")
    p.add_argument("--nvlink", action="store_true", help="Use NCCL peer transfers over NVLink where available.")
    p.add_argument("--eval-every", type=int, default=0, help="Generate the eval panel every N steps.")
    p.add_argument("--eval-before", action="store_true", help="Generate the eval panel before step 1.")
    p.add_argument("--goal-first", action="store_true",
                   help="Plan B: move the clip's last latent frame (the outcome) to sequence slot 1 "
                        "with its true temporal RoPE position, so every later block is generated "
                        "toward a goal the model imagined first. Inference needs no goal input.")
    p.add_argument("--ckpt-every", type=int, default=1,
                   help="Activation-checkpoint every Kth transformer block (1 = all, the 48 GB "
                        "setting; 0 = none). Recompute costs ~30%% of a step on large cards.")
    p.add_argument("--no-ckpt-blocks", type=int, default=0,
                   help="Leave the first N blocks un-checkpointed, checkpoint the rest. Each "
                        "un-checkpointed block costs ~4 GB at F64/480p; overrides --ckpt-every.")
    p.add_argument("--t5-on-gpu", action="store_true",
                   help="Keep UMT5 resident on the GPU instead of streaming it in per step.")
    p.add_argument("--workers", type=int, default=2, help="Training DataLoader workers.")
    p.add_argument("--true-end-goal", action="store_true",
                   help="With --goal-first: goal = the video's real last frame at its real time "
                        "position (needs manifests with video_end_raw24).")
    p.add_argument("--rollout-prob", type=float, default=0.0,
                   help="Probability of replacing the first generated block in teacher-forcing "
                        "history with a no-grad self rollout.")
    p.add_argument("--rollout-steps", type=int, default=4,
                   help="Denoising evaluations used for the self-generated history block.")
    p.add_argument("--rollout-frames", type=int, default=72,
                   help="With --process-rollout: the one latent length that gets the self rollout "
                        "(72 = M/F72; 56 = S/F56, ~30%% of S samples). Other lengths train normally.")
    p.add_argument("--gan-weight", type=float, default=0.0,
                   help="With --process-rollout: weight of a latent-video hinge GAN term on the "
                        "predictions made from self-generated history (real = real video latents).")
    p.add_argument("--gan-source", choices=("pred", "rollout"), default="pred",
                   help="pred: judge the supervised predictions after self history; rollout: judge "
                        "the self-generated first block itself (its t=333 step keeps gradients).")
    p.add_argument("--gan-warmup", type=int, default=20,
                   help="Discriminator-only updates before the generator GAN term switches on.")
    p.add_argument("--hg-train-w", type=float, default=0.0,
                   help="Training-time history guidance (model-guidance style): the prediction is shifted by "
                        "-w*sg(f(real hist) - f(weak hist)) before the loss, so the model itself learns the "
                        "history-guided velocity and inference needs no extra noisy-history branch.")
    p.add_argument("--hg-train-sigma", type=float, default=0.5)
    p.add_argument("--hg-anchor-sigma", type=float, default=0.0,
                   help="Anchor guidance: in the weak branch of training-time history guidance the reference "
                        "(first) frame is re-noised to this level too, so the learned guidance also amplifies "
                        "what the clean first frame contributes (robot / object appearance). 0 = reference clean.")
    p.add_argument("--hg-train-prob", type=float, default=0.5)
    p.add_argument("--joint-prob", type=float, default=0.0,
                   help="Joint (rolling) denoising: per sample, probability that W consecutive blocks get a noise "
                        "staircase (older = cleaner) and every window block but the newest is seen by later blocks "
                        "at its own noisy state instead of clean, as in rolling-window inference.")
    p.add_argument("--joint-window", type=int, default=3)
    p.add_argument("--joint-gap", default="0.15,0.4",
                   help="Window history noise = newer block's noise minus a gap ~ U(lo,hi): always cleaner than the "
                        "block it conditions, never clean enough to copy.")
    p.add_argument("--joint-hist-range", default="0.15,0.6",
                   help="Clamp for window-history noise levels: >= lo blocks pixel copying, <= hi keeps structure.")
    p.add_argument("--train-grid-steps", type=int, default=0,
                   help="Snap every training timestep to the sigma grid of an N-step shifted sampler (8 = the "
                        "inference schedule), so training sees exactly the noise levels inference visits.")
    p.add_argument("--resample-prob", type=float, default=0.0,
                   help="Resampling Forcing: per step, probability that the history stream is the model's own "
                        "one-step reconstruction of the real clip noised to a low level (no gradient), so the "
                        "target is learned from history carrying the model's own errors.")
    p.add_argument("--resample-steps", type=int, default=1,
                   help="Sampling steps of the resampling pass. >1: deterministic DDIM steps from the resampling "
                        "noise level to 0, each conditioned on the model's own previous estimate of every block "
                        "(parallel stand-in for autoregressive rollout, so errors carry across blocks).")
    p.add_argument("--resample-shift", type=float, default=0.6,
                   help="Shift of the logit-normal resampling noise level (< 1 favours low noise).")
    p.add_argument("--hn-couple-prob", type=float, default=0.0,
                   help="With --history-noise: per-sample probability that every target block shares one "
                        "timestep t and the history noise follows t (noise-level-aware history).")
    p.add_argument("--hn-split-t", type=float, default=0.6,
                   help="Coupled samples: t/1000 >= split decides motion (near-clean history), below it "
                        "decides detail (heavily noised history, appearance from the clean reference).")
    p.add_argument("--hn-high-range", type=str, default="0,0.1",
                   help="History noise range for coupled samples with t at or above --hn-split-t.")
    p.add_argument("--hn-low-range", type=str, default="0.3,0.6",
                   help="History noise range for coupled samples with t below --hn-split-t.")
    p.add_argument("--timestep-shift", type=float, default=0.0,
                   help="Override training.timestep_shift (0 keeps the config value).")
    p.add_argument("--history-noise", type=float, default=0.0,
                   help="Max sigma of noise added to the real (clean) history stream: per block "
                        "sigma ~ U(0, max), x = (1-s)*x + s*eps, the model is told via aug_t=1000*s. The "
                        "reference frame stays clean. Inference must add a matching noise level.")
    p.add_argument("--no-error-recycling", action="store_true",
                   help="Disable video error recycling: history is always the real clean video "
                        "(pure teacher forcing, no injected self-errors).")
    p.add_argument("--allow-data-change", action="store_true",
                   help="Resume across a manifest change (new --data-root): the dataset, sampler and "
                        "data-cursor sections of the resume contract may differ; all else stays strict.")
    p.add_argument("--delta-weight", type=float, default=0.0,
                   help="Latent difference regression: MSE between predicted and real frame-to-frame "
                        "changes (x0[t+1]-x0[t] vs z[t+1]-z[t]), weighted by (1 - noise level).")
    p.add_argument("--align-weight", type=float, default=0.0,
                   help="Terminal-slice alignment: cosine between a projection of each predicted change and "
                        "the (stop-grad) pooled remaining change to the clip's last block.")
    p.add_argument("--align-slice", type=int, default=8, help="Latent frames in the terminal slice.")
    p.add_argument("--prompt-drop-prob", type=float, default=0.0,
                   help="Per step, with this probability shorten the prompt (cut the tail or drop random "
                        "words); at least one word always remains, never an empty prompt.")
    p.add_argument("--contrast-prob", type=float, default=0.0,
                   help="Per step, with this probability run a no-grad forward under another clip's prompt "
                        "and push the predicted frame changes away from that wrong-prompt prediction.")
    p.add_argument("--contrast-weight", type=float, default=0.2)
    p.add_argument("--contrast-cap", type=float, default=0.1,
                   help="Cap on the change-space distance the contrast term rewards.")
    p.add_argument("--goal-insert-prob", type=float, default=0.0,
                   help="Per step, with this probability the clip's real last frame (uniform-random noise "
                        "level) becomes a context frame at its true end time, visible to every block; it is "
                        "not supervised. Training-only: inference never inserts it.")
    p.add_argument("--offload-ckpt", action="store_true",
                   help="Keep each checkpointed block's saved input in pinned CPU memory instead of on the "
                        "GPU (exact; frees ~0.9 GB per block at L/192 frames, costs PCIe transfers).")
    p.add_argument("--ffn-chunk", type=int, default=0,
                   help="Run each block's FFN over sequence chunks of this many tokens, each under its "
                        "own activation checkpoint (exact; caps the 14336-wide FFN activation for M/L).")
    p.add_argument("--motion-weight", type=float, default=0.0,
                   help="Up-weight the diffusion MSE where the real video moves: w = 1 + k*min(m/mean(m), cap), "
                        "renormalised to mean 1 per sample (m = |frame t - frame t-1| of the clean latent).")
    p.add_argument("--motion-cap", type=float, default=4.0)
    p.add_argument("--process-rollout", action="store_true",
                   help="Chronological training with self-generated early process history. "
                        "Does not reorder, load, condition on, or predict a goal frame.")
    p.add_argument("--bidirectional-attention", action="store_true",
                   help="Train S/M/L with full-sequence bidirectional self-attention. "
                        "Uses one diffusion timestep for the whole clip and disables the "
                        "causal clean-history concatenation to prevent future-target leakage.")
    p.add_argument("--vae-type", choices=("wan", "mg_lightvae", "mg_lightvae_v2"),
                   default="wan", help="Video VAE used to encode training targets.")
    p.add_argument("--action-loss-weight", type=float, default=0.0,
                   help="Weight of chronological latent-motion imitation loss.")
    p.add_argument("--stop-step", type=int, help="Stop this run after this optimizer step.")
    p.add_argument("--fp8", action="store_true",
                   help="FP8 (e4m3/e5m2) matmuls for the frozen base Linear layers under LoRA.")
    p.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --output.")
    p.add_argument("--plan", action="store_true", help="Print the resolved config and exit.")
    a = p.parse_args()
    for name in ("save_every", "epochs", "gpus", "grad_accum", "batch_size"):
        value = getattr(a, name)
        if value is not None and value < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    launched = int(os.environ.get("WORLD_SIZE", "1"))
    if launched > 1 and a.gpus is not None and a.gpus != launched:
        p.error("--gpus must match the torchrun world size")
    a.world_size = max(launched, a.gpus or 1)
    if a.nvlink and a.world_size < 2:
        p.error("--nvlink requires multiple GPU workers")
    if a.world_size > 1 and a.gan_weight > 0:
        p.error("This auxiliary training mode requires one GPU")
    return a


def build_config(a):
    from omegaconf import OmegaConf
    os.environ["DYNAMICS_FORCING_VIDEO_DATA_ROOT"] = str(a.data_root)
    os.environ.setdefault("DYNAMICS_FORCING_VIDEO_INDEX_CACHE", "outputs/index-cache")
    os.environ[f"DYNAMICS_FORCING_VIDEO_STAGE_{a.stage}_GENERATOR_MODEL"] = str(a.generator)
    os.environ[f"DYNAMICS_FORCING_VIDEO_STAGE_{a.stage}_GENERATOR_SHA256"] = "unspecified"
    cfg = OmegaConf.load(CONFIG / f"train_{a.stage.lower()}.yaml")
    OmegaConf.resolve(cfg)
    cfg.infra.sequence_parallel_size = 1
    cfg.infra.sharding_strategy = "no_shard"
    cfg.data.manifest_receipt_sha256 = hashlib.sha256((a.data_root / "receipt.json").read_bytes()).hexdigest()
    cfg.training.batch_size = a.batch_size
    cfg.training.lr = a.lr
    if a.timestep_shift > 0:
        cfg.training.timestep_shift = a.timestep_shift
    cfg.training.max_grad_norm = a.max_grad_norm
    cfg.training.gradient_accumulation_steps = a.grad_accum
    # 704x1280 with teacher forcing is ~112k tokens; too many activations for one 48 GB card
    shape = list(cfg.data.image_or_video_shape)
    shape[3], shape[4] = a.height // 16, a.width // 16
    cfg.data.image_or_video_shape = shape
    cfg.training.ema_weight = 0.0
    cfg.lora = {"rank": a.rank, "alpha": a.alpha, "targets": LORA_TARGETS}
    if a.epochs is not None:
        with open(cfg.data.data_path) as manifest:
            count = sum(bool(line.strip()) for line in manifest)
        samples_per_worker = (count // a.world_size if cfg.data.drop_last
                              else math.ceil(count / a.world_size))
        batches_per_epoch = math.ceil(samples_per_worker / int(cfg.training.batch_size))
        if batches_per_epoch < 1:
            raise ValueError("The training manifest has no batches for the requested worker count.")
        cfg.training.max_iters = math.ceil(a.epochs * batches_per_epoch / a.grad_accum)
        print(f"[epochs] epochs={a.epochs} batches_per_worker={batches_per_epoch} "
              f"optimizer_updates={cfg.training.max_iters}", flush=True)
    cfg.lr_schedule = {"warmup": a.warmup, "min_ratio": a.min_lr_ratio,
                       "total": int(cfg.training.max_iters)}
    cfg.training.log_iters = a.save_every  # also the checkpoint interval
    if a.no_error_recycling:
        from omegaconf import OmegaConf as _OC
        hit = 0
        for _path in ("training.error_recycling", "error_recycling", "algorithm.error_recycling"):
            if _OC.select(cfg, _path) is not None:
                _OC.update(cfg, _path + ".enabled", False, merge=True); hit += 1
        print(f"[no-error-recycling] disabled in {hit} config section(s)", flush=True)
    cfg.data.shuffle = True  # random sample order (seeded permutation), not manifest order
    cfg.evaluation.interval = a.eval_every
    if a.bidirectional_attention:
        # Full-sequence attention cannot retain causal teacher forcing: its clean
        # half contains future ground truth and would leak the answer into every
        # noisy query.  Data, flow target, masks, error recycling and optimizer
        # stay unchanged; only the architecture-required input semantics change.
        cfg.algorithm.causal = False
        cfg.algorithm.generator_is_causal = False
        cfg.algorithm.teacher_forcing = False
    if a.vae_type != "wan":  # the causal wrapper does not accept vae_type
        cfg.model_kwargs.vae_type = a.vae_type
    # small fixed panel: one clip per source dataset from our val split, at a length the stage
    # actually has (S 32, M 72, L 128) — the config validator rejects an empty panel even
    # when in-training evaluation is off
    panel_frames = {"S": 32, "M": 72}[a.stage]
    picked = {}
    for line in (a.data_root / "manifests" / a.stage / "val.jsonl").open():
        row = json.loads(line)
        # the evaluator requires the clip's native length to equal the group's num_frames
        if row["valid_latent_frames"] == panel_frames and row["source_dataset"] not in picked:
            picked[row["source_dataset"]] = row["id"]
    cfg.evaluation.groups = [{"name": f"panel_f{panel_frames}", "num_frames": panel_frames,
                              "eval_data_path": str(a.data_root / "manifests" / a.stage / "val.jsonl"),
                              "seed": int(cfg.evaluation.seed), "sample_ids": sorted(picked.values())}]
    return cfg


def goal_first_order(num_frames):
    """Generation order [f0, f_last, f1..f_{last-1}] and the RoPE offsets that keep each
    frame at its real time index (positions become [0, F-1, 1, ..., F-2])."""
    import torch
    order = [0, num_frames - 1] + list(range(1, num_frames - 1))
    offset = torch.full((num_frames,), -1.0, dtype=torch.float64)
    offset[0] = 0.0
    offset[1] = float(num_frames - 2)
    return order, offset


def install_true_end_goal():
    """Goal = the episode's real last frame, not the last frame of the truncated training prefix.

    ~19% of M clips (20% of RoVid-X in S) are cut before the task finishes, so their prefix end
    is a half-done state. The dataset additionally decodes the video's true final frame
    (manifest field video_end_raw24) with the same resize/normalize as training frames; the
    collate passes it through; the trainer encodes it on its own and puts it in the goal slot
    at its true temporal position (which can lie beyond the training window).
    """
    import numpy as np
    import torch
    import utils.dataset as ds_mod
    import trainer.diffusion as td

    Dataset = ds_mod.RobotVideoManifestDataset
    original_getitem = Dataset.__getitem__

    def decode_frame(self, video_path, raw24_index):
        h, w = self.video_size
        if ds_mod.decord is not None:
            reader = ds_mod.decord.VideoReader(str(video_path), width=w, height=h)
            src_index = min(len(reader) - 1, int(round(raw24_index * float(reader.get_avg_fps()) / 24.0)))
            frame = reader[src_index].asnumpy()
        else:
            import subprocess
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{raw24_index / 24.0:.4f}", "-i", str(video_path),
                   "-vf", f"scale={w}:{h}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            frame = np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout,
                                  dtype=np.uint8).reshape(h, w, 3)
        t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        if tuple(t.shape[1:]) != tuple(self.video_size):
            t = self.resize_transform(t)
        return self.normalize(t).to(torch.float16)

    def getitem(self, index):
        out = original_getitem(self, index)
        if self.evaluation_mode:
            return out
        row = self._read_row(index)
        end24 = max(int(row.get("video_end_raw24", row["valid_raw_frames"])), int(row["valid_raw_frames"]))
        out["goal_frame"] = decode_frame(self, row["video"], end24 - 1)
        out["goal_end24"] = end24
        return out

    Dataset.__getitem__ = getitem

    original_collate = td.multi_video_collate_fn

    def collate(batch):
        result = original_collate(batch)
        if "goal_frame" in batch[0]:
            result["goal_frame"] = torch.stack([b["goal_frame"] for b in batch], dim=0)  # [B, C, H, W]
            result["goal_end24"] = torch.tensor([b["goal_end24"] for b in batch], dtype=torch.long)
        return result

    td.multi_video_collate_fn = collate


def goal_latent_index(end24):
    """Latent index of raw (24 fps) frame end24-1 under the 4x causal VAE (frame 0 -> latent 0)."""
    return (int(end24) - 1 + 3) // 4


def install_goal_first(trainer_cls, goal_clean_prob=0.15, goal_max_sigma=0.7,
                       true_end=False, rollout_prob=0.0, rollout_steps=4):
    """Plan B training: the outcome frame is denoised in block 0 (supervised by the real last
    frame) and every later block attends to it, so the model learns to imagine where the task
    ends before generating how it gets there.

    Teacher forcing would otherwise hand every later block the *exact* real goal through the
    clean history stream, while at inference that slot holds the model's own imperfect guess.
    With a perfect target in view the process blocks can just drift toward it (or stall and
    jump at the end). So the goal's history copy is noised to a random sigma in
    [0, goal_max_sigma] (kept clean with goal_clean_prob) and its aug_t says how noisy it is:
    the goal becomes a direction to head for, not pixels to copy.
    """
    import random
    import torch
    import utils.wan_5b_wrapper as wrapper_mod
    original_step = trainer_cls.train_one_step
    Wrapper = wrapper_mod.WanDiffusionWrapper
    original_forward = Wrapper.forward
    state = {"active": False, "rollout": False, "gap": None}

    def forward_noisy_goal(self, *args, clean_x=None, aug_t=None, **kwargs):
        if state["active"] and clean_x is not None and clean_x.shape[1] > 2:
            timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
            noisy = kwargs.get("noisy_image_or_video", args[0] if args else None)
            conditional = kwargs.get("conditional_dict", args[1] if len(args) > 1 else None)
            # Scheduled self-conditioning: generate block 0 from noise using only the real
            # conditioning frame and text, then expose that imperfect block (not the real
            # block) as history to the supervised pass.  Because the model is block-causal,
            # block 0 cannot read clean_x from itself or future blocks.
            # The fused AdaLN kernel on this stack is not safe when the repeated rollout
            # call switches between dynamic M bucket sizes.  Keep rollout on the canonical
            # 72-frame M bucket; all longer buckets retain ordinary TF + error recycling.
            do_rollout = (state["rollout"] and noisy is not None and conditional is not None
                          and clean_x.shape[1] == 72)
            state["rollout"] = do_rollout
            if do_rollout:
                block = min(int(getattr(self.model, "num_frame_per_block", 8)), clean_x.shape[1])
                steps = max(1, int(rollout_steps))
                with torch.no_grad():
                    rollout = torch.randn_like(noisy)
                    rollout[:, 0] = clean_x[:, 0]
                    base_noise = rollout.clone()
                    times = torch.linspace(999.0, 1.0, steps, device=noisy.device, dtype=timestep.dtype)
                    for i, t in enumerate(times):
                        rt = torch.full(timestep.shape, t, device=timestep.device, dtype=timestep.dtype)
                        rt[:, 0] = 0
                        _, x0 = original_forward(
                            self, noisy_image_or_video=rollout,
                            conditional_dict=conditional, timestep=rt,
                            # Keep the teacher-forcing stream present so the compiled
                            # training mask retains its 2x sequence shape.  Block 0 is
                            # causally first and cannot consume its own/future clean slots.
                            clean_x=clean_x, aug_t=torch.zeros_like(rt))
                        if i + 1 < steps:
                            next_t = times[i + 1]
                            # Flow-matching interpolation for the next, less noisy point.
                            sigma = (next_t.float() / 1000.0).clamp(0, 1).to(x0.dtype)
                            rollout[:, :block] = ((1 - sigma) * x0[:, :block]
                                                  + sigma * base_noise[:, :block])
                            rollout[:, 0] = clean_x[:, 0]
                    generated = x0[:, :block].detach()
                    generated[:, 0] = clean_x[:, 0]
                    state["gap"] = (generated[:, 1:block].float()
                                    - clean_x[:, 1:block].float()).square().mean().item()
                clean_x = clean_x.clone()
                clean_x[:, 1:block] = generated[:, 1:block]
                if aug_t is None:
                    aug_t = torch.zeros_like(timestep)
                else:
                    aug_t = aug_t.clone()
                aug_t[:, 1:block] = 0
            if aug_t is None:
                aug_t = torch.zeros_like(timestep)
            else:
                aug_t = aug_t.clone()
            if random.random() >= goal_clean_prob:
                sigma = random.uniform(0.0, goal_max_sigma)
                t_goal = torch.full((clean_x.shape[0],), sigma * 1000.0,
                                    device=clean_x.device, dtype=timestep.dtype)
                clean_x = clean_x.clone()
                goal = clean_x[:, 1]
                clean_x[:, 1] = self.scheduler.add_noise(goal, torch.randn_like(goal), t_goal).to(goal.dtype)
                aug_t[:, 1] = t_goal
        return original_forward(self, *args, clean_x=clean_x, aug_t=aug_t, **kwargs)

    Wrapper.forward = forward_noisy_goal

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        if not getattr(self, "_goal_first_ready", False):
            helper = self.sp_helper
            original_encode = helper.encode_raw_video_latents

            def encode_goal_first(batch, *, batch_size):
                clean, image, sharded = original_encode(batch, batch_size=batch_size)
                order, _ = goal_first_order(clean.shape[1])
                clean = clean[:, order].contiguous()
                if true_end and "goal_frame" in batch:
                    # the episode's real final frame, encoded on its own, in the goal slot
                    pix = batch["goal_frame"].to(device=self.device, dtype=self.dtype).unsqueeze(2)
                    with torch.no_grad():
                        goal = self.model.vae.encode_to_latent(pix).to(device=self.device, dtype=clean.dtype)
                    clean[:, 1:2] = goal
                return clean, image, sharded

            helper.encode_raw_video_latents = encode_goal_first
            self._goal_first_ready = True
        wan = self.model.generator.module.model
        num_frames = int(self.config.image_or_video_shape[1])
        _, offset = goal_first_order(num_frames)
        if true_end and "goal_end24" in batch:
            # goal sits at its real time index, which may lie past the end of this window
            goal_index = max(num_frames - 1, goal_latent_index(batch["goal_end24"][0]))
            offset[1] = float(goal_index - 1)
        wan.rope_temporal_offset = offset.to(self.device)
        # The goal frame is 1 of 24-64 frames, so the mean loss hides whether it is learned.
        # Capture the per-frame MSE (before timestep weighting) and log goal vs process.
        F = torch.nn.functional
        original_mse, captured = F.mse_loss, {}

        def mse_capture(a, b, *args, reduction="mean", **kw):
            out = original_mse(a, b, *args, reduction=reduction, **kw)
            if reduction == "none" and out.dim() == 5:
                captured["per_frame"] = out.detach().float().mean(dim=(2, 3, 4))
            return out

        F.mse_loss = mse_capture
        state["active"] = True
        # M has six length buckets.  Sampling every canonical F72 item gives an effective
        # rollout rate close to 1/6 while avoiding unsafe dynamic-shape kernel switching.
        state["rollout"] = rollout_prob > 0
        state["gap"] = None
        try:
            return original_step(self, batch, accumulation_step=accumulation_step,
                                 accumulation_steps=accumulation_steps)
        finally:
            state["active"] = False
            did_rollout, rollout_gap = state["rollout"], state["gap"]
            state["rollout"] = False
            F.mse_loss = original_mse
            wan.rope_temporal_offset = 0.0
            pf = captured.get("per_frame")
            if pf is not None and pf.shape[1] > 2:
                # block 0 = [first(ctx), goal, f1..f6]: goal and f1..f6 share one noise level,
                # so goal_mse vs block0_mse is the like-for-like comparison.
                print(f"[goal] step {self.step:07d} goal_mse={pf[0, 1].item():.4f} "
                      f"block0_mse={pf[0, 2:8].mean().item():.4f} "
                      f"process_mse={pf[0, 2:].mean().item():.4f} frames={pf.shape[1]}", flush=True)
            if did_rollout and rollout_gap is not None:
                print(f"[rollout] step {self.step:07d} blocks=1 steps={rollout_steps} "
                      f"gap_mse={rollout_gap:.6f}", flush=True)

    trainer_cls.train_one_step = train_one_step


GOAL = {"on": False, "sigma": 0.0}


HN = {"clean_x": None, "aug_t": None}
LAZY = []
DELTA_POOL = 4
JOINT = {"rows": None}


def install_history_noise(max_sigma, block=8, couple_prob=0.0, split_t=0.6,
                          high_range=(0.0, 0.1), low_range=(0.3, 0.6), resample_prob=0.0, resample_shift=0.6,
                          resample_steps=1, joint_prob=0.0, joint_window=3,
                          hg_w=0.0, hg_sigma=0.5, hg_prob=0.5,
                          joint_gap=(0.15, 0.4), joint_hist_range=(0.15, 0.6), grid_steps=0,
                          hg_anchor=0.0):
    """Teacher forcing on real history, slightly noised so that small errors in self-generated
    history at inference look like what training already saw (Diffusion-Forcing style)."""
    import torch
    import utils.wan_5b_wrapper as wrapper_mod
    Wrapper = wrapper_mod.WanDiffusionWrapper
    inner = Wrapper.forward

    def resampled_history(self, clean_x, aug_t, kwargs):
        """Resampling Forcing (self-resampling): noise the real clip to a low level drawn from a shifted
        logit-normal, let the current model denoise it in one step (teacher forcing on the real history,
        no gradient) and return its x0 estimate as the history. Reference frame stays real."""
        B, F = clean_x.shape[:2]
        sched = self.scheduler
        tab_s = sched.sigmas.to(clean_x.device).double()
        tab_t = sched.timesteps.to(clean_x.device)
        u = torch.sigmoid(torch.randn(B, device=clean_x.device, dtype=torch.float64))
        target = resample_shift * u / (1 + (resample_shift - 1) * u)
        idx = (tab_s[None] - target[:, None]).abs().argmin(dim=1)
        ts_in = kwargs["timestep"]
        s0 = tab_s[idx]                                        # (B,) start level
        eps = torch.randn_like(clean_x)

        def level(sig):                                        # (B,) sigma -> per-frame (sigma, timestep)
            j = (tab_s[None] - sig[:, None]).abs().argmin(dim=1)
            t_f = tab_t[j][:, None].to(ts_in.dtype).expand(B, F).clone()
            s_f = tab_s[j][:, None].expand(B, F).clone()
            t_f[:, 0] = 0
            s_f[:, 0] = 0
            return s_f, t_f

        s_f, t_f = level(s0)
        s5 = s_f.view(B, F, 1, 1, 1).to(clean_x.dtype)
        x = (1 - s5) * clean_x + s5 * eps
        hist = clean_x
        import wan_5b.modules.causal_model as cm
        triton_adaln = cm._TRITON_ADALN_ENABLED
        cm._TRITON_ADALN_ENABLED = False      # its grid caps rows at 65535 tokens; the full TF sequence is longer
        try:
            with torch.no_grad():
                for k in range(resample_steps):
                    _, x0 = inner(self, clean_x=hist, aug_t=aug_t,
                                  **dict(kwargs, noisy_image_or_video=x, timestep=t_f))
                    x0 = x0.to(clean_x.dtype)
                    x0[:, 0] = clean_x[:, 0]
                    # every block's history becomes the model's own current estimate, so the next step of
                    # block b is conditioned on generated (not real) blocks < b, as in a rollout
                    hist = x0
                    if k + 1 < resample_steps:
                        s_next, t_next = level(s0 * (1 - (k + 1) / resample_steps))
                        sn5 = s_next.view(B, F, 1, 1, 1).to(clean_x.dtype)
                        eps_hat = (x - (1 - s5) * x0) / s5.clamp_min(1e-4)
                        x = (1 - sn5) * x0 + sn5 * eps_hat      # deterministic DDIM step
                        s5, t_f = sn5, t_next
        finally:
            cm._TRITON_ADALN_ENABLED = triton_adaln
        HN["resampled"] = float(tab_s[idx].mean())
        return hist

    def joint_window_history(clean_x, aug_t, kwargs):
        """Rows picked by joint_get_timestep: blocks inside the window (except the newest) enter the history
        stream exactly as they are in the target stream (same noisy latent, same timestep), so each window
        block is learned while attending to cleaner-but-unfinished predecessors."""
        rows = JOINT.get("rows")
        if not rows:
            return clean_x, aug_t
        x_t, ts = kwargs.get("noisy_image_or_video"), kwargs.get("timestep")
        if x_t is None or not torch.is_tensor(ts) or ts.shape[:2] != clean_x.shape[:2]:
            return clean_x, aug_t
        B, F = clean_x.shape[:2]
        clean_x = clean_x.clone()
        aug_t = torch.zeros(B, F, device=clean_x.device, dtype=ts.dtype) if aug_t is None else aug_t.clone().to(ts.dtype)
        for r, blocks_r in rows:
            for b in blocks_r[:-1]:
                lo, hi = max(1, b * block), min(F, (b + 1) * block)   # frame 0 stays the clean reference
                clean_x[r, lo:hi] = x_t[r, lo:hi].to(clean_x.dtype)
                aug_t[r, lo:hi] = ts[r, lo:hi]
        JOINT["rows"] = None
        return clean_x, aug_t

    def history_guided(self, args, kwargs, clean_x, aug_t, out):
        if (hg_w <= 0 or clean_x is None or not torch.is_grad_enabled()
                or float(torch.rand(())) >= hg_prob):
            return out
        flow, x0 = out
        B, F = clean_x.shape[:2]
        weak = (1 - hg_sigma) * clean_x + hg_sigma * torch.randn_like(clean_x)
        if hg_anchor > 0:                                      # weak anchor: reference re-noised as well
            weak[:, 0] = (1 - hg_anchor) * clean_x[:, 0] + hg_anchor * torch.randn_like(clean_x[:, 0])
        else:
            weak[:, 0] = clean_x[:, 0]                         # reference frame stays clean
        ts = kwargs.get("timestep")
        dtype = ts.dtype if torch.is_tensor(ts) else clean_x.dtype
        at = torch.full((B, F), hg_sigma * 1000.0, device=clean_x.device, dtype=dtype)
        if aug_t is not None:
            at = torch.maximum(at, aug_t.to(dtype))
        at[:, 0] = 0 if aug_t is None else aug_t[:, 0].to(dtype)
        if hg_anchor > 0:
            at[:, 0] = hg_anchor * 1000.0
        import wan_5b.modules.causal_model as cm
        triton_adaln = cm._TRITON_ADALN_ENABLED
        cm._TRITON_ADALN_ENABLED = False
        try:
            with torch.no_grad():
                fw, _ = inner(self, *args, clean_x=weak, aug_t=at, **kwargs)
        finally:
            cm._TRITON_ADALN_ENABLED = triton_adaln
        d = (flow.detach().float() - fw.float())
        if hg_anchor > 0:
            d[:, 0] = 0                                        # the reference frame itself is never a target
        else:
            d[:, :block] = 0                                   # block 0 has no generated history
        HN["hg_d"] = float(d.abs().mean())
        return (flow.float() - hg_w * d).to(flow.dtype), x0

    def laziness(real, out):
        try:
            x0 = out[1]
            if real is None or x0 is None or x0.shape != real.shape or real.shape[1] <= block:
                return
            with torch.no_grad():
                r = real[:, block:].float(); prev = real[:, block - 1:-1].float()
                d_pred = (x0[:, block:].float() - prev).abs().mean()
                d_real = (r - prev).abs().mean().clamp_min(1e-6)
                HN["lazy"] = float(d_pred / d_real)
                LAZY.append(HN["lazy"])
                if len(LAZY) >= 1:
                    print(f"[laziness] change_ratio={HN['lazy']:.3f} (1=real motion, <1 lazier)", flush=True)
                    LAZY.clear()
        except Exception:
            pass

    def forward_hist_noise(self, *args, clean_x=None, aug_t=None, **kwargs):
        real_x = clean_x.detach() if (clean_x is not None and torch.is_grad_enabled()) else None
        out = _forward_hist_noise(self, *args, clean_x=clean_x, aug_t=aug_t, **kwargs)
        laziness(real_x, out)
        return out

    def _forward_hist_noise(self, *args, clean_x=None, aug_t=None, **kwargs):
        if (clean_x is not None and torch.is_grad_enabled() and resample_prob > 0
                and "timestep" in kwargs and float(torch.rand(())) < resample_prob):
            clean_x = resampled_history(self, clean_x, aug_t, kwargs)
            clean_x, aug_t = joint_window_history(clean_x, aug_t, kwargs)
            HN["clean_x"] = clean_x.detach()
            return history_guided(self, args, kwargs, clean_x, aug_t,
                                  inner(self, *args, clean_x=clean_x, aug_t=aug_t, **kwargs))
        HN.pop("resampled", None)
        if clean_x is not None and torch.is_grad_enabled():
            B, F = clean_x.shape[:2]
            nb = (F + block - 1) // block
            sig = torch.rand(B, nb, device=clean_x.device) * max_sigma
            ts = kwargs.get("timestep")
            if couple_prob > 0 and torch.is_tensor(ts) and ts.dim() == 2 and ts.shape[1] == F and F > block:
                # Coupled samples (all target blocks share one t, see coupled_get_timestep):
                # high t decides motion -> near-clean history; low t decides detail -> history keeps
                # only low frequencies, so arm appearance has to come from the clean reference frame.
                tgt = ts[:, 1:].float()
                coupled = (tgt == tgt[:, :1]).all(dim=1)
                if coupled.any():
                    t_cur = tgt[:, 0] / 1000.0
                    hi = t_cur >= split_t
                    a0 = torch.where(hi, torch.full_like(t_cur, high_range[0]), torch.full_like(t_cur, low_range[0]))
                    a1 = torch.where(hi, torch.full_like(t_cur, high_range[1]), torch.full_like(t_cur, low_range[1]))
                    sig_c = a0[:, None] + torch.rand(B, nb, device=clean_x.device) * (a1 - a0)[:, None]
                    sig = torch.where(coupled[:, None], sig_c.to(sig.dtype), sig)
                    HN["coupled"] = (int(coupled.sum()), float(t_cur[coupled].mean()))
            sig = sig.repeat_interleave(block, dim=1)[:, :F]
            sig[:, 0] = 0.0                                   # reference frame stays clean
            if GOAL["on"] and F > 1:
                sig[:, 1] = 0.0                               # inserted end frame has its own noise
            s5 = sig.view(B, F, 1, 1, 1).to(clean_x.dtype)
            clean_x = (1 - s5) * clean_x + s5 * torch.randn_like(clean_x)
            t = (sig * 1000.0)
            ts = kwargs.get("timestep")
            dtype = ts.dtype if torch.is_tensor(ts) else clean_x.dtype
            base = aug_t if aug_t is not None else torch.zeros(B, F, device=clean_x.device, dtype=dtype)
            aug_t = torch.maximum(base.to(dtype), t.to(dtype))
            clean_x, aug_t = joint_window_history(clean_x, aug_t, kwargs)
            HN["clean_x"], HN["aug_t"] = clean_x.detach(), aug_t.detach()
        return history_guided(self, args, kwargs, clean_x, aug_t,
                              inner(self, *args, clean_x=clean_x, aug_t=aug_t, **kwargs))

    Wrapper.forward = forward_hist_noise
    if hg_w > 0:
        print(f"[history-guidance-train] w={hg_w} weak-history sigma={hg_sigma} p={hg_prob} "
              f"anchor sigma={hg_anchor}", flush=True)
    print(f"[history-noise] on: sigma ~ U(0, {max_sigma}) per {block}-frame block, reference clean",
          flush=True)
    if resample_prob > 0:
        print(f"[resampling-forcing] p={resample_prob}: history = {resample_steps}-step self-generated re-denoising "
              f"of the real clip from sigma ~ shifted logit-normal (shift {resample_shift}), no gradient", flush=True)
    if couple_prob > 0:
        import model.diffusion as diffusion_mod
        from model.base import BaseModel
        original_get_timestep = BaseModel._get_timestep

        def coupled_get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                                 num_frame_per_block, uniform_timestep=False):
            idx = original_get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                                        num_frame_per_block, uniform_timestep=uniform_timestep)
            if (not uniform_timestep and max_timestep == self.scheduler.num_train_timesteps
                    and idx.dim() == 2 and idx.shape[1] > 1):
                pick = torch.rand(idx.shape[0], device=idx.device) < couple_prob
                if pick.any():
                    idx = idx.clone()
                    idx[pick] = idx[pick, 1:2].expand(-1, idx.shape[1])
            return idx

        diffusion_mod.CausalDiffusion._get_timestep = coupled_get_timestep
        print(f"[history-noise] noise-level-aware: p={couple_prob} split_t={split_t} "
              f"high(t>=split)~U{tuple(high_range)} low~U{tuple(low_range)}", flush=True)
    if joint_prob > 0:
        import model.diffusion as diffusion_mod
        prev_get_timestep = diffusion_mod.CausalDiffusion._get_timestep

        def joint_get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                               num_frame_per_block, uniform_timestep=False):
            idx = prev_get_timestep(self, min_timestep, max_timestep, batch_size, num_frame,
                                    num_frame_per_block, uniform_timestep=uniform_timestep)
            if (uniform_timestep or max_timestep != self.scheduler.num_train_timesteps
                    or idx.dim() != 2 or num_frame < joint_window * block + 1):
                return idx
            nb = (num_frame + block - 1) // block
            sig = self.scheduler.sigmas.to(idx.device).double()
            idx = idx.clone()
            grid = None
            if grid_steps > 0:                                   # sigma levels of the N-step shifted sampler
                shift = 5.0
                u_ = torch.linspace(1.0, 0.0, grid_steps + 1, dtype=torch.float64, device=idx.device)[:-1]
                grid = shift * u_ / (1 + (shift - 1) * u_)

            def snap(target):
                t = float(target)
                if grid is not None:
                    t = float(grid[(grid - t).abs().argmin()])
                return int((sig - t).abs().argmin())

            if grid is not None:                                 # every target frame on the inference grid
                for r in range(idx.shape[0]):
                    for f in range(idx.shape[1]):
                        idx[r, f] = snap(sig[idx[r, f]])
            rows = []
            lo_h, hi_h = joint_hist_range
            for r in range(idx.shape[0]):
                if float(torch.rand(())) >= joint_prob:
                    continue
                e = int(torch.randint(joint_window - 1, nb, ()))
                win = list(range(e - joint_window + 1, e + 1))
                # newest block: any level; each older block = newer level - gap, clamped to [lo_h, hi_h]
                level = float(sig[idx[r, min(e * block, idx.shape[1] - 1)]])
                levels = [level]
                for _ in win[:-1]:
                    gap = joint_gap[0] + float(torch.rand(())) * (joint_gap[1] - joint_gap[0])
                    levels.append(min(max(levels[-1] - gap, lo_h), hi_h))
                levels = levels[::-1]                             # oldest -> newest
                for b, lv in zip(win, levels):
                    idx[r, b * block:(b + 1) * block] = snap(lv)
                rows.append((r, win))
            JOINT["rows"] = rows or None
            return idx

        diffusion_mod.CausalDiffusion._get_timestep = joint_get_timestep
        print(f"[joint-denoising] p={joint_prob}: {joint_window}-block window, history noise = newer level - "
              f"U{tuple(joint_gap)} clamped to {tuple(joint_hist_range)}, grid_steps={grid_steps}", flush=True)


def install_delta_losses(trainer_cls, delta_w, align_w, slice_frames=8, proj_path=None, save_every=50,
                         prompt_drop=0.0, contrast_prob=0.0, contrast_w=0.2, contrast_cap=0.1):
    """Supervise what changes, not what the frame looks like.

    delta:  MSE( x0_hat[t+1]-x0_hat[t] , z[t+1]-z[t] ) per frame pair, weight (1 - t_{t+1}/1000).
    align:  1 - cos( P(pool(dx0_hat_t)) , sg(pool(mean(z[terminal slice]) - z[t])) ) for pairs before
            the terminal slice, same noise weight; P is a small trainable MLP (own optimizer).
    Frames are taken in true time order (the goal-insert reorder is undone); i2v context frames
    (the reference, and the inserted end frame) are never a predicted endpoint.
    """
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    import utils.wan_5b_wrapper as wrapper_mod
    Wrapper = wrapper_mod.WanDiffusionWrapper
    inner_forward = Wrapper.forward
    cap = {"x0": None, "t": None}

    def forward_capture(self, *args, **kwargs):
        out = inner_forward(self, *args, **kwargs)
        if cap.get("armed"):
            cap["x0"], cap["t"] = out[1], kwargs.get("timestep")
            cap["kw"] = dict(kwargs)
        return out

    Wrapper.forward = forward_capture
    proj = opt = None
    if align_w > 0:
        dim = 48 * 16
        proj = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, dim)).cuda()
        opt = torch.optim.Adam(proj.parameters(), lr=1e-4)
        if proj_path and os.path.exists(proj_path):
            ck = torch.load(proj_path, map_location="cuda", weights_only=False)
            proj.load_state_dict(ck["proj"]); opt.load_state_dict(ck["opt"])
            print(f"[delta] restored alignment projector from {proj_path}", flush=True)
    stats = {}
    import random as _rnd
    bank, cur = [], {"prompt": None, "contrast": False}

    def shorten(text):
        words = text.split()
        if len(words) < 3:
            return text
        if _rnd.random() < 0.5:                                   # cut the tail
            return " ".join(words[:_rnd.randint(1, len(words) - 1)])
        kept = [w for w in words if _rnd.random() > 0.3]          # drop random words
        return " ".join(kept) if kept else words[0]

    def pool(x):  # [B,N,C,H,W] -> [B,N,C*16]
        B, N = x.shape[:2]
        return Fn.adaptive_avg_pool2d(x.flatten(0, 1).float(), 4).reshape(B, N, -1)

    original_step = trainer_cls.train_one_step

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        if not getattr(self, "_delta_ready", False):
            inner_loss = self.model.generator_loss

            def generator_loss_delta(**kw):
                if opt is not None and any(p.grad is not None for p in proj.parameters()):
                    opt.step(); opt.zero_grad(set_to_none=True)   # grads from the previous backward
                cap["armed"], cap["x0"], cap["t"] = True, None, None
                try:
                    base, logs = inner_loss(**kw)
                finally:
                    cap["armed"] = False
                pred, t, real = cap["x0"], cap["t"], kw["clean_latent"]
                if pred is None or t is None or pred.shape != real.shape or real.shape[1] < 3:
                    return base, logs
                B, F = real.shape[:2]
                t = t.view(B, -1)[:, -F:] if t.dim() == 1 else t[:, -F:]
                ctx = torch.zeros(F, dtype=torch.bool, device=real.device); ctx[0] = True
                idx = list(range(F))
                if GOAL["on"]:
                    idx = [0] + list(range(2, F)) + [1]                # back to true time order
                    pred, real, t = pred[:, idx], real[:, idx], t[:, idx]
                    ctx[-1] = True                                       # the inserted end frame
                d_hat = pred[:, 1:].float() - pred[:, :-1].float()      # [B,F-1,C,H,W]
                d_real = real[:, 1:].float() - real[:, :-1].float()
                w = (1 - t[:, 1:].float() / 1000.0).clamp(min=0)        # noise of the later frame
                w = w * (~ctx[1:]).float()
                total = base
                if delta_w > 0 and w.sum() > 0:
                    # region-level change: 4x4-pooled latents, so per-pixel texture flicker cancels and the
                    # loss rewards changes of whole regions; plus a stride-2 term for the change over time
                    def pooled(x):
                        B_, N_ = x.shape[:2]
                        y = Fn.avg_pool2d(x.flatten(0, 1).float(), DELTA_POOL)
                        return y.reshape(B_, N_, *y.shape[1:])
                    ph, pr = pooled(pred), pooled(real)
                    per1 = ((ph[:, 1:] - ph[:, :-1]) - (pr[:, 1:] - pr[:, :-1])).square().mean(dim=(2, 3, 4))
                    dl = (per1 * w).sum() / w.sum()
                    if F > 2:
                        w2 = w[:, 1:] * (~ctx[:-2]).float()
                        if w2.sum() > 0:
                            per2 = ((ph[:, 2:] - ph[:, :-2]) - (pr[:, 2:] - pr[:, :-2])).square().mean(dim=(2, 3, 4))
                            dl = dl + 0.5 * (per2 * w2).sum() / w2.sum()
                    total = total + float(delta_w) * dl.to(base.dtype)
                    stats["delta"] = dl.item()
                S = min(slice_frames, F - 2)
                if align_w > 0 and F - S > 1:
                    term = pool(real[:, F - S:].float().mean(dim=1, keepdim=True))   # [B,1,D]
                    target = (term - pool(real[:, :F - 1].float())).detach()          # [B,F-1,D]
                    pairs = slice(0, F - S - 1)
                    pw = w[:, pairs]
                    if pw.sum() > 0:
                        cos = Fn.cosine_similarity(proj(pool(d_hat[:, pairs])), target[:, pairs], dim=-1)
                        al = ((1 - cos) * pw).sum() / pw.sum()
                        total = total + float(align_w) * al.to(base.dtype)
                        stats["align_cos"] = 1 - al.item()
                if cur["contrast"] and cur.get("wrong_emb") is not None and delta_w > 0 \
                        and w.sum() > 0 and cap.get("kw"):
                    if True:
                        kw2 = dict(cap["kw"])
                        cond = dict(kw2["conditional_dict"])
                        n = cond["prompt_embeds"].shape[0]
                        with torch.no_grad():
                            emb = cur["wrong_emb"].to(cond["prompt_embeds"].device)
                            if emb.shape[0] == 1 and n > 1:
                                emb = emb.expand(n, *emb.shape[1:])
                            if emb.shape == cond["prompt_embeds"].shape:
                                cond["prompt_embeds"] = emb.to(cond["prompt_embeds"].dtype)
                                kw2["conditional_dict"] = cond
                                if HN.get("clean_x") is not None and kw2.get("clean_x") is not None \
                                        and HN["clean_x"].shape == kw2["clean_x"].shape:
                                    kw2["clean_x"], kw2["aug_t"] = HN["clean_x"], HN["aug_t"]
                                # the fused Triton adaLN kernel (used only under no_grad) is not safe
                                # for a second forward of a dynamic-length bucket: use plain LayerNorm
                                import wan_5b.modules.causal_model as _cm
                                _prev = getattr(_cm, "_TRITON_ADALN_ENABLED", False)
                                _cm._TRITON_ADALN_ENABLED = False
                                try:
                                    _, x0w = self.model.generator(**kw2)
                                finally:
                                    _cm._TRITON_ADALN_ENABLED = _prev
                            else:
                                x0w = None
                        if x0w is not None:
                            if GOAL["on"]:
                                x0w = x0w[:, idx]
                            d_wrong = (x0w[:, 1:].float() - x0w[:, :-1].float()).detach()
                            per_w = (d_wrong - d_real).square().mean(dim=(2, 3, 4))
                            per_t = (d_hat.detach() - d_real).square().mean(dim=(2, 3, 4))
                            stats["prompt_gap"] = ((per_w - per_t) * w).sum().item() / w.sum().item()
                            sep = (d_hat - d_wrong).square().mean(dim=(2, 3, 4)).clamp(max=contrast_cap)
                            push = (sep * w).sum() / w.sum()
                            total = total - float(contrast_w) * push.to(base.dtype)
                            stats["contrast_sep"] = push.item()
                return total, logs

            self.model.generator_loss = generator_loss_delta
            self._delta_ready = True
        stats.clear()
        cap["kw"] = None
        prompts = batch.get("prompts")
        if prompts:
            full = prompts[0][0] if isinstance(prompts[0], (list, tuple)) else prompts[0]
            cur["prompt"] = full
            cur["contrast"] = contrast_prob > 0 and _rnd.random() < contrast_prob
            cur["wrong_emb"] = None
            others = [q for q in bank if q != full]
            if cur["contrast"] and others:
                # encode the wrong prompt now, before the step holds activations (T5 is streamed)
                with torch.no_grad():
                    cur["wrong_emb"] = self.model.text_encoder(
                        text_prompts=[_rnd.choice(others)])["prompt_embeds"].detach()
            if prompt_drop > 0 and _rnd.random() < prompt_drop:
                batch["prompts"] = [[shorten(q) for q in sub] if isinstance(sub, (list, tuple))
                                    else shorten(sub) for sub in prompts]
                short = batch["prompts"][0][0] if isinstance(batch["prompts"][0], (list, tuple)) \
                    else batch["prompts"][0]
                stats["prompt_words"] = float(len(short.split()))
            bank.append(full)
            del bank[:-64]
        out = original_step(self, batch, accumulation_step=accumulation_step,
                            accumulation_steps=accumulation_steps)
        if stats:
            msg = " ".join(f"{k}={v:.5f}" for k, v in stats.items())
            print(f"[delta] step {self.step:07d} {msg}", flush=True)
        if proj is not None and proj_path and self.step % save_every == 0:
            torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "step": self.step}, proj_path)
        return out

    trainer_cls.train_one_step = train_one_step
    print(f"[delta] on: delta_weight={delta_w} align_weight={align_w} terminal_slice={slice_frames}",
          flush=True)


def install_goal_insert(trainer_cls, prob):
    """Random end-frame conditioning (training only).

    With probability `prob` a step reorders the clip to [f0, f_last, f1..f_{last-1}], keeps every
    frame at its real time via RoPE offsets, and passes [f0, noisy f_last] as the two i2v context
    frames: both are excluded from the loss (so the last frame is never a target that could be
    copied), the end frame carries a uniform-random noise level that the model is told through its
    timestep, and the block-causal mask makes it visible to every later block. Otherwise the step
    is ordinary training, which is also what inference does.
    """
    import random
    import torch
    import utils.wan_5b_wrapper as wrapper_mod
    Wrapper = wrapper_mod.WanDiffusionWrapper
    inner_forward = Wrapper.forward

    def forward_goal(self, *args, **kwargs):
        if GOAL["on"]:
            t_goal = GOAL["sigma"] * 1000.0
            for key in ("timestep", "aug_t"):
                t = kwargs.get(key)
                if torch.is_tensor(t) and t.dim() == 2 and t.shape[1] > 1:
                    t = t.clone(); t[:, 1] = t_goal; kwargs[key] = t
        return inner_forward(self, *args, **kwargs)

    Wrapper.forward = forward_goal
    original_step = trainer_cls.train_one_step

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        if not getattr(self, "_goal_insert_ready", False):
            helper = self.sp_helper
            original_encode = helper.encode_raw_video_latents
            trainer = self

            def encode_goal(batch, *, batch_size):
                clean, image, sharded = original_encode(batch, batch_size=batch_size)
                if GOAL["on"] and clean.shape[1] > 2:
                    order, offset = goal_first_order(clean.shape[1])
                    clean = clean[:, order].contiguous()
                    goal = clean[:, 1:2]
                    sigma = GOAL["sigma"]
                    noisy_goal = (1 - sigma) * goal + sigma * torch.randn_like(goal)
                    image = torch.cat([clean[:, 0:1], noisy_goal], dim=1)
                    trainer.model.generator.module.model.rope_temporal_offset = offset.to(clean.device)
                else:
                    GOAL["on"] = False
                return clean, image, sharded

            helper.encode_raw_video_latents = encode_goal
            self._goal_insert_ready = True
        GOAL["on"] = random.random() < prob
        GOAL["sigma"] = random.random()
        inserted, sigma = GOAL["on"], GOAL["sigma"]
        try:
            return original_step(self, batch, accumulation_step=accumulation_step,
                                 accumulation_steps=accumulation_steps)
        finally:
            if GOAL["on"]:
                print(f"[goal-insert] step {self.step:07d} sigma={sigma:.3f}", flush=True)
            GOAL["on"] = False
            self.model.generator.module.model.rope_temporal_offset = 0.0

    trainer_cls.train_one_step = train_one_step
    print(f"[goal-insert] on: prob={prob} noise=uniform(0,1) end frame at true end time, "
          f"context-only (no loss)", flush=True)


def install_process_rollout(trainer_cls, rollout_steps=4, action_loss_weight=0.0,
                            rollout_frames=72, gan_weight=0.0, gan_warmup=20, gan_path=None,
                            save_every=50, gan_source="pred"):
    """Scheduled process supervision without any goal-frame token or target.

    On the canonical M/F72 bucket, block 0 is rolled out from noise using only the real
    reference frame and text.  Its generated process frames replace the clean teacher-forced
    history for the supervised pass, whose chronological real-video targets stay unchanged.
    Longer dynamic buckets keep the standard path because this Triton build cannot safely
    switch repeated-forward shapes inside one process.
    """
    import torch
    import utils.wan_5b_wrapper as wrapper_mod
    original_step = trainer_cls.train_one_step
    Wrapper = wrapper_mod.WanDiffusionWrapper
    original_forward = Wrapper.forward
    state = {"active": False, "rollout": False, "gap": None, "frames": 0,
             "x0_pred": None, "action_loss": None, "t": None, "gan": None, "gen": None}

    gan = None
    if gan_weight > 0:
        import os
        import torch.nn as nn
        sn = nn.utils.spectral_norm

        class LatentVideoDisc(nn.Module):
            """Patch discriminator over Wan latents [B,F,48,H,W] + the per-frame timestep."""

            def __init__(self, ch=128):
                super().__init__()
                act = lambda: nn.LeakyReLU(0.2)
                self.net = nn.Sequential(
                    sn(nn.Conv3d(49, ch, 3, padding=1)), act(),
                    sn(nn.Conv3d(ch, 2 * ch, (3, 4, 4), stride=(1, 2, 2), padding=1)), act(),
                    sn(nn.Conv3d(2 * ch, 2 * ch, (3, 4, 4), stride=(2, 2, 2), padding=1)), act(),
                    sn(nn.Conv3d(2 * ch, 4 * ch, (3, 4, 4), stride=(2, 2, 2), padding=1)), act(),
                    sn(nn.Conv3d(4 * ch, 1, 3, padding=1)))

            def forward(self, x, t):
                x = x.float().permute(0, 2, 1, 3, 4)                      # [B,C,F,H,W]
                tc = (t.float() / 1000.0)[:, None, :, None, None].expand(
                    -1, 1, -1, x.shape[-2], x.shape[-1])
                return self.net(torch.cat([x, tc], 1))

        disc = LatentVideoDisc().cuda()
        opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.0, 0.99))
        gan = {"disc": disc, "opt": opt, "updates": 0}
        if gan_path and os.path.exists(gan_path):
            ck = torch.load(gan_path, map_location="cuda", weights_only=False)
            disc.load_state_dict(ck["disc"]); opt.load_state_dict(ck["opt"])
            gan["updates"] = int(ck.get("updates", 0))
            print(f"[gan] restored discriminator ({gan['updates']} updates) from {gan_path}",
                  flush=True)
        print(f"[gan] latent discriminator params="
              f"{sum(p.numel() for p in disc.parameters())/1e6:.1f}M weight={gan_weight} "
              f"warmup={gan_warmup}", flush=True)

    def forward_process_rollout(self, *args, clean_x=None, aug_t=None, **kwargs):
        timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
        noisy = kwargs.get("noisy_image_or_video", args[0] if args else None)
        conditional = kwargs.get("conditional_dict", args[1] if len(args) > 1 else None)
        do_rollout = (state["active"] and clean_x is not None and noisy is not None
                      and conditional is not None and clean_x.shape[1] == rollout_frames
                      and not GOAL["on"])
        state["rollout"] = do_rollout
        if state["active"]:
            state["t"] = timestep
        state["frames"] = int(clean_x.shape[1]) if clean_x is not None else 0
        if do_rollout:
            block = min(int(getattr(self.model, "num_frame_per_block", 8)), clean_x.shape[1])
            steps = max(1, int(rollout_steps))
            with torch.no_grad():
                rollout = torch.randn_like(noisy)
                rollout[:, 0] = clean_x[:, 0]
                base_noise = rollout.clone()
                times = torch.linspace(999.0, 1.0, steps, device=noisy.device,
                                       dtype=timestep.dtype)
                # GAN on the rollout itself: the last content-forming step (t~333 of 999/666/333/1)
                # keeps its graph so the discriminator can push the generated block; the near-
                # identity t=1 refinement is skipped.
                grad_i = (steps - 2 if (gan is not None and gan_source == "rollout"
                                        and steps >= 2) else -1)
                for i, t in enumerate(times):
                    rt = torch.full(timestep.shape, t, device=timestep.device,
                                    dtype=timestep.dtype)
                    rt[:, 0] = 0
                    if i == grad_i:
                        with torch.enable_grad():
                            _, x0g = original_forward(
                                self, noisy_image_or_video=rollout,
                                conditional_dict=conditional, timestep=rt,
                                clean_x=clean_x, aug_t=torch.zeros_like(rt))
                            # own storage: the history copy below is edited in place
                            state["gen"] = x0g[:, 1:block].clone()
                        x0 = x0g.detach().clone()
                        break
                    _, x0 = original_forward(
                        self, noisy_image_or_video=rollout,
                        conditional_dict=conditional, timestep=rt,
                        clean_x=clean_x, aug_t=torch.zeros_like(rt))
                    if i + 1 < steps:
                        sigma = (times[i + 1].float() / 1000.0).clamp(0, 1).to(x0.dtype)
                        rollout[:, :block] = ((1 - sigma) * x0[:, :block]
                                              + sigma * base_noise[:, :block])
                        rollout[:, 0] = clean_x[:, 0]
                generated = x0[:, :block].detach()
                generated[:, 0] = clean_x[:, 0]
                state["gap"] = (generated[:, 1:block].float()
                                - clean_x[:, 1:block].float()).square().mean().item()
            clean_x = clean_x.clone()
            clean_x[:, 1:block] = generated[:, 1:block]
            if aug_t is None:
                aug_t = torch.zeros_like(timestep)
            else:
                aug_t = aug_t.clone()
            aug_t[:, 1:block] = 0
        result = original_forward(self, *args, clean_x=clean_x, aug_t=aug_t, **kwargs)
        if state["active"]:
            state["x0_pred"] = result[1]
        return result

    Wrapper.forward = forward_process_rollout

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        if gan is not None and not getattr(self, "_gan_ready", False):
            inner_loss = self.model.generator_loss

            def generator_loss_with_gan(**kwargs):
                state["x0_pred"] = None
                base_loss, logs = inner_loss(**kwargs)
                pred, real, t = state.get("x0_pred"), kwargs["clean_latent"], state.get("t")
                if state["rollout"] and not gan.get("shapes_logged"):
                    gan["shapes_logged"] = True
                    print(f"[gan] shapes pred={None if pred is None else tuple(pred.shape)} "
                          f"real={tuple(real.shape)} t={None if t is None else tuple(t.shape)}",
                          flush=True)
                if gan_source == "rollout":
                    gen = state.get("gen")
                    if not state["rollout"] or gen is None:
                        return base_loss, logs
                    fake, real_c = gen, real[:, 1:1 + gen.shape[1]].detach()
                    tt = torch.zeros(fake.shape[:2], device=fake.device)
                    return gan_step(fake, real_c, tt, base_loss, logs)
                if not state["rollout"] or pred is None or t is None:
                    return base_loss, logs
                B, F = real.shape[0], real.shape[1]
                if pred.shape[1] != F and pred.shape[1] % F == 0:
                    pred = pred[:, -F:]                       # keep the noisy (supervised) stream
                if t.dim() == 1 and t.numel() % B == 0:
                    t = t.view(B, -1)
                if t.shape[1] != F and t.shape[1] % F == 0:
                    t = t[:, -F:]
                if pred.shape != real.shape or t.shape != (B, F) or F <= 8:
                    if not gan.get("skip_logged"):
                        gan["skip_logged"] = True
                        print(f"[gan] skip: pred={tuple(pred.shape)} real={tuple(real.shape)} "
                              f"t={tuple(t.shape)}", flush=True)
                    return base_loss, logs
                block = 8
                fake, real_c, tt = pred[:, block:], real[:, block:].detach(), t[:, block:].detach()
                return gan_step(fake, real_c, tt, base_loss, logs)

            def gan_step(fake, real_c, tt, base_loss, logs):
                disc, opt = gan["disc"], gan["opt"]
                # discriminator update: real video latents vs predictions from self history
                with torch.enable_grad():
                    d_real = disc(real_c, tt); d_fake = disc(fake.detach(), tt)
                    d_loss = torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean()
                    opt.zero_grad(set_to_none=True); d_loss.backward(); opt.step()
                gan["updates"] += 1
                rec = {"d_loss": d_loss.item(), "d_real": d_real.mean().item(),
                       "d_fake": d_fake.mean().item(), "g": None}
                if gan["updates"] > gan_warmup:
                    for p_ in disc.parameters():
                        p_.requires_grad_(False)
                    g_loss = -disc(fake, tt).mean()
                    for p_ in disc.parameters():
                        p_.requires_grad_(True)
                    rec["g"] = g_loss.item()
                    base_loss = base_loss + float(gan_weight) * g_loss.to(base_loss.dtype)
                state["gan"] = rec
                return base_loss, logs

            self.model.generator_loss = generator_loss_with_gan
            self._gan_ready = True
        if action_loss_weight > 0 and not getattr(self, "_action_imitation_ready", False):
            original_generator_loss = self.model.generator_loss

            def generator_loss_with_action(**kwargs):
                state["x0_pred"] = None
                base_loss, logs = original_generator_loss(**kwargs)
                pred = state.get("x0_pred")
                real = kwargs["clean_latent"]
                if pred is None or pred.shape != real.shape:
                    return base_loss, logs
                # Behavior cloning in latent motion space: match the direction and magnitude
                # of every chronological transition, not merely each frame independently.
                pred_delta = pred[:, 1:].float() - pred[:, :-1].float()
                real_delta = real[:, 1:].float() - real[:, :-1].float()
                per_frame = (pred_delta - real_delta).square().mean(dim=(2, 3, 4))
                mask = kwargs.get("loss_mask")
                if mask is not None:
                    pair_mask = (mask[:, 1:] * mask[:, :-1]).to(per_frame.dtype)
                    action_loss = (per_frame * pair_mask).sum() / pair_mask.sum().clamp(min=1)
                else:
                    action_loss = per_frame.mean()
                state["action_loss"] = action_loss.detach().item()
                logs["action_imitation_loss"] = action_loss.detach()
                return base_loss + float(action_loss_weight) * action_loss, logs

            self.model.generator_loss = generator_loss_with_action
            self._action_imitation_ready = True
        state.update(active=True, rollout=False, gap=None, frames=0,
                     x0_pred=None, action_loss=None, t=None, gan=None, gen=None)
        try:
            return original_step(self, batch, accumulation_step=accumulation_step,
                                 accumulation_steps=accumulation_steps)
        finally:
            state["active"] = False
            if state["rollout"] and state["gap"] is not None:
                print(f"[process-rollout] step {self.step:07d} blocks=1 steps={rollout_steps} "
                      f"gap_mse={state['gap']:.6f} frames={state['frames']}", flush=True)
            if state["gan"] is not None:
                r = state["gan"]
                g = "off" if r["g"] is None else f"{r['g']:.4f}"
                print(f"[gan] step {self.step:07d} d_loss={r['d_loss']:.4f} "
                      f"d_real={r['d_real']:.3f} d_fake={r['d_fake']:.3f} g_loss={g} "
                      f"updates={gan['updates']}", flush=True)
            if gan is not None and gan_path and self.step % save_every == 0:
                torch.save({"disc": gan["disc"].state_dict(), "opt": gan["opt"].state_dict(),
                            "updates": gan["updates"], "step": self.step}, gan_path)
            if state["action_loss"] is not None:
                print(f"[action-imitation] step {self.step:07d} "
                      f"loss={state['action_loss']:.6f} weight={action_loss_weight:.3f}", flush=True)

    trainer_cls.train_one_step = train_one_step


def install_motion_weight(trainer_cls, k=2.0, cap=4.0):
    """Moving regions (the robot) carry little of the frame-mean MSE next to the static
    background; weight each latent position by how much the real video changes there."""
    import torch
    import torch.nn.functional as F
    ctx = {"real": None, "stats": None}
    plain_mse = F.mse_loss

    def weight_map(real):
        r = real.float()
        if GOAL["on"] and r.shape[1] > 2:   # slot 1 is the end frame: f1's predecessor is f0
            r = r.clone(); r[:, 1] = r[:, 0]
        m = (r[:, 1:] - r[:, :-1]).abs().mean(dim=2, keepdim=True)        # [B,F-1,1,H,W]
        m = torch.cat([m[:, :1], m], dim=1)                                # frame 0 takes frame 1's
        rel = m / m.mean(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-6)
        w = 1 + k * rel.clamp(max=cap)
        w = w / w.mean(dim=(1, 2, 3, 4), keepdim=True)
        return w

    def mse_motion(input, target, *args, reduction="mean", **kwargs):
        real = ctx["real"]
        if (reduction == "none" and real is not None and input.dim() == 5
                and input.shape == real.shape and target.shape == real.shape):
            w = weight_map(real.detach())
            ctx["stats"] = (w.max().item(), w.amin().item())
            return plain_mse(input, target, *args, reduction="none", **kwargs) * w.to(input.dtype)
        return plain_mse(input, target, *args, reduction=reduction, **kwargs)

    F.mse_loss = mse_motion
    original_step = trainer_cls.train_one_step

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        if not getattr(self, "_motion_ready", False):
            inner = self.model.generator_loss

            def generator_loss_motion(**kw):
                ctx["real"] = kw.get("clean_latent")
                try:
                    return inner(**kw)
                finally:
                    ctx["real"] = None

            self.model.generator_loss = generator_loss_motion
            self._motion_ready = True
        ctx["stats"] = None
        out = original_step(self, batch, accumulation_step=accumulation_step,
                            accumulation_steps=accumulation_steps)
        if ctx["stats"] is not None and self.step % 10 == 0:
            print(f"[motion-weight] step {self.step:07d} k={k} w_max={ctx['stats'][0]:.2f} "
                  f"w_min={ctx['stats'][1]:.2f}", flush=True)
        return out

    trainer_cls.train_one_step = train_one_step
    print(f"[motion-weight] on: k={k} cap={cap}", flush=True)


def install_efficiency(trainer_cls, model_cls, lr_cfg):
    """No-padding training, warmup+cosine LR, logit-normal timesteps."""
    import math
    import torch

    original_step = trainer_cls.train_one_step
    mask_cache = {}  # CausalWanModel caches one flex-attention block mask keyed only by batch size

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        shape = self.config.image_or_video_shape
        carrier = shape[1]
        wan = self.model.generator.module.model
        # The stage carrier pads every clip to F64 for 128-way sequence parallelism.
        # On one GPU, encode and train only the valid prefix (buckets are multiples of 8).
        if "num_valid_latent_frames" in batch:
            valid_latent = int(torch.as_tensor(batch["num_valid_latent_frames"]).max())
            valid_raw = int(torch.as_tensor(batch["num_valid_raw_frames"]).max())
            batch["frames"] = batch["frames"][:, :, :valid_raw]
            shape[1] = valid_latent
        step = self.step
        if step < lr_cfg["warmup"]:
            scale = (step + 1) / lr_cfg["warmup"]
        else:
            progress = min(1.0, (step - lr_cfg["warmup"]) / max(1, lr_cfg["total"] - lr_cfg["warmup"]))
            scale = lr_cfg["min_ratio"] + (1 - lr_cfg["min_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in self.generator_optimizer.param_groups:
            group["lr"] = float(self.config.lr) * scale
        cache_key = (shape[1], len(batch["prompts"]))
        wan.block_mask = mask_cache.get(cache_key)
        if wan.block_mask is None:
            wan._block_mask_batch_size = None  # force a rebuild for this length
        try:
            return original_step(self, batch, accumulation_step=accumulation_step,
                                 accumulation_steps=accumulation_steps)
        finally:
            mask_cache[cache_key] = wan.block_mask
            shape[1] = carrier

    trainer_cls.train_one_step = train_one_step

    original_t = model_cls._get_timestep

    def logit_normal_timestep(self, min_timestep, max_timestep, *args, **kwargs):
        # Monotone map of the original uniform draw through the logit-normal inverse CDF:
        # mid noise levels get more samples, per-block equality is preserved.
        t = original_t(self, min_timestep, max_timestep, *args, **kwargs)
        span = max_timestep - min_timestep
        u = ((t - min_timestep).float() + 0.5) / span
        z = math.sqrt(2.0) * torch.special.erfinv(2 * u - 1)
        return (min_timestep + torch.sigmoid(z) * span).long().clamp_(min_timestep, max_timestep - 1)

    model_cls._get_timestep = logit_normal_timestep


def lora_param_names(module):
    return {n for n, _ in module.named_parameters() if "lora_" in n}


def _bf16_linear(lin):
    """fp32 master weights, matmul in the activation dtype (bf16 under FSDP mixed precision)."""
    import torch.nn.functional as F

    def forward(x):
        b = lin.bias.to(x.dtype) if lin.bias is not None else None
        return F.linear(x, lin.weight.to(x.dtype), b)
    return forward


def install_throughput(ckpt_every, workers, no_ckpt_blocks=0, offload_ckpt=False):
    """Big-card throughput knobs that leave the checkpointed config (and resume contract) alone."""
    import functools
    import torch
    import torch.utils.checkpoint as cp
    import utils.wan_5b_wrapper as wrapper_mod

    original_enable = wrapper_mod.WanDiffusionWrapper.enable_gradient_checkpointing

    def offloaded_checkpoint(f, *a, **k):
        # Only the checkpoint's saved inputs see this hook; its backward recompute installs its own.
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return cp.checkpoint(f, *a, use_reentrant=False, **k)

    def enable_selective(self):
        original_enable(self)
        if offload_ckpt:
            model = self.model
            model.gradient_checkpointing = False
            for block in model.blocks:
                block.forward = functools.partial(offloaded_checkpoint, block.forward)
            print(f"[throughput] activation checkpointing on {len(model.blocks)}/{len(model.blocks)} blocks, "
                  f"saved inputs offloaded to pinned CPU memory", flush=True)
            return
        if ckpt_every == 1 and no_ckpt_blocks == 0:
            return
        model = self.model
        model.gradient_checkpointing = False  # the model's own all-blocks checkpointing
        if no_ckpt_blocks > 0:
            for i, block in enumerate(model.blocks):
                if i >= no_ckpt_blocks:
                    fwd = block.forward
                    block.forward = functools.partial(
                        lambda f, *a, **k: cp.checkpoint(f, *a, use_reentrant=False, **k), fwd)
            print(f"[throughput] activation checkpointing on {len(model.blocks) - no_ckpt_blocks}/"
                  f"{len(model.blocks)} blocks (first {no_ckpt_blocks} kept in memory)", flush=True)
            return
        if ckpt_every > 1:
            for i, block in enumerate(model.blocks):
                if i % ckpt_every == 0:
                    fwd = block.forward
                    block.forward = functools.partial(
                        lambda f, *a, **k: cp.checkpoint(f, *a, use_reentrant=False, **k), fwd)
        n = 0 if ckpt_every == 0 else (len(model.blocks) + ckpt_every - 1) // ckpt_every
        print(f"[throughput] activation checkpointing on {n}/{len(model.blocks)} blocks", flush=True)

    wrapper_mod.WanDiffusionWrapper.enable_gradient_checkpointing = enable_selective

    if workers != 2:
        Base = torch.utils.data.DataLoader

        class TrainLoader(Base):
            def __init__(self, *a, **k):
                if k.get("num_workers") == 2:  # the training loader; eval uses 0
                    k["num_workers"] = workers
                    k.setdefault("prefetch_factor", 4)
                    k.setdefault("persistent_workers", True)
                super().__init__(*a, **k)

        torch.utils.data.DataLoader = TrainLoader


def install_lora(rank, alpha, base_ckpt, t5_on_gpu=False,
                 bidirectional_attention=False, fp8=False, ffn_chunk=0):
    """Patch generator construction, strict loading and FSDP wrapping for LoRA."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    # peft would upcast 56k-token activations to the fp32 adapter dtype (6 GB on ffn.0)
    BaseTunerLayer._cast_input_dtype = lambda self, x, dtype: x
    import utils.wan_5b_wrapper as wrapper_mod
    import trainer.diffusion as td

    Wrapper = wrapper_mod.WanDiffusionWrapper
    original_init = Wrapper.__init__

    def init_with_lora(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not kwargs.get("is_causal", False) and not bidirectional_attention:
            return
        lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS)
        self.model = inject_adapter_in_model(lcfg, self.model)
        for n, p in self.model.named_parameters():
            p.requires_grad_("lora_" in n)
        if ffn_chunk > 0:
            import torch.utils.checkpoint as _cp
            n_ffn = 0
            for blk in getattr(self.model, "blocks", []):
                ffn = getattr(blk, "ffn", None)
                if ffn is None:
                    continue
                inner = ffn.forward

                def chunked(x, _inner=inner):
                    L = x.shape[1]
                    if L <= ffn_chunk or not torch.is_grad_enabled():
                        return _inner(x)
                    return torch.cat([_cp.checkpoint(_inner, x[:, i:i + ffn_chunk], use_reentrant=False)
                                      for i in range(0, L, ffn_chunk)], dim=1)

                ffn.forward = chunked
                n_ffn += 1
            print(f"[ffn-chunk] {n_ffn} FFNs chunked at {ffn_chunk} tokens", flush=True)
        if fp8:
            sys.path.insert(0, "/data")
            from fp8_lora import install_fp8
            print(f"[fp8] frozen base Linear layers on FP8: {install_fp8(self.model)}", flush=True)

    Wrapper.__init__ = init_with_lora

    # Runs resumed from a manifest-order checkpoint may switch to shuffled order: allow only the
    # sampler's shuffle flag to differ from the stored resume contract; everything else stays strict.
    original_contract = td._validate_resume_runtime_contract

    def contract_allow_shuffle(contract, **kw):
        if os.environ.get("LORA_ALLOW_DATA_CHANGE") == "1":
            changed = [k for k in ("dataset", "sampler", "data_cursor")
                       if contract.get(k) != dict(kw.get(k) or {})]
            if changed:
                print(f"[resume] data change accepted for {changed}", flush=True)
                contract = dict(contract)
                for k in changed:
                    contract[k] = dict(kw[k])
                return original_contract(contract, **kw)
        stored = dict(contract.get("sampler") or {})
        runtime = dict(kw.get("sampler") or {})
        if stored.get("shuffle") != runtime.get("shuffle") and \
                {k: v for k, v in stored.items() if k != "shuffle"} == \
                {k: v for k, v in runtime.items() if k != "shuffle"}:
            print(f"[resume] sampler shuffle {stored.get('shuffle')} -> {runtime.get('shuffle')}", flush=True)
            contract = dict(contract); contract["sampler"] = runtime
        return original_contract(contract, **kw)

    td._validate_resume_runtime_contract = contract_allow_shuffle
    original_strict = td._strict_load_generator_state_dict
    target_re = re.compile(LORA_TARGETS)

    def strict_with_lora(generator, state_dict):
        has_lora = any("lora_" in k for k in state_dict)
        if has_lora and all("lora_" in k for k in state_dict):
            # LoRA-only checkpoint (see lora_only_state_dict): base comes from the frozen
            # astronex weights, adapters from the checkpoint.
            base = torch.load(base_ckpt, map_location="cpu", weights_only=False, mmap=True)
            base = base["generator"] if "generator" in base else base
            merged = remap_base(base, generator)
            merged.update(state_dict)
            return original_strict(generator, merged)
        if has_lora or not lora_param_names(generator):
            return original_strict(generator, state_dict)
        return original_strict(generator, remap_base(state_dict, generator))

    def remap_base(state_dict, generator):
        # Plain (pre-LoRA) checkpoint: route base weights to peft's base_layer and
        # keep the freshly initialised adapters so the strict key check passes.
        prefix = "model." if any(k.startswith("model.") for k in state_dict) else ""
        remapped = {}
        for k, v in state_dict.items():
            parts = k.rsplit(".", 1)
            mod = parts[0][len(prefix):] if prefix else parts[0]
            if len(parts) == 2 and target_re.fullmatch(mod):
                remapped[f"{parts[0]}.base_layer.{parts[1]}"] = v
            else:
                remapped[k] = v
        level = generator if prefix else generator.model
        for k, v in level.state_dict().items():
            if "lora_" in k:
                remapped[k] = v
        return remapped

    td._strict_load_generator_state_dict = strict_with_lora

    original_wrap = td.fsdp_wrap
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from functools import partial

    def wrap_with_lora(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", **kw):
        device = torch.cuda.current_device()
        if isinstance(module, Wrapper) and lora_param_names(module):
            lora_mods = [m for n, m in module.named_modules() if n.endswith(("lora_A", "lora_B"))]
            lora_ids = {id(p) for m in lora_mods for p in m.parameters()}
            # model/diffusion.py re-enables grads on the whole generator after
            # construction, so the freeze has to happen here, right before wrapping.
            for p in module.parameters():
                p.requires_grad_(id(p) in lora_ids)
                p.data = p.data.float() if id(p) in lora_ids else p.data.to(torch.bfloat16)
            for m in lora_mods:
                m.to(device)
                for lin in m.modules():
                    if isinstance(lin, torch.nn.Linear):
                        lin.forward = _bf16_linear(lin)
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"[LoRA] rank={rank} alpha={alpha} trainable={trainable / 1e6:.1f}M "
                  f"frozen base bf16={sum(p.numel() for p in module.parameters() if not p.requires_grad) / 1e9:.2f}B")
            # One GPU: FSDP degrades to NO_SHARD, whose use_orig_params writeback breaks on
            # frozen params from step 2 ("Cannot writeback when the parameter shape changes").
            # Keep FSDP only as the interface the trainer expects; it manages no parameters.
            module.to(device)
            wrapped = FSDP(
                module,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                device_id=device, use_orig_params=True,
                ignored_states=list(module.parameters()),
                sync_module_states=False,
            )
            # Checkpoints carry only the adapters (~0.65 GB, not 12 GB): the base is frozen
            # and identical to --generator, which strict_with_lora reloads on resume.
            full_state_dict = wrapped.state_dict

            def lora_only_state_dict(*a, **k):
                return {key: v for key, v in full_state_dict(*a, **k).items() if "lora_" in key}

            wrapped.state_dict = lora_only_state_dict
            wrapped._lora_unmanaged = True
            if torch.distributed.get_world_size() > 1:
                from adapter_sync import sync_initial_parameters, average_gradients

                params = [p for p in module.parameters() if p.requires_grad]
                sync_initial_parameters(params)

                def clip_distributed_adapters(max_norm):
                    average_gradients(params)
                    return torch.nn.utils.clip_grad_norm_(params, max_norm)

                wrapped.clip_grad_norm_ = clip_distributed_adapters
            return wrapped
        # T5 (frozen, ~11 GB bf16) is only used to encode prompts at the start of a step.
        # Keep it in pinned CPU memory and stream it onto the GPU just for that call;
        # afterwards the GPU copy is dropped, leaving backward the extra headroom.
        if type(module).__name__ == "WanTextEncoder" and not t5_on_gpu:
            module.to(torch.bfloat16)
            params = list(module.parameters()) + list(module.buffers())
            # WanTextEncoder moves itself to CUDA in __init__; bring it back to pinned host memory
            cpu_copies = [t.data.cpu().pin_memory() for t in params]
            for t, c in zip(params, cpu_copies):
                t.data = c
            torch.cuda.empty_cache()
            original_forward = module.forward

            def streamed_forward(*args, **kwargs):
                for t, c in zip(params, cpu_copies):
                    t.data = c.to(device, non_blocking=True)
                try:
                    with torch.no_grad():
                        out = original_forward(*args, **kwargs)
                finally:
                    for t, c in zip(params, cpu_copies):
                        t.data = c
                return out

            module.forward = streamed_forward
            return module
        if not any(p.requires_grad for p in module.parameters()):
            module.to(torch.bfloat16)
        return original_wrap(module, sharding_strategy=sharding_strategy, mixed_precision=mixed_precision,
                             wrap_strategy=wrap_strategy, **kw)

    td.fsdp_wrap = wrap_with_lora

    # FSDP manages no generator params, so its optimizer-state flattening has nothing to
    # anchor on (StopIteration on resume). The saved state is keyed by parameter name;
    # map it back to this optimizer's integer ids directly.
    original_to_load = FSDP.optim_state_dict_to_load

    def optim_state_dict_to_load(model, optim, optim_state_dict, *a, **k):
        if not getattr(model, "_lora_unmanaged", False):
            return original_to_load(model, optim, optim_state_dict, *a, **k)
        clean = lambda n: n.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        name_of = {id(p): clean(n) for n, p in model.named_parameters()}
        state, groups, idx = {}, [], 0
        for gi, group in enumerate(optim.param_groups):
            ids = []
            for p in group["params"]:
                name = name_of[id(p)]
                if name in optim_state_dict["state"]:
                    state[idx] = optim_state_dict["state"][name]
                ids.append(idx)
                idx += 1
            saved = optim_state_dict["param_groups"][gi]
            groups.append({**{k2: v for k2, v in saved.items() if k2 != "params"}, "params": ids})
        print(f"[LoRA] restored optimizer state for {len(state)}/{idx} params")
        return {"state": state, "param_groups": groups}

    FSDP.optim_state_dict_to_load = staticmethod(optim_state_dict_to_load)


def install_bidirectional_training():
    """Use the repository's native WanModel full attention in the S trainer.

    The diffusion loss class historically hard-codes the causal wrapper even
    when the config requests a bidirectional generator.  Patch only that
    constructor and timestep sampling; all loss/data/error-recycling code is
    left in place.
    """
    import torch
    import model.diffusion as diffusion_mod
    from model.base import BaseModel
    from utils.wan_5b_wrapper import (
        WanDiffusionWrapper, WanTextEncoder, build_vae_5b)

    def initialize_models_bidirectional(self, args, device):
        model_name = getattr(args.model_kwargs, "model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported, got {model_name}")
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        vae_type = model_kwargs.pop("vae_type", "wan")
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=False)
        self.generator.model.requires_grad_(True)
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)
        self.vae = build_vae_5b(type("VAEArgs", (), {"vae_type": vae_type})())
        self.vae.requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        print(f"[bidirectional] native WanModel full attention; no causal mask/KV/sink; vae={vae_type}",
              flush=True)

    diffusion_mod.CausalDiffusion._initialize_models = initialize_models_bidirectional

    original_get_timestep = BaseModel._get_timestep

    def uniform_clip_timestep(self, min_timestep, max_timestep, batch_size,
                              num_frame, num_frame_per_block,
                              uniform_timestep=False):
        return original_get_timestep(
            self, min_timestep, max_timestep, batch_size, num_frame,
            num_frame_per_block, uniform_timestep=True)

    diffusion_mod.CausalDiffusion._get_timestep = uniform_clip_timestep


def main():
    # 7 M length buckets x (training forward, no-grad contrast forward) exceed dynamo's default of
    # 8 graphs per frame; past the limit flex_attention runs eagerly as dense math (~140 GB) -> OOM.
    import torch._dynamo
    for _k, _v in (("recompile_limit", 64), ("cache_size_limit", 64), ("accumulated_recompile_limit", 1024),
                   ("accumulated_cache_size_limit", 1024)):
        if hasattr(torch._dynamo.config, _k):
            setattr(torch._dynamo.config, _k, _v)
    a = arguments()
    a.assets_root = a.assets_root.expanduser().resolve()
    a.generator = a.generator.expanduser().resolve()
    a.data_root = a.data_root.expanduser().resolve()
    os.environ.setdefault("DYNAMICS_FORCING_VIDEO_INDEX_CACHE", str(a.output.resolve() / "index-cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(a.output.resolve() / "torch-cache"))
    cfg = build_config(a)
    from omegaconf import OmegaConf
    stop = a.stop_step or int(cfg.training.max_iters)
    if not 1 <= stop <= int(cfg.training.max_iters):
        raise ValueError("--stop-step must lie within the configured optimizer update count.")
    if a.nvlink:
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
    if a.plan:
        print(OmegaConf.to_yaml(cfg))
        return
    if a.world_size > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        import torch

        if torch.cuda.device_count() < a.world_size:
            raise RuntimeError(f"Requested {a.world_size} GPUs, but only {torch.cuda.device_count()} are visible.")
        launcher = Path(__file__).resolve().parents[1] / "df.py"
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc-per-node", str(a.world_size), str(launcher), "--train",
                        *sys.argv[1:]], check=True)
        return
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    config_path = a.output / f"train_{a.stage.lower()}_lora.yaml"
    staged = config_path.with_suffix(f".{os.getpid()}.tmp")
    OmegaConf.save(cfg, staged)
    os.replace(staged, config_path)
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                 "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511"}.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Inductor's default cache lives in /tmp, which Ubuntu wipes at boot; every reboot
    # then re-autotunes flex attention for all six S lengths (~30-40 min).
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/torch_cache")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(CORE))
    sys.stdout.reconfigure(line_buffering=True)  # real-time log lines
    install_throughput(a.ckpt_every, a.workers, a.no_ckpt_blocks, a.offload_ckpt)
    install_lora(a.rank, a.alpha, str(a.generator), t5_on_gpu=a.t5_on_gpu,
                 bidirectional_attention=a.bidirectional_attention, fp8=a.fp8,
                 ffn_chunk=a.ffn_chunk)
    if a.bidirectional_attention:
        install_bidirectional_training()
    import train
    from trainer import DiffusionTrainer
    from model.base import BaseModel
    if a.goal_first:
        if a.true_end_goal:
            install_true_end_goal()
        # inner wrapper: must see the per-sample frame count that install_efficiency sets
        install_goal_first(DiffusionTrainer, true_end=a.true_end_goal,
                           rollout_prob=a.rollout_prob, rollout_steps=a.rollout_steps)
    elif a.process_rollout:
        install_process_rollout(DiffusionTrainer, rollout_steps=a.rollout_steps,
                                action_loss_weight=a.action_loss_weight,
                                rollout_frames=a.rollout_frames, gan_weight=a.gan_weight,
                                gan_warmup=a.gan_warmup, gan_path=str(a.output / "disc_latest.pt"),
                                save_every=a.save_every, gan_source=a.gan_source)
    if a.allow_data_change:
        os.environ["LORA_ALLOW_DATA_CHANGE"] = "1"
    if a.history_noise > 0:
        install_history_noise(a.history_noise, couple_prob=a.hn_couple_prob, split_t=a.hn_split_t,
                              high_range=tuple(float(x) for x in a.hn_high_range.split(",")),
                              low_range=tuple(float(x) for x in a.hn_low_range.split(",")),
                              resample_prob=a.resample_prob, resample_shift=a.resample_shift,
                              resample_steps=a.resample_steps,
                              joint_prob=a.joint_prob, joint_window=a.joint_window,
                              hg_w=a.hg_train_w, hg_sigma=a.hg_train_sigma, hg_prob=a.hg_train_prob,
                              joint_gap=tuple(float(v) for v in a.joint_gap.split(",")),
                              joint_hist_range=tuple(float(v) for v in a.joint_hist_range.split(",")),
                              grid_steps=a.train_grid_steps, hg_anchor=a.hg_anchor_sigma)
    if a.goal_insert_prob > 0:
        install_goal_insert(DiffusionTrainer, a.goal_insert_prob)
    if a.delta_weight > 0 or a.align_weight > 0:
        install_delta_losses(DiffusionTrainer, a.delta_weight, a.align_weight, a.align_slice,
                             proj_path=str(a.output / "delta_proj.pt"), save_every=a.save_every,
                             prompt_drop=a.prompt_drop_prob, contrast_prob=a.contrast_prob,
                             contrast_w=a.contrast_weight, contrast_cap=a.contrast_cap)
    install_efficiency(DiffusionTrainer, BaseModel, dict(cfg.lr_schedule))
    if a.motion_weight > 0:
        install_motion_weight(DiffusionTrainer, k=a.motion_weight, cap=a.motion_cap)
    os.chdir(a.assets_root)
    sys.argv = ["train.py", "--config_path", str(config_path), "--logdir", str(a.output),
                "--wandb-save-dir", str(a.output / "wandb"), "--disable-wandb",
                "--stop-after-step", str(stop)]
    if a.eval_before:
        sys.argv.append("--generate-before-train")
    if not a.resume:
        sys.argv.append("--no-auto-resume")
    train.main()


if __name__ == "__main__":
    main()
