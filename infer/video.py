"""Video decoding and flow sampling helpers."""
import numpy as np
import torch


def shifted_sigmas(steps, shift):
    s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return shift * s / (1 + (shift - 1) * s)


def read_video(path, n_raw, height, width):
    """Same decode as the training dataset: resampled to 24 fps, stretched to height x width."""
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=24,scale={width}:{height}",
           "-frames:v", str(n_raw), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
