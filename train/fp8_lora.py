"""FP8 (e4m3 fwd / e5m2 grad) matmuls for the frozen base Linear layers under LoRA.
The base weight never needs a gradient, so backward only computes dX = dY @ W, also in FP8.
Per-tensor dynamic scaling (sm_89 _scaled_mm). LoRA branches stay bf16."""
import torch

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
M4, M5 = torch.finfo(E4).max, torch.finfo(E5).max


def _q(t, dtype, fmax):
    amax = t.abs().amax().float().clamp(min=1e-12)
    scale = fmax / amax
    return (t.float() * scale).clamp(-fmax, fmax).to(dtype), scale.reciprocal()


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_f8, w_t_f8, w_inv, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xq, x_inv = _q(x2, E4, M4)
        # B must be column-major: w_f8 is [out,in] row-major == [in,out] column-major
        y = torch._scaled_mm(xq, w_f8.t(), scale_a=x_inv, scale_b=w_inv, out_dtype=x.dtype,
                             bias=bias, use_fast_accum=True)
        ctx.save_for_backward(w_t_f8, w_inv)
        ctx.shp = shp
        return y.reshape(*shp[:-1], -1)

    @staticmethod
    def backward(ctx, gy):
        w_t_f8, w_inv = ctx.saved_tensors
        g2 = gy.reshape(-1, gy.shape[-1])
        gq, g_inv = _q(g2, E5, M5)
        gx = torch._scaled_mm(gq, w_t_f8.t(), scale_a=g_inv, scale_b=w_inv, out_dtype=gy.dtype)
        return gx.reshape(ctx.shp), None, None, None, None


def fp8_forward(lin, x):
    if getattr(lin, "_f8", None) is None or lin._f8[0].device != lin.weight.device:
        w = lin.weight.detach()
        wq, w_inv = _q(w, E4, M4)
        lin._f8 = (wq.contiguous(), wq.t().contiguous(), w_inv)  # [out,in] and [in,out]
    wq, wtq, w_inv = lin._f8
    b = None if lin.bias is None else lin.bias.to(torch.bfloat16)
    if x.shape[-1] % 16 or wq.shape[0] % 16:
        return torch.nn.functional.linear(x, lin.weight, lin.bias)
    return _FP8Linear.apply(x.to(torch.bfloat16), wq, wtq, w_inv, b)


def install_fp8(model, drop_bf16=False):
    """Route every peft LoRA layer's frozen base_layer through FP8. Returns #layers."""
    n = 0
    for name, mod in model.named_modules():
        base = getattr(mod, "base_layer", None)
        if isinstance(base, torch.nn.Linear) and not base.weight.requires_grad:
            base.forward = (lambda lin: (lambda x: fp8_forward(lin, x)))(base)
            n += 1
    return n


if __name__ == "__main__":   # micro-benchmark on the 5B block shapes
    import time
    torch.manual_seed(0)
    T, D, H = 30 * 52 * 16, 3072, 14336   # ~16 latent frames worth of tokens
    for (i, o) in ((D, D), (D, H), (H, D)):
        lin = torch.nn.Linear(i, o, device="cuda", dtype=torch.bfloat16).requires_grad_(False)
        x = (torch.randn(T, i, device="cuda", dtype=torch.bfloat16)).requires_grad_(True)
        def run(f):
            y = f(x); y.backward(torch.ones_like(y) * 0.01); return y
        for tag, f in (("bf16", lin), ("fp8", lambda z: fp8_forward(lin, z))):
            for _ in range(3): run(f)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(20): run(f)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 20 * 1e3
            print(f"{i:>5}->{o:<5} {tag}: {ms:6.2f} ms fwd+bwd", flush=True)
        with torch.no_grad():
            ref = lin(x); got = fp8_forward(lin, x)
        print(f"   rel err {((got - ref).float().norm() / ref.float().norm()).item():.4f}", flush=True)
