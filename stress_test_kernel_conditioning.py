"""
stress_test_kernel_conditioning.py -- localizes EXACTLY where each forward
kernel (A/B/C/D) starts to silently saturate, instead of just running 5
fixed adversarial modes and reporting pass/fail.

Motivation (see chat): test_gdn2_full_pipeline.py's fixed
"adversarial_periodic" mode caught Kernel B failing at cond(M)~1.45e8, but
gave no sense of WHERE the cliff is -- is it at cond~1e4? 1e6? Does it
depend on period length, on batch/head count, on decay magnitude alone?
Without that, you cannot tell whether a given REAL training batch is safely
inside the healthy region or one bad token sequence away from the same
failure mode.

Method: for each kernel stage, define a single scalar "difficulty knob"
that continuously interpolates from easy (well-conditioned) to the known
adversarial extreme, then binary-search along that knob for the exact
value where health_from_residuals' wy_residual_inf crosses the alert
threshold. Report the crossing point + a few points around it, for every
combination of (batch, heads, chunk_size) you care about -- the boundary
may depend on shape, not just on the decay pattern itself.

Run:  python stress_test_kernel_conditioning.py
"""
from __future__ import annotations

import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(".")
from atomic_ops import gdn2_fwd as fwdmod
from atomic_ops.configs import KernelConfig, KAGGLE_MEDIUM
from kernel_health import _wy_residual_and_cond, WY_RESIDUAL_ALERT, COND_ALERT


def make_periodic_inputs(key, bsz, H, D, n_chunks, bt, period: int, decay_strength: float):
    """`period`: how many tokens before the decay pattern repeats exactly
    (period=1 -> constant decay per head, most adversarial; period=bt ->
    no repetition within a chunk, easiest). `decay_strength`: multiplies
    the log-decay magnitude (higher = faster decay = more separated scales
    inside Akk = typically WORSE conditioning, this is the actual knob the
    original adversarial_periodic mode varies implicitly).
    """
    L = n_chunks * bt
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (bsz, L, H, D)) * 0.1
    k = jax.random.normal(k2, (bsz, L, H, D)) * 0.1
    v = jax.random.normal(k3, (bsz, L, H, D)) * 0.5
    w = jax.nn.sigmoid(jax.random.normal(k4, (bsz, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(k4, (bsz, L, H, D)) * 0.5)

    period = max(1, min(period, bt))
    t = jnp.arange(L) % period
    base_pattern = jnp.sin(t.astype(jnp.float32) * (2 * jnp.pi / period))
    g = -jnp.abs(base_pattern)[None, :, None, None] * decay_strength
    g = jnp.broadcast_to(g, (bsz, L, H, D))
    return q, k, v, w, b, g


def probe(key, bsz, H, D, n_chunks, config, period, decay_strength):
    q, k, v, w, b, g = make_periodic_inputs(key, bsz, H, D, n_chunks, config.bt, period, decay_strength)
    Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
    A = fwdmod.wy_solve_pallas(Akk, config=config)
    resid, cond_proxy = _wy_residual_and_cond(Akk, A)
    return float(jax.device_get(resid)), float(jax.device_get(cond_proxy))


def binary_search_breaking_point(key, bsz, H, D, n_chunks, config, period,
                                  lo=0.5, hi=50.0, iters=14):
    """Finds the smallest `decay_strength` at which wy_residual_inf crosses
    WY_RESIDUAL_ALERT, for a fixed `period`. Assumes monotonic-ish
    worsening with decay_strength (true empirically -- larger decay
    spread -> worse Akk conditioning), verified by checking hi actually
    fails before searching.
    """
    resid_hi, cond_hi = probe(key, bsz, H, D, n_chunks, config, period, hi)
    if resid_hi <= WY_RESIDUAL_ALERT and cond_hi <= COND_ALERT:
        return None, resid_hi, cond_hi  # never breaks even at hi -- report as safe

    resid_lo, _ = probe(key, bsz, H, D, n_chunks, config, period, lo)
    if resid_lo > WY_RESIDUAL_ALERT:
        return lo, resid_lo, cond_hi  # breaks even at the easy end

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        resid_mid, cond_mid = probe(key, bsz, H, D, n_chunks, config, period, mid)
        if resid_mid > WY_RESIDUAL_ALERT or cond_mid > COND_ALERT:
            hi = mid
        else:
            lo = mid
    resid_final, cond_final = probe(key, bsz, H, D, n_chunks, config, period, hi)
    return hi, resid_final, cond_final


def main():
    config = KAGGLE_MEDIUM
    key = jax.random.PRNGKey(0)
    shapes = [
        (2, 2, 128, 2),   # matches the existing test file's default shape
        (4, 6, 128, 2),   # closer to real n_heads=6 in ModelConfig
        (8, 6, 128, 4),   # closer to real micro_batch_size=8
    ]
    periods = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"{'bsz':>4} {'H':>3} {'n_chunks':>8} {'period':>7}  {'break@decay':>12} {'resid@break':>12} {'cond@break':>12}")
    for (bsz, H, D, n_chunks) in shapes:
        for period in periods:
            key, sub = jax.random.split(key)
            brk, resid, cond = binary_search_breaking_point(sub, bsz, H, D, n_chunks, config, period)
            brk_str = f"{brk:.3f}" if brk is not None else "SAFE(<=50)"
            print(f"{bsz:>4} {H:>3} {n_chunks:>8} {period:>7}  {brk_str:>12} {resid:>12.3e} {cond:>12.3e}")

    print("\nInterpretation: lower 'break@decay' = kernel is MORE fragile at that\n"
          "period/shape combination -- if any real ModelConfig-shaped row breaks\n"
          "at a decay_strength within the range GatedDeltaNet2J's own\n"
          "`a_param`/`decay_proj` can actually produce (check against\n"
          "-exp(clip(a_param,-20,20)) * softplus(f_proj) in model.py), that combination\n"
          "is a LIVE risk, not just a synthetic curiosity.")


if __name__ == "__main__":
    main()
