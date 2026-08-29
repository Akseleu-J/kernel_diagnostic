"""
atomic_ops/kernel_health.py -- always-on, CHEAP per-kernel health diagnostics
that catch exactly the failure class found in the B/B1/B2 stress-test run:
Kernel B (WY-solve) silently saturating on near-singular Akk (cond~1.45e8),
producing a clipped/garbage `A` that is numerically finite (so no existing
guard fires) but structurally wrong.

WHY THIS IS DIFFERENT FROM kernel_diag.py
-----------------------------------------------------------------------
kernel_diag.py's gdn2_kernel_stage_diagnostics already exists and is cheap
enough to be always-on (it says so in its own docstring), but it only
checks `isfinite` + `maxabs` -- exactly the two things clip+nan_to_num are
DESIGNED to make look healthy. A saturated A (raw=2.5e6, sanitized=1e4) is
`isfinite=True` and `maxabs=1e4` -- looks completely fine to kernel_diag.py.
This module adds the two signals that actually distinguish "healthy" from
"saturated-but-hidden":

  1. `was_clipped`: raw value (BEFORE sanitize) != sanitized value, per
     stage. Cheap: one extra elementwise compare + any(), computed from
     the SAME values kernel_diag.py already recomputes -- no extra kernel
     calls beyond what it already does.
  2. `wy_residual`: ||Akk @ A + A - I||_inf for Kernel B specifically --
     the direct, unambiguous test for "is A actually close to
     (I+Akk)^-1", not just "is A finite and bounded". This is the exact
     quantity the stress test used to catch the adversarial_periodic
     failure (residual 2.4e4 vs healthy ~1e-8). Computed only on the
     already-materialized (per-chunk, per-head) Akk/A -- O(BT^2 * D) per
     chunk, same cost class as one extra matmul, not a full kernel
     re-run.

Both signals are pure jnp reductions -- no jax.debug.print, safe under jit,
same "chistye jnp scalars" convention as kernel_diag.py/diagnostics.py.
Intended call site: same place kernel_diag.py is called from (right after
gdn2_pallas_forward_trainable in GatedDeltaNet2J.__call__), OR standalone
against the residuals dict returned by gdn2_pallas_forward_with_residuals
if you want zero extra kernel invocations at all (see
`health_from_residuals` below -- preferred, since it reuses residuals the
backward pass will recompute anyway rather than triggering yet another
stop_gradient forward pass).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST

# Same clip bound as configs.py's KernelConfig.clip default -- kept as an
# explicit arg so callers using a non-default config stay consistent.
_DEFAULT_CLIP = 1e4

# Above this, WY residual is considered "silently saturated" -- calibrated
# from the stress test: healthy modes gave residual ~1e-8..1e-6, the
# adversarial_periodic failure gave 2.4e4. 1.0 is generously conservative
# (3+ orders of magnitude below the observed failure, 3+ above healthy
# noise floor) so it won't false-positive on ordinary fp32 rounding.
WY_RESIDUAL_ALERT = 1.0
COND_ALERT = 1e5  # matches the ~1e5-1e8 range that broke Kernel B in tests


def _was_clipped(raw, sanitized):
    return jnp.any(jnp.abs(raw - sanitized) > 1e-6 * (jnp.abs(raw) + 1.0))


def _wy_residual_and_cond(Akk, A):
    """||Akk @ A + A - I||_inf per (batch,head,chunk), reduced to max over
    the batch -- same quantity test_gdn2_full_pipeline.py's STAGE B check
    uses (`max||M@A - I||_inf`), where M = I + Akk. Also returns a cheap
    conditioning PROXY (not a true condition number -- an SVD/eig would be
    far too expensive to run every step -- but ||A||_inf * ||Akk||_inf is a
    standard, well-known upper bound on cond(I+Akk) in the operator norm,
    cheap to compute from tensors already in hand)."""
    eye = jnp.eye(Akk.shape[-1], dtype=jnp.float32)
    M = eye + Akk.astype(jnp.float32)
    resid = jnp.einsum("...ij,...jk->...ik", M, A.astype(jnp.float32), precision=_HIGHEST) - eye
    resid_inf = jnp.max(jnp.sum(jnp.abs(resid), axis=-1))

    A_inf = jnp.max(jnp.sum(jnp.abs(A.astype(jnp.float32)), axis=-1))
    M_inf = jnp.max(jnp.sum(jnp.abs(M), axis=-1))
    cond_proxy = A_inf * M_inf
    return resid_inf, cond_proxy


def health_from_residuals(residuals: dict, config=None, axis_name=None):
    """Preferred call site: pass the `residuals` dict already returned by
    gdn2_pallas_forward_with_residuals (kernel_trainable_B6._gdn2_core_fwd
    already computes and stashes exactly Aqk/Akk/A/w_pseudo/u/kg/qg for the
    backward pass) -- this adds ZERO extra Pallas kernel invocations beyond
    what training already pays for, unlike kernel_diag.py's separate
    stop_gradient recompute of A/B/C.

    Returns a flat dict of scalars: wy_residual_inf, wy_cond_proxy,
    wy_saturated (bool as float32), plus per-stage was_clipped flags for
    aqk/akk/a_wy_inverse/w_pseudo/u/kg/qg IF the raw (pre-sanitize) value
    is available in `residuals` -- current kernel_trainable_B6.py does NOT
    stash the pre-sanitize raw value (each Pallas kernel sanitizes
    internally and only the sanitized value ever leaves the kernel), so
    was_clipped detection needs one extra Pallas output per kernel (see
    module docstring's "requires touching kernel signatures" caveat) --
    NOT done here to keep this a zero-extra-kernel-call diagnostic. The
    wy_residual/cond check below does NOT have this limitation: it needs
    only Akk and A, both already in `residuals` as sanitized outputs, and
    is a DIRECT correctness check (not a "was it clipped" proxy) -- this
    is deliberately the first diagnostic added because it's the one that
    actually caught the real bug in the stress test, at zero extra cost.
    """
    Akk = residuals["Akk"]
    A = residuals["A"]
    resid_inf, cond_proxy = _wy_residual_and_cond(Akk, A)

    if axis_name is not None:
        resid_inf = jax.lax.pmax(resid_inf, axis_name=axis_name)
        cond_proxy = jax.lax.pmax(cond_proxy, axis_name=axis_name)

    saturated = jnp.logical_or(
        resid_inf > WY_RESIDUAL_ALERT, cond_proxy > COND_ALERT
    ).astype(jnp.float32)

    return {
        "wy_residual_inf": resid_inf,
        "wy_cond_proxy": cond_proxy,
        "wy_saturated": saturated,
    }
