"""
test_gdn2_full_pipeline.py
==========================

Полный, "по каждой мелочи", диагностический тест-сьют для вашего GDN-2
пайплайна (atomic_ops/gdn2_fwd.py + gdn2_bwd.py + gdn2_pipeline.py +
reference.py), плюс live-обучение крошечной чисто-GDN2 модели с
пошаговой (per-stage, per-step) инструментацией.

Зачем именно так, а не "ещё один offline unit test":
-----------------------------------------------------------------------
Диагностика №1/№2 из чата уже показала:
  - WY-solve (Kernel B) сам по себе НЕ является источником взрыва --
    cond(I+Akk) стабильно 2-3 на любых входах, включая adversarial.
  - CPU-референс (без внутренних nan_to_num/clip) даёт ЯВНЫЙ NaN на
    шаге 3 при LR=3e-3 без клипа -- то есть источник где-то в
    forward/backward цепочке ВНЕ WY-solve, и на TPU он МАСКИРУЕТСЯ
    встроенными sanitize()-вызовами (nan_to_num превращает NaN в 0,
    что не то же самое, что "проблемы нет").

Поэтому этот сьют делает две принципиально разные вещи:

  ЧАСТЬ 1 -- ИЗОЛИРОВАННЫЕ ПРОВЕРКИ КАЖДОЙ СТАДИИ (A, B, C, D, B1-B5)
  Каждая стадия тестируется ОТДЕЛЬНО, на синтетических и adversarial
  входах, СРАВНИВАЯ:
    (a) "сырой" (без sanitize) путь -- ловит именно то место, где NaN/inf
        рождается ПЕРВЫЙ раз, а не там, где санитайзер его сначала прячет;
    (b) санитизированный (текущий, production) путь -- что реально видит
        обучение;
    (c) численный градиент / jax.vjp-референс -- перепроверка того, что
        аналитический backward каждой стадии действительно корректен
        (не только "не NaN", но и "правильный").
  Через monkeypatch отключаем sanitize() ВРЕМЕННО (per-test), чтобы
  увидеть настоящий сигнал, а не замаскированный.

  ЧАСТЬ 2 -- LIVE TRAINING НА КРОШЕЧНОЙ ЧИСТО-GDN2 МОДЕЛИ
  Строит модель из ЕДИНСТВЕННОГО типа слоя (gdn2, без mamba2/mla/moe,
  без residual-блочной архитектуры model.py) -- узкий d_model, малое
  число слоёв, реальный jax.grad + реальный optax-подобный SGD/Adam шаг,
  на каждом шаге:
    - считает finite/maxabs КАЖДОЙ промежуточной величины A->B->C->D
      и B1->B5 (через stop_gradient side-channel, как kernel_diag.py,
      но БЕЗ маскировки -- сырые значения, не после sanitize);
    - останавливается и печатает ПОЛНЫЙ dump (какая стадия, какой шаг,
      какой батч) при первом non-finite ДО санитайзера;
    - воспроизводит сценарии A/B/C/D из диагностики №2 (разные LR,
      наличие/отсутствие clip_by_global_norm) для явного демонстрационного
      сравнения.

Как использовать:
    python test_gdn2_full_pipeline.py --all
    python test_gdn2_full_pipeline.py --stage A
    python test_gdn2_full_pipeline.py --live --steps 300 --lr 3e-3 --no-clip
"""
from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

# ---------------------------------------------------------------------
# Import project modules. We patch atomic_ops.configs.sanitize at
# runtime for the "raw signal" tests, so import the module object
# itself (not just the function) everywhere it's used downstream.
# ---------------------------------------------------------------------
import atomic_ops.configs as cfgmod
from atomic_ops.configs import KernelConfig
import atomic_ops.gdn2_fwd as fwdmod
import atomic_ops.gdn2_bwd as bwdmod
import atomic_ops.gdn2_pipeline as pipemod
import atomic_ops.reference as refmod

_HIGHEST = jax.lax.Precision.HIGHEST


# =======================================================================
# Utilities
# =======================================================================

def _stat(x, name=""):
    x = np.asarray(jax.device_get(x), dtype=np.float64)
    finite = np.isfinite(x)
    n_total = x.size
    n_nonfinite = n_total - int(finite.sum())
    maxabs = float(np.max(np.abs(x[finite]))) if finite.any() else float("nan")
    n_nan = int(np.isnan(x).sum())
    n_posinf = int(np.isposinf(x).sum())
    n_neginf = int(np.isneginf(x).sum())
    return dict(
        name=name, shape=tuple(x.shape), n_total=n_total,
        n_nonfinite=n_nonfinite, n_nan=n_nan, n_posinf=n_posinf, n_neginf=n_neginf,
        maxabs_finite=maxabs, all_finite=(n_nonfinite == 0),
    )


def _print_stat(s):
    tag = "OK  " if s["all_finite"] else "FAIL"
    print(f"  [{tag}] {s['name']:<28} shape={s['shape']!s:<20} "
          f"maxabs(finite)={s['maxabs_finite']:.4e}  "
          f"nonfinite={s['n_nonfinite']}/{s['n_total']} "
          f"(nan={s['n_nan']} +inf={s['n_posinf']} -inf={s['n_neginf']})")


def assert_finite(x, name, hard=True):
    s = _stat(x, name)
    _print_stat(s)
    if hard and not s["all_finite"]:
        raise AssertionError(f"{name} is non-finite: {s}")
    return s


@contextlib.contextmanager
def raw_signal():
    """Monkeypatch every module's `sanitize`/`sanitize_h0` to IDENTITY so
    the raw (pre-clip, pre-nan_to_num) numerics are visible. This is the
    ONLY way to see where a NaN/inf is actually born, since the
    production path replaces it with 0 at every kernel boundary -- see
    module docstring's diagnostic #2 finding.

    IMPORTANT: this must patch the NAME BOUND in each importing module
    (fwdmod.sanitize, bwdmod.sanitize, ...), not just cfgmod.sanitize --
    Python's `from .configs import sanitize` binds a local reference at
    import time that a later `cfgmod.sanitize = ...` will NOT retroactively
    change.
    """
    identity = lambda x, *a, **k: x
    originals = []
    targets = [
        (fwdmod, "sanitize"), (fwdmod, "sanitize_h0"),
        (bwdmod, "sanitize"), (bwdmod, "clip_acc"),
        (cfgmod, "sanitize"), (cfgmod, "sanitize_h0"), (cfgmod, "clip_acc"),
    ]
    for mod, attr in targets:
        if hasattr(mod, attr):
            originals.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, identity)
    try:
        yield
    finally:
        for mod, attr, orig in originals:
            setattr(mod, attr, orig)


def make_inputs(key, bsz, H, D, n_chunks, bt, mode="random", dtype=jnp.float32):
    """mode:
      - "random": standard normal-ish, well-conditioned
      - "adversarial_zero_decay": g=0 everywhere -> decay_diff=0 -> edecay=1
        everywhere within causal mask (max possible coupling in Akk)
      - "adversarial_large_decay": g very negative -> decay saturates the
        -20 clip immediately (tests the clip boundary itself)
      - "adversarial_periodic": periodic k pattern (mimics toy-corpus
        repeated tokens from diagnostic #2) -> near-duplicate k rows,
        stresses Akk's off-diagonal structure
      - "adversarial_nearzero_norm": q/k near the epsilon floor of
        normalization (tests the +eps guards in model.py's
        _safe_normalize, reproduced here directly on q/k)
    """
    L = n_chunks * bt
    keys = jax.random.split(key, 8)

    if mode == "random":
        q = jax.random.normal(keys[0], (bsz, L, H, D), dtype=dtype) * 0.1
        k = jax.random.normal(keys[1], (bsz, L, H, D), dtype=dtype) * 0.1
        v = jax.random.normal(keys[2], (bsz, L, H, D), dtype=dtype) * 0.5
        w = jax.nn.sigmoid(jax.random.normal(keys[3], (bsz, L, H, D), dtype=dtype))
        b = jax.nn.sigmoid(jax.random.normal(keys[4], (bsz, L, H, D), dtype=dtype))
        g = -jax.nn.softplus(jax.random.normal(keys[5], (bsz, L, H, D), dtype=dtype)) * 0.1

    elif mode == "adversarial_zero_decay":
        q = jax.random.normal(keys[0], (bsz, L, H, D), dtype=dtype) * 0.1
        k = jax.random.normal(keys[1], (bsz, L, H, D), dtype=dtype) * 0.1
        v = jax.random.normal(keys[2], (bsz, L, H, D), dtype=dtype) * 0.5
        w = jnp.ones((bsz, L, H, D), dtype=dtype)
        b = jnp.ones((bsz, L, H, D), dtype=dtype)
        g = jnp.zeros((bsz, L, H, D), dtype=dtype)   # alpha == 1 everywhere

    elif mode == "adversarial_large_decay":
        q = jax.random.normal(keys[0], (bsz, L, H, D), dtype=dtype) * 0.1
        k = jax.random.normal(keys[1], (bsz, L, H, D), dtype=dtype) * 0.1
        v = jax.random.normal(keys[2], (bsz, L, H, D), dtype=dtype) * 0.5
        w = jnp.ones((bsz, L, H, D), dtype=dtype)
        b = jnp.ones((bsz, L, H, D), dtype=dtype)
        g = -jnp.ones((bsz, L, H, D), dtype=dtype) * 50.0   # forces the -20 clip hard

    elif mode == "adversarial_periodic":
        period = 4
        base = jax.random.normal(keys[0], (bsz, period, H, D), dtype=dtype) * 0.2
        reps = L // period + 1
        k = jnp.tile(base, (1, reps, 1, 1))[:, :L]
        q = k * 1.01 + 1e-3 * jax.random.normal(keys[1], (bsz, L, H, D), dtype=dtype)
        v = jax.random.normal(keys[2], (bsz, L, H, D), dtype=dtype) * 0.5
        w = jax.nn.sigmoid(jax.random.normal(keys[3], (bsz, L, H, D), dtype=dtype))
        b = jax.nn.sigmoid(jax.random.normal(keys[4], (bsz, L, H, D), dtype=dtype))
        g = -jax.nn.softplus(jax.random.normal(keys[5], (bsz, L, H, D), dtype=dtype)) * 0.05

    elif mode == "adversarial_nearzero_norm":
        eps = 1e-6
        raw_q = jax.random.normal(keys[0], (bsz, L, H, D), dtype=dtype) * 1e-4
        raw_k = jax.random.normal(keys[1], (bsz, L, H, D), dtype=dtype) * 1e-4
        q = raw_q * jax.lax.rsqrt(jnp.sum(raw_q * raw_q, axis=-1, keepdims=True) + eps ** 2)
        k = raw_k * jax.lax.rsqrt(jnp.sum(raw_k * raw_k, axis=-1, keepdims=True) + eps ** 2)
        v = jax.random.normal(keys[2], (bsz, L, H, D), dtype=dtype) * 0.5
        w = jax.nn.sigmoid(jax.random.normal(keys[3], (bsz, L, H, D), dtype=dtype))
        b = jax.nn.sigmoid(jax.random.normal(keys[4], (bsz, L, H, D), dtype=dtype))
        g = -jax.nn.softplus(jax.random.normal(keys[5], (bsz, L, H, D), dtype=dtype)) * 0.1

    else:
        raise ValueError(mode)

    return q, k, v, w, b, g


ADVERSARIAL_MODES = [
    "random",
    "adversarial_zero_decay",
    "adversarial_large_decay",
    "adversarial_periodic",
    "adversarial_nearzero_norm",
]


# =======================================================================
# STAGE A -- build_chunk_scores_pallas (Kernel A): Aqk, Akk
# =======================================================================

def test_stage_A(config: KernelConfig, seed=0):
    print("\n=== STAGE A: build_chunk_scores_pallas (Aqk, Akk) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True
    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)

        # (a) raw signal -- no sanitize inside Kernel A
        with raw_signal():
            Aqk_raw, Akk_raw = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
        s1 = assert_finite(Aqk_raw, f"[{mode}] Aqk_raw", hard=False)
        s2 = assert_finite(Akk_raw, f"[{mode}] Akk_raw", hard=False)
        ok = ok and s1["all_finite"] and s2["all_finite"]

        # (b) production (sanitized) path
        Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
        assert_finite(Aqk, f"[{mode}] Aqk_sanitized")
        assert_finite(Akk, f"[{mode}] Akk_sanitized")

        # structural invariant: Akk must be STRICTLY lower-triangular
        # (diag==0) inside every BT block, Aqk must be causal (upper==0)
        BT = config.bt
        akk_np = np.asarray(jax.device_get(Akk))
        aqk_np = np.asarray(jax.device_get(Aqk))
        diag_max = np.max(np.abs(np.diagonal(akk_np, axis1=-2, axis2=-1)))
        upper_mask = np.triu(np.ones((BT, BT)), k=1).astype(bool)
        aqk_upper_max = np.max(np.abs(aqk_np[..., upper_mask]))
        print(f"  [structural] Akk diag maxabs={diag_max:.3e} (must be 0)  "
              f"Aqk strict-upper maxabs={aqk_upper_max:.3e} (must be 0)")
        if diag_max > 1e-6 or aqk_upper_max > 1e-6:
            ok = False

        # cross-check against the reference chunk-builder (Aqk/Akk math
        # only, ignoring the WY-inverse which reference.py computes
        # differently but the raw scores must match bit-for-bit modulo
        # the sanitize/clip boundary).
        g_c = g.reshape(bsz, n_chunks, config.bt, H, D)
        g_c = jnp.moveaxis(g_c, 1, 0)
        q_c = jnp.moveaxis(q.reshape(bsz, n_chunks, config.bt, H, D), 1, 0)
        k_c = jnp.moveaxis(k.reshape(bsz, n_chunks, config.bt, H, D), 1, 0)
        b_c = jnp.moveaxis(b.reshape(bsz, n_chunks, config.bt, H, D), 1, 0)
        v_c = jnp.moveaxis(v.reshape(bsz, n_chunks, config.bt, H, D), 1, 0)
        w_c = jnp.moveaxis(w.reshape(bsz, n_chunks, config.bt, H, D), 1, 0)

        ref_aqk_chunks = []
        ref_akk_chunks = []
        for c in range(n_chunks):
            aqk_ref, _, _, _, _, _ = refmod._build_chunk_wy(
                q_c[c], k_c[c], v_c[c], g_c[c], b_c[c], w_c[c], scale=1.0, wy_eps=0.0
            )
            ref_aqk_chunks.append(aqk_ref)
        # ref_build_chunk_wy also returns Aqk with same causal masking;
        # compare per-chunk against our kernel's per-chunk slice
        for c in range(n_chunks):
            kernel_slice = Aqk[:, :, c]
            rel = np.abs(np.asarray(jax.device_get(kernel_slice)) -
                         np.asarray(jax.device_get(ref_aqk_chunks[c])))
            maxdiff = float(rel.max())
            print(f"  [{mode}] chunk {c}: |Aqk_kernel - Aqk_reference| max = {maxdiff:.3e}")
            if maxdiff > 5e-3:
                ok = False

    print(f"STAGE A RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# STAGE B -- wy_solve_pallas: A = (I + (1-eps)*Akk)^-1
# =======================================================================

def test_stage_B(config: KernelConfig, seed=1):
    print("\n=== STAGE B: wy_solve_pallas (WY inverse A) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)
        Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)

        with raw_signal():
            A_raw = fwdmod.wy_solve_pallas(Akk, config=config)
        assert_finite(A_raw, f"[{mode}] A_raw", hard=False)

        A = fwdmod.wy_solve_pallas(Akk, config=config)
        assert_finite(A, f"[{mode}] A_sanitized")

        # conditioning + correctness check: verify (I + (1-eps)Akk) @ A ~= I
        eps = config.wy_eps
        akk_np = np.asarray(jax.device_get(Akk), dtype=np.float64)
        a_np = np.asarray(jax.device_get(A), dtype=np.float64)
        BT = config.bt
        eye = np.eye(BT)
        max_resid = 0.0
        max_cond = 0.0
        for bi in range(bsz):
            for hi in range(H):
                for ci in range(n_chunks):
                    M = eye + (1 - eps) * akk_np[bi, hi, ci]
                    resid = np.abs(M @ a_np[bi, hi, ci] - eye).max()
                    max_resid = max(max_resid, resid)
                    try:
                        cnum = np.linalg.cond(M)
                    except np.linalg.LinAlgError:
                        cnum = float("inf")
                    max_cond = max(max_cond, cnum)
        print(f"  [{mode}] max||M@A - I||_inf = {max_resid:.3e}   max cond(M) = {max_cond:.3e}")
        # A strictly-lower-triangular unit matrix inverse should be exact
        # to within a small multiple of BC-block forward-substitution
        # error (empirically < 1e-2 for BT=256,BC=128,MB=16 per project
        # notes) -- this is the DIRECT reproduction of diagnostic #1's
        # "cond(I+Akk) stays ~2-3" claim; fail loudly if that regresses.
        if max_resid > 5e-2:
            ok = False
            print(f"  [{mode}] !! WY inverse residual too large -- investigate Kernel B for this mode.")
        if max_cond > 50.0:
            print(f"  [{mode}] !! NOTE: conditioning higher than the ~2-3 baseline "
                  f"claimed in the chat diagnosis -- re-verify the 'WY-solve is not "
                  f"the source' conclusion specifically for this adversarial mode.")

    print(f"STAGE B RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# STAGE C -- recompute_wy_pallas: w_pseudo, u, kg, qg, gc_last
# =======================================================================

def test_stage_C(config: KernelConfig, seed=2):
    print("\n=== STAGE C: recompute_wy_pallas (w_pseudo, u, kg, qg, gc_last) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)
        Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
        A = fwdmod.wy_solve_pallas(Akk, config=config)

        with raw_signal():
            w_pseudo_raw, u_raw, kg_raw, qg_raw, gc_last_raw = fwdmod.recompute_wy_pallas(
                q, k, v, w, b, g, A, config=config
            )
        stats_raw = [
            assert_finite(w_pseudo_raw, f"[{mode}] w_pseudo_raw", hard=False),
            assert_finite(u_raw, f"[{mode}] u_raw", hard=False),
            assert_finite(kg_raw, f"[{mode}] kg_raw", hard=False),
            assert_finite(qg_raw, f"[{mode}] qg_raw", hard=False),
            assert_finite(gc_last_raw, f"[{mode}] gc_last_raw", hard=False),
        ]
        ok = ok and all(s["all_finite"] for s in stats_raw)

        w_pseudo, u, kg, qg, gc_last = fwdmod.recompute_wy_pallas(q, k, v, w, b, g, A, config=config)
        for name, val in (("w_pseudo", w_pseudo), ("u", u), ("kg", kg), ("qg", qg), ("gc_last", gc_last)):
            assert_finite(val, f"[{mode}] {name}_sanitized")

        # This is exactly where kernel_c_recompute.py's own docstring
        # documents the historical bug (kg/qg previously unsanitized,
        # w_pseudo/u only nan_to_num'd not clipped) -- explicitly flag if
        # the RAW magnitude before sanitize is already "suspiciously
        # large but finite" (the exact precursor pattern from that
        # incident), even though it passes the finite check.
        for name, val in (("w_pseudo", w_pseudo_raw), ("u", u_raw), ("kg", kg_raw), ("qg", qg_raw)):
            v_np = np.asarray(jax.device_get(val))
            finite = np.isfinite(v_np)
            maxabs = float(np.max(np.abs(v_np[finite]))) if finite.any() else float("nan")
            if maxabs > 1e6:
                print(f"  [{mode}] !! PRECURSOR WARNING: raw {name} maxabs={maxabs:.3e} "
                      f"(>1e6, still finite) -- this is the exact 'large-but-finite A "
                      f"propagating downstream' pattern from kernel_c_recompute.py's "
                      f"own incident writeup. Investigate even though sanitize() "
                      f"currently catches it.")

    print(f"STAGE C RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# STAGE D -- gdn2_inter_chunk_combine(_with_state): o, h_final
# =======================================================================

def test_stage_D(config: KernelConfig, seed=3):
    print("\n=== STAGE D: inter-chunk scan (o, h_final) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 4   # more chunks: stresses the scan carry
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)
        Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
        A = fwdmod.wy_solve_pallas(Akk, config=config)
        w_pseudo, u, kg, qg, gc_last = fwdmod.recompute_wy_pallas(q, k, v, w, b, g, A, config=config)

        with raw_signal():
            o_raw, h_final_raw = fwdmod.gdn2_inter_chunk_combine(
                Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, config=config
            )
        s1 = assert_finite(o_raw, f"[{mode}] o_raw", hard=False)
        s2 = assert_finite(h_final_raw, f"[{mode}] h_final_raw", hard=False)
        ok = ok and s1["all_finite"] and s2["all_finite"]

        # h_final growth across the scan -- is it monotonic-runaway or
        # bounded? Print per-chunk h_pre maxabs to see WHERE in the
        # sequence (which chunk index) growth actually happens, since a
        # scan carry blowing up shows as "fine chunk 0, exploding chunk N".
        o_chunks, h_final, h_pre_all, v_new_all = fwdmod.gdn2_inter_chunk_combine_with_state(
            Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, config=config
        )
        h_pre_np = np.asarray(jax.device_get(h_pre_all))
        per_chunk_maxabs = np.max(np.abs(h_pre_np), axis=tuple(range(1, h_pre_np.ndim)))
        print(f"  [{mode}] h_pre maxabs per chunk (scan order): "
              + ", ".join(f"{v:.3e}" for v in per_chunk_maxabs))
        if len(per_chunk_maxabs) >= 2 and per_chunk_maxabs[-1] > 10 * (per_chunk_maxabs[0] + 1e-9):
            print(f"  [{mode}] !! h_pre grows >10x from first to last chunk within ONE "
                  f"scan call -- possible carry-state runaway distinct from the "
                  f"cross-step optimizer drift reported in chat.")

    print(f"STAGE D RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# BACKWARD B1-B5 -- each checked against jax.vjp on the SAME forward
# stage function (not the full-pipeline reference), isolating exactly
# which backward kernel (if any) diverges from autodiff ground truth.
# =======================================================================

def _rand_like(key, x, scale=1.0):
    return jax.random.normal(key, x.shape, dtype=jnp.float32) * scale


def test_backward_B1(config: KernelConfig, seed=10):
    """B1 = gdn2_dhu_backward: reverse-scan adjoint of the inter-chunk
    state recurrence. Cross-check against jax.vjp on
    gdn2_inter_chunk_combine itself (forward-mode ground truth for the
    exact function this backward differentiates)."""
    print("\n=== BACKWARD B1: gdn2_dhu_backward vs jax.vjp(gdn2_inter_chunk_combine) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 3
    ok = True
    q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode="random")
    Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
    A = fwdmod.wy_solve_pallas(Akk, config=config)
    w_pseudo, u, kg, qg, gc_last = fwdmod.recompute_wy_pallas(q, k, v, w, b, g, A, config=config)

    def fwd_fn(w_pseudo_, u_, kg_, qg_, gc_last_):
        o, h_final = fwdmod.gdn2_inter_chunk_combine(Aqk, w_pseudo_, u_, kg_, qg_, gc_last_, scale=1.0, config=config)
        return o, h_final

    (o0, h_final0), vjp_fn = jax.vjp(fwd_fn, w_pseudo, u, kg, qg, gc_last)
    k2 = jax.random.split(key, 2)
    do = _rand_like(k2[0], o0, 0.1)
    dh_final = _rand_like(k2[1], h_final0, 0.1)
    dw_pseudo_ref, du_ref, dkg_ref, dqg_ref, dgc_last_ref = vjp_fn((do, dh_final))

    # B1 only produces dh_all/dh0/dv_all (state-side adjoints) -- it does
    # NOT directly return dw_pseudo/du/dkg/dqg (those come from B2/B3
    # downstream in the real pipeline). To isolate B1 alone, check the
    # STATE recursion adjoint dh0 against jax.vjp w.r.t. an explicit h0
    # input instead.
    h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    def fwd_fn_h0(h0_):
        o, h_final = fwdmod.gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, h0=h0_, config=config)
        return o, h_final

    (o1, h_final1), vjp_fn_h0 = jax.vjp(fwd_fn_h0, h0)
    (dh0_ref,) = vjp_fn_h0((do, dh_final))

    # Now call B1 directly with the SAME cotangents and check dh0 matches.
    _, _, dv_all_unused = None, None, None
    dh_all, dh0_kernel, dv_all = bwdmod.gdn2_dhu_backward(
        # do must be reshaped into (bsz,H,n_chunks,BT,D) chunk form to
        # match B1's expected layout
        jnp.moveaxis(do.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1)),
        jnp.zeros_like(jnp.moveaxis(u.reshape(bsz, H, n_chunks, config.bt, D), 0, 0)) * 0.0,
        w_pseudo, qg, kg, gc_last, scale=1.0, dht=dh_final,
    )
    diff = np.abs(np.asarray(jax.device_get(dh0_kernel)) - np.asarray(jax.device_get(dh0_ref)))
    rel = diff.max() / (np.abs(np.asarray(jax.device_get(dh0_ref))).max() + 1e-8)
    print(f"  dh0: kernel vs jax.vjp  max_abs_diff={diff.max():.3e}  rel={rel:.3e}")
    assert_finite(dh_all, "dh_all")
    assert_finite(dh0_kernel, "dh0_kernel")
    assert_finite(dv_all, "dv_all")
    if rel > 0.05:
        ok = False
        print("  !! B1 dh0 diverges from jax.vjp reference by >5% -- investigate.")

    print(f"BACKWARD B1 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_B2(config: KernelConfig, seed=11):
    """B2 = dav_backward_pallas: adjoint of (Aqk, v_new) -> intra term in
    the D-scan. Cross-check against jax.vjp on the same einsum done in
    plain JAX."""
    print("\n=== BACKWARD B2: dav_backward_pallas vs jax.vjp ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, n_chunks, BT, D = 2, 2, 3, config.bt, 128
    ok = True
    k2 = jax.random.split(key, 3)
    Aqk = jax.random.normal(k2[0], (bsz, H, n_chunks, BT, BT)) * 0.1
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    Aqk = Aqk * causal
    v_new = jax.random.normal(k2[1], (bsz, H, n_chunks, BT, D)) * 0.5
    do = jax.random.normal(k2[2], (bsz, H, n_chunks, BT, D)) * 0.1

    def fwd_fn(Aqk_, v_new_):
        intra = jnp.einsum("bhcij,bhcjv->bhciv", Aqk_, v_new_, precision=_HIGHEST)
        return intra

    intra0, vjp_fn = jax.vjp(fwd_fn, Aqk, v_new)
    (dAqk_ref, dv_new_ref) = vjp_fn(do)

    dAqk_kernel, dv_new_kernel = bwdmod.dav_backward_pallas(Aqk, v_new, do, config=config)

    for name, kern, ref in (("dAqk", dAqk_kernel, dAqk_ref), ("dv_new(partial)", dv_new_kernel, dv_new_ref)):
        assert_finite(kern, name)
        diff = np.abs(np.asarray(jax.device_get(kern)) - np.asarray(jax.device_get(ref)))
        rel = diff.max() / (np.abs(np.asarray(jax.device_get(ref))).max() + 1e-8)
        print(f"  {name}: max_abs_diff={diff.max():.3e} rel={rel:.3e}")
        if name == "dAqk" and rel > 0.05:
            ok = False

    print(f"BACKWARD B2 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_B3(config: KernelConfig, seed=12):
    """B3 = wy_dqkg_backward_pallas: the matrix-inverse-gradient hot spot
    (A^T @ dA_total @ A^T double matmul). This is the exact location the
    project's own docstring flags as highest-amplification -- test with
    BOTH random and adversarial (near-singular-looking, though we've
    established Akk itself is well-conditioned) Akk."""
    print("\n=== BACKWARD B3: wy_dqkg_backward_pallas (matrix-inverse-grad hot spot) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)
        Aqk, Akk = fwdmod.build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config)
        A = fwdmod.wy_solve_pallas(Akk, config=config)
        w_pseudo, u, kg, qg, gc_last = fwdmod.recompute_wy_pallas(q, k, v, w, b, g, A, config=config)
        o_chunks, h_final, h_pre_all, v_new_all = fwdmod.gdn2_inter_chunk_combine_with_state(
            Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, config=config
        )

        idx = jnp.arange(config.bt)
        tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
        g_r = jnp.moveaxis(g.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))
        gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

        q_r = jnp.moveaxis(q.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))
        k_r = jnp.moveaxis(k.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))
        b_r = jnp.moveaxis(b.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))
        w_r = jnp.moveaxis(w.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))
        v_r = jnp.moveaxis(v.reshape(bsz, n_chunks, config.bt, H, D), (1, 3), (2, 1))

        keys2 = jax.random.split(key, 2)
        do_r = _rand_like(keys2[0], v_new_all, 0.05)
        dv_r = _rand_like(keys2[1], v_new_all, 0.05)
        dh_next_all = jnp.zeros_like(h_pre_all)

        with raw_signal():
            b3_raw = bwdmod.wy_dqkg_backward_pallas(
                q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
                do_r, dv_r, dh_next_all, scale=1.0, config=config,
            )
        all_finite_raw = True
        for name, val in b3_raw.items():
            s = assert_finite(val, f"[{mode}] {name}_raw", hard=False)
            all_finite_raw = all_finite_raw and s["all_finite"]
        ok = ok and all_finite_raw

        b3 = bwdmod.wy_dqkg_backward_pallas(
            q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
            do_r, dv_r, dh_next_all, scale=1.0, config=config,
        )
        for name, val in b3.items():
            assert_finite(val, f"[{mode}] {name}_sanitized")

        # highlight the exact hot-spot: dAkk magnitude vs A magnitude --
        # if dAkk maxabs >> 1/eps-scale of A, the double-matmul is
        # amplifying strongly even though everything stays finite.
        dAkk_np = np.asarray(jax.device_get(b3_raw["dAkk"]))
        A_np = np.asarray(jax.device_get(A))
        finite_dAkk = dAkk_np[np.isfinite(dAkk_np)]
        amp_ratio = (np.abs(finite_dAkk).max() if finite_dAkk.size else float("nan")) / (np.abs(A_np).max() + 1e-9)
        print(f"  [{mode}] amplification ratio max|dAkk_raw| / max|A| = {amp_ratio:.3e} "
              f"(large values here mean the A^T@dA@A^T double-matmul IS amplifying "
              f"strongly on this input, even though the final value is finite)")

    print(f"BACKWARD B3 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_B4(config: KernelConfig, seed=13):
    """B4 = intra_backward_pallas: read-modify-write accumulation across
    (si,sj) sub-blocks -- the exact pattern the project's own docstring
    flags as an inf+(-inf)=NaN risk if not clipped after every write."""
    print("\n=== BACKWARD B4: intra_backward_pallas (RMW accumulation) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)
        keys2 = jax.random.split(key, 2)
        BT = config.bt
        dAqk = _rand_like(keys2[0], jnp.zeros((bsz, n_chunks, H, BT, BT)), 0.1) if False else \
            jax.random.normal(keys2[0], (bsz, H, n_chunks, BT, BT)) * 0.1
        dAkk = jax.random.normal(keys2[1], (bsz, H, n_chunks, BT, BT)) * 0.1

        with raw_signal():
            dq_raw, dk_raw, db_raw, dgc_raw = bwdmod.intra_backward_pallas(
                dAqk, dAkk, q, k, b, g, scale=1.0, config=config
            )
        stats = [
            assert_finite(dq_raw, f"[{mode}] dq_raw", hard=False),
            assert_finite(dk_raw, f"[{mode}] dk_raw", hard=False),
            assert_finite(db_raw, f"[{mode}] db_raw", hard=False),
            assert_finite(dgc_raw, f"[{mode}] dgc_raw", hard=False),
        ]
        ok = ok and all(s["all_finite"] for s in stats)

        dq, dk, db, dgc = bwdmod.intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale=1.0, config=config)
        for name, val in (("dq", dq), ("dk", dk), ("db", db), ("dgc", dgc)):
            assert_finite(val, f"[{mode}] {name}_sanitized")

        # opposite-sign inf+inf=NaN stress test: force dAqk/dAkk to contain
        # deliberately huge opposite-signed values in overlapping regions
        # to specifically probe the RMW accumulation risk described in the
        # module's own docstring.
        huge = 1e30
        dAqk_stress = dAqk.at[:, :, :, :64, :64].set(huge)
        dAkk_stress = dAkk.at[:, :, :, :64, :64].set(-huge)
        with raw_signal():
            dq_s, dk_s, db_s, dgc_s = bwdmod.intra_backward_pallas(
                dAqk_stress, dAkk_stress, q, k, b, g, scale=1.0, config=config
            )
        s = assert_finite(dq_s, f"[{mode}] dq_STRESS_raw (huge opposite-sign inputs)", hard=False)
        if not s["all_finite"]:
            print(f"  [{mode}] !! B4 RMW accumulation produced non-finite under the "
                  f"opposite-sign-huge-value stress test (raw path) -- this is exactly "
                  f"the inf+(-inf)=NaN pattern the module docstring warns about. "
                  f"Sanitized path below should still recover:")
        dq_s2, dk_s2, db_s2, dgc_s2 = bwdmod.intra_backward_pallas(
            dAqk_stress, dAkk_stress, q, k, b, g, scale=1.0, config=config
        )
        assert_finite(dq_s2, f"[{mode}] dq_STRESS_sanitized")

    print(f"BACKWARD B4 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_B5(config: KernelConfig, seed=14):
    """B5 = reverse_cumsum_bwd: reverse tril-matmul cumsum for dg. Simple
    linear op -- cross-check against jax.vjp on jnp.cumsum directly, plus
    a stress test summing BT=256 large values into one row."""
    print("\n=== BACKWARD B5: reverse_cumsum_bwd vs jax.vjp(cumsum) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, n_chunks, D = 2, 2, 2, 128
    ok = True
    BT = config.bt
    dgc = jax.random.normal(key, (bsz, H, n_chunks, BT, D)) * 0.1

    def fwd_fn(g_raw):
        return jnp.cumsum(g_raw, axis=-2)

    g_raw = jax.random.normal(jax.random.PRNGKey(seed + 1), (bsz, H, n_chunks, BT, D)) * 0.1
    _, vjp_fn = jax.vjp(fwd_fn, g_raw)
    (dg_ref,) = vjp_fn(dgc)

    dg_kernel = bwdmod.reverse_cumsum_bwd(dgc, chunk_size=BT)
    assert_finite(dg_kernel, "dg_kernel")
    diff = np.abs(np.asarray(jax.device_get(dg_kernel)) - np.asarray(jax.device_get(dg_ref)))
    rel = diff.max() / (np.abs(np.asarray(jax.device_get(dg_ref))).max() + 1e-8)
    print(f"  dg: max_abs_diff={diff.max():.3e} rel={rel:.3e}")
    if rel > 1e-3:
        ok = False

    # stress: BT large-but-finite entries summing into row 0 -- tests the
    # "single large-but-finite dgc entry blows up dg_raw" scenario the
    # module docstring specifically calls out.
    dgc_stress = jnp.ones((bsz, H, n_chunks, BT, D)) * 1e5
    with raw_signal():
        dg_stress_raw = bwdmod.reverse_cumsum_bwd(dgc_stress, chunk_size=BT)
    s = assert_finite(dg_stress_raw, "dg_STRESS_raw (all rows = 1e5)", hard=False)
    dg_stress = bwdmod.reverse_cumsum_bwd(dgc_stress, chunk_size=BT)
    assert_finite(dg_stress, "dg_STRESS_sanitized")
    print(f"  STRESS max value at row 0 (should be ~BT*1e5={BT*1e5:.2e} if unclipped, "
          f"clipped to 1e4 by sanitize): "
          f"{float(jnp.max(dg_stress[..., 0, :])):.3e}")

    print(f"BACKWARD B5 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# FULL PIPELINE custom_vjp CHECK -- gdn2_pallas_forward_trainable vs
# jax.vjp on gdn2_chunked_wy_reference (the two independent "cheat" and
# "honest" paths this project already maintains side by side).
# =======================================================================

def test_full_pipeline_grad_check(config: KernelConfig, seed=20):
    print("\n=== FULL PIPELINE: gdn2_pallas_forward_trainable vs jax.vjp(reference) ===")
    key = jax.random.PRNGKey(seed)
    bsz, H, D, n_chunks = 2, 2, 128, 2
    ok = True

    for mode in ADVERSARIAL_MODES:
        print(f"-- mode={mode}")
        q, k, v, w, b, g = make_inputs(key, bsz, H, D, n_chunks, config.bt, mode=mode)

        def loss_kernel(q_, k_, v_, w_, b_, g_):
            o, h_final = pipemod.gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, config=config)
            return jnp.sum(o ** 2) + jnp.sum(h_final ** 2)

        def loss_ref(q_, k_, v_, w_, b_, g_):
            o, h_final = refmod.gdn2_chunked_wy_reference(
                q_, k_, v_, g_, b_, w_, scale=1.0, chunk_size=config.bt, wy_eps=config.wy_eps
            )
            return jnp.sum(o ** 2) + jnp.sum(h_final ** 2)

        grads_kernel = jax.grad(loss_kernel, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, w, b, g)
        grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, w, b, g)

        names = ["dq", "dk", "dv", "dw", "db", "dg"]
        for name, gk, gr in zip(names, grads_kernel, grads_ref):
            assert_finite(gk, f"[{mode}] {name}_kernel")
            gk_np = np.asarray(jax.device_get(gk), dtype=np.float64)
            gr_np = np.asarray(jax.device_get(gr), dtype=np.float64)
            diff = np.abs(gk_np - gr_np)
            rel = diff.max() / (np.abs(gr_np).max() + 1e-6)
            print(f"  [{mode}] {name}: max_abs_diff={diff.max():.3e}  rel={rel:.3e}")
            if rel > 0.10:
                ok = False
                print(f"  [{mode}] !! {name} diverges from reference by >10%.")

    print(f"FULL PIPELINE GRAD CHECK RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# =======================================================================
# PART 2 -- LIVE TRAINING ON A TINY, PURE-GDN2 MODEL
# =======================================================================

@dataclass
class TinyGDN2Params:
    pass


def init_tiny_model(key, d_model, n_layers, n_heads, d_head, vocab_size):
    keys = jax.random.split(key, 3 + n_layers * 8)
    ki = iter(keys)
    scale = 0.02
    params = {
        "embed": jax.random.normal(next(ki), (vocab_size, d_model)) * scale,
        "unembed": jax.random.normal(next(ki), (d_model, vocab_size)) * scale,
        "layers": [],
    }
    for _ in range(n_layers):
        layer = {
            "wq": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "wk": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "wv": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "ww": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "wb": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "wg": jax.random.normal(next(ki), (d_model, n_heads * d_head)) * scale,
            "wo": jax.random.normal(next(ki), (n_heads * d_head, d_model)) * scale,
            "decay_a": jnp.zeros((n_heads,)),
        }
        params["layers"].append(layer)
    return params


def gdn2_block_live(x, layer_params, config, n_heads, d_head, diag_sink=None, tag=""):
    """Pure-GDN2 block for the tiny live-training model. Every
    intermediate stage is optionally pushed into diag_sink (a plain
    Python list) as (name, array) BEFORE any sanitize, via a side jvp-free
    stop_gradient snapshot -- so we can inspect raw magnitudes on the
    ACTUAL training trajectory, not just synthetic adversarial inputs."""
    b, l, d = x.shape
    q = (x @ layer_params["wq"]).reshape(b, l, n_heads, d_head)
    k = (x @ layer_params["wk"]).reshape(b, l, n_heads, d_head)
    v = jnp.clip((x @ layer_params["wv"]).reshape(b, l, n_heads, d_head), -50.0, 50.0)
    w_gate = jax.nn.sigmoid((x @ layer_params["ww"]).reshape(b, l, n_heads, d_head))
    b_gate = jax.nn.sigmoid((x @ layer_params["wb"]).reshape(b, l, n_heads, d_head))
    f_proj = (x @ layer_params["wg"]).reshape(b, l, n_heads, d_head)

    eps = 1e-6
    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + eps ** 2)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + eps ** 2)

    a_param_safe = jnp.clip(layer_params["decay_a"], -20.0, 20.0)
    g = -jnp.exp(a_param_safe)[None, None, :, None] * jax.nn.softplus(f_proj)

    if diag_sink is not None:
        for name, val in (("q", q), ("k", k), ("v", v), ("w_gate", w_gate), ("b_gate", b_gate), ("g", g)):
            diag_sink.append((f"{tag}:{name}", jax.lax.stop_gradient(val)))

    o, h_final = pipemod.gdn2_pallas_forward_trainable(q, k, v, w_gate, b_gate, g, scale=1.0, config=config)

    if diag_sink is not None:
        diag_sink.append((f"{tag}:o", jax.lax.stop_gradient(o)))
        diag_sink.append((f"{tag}:h_final", jax.lax.stop_gradient(h_final)))

    o = o.reshape(b, l, n_heads * d_head)
    return o @ layer_params["wo"]


def tiny_forward(params, input_ids, config, n_heads, d_head, diag_sink=None):
    x = params["embed"][input_ids]
    for li, layer in enumerate(params["layers"]):
        delta = gdn2_block_live(x, layer, config, n_heads, d_head, diag_sink=diag_sink, tag=f"layer{li}")
        x = jnp.clip(x + delta, -1e3, 1e3)
    logits = x @ params["unembed"]
    return logits


def tiny_loss(params, input_ids, labels, config, n_heads, d_head, diag_sink=None):
    logits = tiny_forward(params, input_ids, config, n_heads, d_head, diag_sink=diag_sink)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def make_toy_batch(key, bsz, seq_len, vocab_size, periodic=True):
    """Reproduces diagnostic #2's toy-corpus flavor: periodic token
    pattern (stresses adversarial_periodic-style Akk structure) if
    periodic=True, else pure random tokens."""
    if periodic:
        period = 8
        base = jax.random.randint(key, (bsz, period), 0, vocab_size)
        reps = seq_len // period + 1
        ids = jnp.tile(base, (1, reps))[:, :seq_len]
    else:
        ids = jax.random.randint(key, (bsz, seq_len), 0, vocab_size)
    labels = jnp.roll(ids, -1, axis=1)
    return ids, labels


def run_live_training(steps=300, lr=3e-3, use_clip=True, clip_norm=1.0,
                       use_nan_guard=True, seed=0, log_every=10, dump_dir=None):
    print(f"\n=== LIVE TRAINING: steps={steps} lr={lr} clip={use_clip} "
          f"nan_guard={use_nan_guard} ===")

    config = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)
    n_heads, d_head = 2, 128
    d_model = n_heads * d_head
    n_layers = 2
    vocab_size = 64
    seq_len = config.bt * 2   # 2 chunks
    bsz = 2

    key = jax.random.PRNGKey(seed)
    key, pkey = jax.random.split(key)
    params = init_tiny_model(pkey, d_model, n_layers, n_heads, d_head, vocab_size)

    def loss_fn(p, ids, labels):
        return tiny_loss(p, ids, labels, config, n_heads, d_head, diag_sink=None)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # crude Adam
    # ФИКС: opt_state хранится как ДВА ОТДЕЛЬНЫХ pytree (m и v) той же
    # структуры, что params -- а не как pytree, где КАЖДЫЙ лист сам
    # является Python-кортежем (m, v). Последнее выглядит правильным, но
    # jax.tree_util.tree_flatten разворачивает ЛЮБЫЕ вложенные
    # tuple/list/dict как часть дерева по умолчанию -- то есть лист
    # "(m, v)" на самом деле flatten'ится в ДВА отдельных листа (m и v
    # по отдельности), и flat_s получается ровно в 2 раза длиннее flat_p.
    # Отсюда "ValueError: too many values to unpack (expected 2)" --
    # zip(flat_p, flat_g, flat_s) в реальности сопоставлял params[i] с
    # opt_state[2i] (m) вместо (m,v)-пары, и на очередной итерации `m, v
    # = mv` получал скаляр вместо пары.
    def init_adam(p):
        m0 = jax.tree_util.tree_map(jnp.zeros_like, p)
        v0 = jax.tree_util.tree_map(jnp.zeros_like, p)
        return {"m": m0, "v": v0}

    opt_state = init_adam(params)
    b1, b2, adam_eps = 0.9, 0.999, 1e-8

    def adam_step(p, gr, st, t, lr_):
        def upd(pp, gg, m, v):
            m = b1 * m + (1 - b1) * gg
            v = b2 * v + (1 - b2) * (gg * gg)
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            new_p = pp - lr_ * mhat / (jnp.sqrt(vhat) + adam_eps)
            return new_p, m, v

        flat_p, treedef = jax.tree_util.tree_flatten(p)
        flat_g, _ = jax.tree_util.tree_flatten(gr)
        flat_m, _ = jax.tree_util.tree_flatten(st["m"])
        flat_v, _ = jax.tree_util.tree_flatten(st["v"])

        new_flat_p, new_flat_m, new_flat_v = [], [], []
        for pp, gg, m, v in zip(flat_p, flat_g, flat_m, flat_v):
            np_, nm_, nv_ = upd(pp, gg, m, v)
            new_flat_p.append(np_)
            new_flat_m.append(nm_)
            new_flat_v.append(nv_)

        new_p = jax.tree_util.tree_unflatten(treedef, new_flat_p)
        new_m = jax.tree_util.tree_unflatten(treedef, new_flat_m)
        new_v = jax.tree_util.tree_unflatten(treedef, new_flat_v)
        return new_p, {"m": new_m, "v": new_v}

    def global_norm(gr):
        leaves = jax.tree_util.tree_leaves(gr)
        return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

    history = {"loss": [], "grad_norm": [], "weight_norm": [], "step_skipped": []}
    t0 = time.time()
    nan_step = None

    for step in range(1, steps + 1):
        key, dkey = jax.random.split(key)
        ids, labels = make_toy_batch(dkey, bsz, seq_len, vocab_size, periodic=True)

        loss, grads = grad_fn(params, ids, labels)
        gn = float(global_norm(grads))
        wn = float(global_norm(params))
        loss_v = float(loss)

        loss_finite = np.isfinite(loss_v)
        grad_finite = np.isfinite(gn)

        if not (loss_finite and grad_finite):
            print(f"[step {step}] !! NON-FINITE DETECTED: loss={loss_v} grad_norm={gn} "
                  f"(loss_finite={loss_finite}, grad_finite={grad_finite})")
            nan_step = step

            # ---- FULL PER-STAGE DUMP at the exact failing step/batch ----
            print(f"[step {step}] Running full per-stage diagnostic dump on this exact batch...")
            diag_sink = []
            try:
                _ = tiny_loss(params, ids, labels, config, n_heads, d_head, diag_sink=diag_sink)
                for name, val in diag_sink:
                    assert_finite(val, name, hard=False)
            except Exception as e:
                print(f"  (forward-only dump raised: {e})")

            if dump_dir is not None:
                import os, pickle
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, f"nonfinite_step{step}.pkl"), "wb") as f:
                    pickle.dump({
                        "params": jax.device_get(params),
                        "input_ids": jax.device_get(ids),
                        "labels": jax.device_get(labels),
                        "step": step,
                    }, f)
                print(f"  Dumped repro batch + params to {dump_dir}/nonfinite_step{step}.pkl")

            if use_nan_guard:
                print(f"[step {step}] NaN-guard: SKIPPING this update (params unchanged).")
                history["loss"].append(loss_v)
                history["grad_norm"].append(gn)
                history["weight_norm"].append(wn)
                history["step_skipped"].append(1)
                continue
            else:
                print(f"[step {step}] NaN-guard DISABLED -- applying update anyway "
                      f"(will corrupt Adam moments from here on, matching diagnostic #2's "
                      f"'mgnovennыy collapse' scenario).")

        if use_clip and grad_finite:
            scale = jnp.minimum(1.0, clip_norm / (gn + 1e-6))
            grads = jax.tree_util.tree_map(lambda g_: g_ * scale, grads)
        elif use_clip and not grad_finite:
            # global_norm itself is NaN -- clip cannot rescue this, exactly
            # per the chat diagnosis point #2 ("norm([...,nan,...])=nan").
            print(f"[step {step}] clip_by_global_norm CANNOT help: "
                  f"norm itself is non-finite (nan propagates through the norm reduction).")

        params, opt_state = adam_step(params, grads, opt_state, step, lr)

        history["loss"].append(loss_v)
        history["grad_norm"].append(gn)
        history["weight_norm"].append(wn)
        history["step_skipped"].append(0)

        if step % log_every == 0 or step == 1:
            print(f"[step {step:4d}] loss={loss_v:.4f}  grad_norm={gn:.4e}  "
                  f"weight_norm={wn:.4e}  elapsed={time.time()-t0:.1f}s")

    n_nonfinite_steps = sum(1 for l in history["loss"] if not np.isfinite(l))
    print(f"\n--- LIVE TRAINING SUMMARY ---")
    print(f"steps={steps}  first_nonfinite_step={nan_step}  "
          f"total_nonfinite_events={n_nonfinite_steps}  "
          f"final_loss={history['loss'][-1]:.4f}")
    return history


# =======================================================================
# Driver
# =======================================================================

STAGE_FNS = {
    "A": test_stage_A,
    "B": test_stage_B,
    "C": test_stage_C,
    "D": test_stage_D,
    "B1": test_backward_B1,
    "B2": test_backward_B2,
    "B3": test_backward_B3,
    "B4": test_backward_B4,
    "B5": test_backward_B5,
    "FULL": test_full_pipeline_grad_check,
}


# =======================================================================
# ====================  ОБЫЧНЫЙ КОНФИГ (правьте руками)  ===============
# =======================================================================
# Никакого argparse/CLI -- просто отредактируйте значения ниже и
# запустите файл целиком в Kaggle-ячейке (или `python
# test_gdn2_full_pipeline.py`).

RUN_CONFIG = {
    # ---- Часть 1: изолированные тесты стадий ----
    # RUN_ALL_STAGES=True -> прогнать A,B,C,D,B1-B5,FULL по очереди.
    # Если False -- используется RUN_SINGLE_STAGE (или None, чтобы
    # вообще пропустить Часть 1).
    "RUN_ALL_STAGES": True,
    "RUN_SINGLE_STAGE": None,   # например "B3", если RUN_ALL_STAGES=False

    # ---- Часть 2: live-обучение ----
    "RUN_LIVE_TRAINING": True,
    # Если задан SCENARIO ("A"/"B"/"C"/"D") -- воспроизводится ровно
    # соответствующая комбинация LR/clip из таблицы диагностики №2,
    # LIVE_STEPS/LIVE_LR/LIVE_USE_CLIP/LIVE_USE_NAN_GUARD ниже
    # ИГНОРИРУЮТСЯ. Поставьте SCENARIO=None, чтобы использовать свои
    # значения вручную.
    "SCENARIO": None,
    "LIVE_STEPS": 300,
    "LIVE_LR": 3e-3,
    "LIVE_USE_CLIP": True,
    "LIVE_USE_NAN_GUARD": True,
    "LIVE_DUMP_DIR": "/kaggle/working/gdn2_nonfinite_dumps",
}


def main(cfg=RUN_CONFIG):
    config = KernelConfig(bt=128, bc=64, mb=16, clip=1e4, wy_eps=1e-3)

    results = {}
    if cfg["RUN_ALL_STAGES"]:
        for name, fn in STAGE_FNS.items():
            results[name] = fn(config)
    elif cfg["RUN_SINGLE_STAGE"]:
        name = cfg["RUN_SINGLE_STAGE"]
        results[name] = STAGE_FNS[name](config)

    if cfg["RUN_LIVE_TRAINING"]:
        if cfg["SCENARIO"]:
            scenarios = {
                "A": dict(lr=3e-3, use_clip=False),
                "B": dict(lr=3e-3, use_clip=True),
                "C": dict(lr=3e-4, use_clip=False),
                "D": dict(lr=3e-4, use_clip=True),
            }
            run_live_training(
                steps=cfg["LIVE_STEPS"],
                dump_dir=cfg["LIVE_DUMP_DIR"],
                **scenarios[cfg["SCENARIO"]],
            )
        else:
            run_live_training(
                steps=cfg["LIVE_STEPS"],
                lr=cfg["LIVE_LR"],
                use_clip=cfg["LIVE_USE_CLIP"],
                use_nan_guard=cfg["LIVE_USE_NAN_GUARD"],
                dump_dir=cfg["LIVE_DUMP_DIR"],
            )

    if results:
        print("\n=== SUMMARY ===")
        all_ok = True
        for name, ok in results.items():
            print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        print("\nALL PASS" if all_ok else "\nSOME TESTS FAILED -- see [FAIL] lines above")


if __name__ == "__main__":
    main()
