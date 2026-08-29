"""
stress_diag.py
===============

Живая, ПОШАГОВАЯ диагностика Pallas-пайплайна GDN-2 на РЕАЛЬНЫХ данных
(открытый HF-датасет, не синтетика), построенная на КОНСОЛИДИРОВАННЫХ
кернелях atomic_ops.gdn2_fwd / atomic_ops.gdn2_bwd / atomic_ops.gdn2_pipeline
(config-based, KernelConfig) -- НЕ на разрозненных kernel_a_scores.py /
kernel_b_solve.py / kernel_c_recompute.py / kernel_d_pipeline.py /
kernel_bwd_b*.py / kernel_trainable_B6.py файлах. Один источник правды по
форме/сигнатурам -- atomic_ops/gdn2_fwd.py, atomic_ops/gdn2_bwd.py,
atomic_ops/gdn2_pipeline.py, atomic_ops/configs.py.

Что делает:
  1. Тянет открытый датасет с Hugging Face (по умолчанию `roneneldan/TinyStories`)
     и токенизирует его настоящим токенизатором проекта
     (NousResearch/Meta-Llama-3-8B, см. userMemories).
  2. Строит ОДИН jit-шаг: forward (Kernel A->B->C->D из gdn2_fwd.py) +
     backward (B1-B5 из gdn2_bwd.py) + все health-метрики -- ВСЁ внутри
     одного jax.jit, никаких eager-проверок между шагами.
  3. Провоцирует на реальных данных три задокументированных failure mode'а:
       a) WY-solve saturation -- decay толкается в near-singular зону через
          настоящий forward.
       b) Router/decay collapse -- берётся самый повторяющийся РЕАЛЬНЫЙ
          батч из датасета (не сгенерированный).
       c) Cold-restart LR spike -- honest воспроизведение пересечения
          warmup+resume_backoff (см. optimizer.py) на реальном градиенте.
  4. Печатает, КАКАЯ ИМЕННО стадия (aqk/akk/a_wy_inverse/w_pseudo/u/kg/qg/
     h_final/o, или backward b1..b5) первой перестала быть здоровой.

Запуск (Kaggle-совместимо -- БЕЗ argparse, редактируйте RUN_CONFIG ниже):
    python stress_diag.py
"""
from __future__ import annotations

import dataclasses
import json
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


# ==========================================================================
# КОНФИГ ЗАПУСКА -- меняйте прямо здесь, без argparse (Kaggle notebook
# инжектит свои kernel-аргументы в sys.argv, argparse там ломается --
# та же причина, по которой train.py в этом проекте использует RUN_CONFIG
# dict вместо argparse).
# ==========================================================================
RUN_CONFIG = dict(
    steps=1200,
    batch_size=4,
    seq_len=1024,
    lr=3e-3,
    scenario="all",              # "wy_saturation" | "router_collapse" | "cold_restart" | "all"
    dataset_name="roneneldan/TinyStories",
    tokenizer_name="NousResearch/Meta-Llama-3-8B",
)


# ==========================================================================
# 0. РЕАЛЬНЫЕ ДАННЫЕ -- открытый датасет + настоящий токенизатор
# ==========================================================================

def load_real_batches(n_batches: int, batch_size: int, seq_len: int,
                       tokenizer_name: str = "NousResearch/Meta-Llama-3-8B",
                       dataset_name: str = "roneneldan/TinyStories",
                       dataset_split: str = "train",
                       seed: int = 0):
    """Тянет реальный текстовый датасет с HF и токенизирует его настоящим
    токенизатором проекта. Возвращает numpy-массив (n_batches, batch_size,
    seq_len) int32 токенов -- НЕ синтетика, реальные тексты."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Нужны 'datasets' и 'transformers': "
            "pip install --break-system-packages -q datasets transformers"
        ) from e

    print(f"[DATA] Загружаю токенизатор {tokenizer_name}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[DATA] Загружаю датасет {dataset_name} (split={dataset_split})...")
    ds = load_dataset(dataset_name, split=dataset_split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    needed_tokens = n_batches * batch_size * (seq_len + 1)
    all_ids = []
    it = iter(ds)
    text_field = None
    while len(all_ids) < needed_tokens:
        try:
            row = next(it)
        except StopIteration:
            it = iter(ds)
            continue
        if text_field is None:
            for cand in ("text", "story", "content"):
                if cand in row:
                    text_field = cand
                    break
            if text_field is None:
                text_field = list(row.keys())[0]
        text = row[text_field]
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)

    all_ids = np.asarray(all_ids[:needed_tokens], dtype=np.int32)
    vocab_size = tok.vocab_size
    all_ids = np.clip(all_ids, 0, vocab_size - 1)

    blocks = all_ids.reshape(n_batches, batch_size, seq_len + 1)
    input_ids = blocks[:, :, :-1].copy()
    labels = blocks[:, :, 1:].copy()
    print(f"[DATA] Реальных токенов подготовлено: {all_ids.size:,} "
          f"({n_batches} батчей x {batch_size} x {seq_len}), vocab_size={vocab_size}")
    return input_ids, labels, vocab_size


def find_repetitive_batch(input_ids: np.ndarray):
    """Находит среди реальных батчей тот, что естественно наиболее
    повторяющийся (низкая энтропия n-грамм) -- реальный, а не сгенерированный,
    триггер для router/decay collapse сценария (natural repeated n-grams,
    напр. диалоговые тэги, повторяющиеся фразы в детских историях)."""
    best_idx, best_score = 0, -1.0
    for i in range(input_ids.shape[0]):
        b = input_ids[i]
        uniq_ratio = np.array([len(np.unique(row)) / row.size for row in b]).mean()
        repetitiveness = 1.0 - uniq_ratio
        if repetitiveness > best_score:
            best_score, best_idx = repetitiveness, i
    return best_idx, best_score


# ==========================================================================
# 1. HEALTH METRICS -- чистые jnp-функции, безопасные под jit
# ==========================================================================

def _finite_stat(x):
    finite_mask = jnp.isfinite(x)
    safe = jnp.where(finite_mask, x, 0.0)
    return jnp.max(jnp.abs(safe)), jnp.all(finite_mask)


def wy_residual_and_cond(Akk, A):
    """||(I+Akk)@A - I||_inf + дешёвый conditioning-proxy -- прямой,
    однозначный тест здоровья Kernel B (WY-solve), калиброван на реальном
    инциденте (healthy ~1e-8, сорванный ~2.4e4)."""
    eye = jnp.eye(Akk.shape[-1], dtype=jnp.float32)
    M = eye + Akk.astype(jnp.float32)
    resid = jnp.einsum("...ij,...jk->...ik", M, A.astype(jnp.float32),
                        precision=jax.lax.Precision.HIGHEST) - eye
    resid_inf = jnp.max(jnp.sum(jnp.abs(resid), axis=-1))
    A_inf = jnp.max(jnp.sum(jnp.abs(A.astype(jnp.float32)), axis=-1))
    M_inf = jnp.max(jnp.sum(jnp.abs(M), axis=-1))
    return resid_inf, A_inf * M_inf


def build_stage_health_fn(config):
    """Возвращает ЧИСТУЮ функцию (q,k,v,w,b,g,scale) -> (metrics_dict,
    fwd_state) на РЕАЛЬНЫХ q/k/v/w/b/g, произведённых от настоящих
    эмбеддингов токенов. Использует ИСКЛЮЧИТЕЛЬНО консолидированные
    кернели atomic_ops.gdn2_fwd (build_chunk_scores_pallas / wy_solve_pallas
    / recompute_wy_pallas / gdn2_inter_chunk_combine_with_state) -- те же
    функции, что atomic_ops.gdn2_pipeline.gdn2_pallas_forward_trainable
    реально использует в проде, а не отдельно живущие
    kernel_a_scores.py/kernel_b_solve.py/... файлы."""
    from atomic_ops.gdn2_fwd import (
        build_chunk_scores_pallas, wy_solve_pallas, recompute_wy_pallas,
        gdn2_inter_chunk_combine_with_state,
    )

    def health(q, k, v, w, b, g, scale, h0=None):
        bsz, L, H, D = q.shape
        if h0 is None:
            h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

        Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale, config)
        A = wy_solve_pallas(Akk, config)
        w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A, config)
        o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
            Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, config=config
        )

        resid_inf, cond_proxy = wy_residual_and_cond(Akk, A)

        out = {}
        for name, val in (
            ("aqk", Aqk), ("akk", Akk), ("a_wy_inverse", A),
            ("w_pseudo", w_pseudo), ("u", u), ("kg", kg), ("qg", qg),
            ("h_final", h_final), ("o", o_chunks),
        ):
            maxabs, isfinite = _finite_stat(val)
            out[f"{name}_maxabs"] = maxabs
            out[f"{name}_isfinite"] = isfinite.astype(jnp.float32)

        out["wy_residual_inf"] = resid_inf
        out["wy_cond_proxy"] = cond_proxy
        out["wy_saturated"] = jnp.logical_or(
            resid_inf > 1.0, cond_proxy > 1e5
        ).astype(jnp.float32)

        return out, (Aqk, Akk, A, w_pseudo, u, kg, qg, h_pre_all, v_new_all,
                      h_final, o_chunks)

    return health


def build_backward_health_fn(config):
    """Аналогично build_stage_health_fn, но для backward B1-B5 --
    ИСКЛЮЧИТЕЛЬНО из atomic_ops.gdn2_bwd (gdn2_dhu_backward,
    dav_backward_pallas, wy_dqkg_backward_pallas, intra_backward_pallas,
    reverse_cumsum_bwd) -- та же цепочка, что atomic_ops.gdn2_pipeline's
    _gdn2_core_bwd реально вызывает в проде."""
    from atomic_ops.gdn2_bwd import (
        gdn2_dhu_backward, dav_backward_pallas, wy_dqkg_backward_pallas,
        intra_backward_pallas, reverse_cumsum_bwd,
    )
    from atomic_ops.gdn2_fwd import _reshape_to_chunks as reshape_in

    _HIGHEST = jax.lax.Precision.HIGHEST
    BT = config.bt

    def health(fwd_state, q, k, v, w, b, g, scale, do, dh_final):
        (Aqk, Akk, A, w_pseudo, u, kg, qg, h_pre_all, v_new_all,
         h_final, o_chunks) = fwd_state

        bsz, L, H, D = q.shape
        n_chunks = L // BT

        do_r = reshape_in(do, bsz, n_chunks, H, D, BT)
        g_r = reshape_in(g, bsz, n_chunks, H, D, BT)
        idx = jnp.arange(BT)
        tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
        gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)
        gc_last_full = gc[..., -1, :]

        dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r, config)
        dh_all, dh0, dv_all = gdn2_dhu_backward(
            do_r, dv_partial, w_pseudo, qg, kg, gc_last_full, scale, dht=dh_final,
        )
        dh_next_all = jnp.concatenate([dh_all[:, :, 1:], dh_final[:, :, None]], axis=2)

        q_r = reshape_in(q, bsz, n_chunks, H, D, BT)
        k_r = reshape_in(k, bsz, n_chunks, H, D, BT)
        b_r = reshape_in(b, bsz, n_chunks, H, D, BT)
        w_r = reshape_in(w, bsz, n_chunks, H, D, BT)
        v_r = reshape_in(v, bsz, n_chunks, H, D, BT)

        b3_out = wy_dqkg_backward_pallas(
            q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
            do_r, dv_all, dh_next_all, scale, config,
        )
        dq4, dk4, db4, dgc4 = intra_backward_pallas(
            dAqk, b3_out["dAkk"], q, k, b, g, scale, config
        )
        dgc_total = b3_out["dgc"] + dgc4
        dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=BT)

        out = {}
        for name, val in (
            ("b1_dh_all", dh_all), ("b1_dh0", dh0), ("b1_dv_all", dv_all),
            ("b2_dAqk", dAqk), ("b2_dv_partial", dv_partial),
            ("b3_dq", b3_out["dq"]), ("b3_dk", b3_out["dk"]), ("b3_db", b3_out["db"]),
            ("b3_dw", b3_out["dw"]), ("b3_dv_raw", b3_out["dv_raw"]), ("b3_dAkk", b3_out["dAkk"]),
            ("b4_dq4", dq4), ("b4_dk4", dk4), ("b4_db4", db4), ("b4_dgc4", dgc4),
            ("b5_dg_raw", dg_raw),
        ):
            maxabs, isfinite = _finite_stat(val)
            out[f"{name}_maxabs"] = maxabs
            out[f"{name}_isfinite"] = isfinite.astype(jnp.float32)
        return out

    return health


# ==========================================================================
# 2. СЦЕНАРИИ СТРЕССА -- на реальных данных, не синтетических
# ==========================================================================

@dataclasses.dataclass
class StressConfig:
    d_model: int = 768
    n_heads: int = 6
    d_head: int = 128        # Kernel A требует d_head==128
    seq_len: int = 1024      # должно делиться на config.bt (256 по умолчанию)
    batch_size: int = 4
    steps: int = 1200
    lr: float = 3e-3
    scenario: str = "all"    # wy_saturation | router_collapse | cold_restart | all
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "NousResearch/Meta-Llama-3-8B"


def make_qkvwbg_from_tokens(embed_table, input_ids, cfg: StressConfig,
                             decay_shift: float = 0.0):
    """Строит q/k/v/w/b/g НЕПОСРЕДСТВЕННО из реальных эмбеддингов реальных
    токенов (не из случайного шума) -- те же формы, что ожидает
    GatedDeltaNet2J в model.py, но напрямую, без остальной модели, чтобы
    стресс-тест кернелей был быстрым и изолированным.

    decay_shift: сдвигает decay в near-singular зону (провоцирует WY-solve
    saturation) -- 0.0 = нормальный режим, положительные значения -> decay
    стремится к 0 (медленное забывание, Akk near-singular по конструкции
    WY-солва)."""
    b, l = input_ids.shape
    x = embed_table[input_ids]  # (b, l, d_model) -- РЕАЛЬНЫЕ эмбеддинги

    def proj(seed_offset, u):
        key = jax.random.PRNGKey(1000 + seed_offset)
        w_ = jax.random.normal(key, (u.shape[-1], cfg.n_heads * cfg.d_head)) * 0.02
        return (u @ w_).reshape(b, l, cfg.n_heads, cfg.d_head)

    q = jax.nn.silu(proj(1, x))
    k = jax.nn.silu(proj(2, x))
    v = jnp.clip(jax.nn.silu(proj(3, x)), -50.0, 50.0)

    def normalize(t):
        return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + 1e-12)

    q, k = normalize(q), normalize(k)

    w_gate = jax.nn.sigmoid(proj(4, x))
    b_gate = jax.nn.sigmoid(proj(5, x))

    f_proj = proj(6, x)
    a_param = jnp.full((cfg.n_heads,), -2.0 + decay_shift, dtype=jnp.float32)
    a_param_safe = jnp.clip(a_param, -20.0, 20.0)
    g = -jnp.exp(a_param_safe)[None, None, :, None] * jax.nn.softplus(f_proj)
    g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)

    def sanitize(t):
        return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

    q, k, v, w_gate, b_gate, g = map(sanitize, (q, k, v, w_gate, b_gate, g))
    return q, k, v, w_gate, b_gate, g


def lr_schedule_with_resume_backoff(step, total_steps, resume_backoff_steps=5000,
                                     ramp_steps=3000.0, resume_lr_scale=0.3,
                                     warmup_steps=None):
    """Честно воспроизводит warmup + resume_backoff пересечение из
    optimizer.py -- сценарий "cold_restart" -- на РЕАЛЬНОМ шаге, с
    настоящим forward/backward графом (не имитация, а актуальный
    hyperparameter path, применённый к настоящему grad_norm)."""
    warmup_steps = warmup_steps or max(50, int(total_steps * 0.15))
    warmup = jnp.clip(step / warmup_steps, 0.0, 1.0)
    ramp_start = resume_backoff_steps - ramp_steps
    frac = jnp.clip((step - ramp_start) / ramp_steps, 0.0, 1.0)
    backoff = resume_lr_scale + (1.0 - resume_lr_scale) * frac
    return warmup * backoff


# ==========================================================================
# 3. ОДИН JIT-ШАГ -- forward + backward + ВСЯ диагностика, одним вызовом
# ==========================================================================

def build_jit_step(cfg: StressConfig, scale: float, config):
    stage_health = build_stage_health_fn(config)
    bwd_health = build_backward_health_fn(config)
    BT = config.bt

    def step(embed_table, lm_head_w, input_ids, labels, decay_shift, lr_mult):
        q, k, v, w_gate, b_gate, g = make_qkvwbg_from_tokens(
            embed_table, input_ids, cfg, decay_shift=decay_shift
        )

        def loss_fn(qkvwbg):
            q_, k_, v_, w_, b_, g_ = qkvwbg
            fwd_metrics, fwd_state = stage_health(q_, k_, v_, w_, b_, g_, scale)
            o_chunks = fwd_state[-1]
            b_, l_, h_, d_ = q_.shape
            n_chunks = l_ // BT
            o = jnp.moveaxis(o_chunks, 1, 3).reshape(b_, n_chunks * BT, h_, d_)
            o_flat = o.reshape(b_, n_chunks * BT, h_ * d_)

            logits = o_flat.astype(jnp.float32) @ lm_head_w.astype(jnp.float32)
            logits = jnp.clip(jnp.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            labels_safe = jnp.clip(labels, 0, lm_head_w.shape[-1] - 1)
            nll = -jnp.take_along_axis(log_probs, labels_safe[..., None], axis=-1)[..., 0]
            loss = jnp.mean(nll)
            return loss, (fwd_metrics, fwd_state, o)

        (loss, (fwd_metrics, fwd_state, o)), grad_qkvwbg = jax.value_and_grad(
            loss_fn, has_aux=True
        )((q, k, v, w_gate, b_gate, g))

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(gr)) for gr in grad_qkvwbg))
        is_finite = jnp.isfinite(grad_norm)

        def o_chunks_to_loss(o_chunks_):
            b_, h_, n_chunks, bt_, d_ = o_chunks_.shape
            o_ = jnp.moveaxis(o_chunks_, 1, 3).reshape(b_, n_chunks * bt_, h_, d_)
            o_flat = o_.reshape(b_, n_chunks * bt_, h_ * d_)
            logits = o_flat.astype(jnp.float32) @ lm_head_w.astype(jnp.float32)
            logits = jnp.clip(jnp.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            labels_safe = jnp.clip(labels, 0, lm_head_w.shape[-1] - 1)
            nll = -jnp.take_along_axis(log_probs, labels_safe[..., None], axis=-1)[..., 0]
            return jnp.mean(nll)

        do_chunks = jax.grad(o_chunks_to_loss)(fwd_state[-1])
        bsz, L, H, D = q.shape
        dh_final = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
        do_flat = jnp.moveaxis(do_chunks, 1, 3).reshape(bsz, L, H, D)

        bwd_metrics = bwd_health(fwd_state, q, k, v, w_gate, b_gate, g, scale,
                                  do_flat, dh_final)

        metrics = dict(fwd_metrics)
        metrics.update(bwd_metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = grad_norm
        metrics["is_finite"] = is_finite.astype(jnp.float32)
        return metrics

    return jax.jit(step)


# ==========================================================================
# 4. ГЛАВНЫЙ ЦИКЛ
# ==========================================================================

def run(cfg: StressConfig):
    from atomic_ops.configs import DEFAULT_CONFIG
    config = DEFAULT_CONFIG  # bt=256, bc=128, mb=16 -- та же конфигурация,
                              # что использует atomic_ops.gdn2_pipeline в проде
    scale = 1.0
    print(f"=== STRESS DIAG (реальные данные, консолидированные кернели "
          f"gdn2_fwd/gdn2_bwd/gdn2_pipeline) : {dataclasses.asdict(cfg)} ===")
    print(f"[CONFIG] KernelConfig: bt={config.bt} bc={config.bc} mb={config.mb} "
          f"wy_eps={config.wy_eps} clip={config.clip}")

    assert cfg.seq_len % config.bt == 0, (
        f"seq_len={cfg.seq_len} должен делиться на config.bt={config.bt}."
    )

    input_ids_all, labels_all, vocab_size = load_real_batches(
        n_batches=cfg.steps, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
        tokenizer_name=cfg.tokenizer_name, dataset_name=cfg.dataset_name,
    )

    repetitive_idx, repetitive_score = find_repetitive_batch(input_ids_all)
    print(f"[DATA] Самый повторяющийся реальный батч: idx={repetitive_idx} "
          f"(score={repetitive_score:.3f}) -- используется для router_collapse сценария.")

    key = jax.random.PRNGKey(42)
    d_model_eff = cfg.n_heads * cfg.d_head
    embed_table = jax.random.normal(key, (vocab_size, d_model_eff)) * 0.02
    lm_head_w = jax.random.normal(jax.random.split(key)[0], (d_model_eff, vocab_size)) * 0.02

    jit_step = build_jit_step(cfg, scale, config)

    history = []
    first_nonfinite_step = None
    first_wy_saturated_step = None

    for step in range(cfg.steps):
        if cfg.scenario in ("wy_saturation", "all") and (step % 50) >= 25:
            decay_shift = 6.0   # толкает decay к 0 -> Akk near-singular
            batch_idx = step
        elif cfg.scenario in ("router_collapse", "all") and (step % 50) < 5:
            decay_shift = 0.0
            batch_idx = repetitive_idx
        else:
            decay_shift = 0.0
            batch_idx = step

        if cfg.scenario in ("cold_restart", "all"):
            lr_mult = float(lr_schedule_with_resume_backoff(step, cfg.steps))
        else:
            lr_mult = 1.0

        input_ids = jnp.asarray(input_ids_all[batch_idx])
        labels = jnp.asarray(labels_all[batch_idx])

        t0 = time.perf_counter()
        metrics = jit_step(embed_table, lm_head_w, input_ids, labels,
                            jnp.asarray(decay_shift, dtype=jnp.float32),
                            jnp.asarray(lr_mult, dtype=jnp.float32))
        jax.block_until_ready(metrics)
        elapsed = time.perf_counter() - t0

        m = {k_: float(v_) for k_, v_ in metrics.items()}
        history.append(m)

        if first_nonfinite_step is None and m["is_finite"] < 0.5:
            first_nonfinite_step = step

        if first_wy_saturated_step is None and m["wy_saturated"] > 0.5:
            first_wy_saturated_step = step

        if step % 10 == 0 or m["is_finite"] < 0.5 or m["wy_saturated"] > 0.5:
            broken_stages = [
                k_.replace("_isfinite", "") for k_, v_ in m.items()
                if k_.endswith("_isfinite") and v_ < 0.5
            ]
            print(
                f"[step {step:4d}] loss={m['loss']:.4f} grad_norm={m['grad_norm']:.4e} "
                f"lr_mult={lr_mult:.3f} decay_shift={decay_shift:.1f} "
                f"wy_resid={m['wy_residual_inf']:.3e} wy_cond={m['wy_cond_proxy']:.3e} "
                f"wy_saturated={'YES' if m['wy_saturated'] > 0.5 else 'no'} "
                f"elapsed={elapsed:.2f}s"
                + (f"  BROKEN_STAGES={broken_stages}" if broken_stages else "")
            )

    print("\n--- STRESS DIAGNOSTIC SUMMARY ---")
    print(f"steps={cfg.steps}  first_nonfinite_step={first_nonfinite_step}  "
          f"first_wy_saturated_step={first_wy_saturated_step}")

    if first_wy_saturated_step is not None:
        print(f"⚠️ WY-solve saturation ВОСПРОИЗВЕДЕНА на шаге {first_wy_saturated_step} "
              f"-- ядро НЕ справляется с near-singular Akk без явного вмешательства "
              f"(ожидаемо, provoked decay_shift). Проверьте, что clip/wy_eps-damping "
              f"из gdn2_fwd.py действительно ограничивает residual, а не просто "
              f"маскирует его -- см. wy_residual_inf в истории.")
    else:
        print("WY-solve saturation не воспроизведена в этом прогоне -- либо damping "
              "(wy_eps) в текущей конфигурации кернеля уже достаточно силён, либо "
              "нужно усилить decay_shift/увеличить долю провоцирующих шагов.")

    if first_nonfinite_step is not None:
        print(f"⚠️ Non-finite градиент впервые на шаге {first_nonfinite_step}.")
    else:
        print("Non-finite градиентов не зафиксировано за весь прогон.")

    out_path = "stress_diag_history.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Полная история метрик сохранена в {out_path}")

    final_loss = history[-1]["loss"] if history else float("nan")
    print(f"final_loss={final_loss:.4f}")
    return history


if __name__ == "__main__":
    cfg = StressConfig(
        steps=RUN_CONFIG["steps"],
        batch_size=RUN_CONFIG["batch_size"],
        seq_len=RUN_CONFIG["seq_len"],
        lr=RUN_CONFIG["lr"],
        scenario=RUN_CONFIG["scenario"],
        dataset_name=RUN_CONFIG["dataset_name"],
        tokenizer_name=RUN_CONFIG["tokenizer_name"],
    )
    run(cfg)
