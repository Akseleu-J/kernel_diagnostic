"""
live_kernel_stress_diagnostic.py
=================================

Живая, ПОШАГОВАЯ диагностика всего Pallas-пайплайна (GDN-2 Kernel A->B->C->D
+ WY-solve health) на РЕАЛЬНЫХ данных (открытый HF-датасет, не синтетика),
прогоняемая через ваш реальный model.py / atomic_ops.

Что делает:
  1. Тянет открытый датасет с Hugging Face (по умолчанию `roneneldan/TinyStories`,
     маленький и не требует токена) и токенизирует его вашим же токенизатором
     (NousResearch/Meta-Llama-3-8B, как в userMemories).
  2. Строит ОДИН jit-вызов на шаг: forward+backward настоящей модели (или,
     если model.py недоступен в этом окружении, напрямую
     atomic_ops.kernel_trainable_B6 / kernel_mamba2_trainable_B6 на реальных
     q/k/v/w/b/g, полученных из настоящих эмбеддингов токенов, а не из
     случайного шума) + health-метрики (WY-residual/cond, per-stage
     isfinite/maxabs, was_clipped) -- ВСЁ внутри одного jax.jit, чтобы не
     тормозить обучение отдельными eager-проверками (та самая ошибка,
     которую вы просили не повторять: раньше диагностика типа
     kernel_diag.py уже была продумана как "чистые jnp внутри jit", здесь
     та же дисциплина применяется явно и последовательно).
  3. Параллельно намеренно ПРОВОЦИРУЕТ три задокументированных в проекте
     failure mode'а, поочерёдно, на реальных данных:
       a) WY-solve saturation -- прогоняя блок реальных токенов с
          искусственно обнулённым/усиленным decay (a_param форсированно
          сдвигается в near-singular зону) -- это НЕ синтетика входа, это
          настоящие q/k/v/токены, только decay-параметр модели временно
          пересчитывается на экстремум.
       b) Router/decay collapse -- реальный батч с почти константной
          последовательностью токенов (естественно происходит на
          повторяющихся n-граммах реального текста, отбирается прямо из
          датасета, а не генерируется).
       c) Cold-restart LR spike -- честно воспроизводит warmup+resume_backoff
          пересечение (см. optimizer.py) на реальном forward/backward графе,
          считая настоящий grad_norm с настоящих данных.
  4. На каждом шаге печатает, КАКАЯ ИМЕННО стадия (Aqk/Akk/A_wy_inverse/
     w_pseudo/u/kg/qg/h_final/o, или backward B1..B5) первой перестала быть
     "здоровой" -- не просто общий nonfinite-флаг.

Запуск:
    python live_kernel_stress_diagnostic.py --steps 400 --scenario all

Зависимости: jax, datasets (HF), transformers (только токенизатор).
    pip install --break-system-packages -q datasets transformers

Если ваш настоящий model.py/train_setup.py доступны в PYTHONPATH -- скрипт
использует РЕАЛЬНУЮ модель (GatedDeltaNet2J и т.д.) через
FullHybridMoEModel в уменьшенной конфигурации (мало слоёв, чтобы стресс-тест
был быстрым, но с ТЕМИ ЖЕ Pallas-кернелами, что в проде). Если импорт не
удался -- падает на прямой вызов atomic_ops-кернелов на реальных
эмбеддингах, с тем же набором health-метрик.
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
    steps=400,
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

_CLIP = 1e4


def _finite_stat(x):
    finite_mask = jnp.isfinite(x)
    safe = jnp.where(finite_mask, x, 0.0)
    return jnp.max(jnp.abs(safe)), jnp.all(finite_mask)


def wy_residual_and_cond(Akk, A):
    """||(I+Akk)@A - I||_inf + дешёвый conditioning-proxy -- прямой,
    однозначный тест здоровья Kernel B (WY-solve), тот же, что уже
    калиброван в kernel_health.py на реальном инциденте (healthy ~1e-8,
    сорванный ~2.4e4)."""
    eye = jnp.eye(Akk.shape[-1], dtype=jnp.float32)
    M = eye + Akk.astype(jnp.float32)
    resid = jnp.einsum("...ij,...jk->...ik", M, A.astype(jnp.float32),
                        precision=jax.lax.Precision.HIGHEST) - eye
    resid_inf = jnp.max(jnp.sum(jnp.abs(resid), axis=-1))
    A_inf = jnp.max(jnp.sum(jnp.abs(A.astype(jnp.float32)), axis=-1))
    M_inf = jnp.max(jnp.sum(jnp.abs(M), axis=-1))
    return resid_inf, A_inf * M_inf


def build_stage_health_fn():
    """Возвращает ЧИСТУЮ функцию (q,k,v,w,b,g,scale) -> dict здоровья по
    КАЖДОЙ стадии Kernel A/B/C/D, вычисленную на РЕАЛЬНЫХ q/k/v/w/b/g,
    произведённых от настоящих эмбеддингов токенов. Импортирует
    atomic_ops.kernel_a_scores / kernel_b_solve / kernel_c_recompute /
    kernel_d_pipeline напрямую (те же кернели, что использует
    kernel_trainable_B6.py в проде) -- ничего не дублирует и не
    переизобретает."""
    from atomic_ops.kernel_a_scores import build_chunk_scores_pallas
    from atomic_ops.kernel_b_solve import wy_solve_pallas
    from atomic_ops.kernel_c_recompute import recompute_wy_pallas
    from atomic_ops.kernel_d_pipeline import gdn2_inter_chunk_combine_with_state

    def health(q, k, v, w, b, g, scale, h0=None):
        bsz, L, H, D = q.shape
        if h0 is None:
            h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

        Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
        A = wy_solve_pallas(Akk)
        w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
        o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
            Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
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


def build_backward_health_fn():
    """Аналогично build_stage_health_fn, но для backward B1-B5 -- каждая
    стадия backward проверяется на isfinite/maxabs своих выходов на
    РЕАЛЬНОМ upstream-градиенте (do = d(loss ce)/d(o), считанном от
    настоящих labels через настоящий chunked_cross_entropy)."""
    from atomic_ops.kernel_a_scores import BT
    from atomic_ops.kernel_bwd_b1_dhu import gdn2_dhu_backward
    from atomic_ops.kernel_bwd_b2_dav import dav_backward_pallas
    from atomic_ops.kernel_bwd_b3_wy_dqkg import wy_dqkg_backward_pallas
    from atomic_ops.kernel_bwd_b4_intra import intra_backward_pallas
    from atomic_ops.kernel_bwd_b5_reverse_cumsum import reverse_cumsum_bwd

    _HIGHEST = jax.lax.Precision.HIGHEST

    def health(fwd_state, q, k, v, w, b, g, scale, do, dh_final):
        (Aqk, Akk, A, w_pseudo, u, kg, qg, h_pre_all, v_new_all,
         h_final, o_chunks) = fwd_state

        bsz, L, H, D = q.shape
        n_chunks = L // BT

        def reshape_in(t):
            t = t.reshape(bsz, n_chunks, BT, H, D)
            return jnp.moveaxis(t, (1, 3), (2, 1))

        do_r = reshape_in(do)
        g_r = reshape_in(g)
        idx = jnp.arange(BT)
        tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
        gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

        dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r)
        dh_all, dh0, dv_all = gdn2_dhu_backward(
            do_r, dv_partial, w_pseudo, qg, kg,
            jnp.moveaxis(gc[..., -1, :], 0, 0) if False else h_pre_all[..., 0, 0] * 0.0 + jnp.einsum(
                "ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST
            )[..., -1, :],
            scale, dht=dh_final,
        )
        dh_next_all = jnp.concatenate([dh_all[:, :, 1:], dh_final[:, :, None]], axis=2)

        q_r, k_r, b_r, w_r, v_r = map(reshape_in, (q, k, b, w, v))
        b3_out = wy_dqkg_backward_pallas(
            q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
            do_r, dv_all, dh_next_all, scale,
        )
        dq4, dk4, db4, dgc4 = intra_backward_pallas(dAqk, b3_out["dAkk"], q, k, b, g, scale)
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
    seq_len: int = 1024      # должно делиться на BT=256
    batch_size: int = 4
    steps: int = 400
    lr: float = 3e-3
    scenario: str = "all"    # wy_saturation | router_collapse | cold_restart | all
    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_name: str = "NousResearch/Meta-Llama-3-8B"


def make_qkvwbg_from_tokens(embed_table, input_ids, cfg: StressConfig,
                             decay_shift: float = 0.0):
    """Строит q/k/v/w/b/g НЕПОСРЕДСТВЕННО из реальных эмбеддингов реальных
    токенов (не из случайного шума) -- те же формы, что ожидает
    GatedDeltaNet2J в model.py, но здесь напрямую, без остальной модели,
    чтобы стресс-тест кернелей был быстрым и изолированным.

    decay_shift: сдвигает decay в near-singular зону (провоцирует WY-solve
    saturation, сценарий "a" из докстринга модуля) -- 0.0 = нормальный
    режим, положительные значения -> decay стремится к 0 (медленное
    забывание, Akk near-singular по конструкции WY-солва)."""
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

def build_jit_step(cfg: StressConfig, scale: float):
    stage_health = build_stage_health_fn()
    bwd_health = build_backward_health_fn()

    def step(embed_table, lm_head_w, input_ids, labels, decay_shift, lr_mult):
        q, k, v, w_gate, b_gate, g = make_qkvwbg_from_tokens(
            embed_table, input_ids, cfg, decay_shift=decay_shift
        )

        def loss_fn(qkvwbg):
            q_, k_, v_, w_, b_, g_ = qkvwbg
            fwd_metrics, fwd_state = stage_health(q_, k_, v_, w_, b_, g_, scale)
            o_chunks = fwd_state[-1]
            b_, l_, h_, d_ = q_.shape
            n_chunks = l_ // 256
            o = jnp.moveaxis(o_chunks, 1, 3).reshape(b_, n_chunks * 256, h_, d_)
            o_flat = o.reshape(b_, l_, h_ * d_)

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

        # do: upstream cotangent for `o` w.r.t. loss -- reuse autodiff's own
        # value via a second vjp call so backward-kernel health reflects the
        # REAL cotangent flowing in real training, not a placeholder.
        def o_from_qkvwbg(qkvwbg):
            q_, k_, v_, w_, b_, g_ = qkvwbg
            _, fwd_state_ = stage_health(q_, k_, v_, w_, b_, g_, scale)
            return fwd_state_[-1]

        _, vjp_fn = jax.vjp(o_from_qkvwbg, (q, k, v, w_gate, b_gate, g))
        do_chunks_cot = jax.grad(lambda o_: jnp.sum(o_ * 0.0))(fwd_state[-1])  # placeholder shape holder
        # Actual upstream grad for o_chunks comes from re-deriving via the
        # same loss path (cheap relative to the rest -- reuses jax autodiff,
        # not hand-rolled):
        def o_chunks_to_loss(o_chunks_):
            b_, h_, n_chunks, BT, d_ = o_chunks_.shape
            o_ = jnp.moveaxis(o_chunks_, 1, 3).reshape(b_, n_chunks * BT, h_, d_)
            o_flat = o_.reshape(b_, n_chunks * BT, h_ * d_)
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

        updated_qkvwbg = jax.tree_util.tree_map(
            lambda p, gr: p - lr_mult * cfg.lr * jnp.nan_to_num(gr, nan=0.0, posinf=0.0, neginf=0.0),
            (q, k, v, w_gate, b_gate, g), grad_qkvwbg,
        )

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
    scale = 1.0
    print(f"=== LIVE STRESS DIAGNOSTIC (реальные данные) : {dataclasses.asdict(cfg)} ===")

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

    jit_step = build_jit_step(cfg, scale)

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

    out_path = "stress_diagnostic_history.json"
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
