"""
grid_bt_bc_condition_diag.py

Единый диагностический скрипт: проверяет гипотезу "BT/BC определяют
обусловленность Akk" БЕЗ полного обучения -- только прямой forward
Kernel A (build_chunk_scores_pallas) под фиксированными случайными
q/k/b/g и замер cond(Akk) через batched SVD, для сетки (BT, BC).

Дёшево: секунды на комбинацию, можно гонять на CPU (interpret=True) --
cond(Akk) чисто численный эффект самой арифметики, не зависит от того,
компилируется кернел через Mosaic на TPU или через pure-JAX интерпретатор
pl.pallas_call(interpret=True). ПОЭТОМУ ЭТОТ ФАЙЛ МОЖНО ГОНЯТЬ ЛОКАЛЬНО /
В ЛЮБОМ CPU-окружении с установленным jax -- TPU не нужен для основной
сетки (Часть 1).

Часть 2 (опциональная, ЗАКОММЕНТИРОВАНА ПО УМОЛЧАНИЮ, см. RUN_CONFIG) --
короткий train-прогон (несколько сотен шагов) РЕАЛЬНОГО Pallas-кернела
(interpret=False, нужен TPU) для лучших 2-3 кандидатов из Части 1, чтобы
подтвердить, что снижение cond(Akk) реально транслируется в
first_nonfinite_step/final_loss, а не только в "числа стали меньше" на
чистой диагностике.

Гоняется одной командой, БЕЗ argparse -- всё управляется через RUN_CONFIG
внизу файла (тот же паттерн config-dict, что train.py/train_setup.py):

    python grid_bt_bc_condition_diag.py

Результаты пишутся в grid_condition_results.json рядом со скриптом.

ГДЕ ГОНЯТЬ:
- Часть 1 (сетка BT x BC, только forward, interpret=True) -- ЛЮБОЕ
  окружение с jax: локально, Colab CPU, Kaggle CPU-сессия. TPU НЕ нужен.
  Это самый дешёвый и быстрый способ получить ответ на вопрос "помогает
  ли уменьшение BC/BT" -- рекомендуется прогнать ЭТО первым, до того как
  тратить TPU-время.
- Часть 2 (короткий train на реальном Pallas, interpret=False) -- нужен
  Kaggle TPU v5e-8, как и остальные *_real_kernel тесты в этом проекте.
  Включается флагом RUN_CONFIG["run_part2_on_tpu"] = True.
"""
from __future__ import annotations

import dataclasses
import json
import time

import jax
import jax.numpy as jnp

from Atomic_ops.configs import KernelConfig
from Atomic_ops.gdn2_fwd import build_chunk_scores_pallas, wy_solve_pallas


# ==========================================================================
# Общие утилиты
# ==========================================================================
# ФИКС (Kaggle CPU-окружение с несовместимыми numpy/scipy): jnp.linalg.svd
# на CPU-бэкенде тянет scipy.linalg как часть своей лидировки, и на
# сломанном образе (numpy._core.umath без '_center', ABI mismatch между
# numpy и scipy) падает ImportError ещё ДО того, как код диагностики
# вообще начинает работать -- см. traceback: JaxStackTraceBeforeTransformation
# -> jax/_src/numpy/linalg.py svd() -> ImportError: cannot import name
# '_center' from 'numpy._core.umath'.
#
# Обходим ЛЮБУЮ зависимость от LAPACK/scipy: обусловленность Akk через
# степенную итерацию на матрично-векторных умножениях (только jnp.matmul/
# einsum, лидируются в XLA напрямую, без scipy) -- power iteration для
# max sigma (top singular value через A^T A), и inverse power iteration
# для min sigma через jnp.linalg.solve (тоже LAPACK-based, но идёт другим
# путём кодогенерации, чем svd -- если solve тоже упадёт, см. фолбэк ниже
# через Neumann-ряд, который вообще не использует ни solve, ни svd).
def _power_iteration_max_sigma(AtA, iters=60, key=None):
    """AtA: (N, n, n) батч симметричных PSD матриц A^T A. Возвращает
    (N,) оценку top eigenvalue (== sigma_max(A)^2) через степенную
    итерацию -- чистые матмулы, без LAPACK/scipy."""
    N, n, _ = AtA.shape
    if key is None:
        key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (N, n))
    v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    def body(_, v):
        v = jnp.einsum("nij,nj->ni", AtA, v)
        v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
        return v

    v = jax.lax.fori_loop(0, iters, body, v)
    Av = jnp.einsum("nij,nj->ni", AtA, v)
    eig_max = jnp.sum(v * Av, axis=-1) / (jnp.sum(v * v, axis=-1) + 1e-12)
    return jnp.abs(eig_max)


def _neumann_min_sigma_proxy(A, iters=30):
    """Оценка НИЖНЕЙ границы sigma_min(A) БЕЗ solve/svd -- через ряд
    Неймана для (I - (I-A))^{-1} = sum (I-A)^k, сходится только если
    ||I-A||<1, что для наших unit-lower-triangular Akk-подобных матриц не
    гарантировано в общем случае. Поэтому это ГРУБЫЙ proxy (может
    недооценить реальный sigma_min, если ряд расходится) -- используется
    ТОЛЬКО как аварийный фолбэк, если jnp.linalg.solve тоже недоступен на
    сломанном окружении. Основной путь -- _akk_condition_stats ниже,
    который сначала пробует solve, и падает на этот proxy только при
    исключении."""
    N, n, _ = A.shape
    eye = jnp.eye(n, dtype=A.dtype)[None]
    diff = eye - A
    acc = eye
    term = eye
    for _ in range(iters):
        term = jnp.einsum("nij,njk->nik", term, diff)
        acc = acc + term
    # acc приближает A^{-1}; sigma_min(A) ~= 1 / sigma_max(A^{-1})
    AtA_inv_approx = jnp.einsum("nij,nik->njk", acc, acc)
    sigma_max_inv_sq = _power_iteration_max_sigma(AtA_inv_approx, iters=60)
    sigma_min = 1.0 / jnp.sqrt(jnp.maximum(sigma_max_inv_sq, 1e-24))
    return sigma_min


def _akk_condition_stats(Akk):
    """Akk: (bsz, H, n_chunks, BT, BT). Обусловленность БЕЗ jnp.linalg.svd
    (см. ФИКС выше) -- top sigma через power iteration на A^T A, bottom
    sigma через inverse power iteration на решении linear system
    (A^T A) x = v, с фолбэком на Neumann-ряд если solve недоступен на
    этом окружении (см. _neumann_min_sigma_proxy)."""
    Akk_f = Akk.astype(jnp.float32)
    shp = Akk_f.shape
    flat = Akk_f.reshape(-1, shp[-2], shp[-1])
    N, n, _ = flat.shape

    AtA = jnp.einsum("nij,nik->njk", flat, flat)
    eig_max = _power_iteration_max_sigma(AtA, iters=60)
    sigma_max = jnp.sqrt(jnp.maximum(eig_max, 0.0))

    try:
        # Inverse power iteration: (A^T A + eps*I) x_{k+1} = x_k, через
        # jnp.linalg.solve -- на CPU лидируется через LAPACK getrs, ЧАЩЕ
        # переживает окружения со сломанным scipy, чем svd (другой путь
        # кодогенерации), но не гарантированно -- отсюда try/except с
        # Neumann-фолбэком ниже.
        eps = 1e-6
        AtA_reg = AtA + eps * jnp.eye(n, dtype=AtA.dtype)[None]
        key = jax.random.PRNGKey(1)
        v = jax.random.normal(key, (N, n))
        v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

        def inv_body(_, v):
            v_new = jnp.linalg.solve(AtA_reg, v[..., None])[..., 0]
            v_new = v_new / (jnp.linalg.norm(v_new, axis=-1, keepdims=True) + 1e-12)
            return v_new

        v = jax.lax.fori_loop(0, 40, inv_body, v)
        Av = jnp.linalg.solve(AtA_reg, v[..., None])[..., 0]
        eig_min_inv = jnp.sum(v * Av, axis=-1) / (jnp.sum(v * v, axis=-1) + 1e-12)
        sigma_min = 1.0 / jnp.sqrt(jnp.maximum(eig_min_inv, 1e-24))
        sigma_min = jax.device_get(sigma_min)  # форсируем выполнение здесь, чтобы поймать ImportError сразу
    except Exception as e:
        print(f"[GRID] ⚠️ jnp.linalg.solve недоступен ({type(e).__name__}: {e}) -- "
              f"использую Neumann-ряд фолбэк для sigma_min (грубее, но не "
              f"зависит от LAPACK/scipy вообще).")
        sigma_min = jax.device_get(_neumann_min_sigma_proxy(flat, iters=30))

    sigma_max_host = jax.device_get(sigma_max)
    cond = sigma_max_host / jnp.maximum(sigma_min, 1e-12)
    cond = jnp.where(jnp.isfinite(cond), cond, 0.0)

    return (
        float(jnp.max(cond)), float(jnp.mean(cond)), float(jnp.median(cond)),
        float(jnp.max(sigma_max_host)), float(jnp.min(sigma_min)),
    )


def _make_fixed_inputs(seed, bsz, seq_len, n_heads, d_head, decay_scale):
    """Строит q/k/b/g того же вида, что реальная модель подаёт в Kernel A:
    L2-нормализованные q/k (как model.py's _safe_normalize), b в (0.2,1.0)
    (как erase_gate sigmoid-выход в реалистичном диапазоне), g -- лог-decay
    масштаба decay_scale (0.0 -- имитирует самое начало обучения, decay_a
    инициализирован в 0; 0.05 -- типичный масштаб после нескольких сотен
    шагов, по аналогии с ablation-тестом)."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (bsz, seq_len, n_heads, d_head)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    b = jax.random.uniform(k3, shape, minval=0.2, maxval=1.0)

    if decay_scale <= 0.0:
        g = jnp.zeros(shape, dtype=jnp.float32)
    else:
        g = -jnp.abs(jax.random.normal(k4, shape)) * decay_scale

    return q, k, b, g


# ==========================================================================
# Часть 1: сетка BT x BC, только forward Kernel A, interpret=True (CPU-ok)
# ==========================================================================
@dataclasses.dataclass(frozen=True)
class GridPoint:
    bt: int
    bc: int
    mb: int = 16


def run_grid_part1(cfg):
    bsz = cfg["bsz"]
    seq_len = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    d_head = cfg["d_head"]
    seeds = cfg["seeds"]
    decay_scales = cfg["decay_scales"]
    bt_values = cfg["bt_values"]
    bc_divisors = cfg["bc_divisors"]

    grid_points = []
    for bt in bt_values:
        assert seq_len % bt == 0, f"seq_len={seq_len} must be divisible by bt={bt}"
        for bc in bc_divisors:
            if bt % bc != 0:
                continue
            mb = min(16, bc)
            if bc % mb != 0:
                continue
            grid_points.append(GridPoint(bt=bt, bc=bc, mb=mb))

    print(f"[GRID] Всего точек сетки (bt,bc): {len(grid_points)}")
    for gp in grid_points:
        print(f"  bt={gp.bt} bc={gp.bc} mb={gp.mb}")

    results = []
    t_start = time.time()

    for gp in grid_points:
        config = KernelConfig(bt=gp.bt, bc=gp.bc, mb=gp.mb, use_centering=False, wy_eps=0.0)

        for decay_scale in decay_scales:
            per_seed_stats = []
            for seed in seeds:
                q, k, b, g = _make_fixed_inputs(seed, bsz, seq_len, n_heads, d_head, decay_scale)

                t0 = time.time()
                Aqk, Akk = build_chunk_scores_pallas(
                    q, k, b, g, scale=1.0, config=config, interpret=True
                )
                elapsed = time.time() - t0

                max_cond, mean_cond, median_cond, max_sigma, min_sigma = _akk_condition_stats(Akk)
                per_seed_stats.append({
                    "seed": seed,
                    "max_cond": max_cond, "mean_cond": mean_cond,
                    "median_cond": median_cond,
                    "max_sigma": max_sigma, "min_sigma": min_sigma,
                    "elapsed_s": elapsed,
                })

            agg_max_cond = max(s["max_cond"] for s in per_seed_stats)
            agg_mean_cond = sum(s["mean_cond"] for s in per_seed_stats) / len(per_seed_stats)

            entry = {
                "bt": gp.bt, "bc": gp.bc, "mb": gp.mb,
                "decay_scale": decay_scale,
                "n_chunks": seq_len // gp.bt,
                "agg_max_cond": agg_max_cond,
                "agg_mean_cond": agg_mean_cond,
                "per_seed": per_seed_stats,
            }
            results.append(entry)
            print(f"    bt={gp.bt:4d} bc={gp.bc:4d} decay_scale={decay_scale:.3f} "
                  f"-> max_cond={agg_max_cond:.3e}  mean_cond={agg_mean_cond:.3e}")

    total_elapsed = time.time() - t_start
    print(f"[GRID] Готово за {total_elapsed:.1f}с.")

    results_sorted = sorted(results, key=lambda r: r["agg_max_cond"])
    print("\n=== ТОП-5 лучших (bt,bc,decay_scale) по agg_max_cond ===")
    for r in results_sorted[:5]:
        print(f"  bt={r['bt']} bc={r['bc']} decay_scale={r['decay_scale']:.3f} "
              f"max_cond={r['agg_max_cond']:.3e} mean_cond={r['agg_mean_cond']:.3e}")

    print("\n=== ТОП-5 худших (bt,bc,decay_scale) по agg_max_cond ===")
    for r in results_sorted[-5:]:
        print(f"  bt={r['bt']} bc={r['bc']} decay_scale={r['decay_scale']:.3f} "
              f"max_cond={r['agg_max_cond']:.3e} mean_cond={r['agg_mean_cond']:.3e}")

    return results


# ==========================================================================
# Часть 1b: ось decay_a_init -- проверка гипотезы "decay_a инициализирован
# в 0.0 (model.py's nn.initializers.zeros) -> g~=0 на первом шаге -> Akk
# в самом плохом (наименее демпфированном decay'ем) режиме именно в
# момент старта обучения". В отличие от Части 1 (decay_scale двигает
# РАЗБРОС log-decay между токенами), здесь мы напрямую имитируем реальный
# a_param = self.param("decay_a", zeros, ...) со СКАЛЯРНЫМ сдвигом
# a_param_init -- т.е. тот же decay-масштаб, что реально задаёт
# model.py::GatedDeltaNet2J при a_param_init != 0 вместо инициализации в 0.
# ==========================================================================
def _make_fixed_inputs_with_decay_a(seed, bsz, seq_len, n_heads, d_head, a_param_init, f_proj_scale=0.5):
    """Имитирует РЕАЛЬНУЮ формулу из model.py::GatedDeltaNet2J:
        g = -exp(clip(a_param,-20,20)) * softplus(f_proj)
    вместо искусственного decay_scale*|normal| из _make_fixed_inputs.
    a_param_init -- то самое значение, в которое СЕЙЧАС инициализируется
    decay_a (nn.initializers.zeros -> 0.0); мы проверяем альтернативные
    инициализации (например, -2.0) вместо изменения кода модели напрямую."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (bsz, seq_len, n_heads, d_head)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    b = jax.random.uniform(k3, shape, minval=0.2, maxval=1.0)

    # f_proj -- имитация выхода decay_proj Dense-слоя на случайных весах
    # при случайном входе; масштаб f_proj_scale~0.5 -- типичный масштаб
    # выхода Dense со стандартной lecun_normal-инициализацией на
    # нормализованном входе (не точная копия, но достаточно для сравнения
    # РАЗНИЦЫ между a_param_init, при фиксированном f_proj_scale).
    f_proj = jax.random.normal(k4, shape) * f_proj_scale

    a_safe = jnp.clip(jnp.asarray(a_param_init, dtype=jnp.float32), -20.0, 20.0)
    g = -jnp.exp(a_safe) * jax.nn.softplus(f_proj)
    g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)

    return q, k, b, g


def run_grid_part1b_decay_init(cfg):
    """Фиксирует bt/bc на разумном дефолте (берётся из cfg) и меняет ТОЛЬКО
    a_param_init -- прямая проверка "стоит ли менять инициализацию decay_a
    с 0.0 на что-то отрицательное", независимо от вопроса BT/BC из Части 1.
    """
    bsz = cfg["bsz"]
    seq_len = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    d_head = cfg["d_head"]
    seeds = cfg["seeds"]
    bt = cfg["decay_init_probe_bt"]
    bc = cfg["decay_init_probe_bc"]
    a_param_inits = cfg["decay_a_init_values"]

    config = KernelConfig(bt=bt, bc=bc, mb=min(16, bc), use_centering=False, wy_eps=0.0)
    results = []

    print(f"[GRID-1b] Проба decay_a_init при фиксированных bt={bt}, bc={bc}")
    for a_init in a_param_inits:
        per_seed_stats = []
        for seed in seeds:
            q, k, b, g = _make_fixed_inputs_with_decay_a(seed, bsz, seq_len, n_heads, d_head, a_init)
            Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=True)
            max_cond, mean_cond, median_cond, max_sigma, min_sigma = _akk_condition_stats(Akk)
            per_seed_stats.append({
                "seed": seed, "max_cond": max_cond, "mean_cond": mean_cond,
                "median_cond": median_cond,
            })

        agg_max_cond = max(s["max_cond"] for s in per_seed_stats)
        agg_mean_cond = sum(s["mean_cond"] for s in per_seed_stats) / len(per_seed_stats)
        entry = {
            "a_param_init": a_init,
            "effective_decay_scale": float(jnp.exp(jnp.clip(jnp.asarray(a_init, jnp.float32), -20.0, 20.0))),
            "agg_max_cond": agg_max_cond, "agg_mean_cond": agg_mean_cond,
            "per_seed": per_seed_stats,
        }
        results.append(entry)
        print(f"    decay_a_init={a_init:+.2f} (effective decay scale={entry['effective_decay_scale']:.4f}) "
              f"-> max_cond={agg_max_cond:.3e}  mean_cond={agg_mean_cond:.3e}")

    results_sorted = sorted(results, key=lambda r: r["agg_max_cond"])
    print("\n[GRID-1b] Лучшие decay_a_init по agg_max_cond:")
    for r in results_sorted[:5]:
        print(f"    decay_a_init={r['a_param_init']:+.2f} -> max_cond={r['agg_max_cond']:.3e}")

    return results


# ==========================================================================
# Часть 3: точность WY-решателя (wy_solve_pallas) как функция BC, ПРИ
# ФИКСИРОВАННОЙ плохо обусловленной Akk. Часть 1 показала, что BC не влияет
# на саму cond(Akk) (это ожидаемо -- Akk не зависит от того, как её потом
# инвертируют), но НЕ проверяла, влияет ли BC на ТОЧНОСТЬ самого forward
# substitution при инвертировании ОДНОЙ И ТОЙ ЖЕ плохо обусловленной Akk --
# это отдельный, более узкий вопрос про численную устойчивость решателя,
# а не про геометрию входа.
#
# Эталон A_exact строится через Neumann-ряд высокого порядка (не
# jnp.linalg.inv/solve -- та же защита от сломанного scipy/LAPACK на этом
# окружении, что и в _akk_condition_stats), и точность решателя измеряется
# как ||I - (I+Akk)@A||_F относительно этого эталона.
# ==========================================================================
def _reference_inverse_neumann(Akk, iters=200):
    """(I+T)^{-1} через ряд Неймана sum_k (-T)^k -- T строго нижняя
    треугольная (nilpotent при конечном BT), поэтому ряд ТОЧНО обрывается
    на T^BT=0 и iters>=BT гарантирует точный результат (не приближение) --
    в отличие от _neumann_min_sigma_proxy выше (та функция для ОБЩЕЙ
    Akk-подобной матрицы, здесь -- специально для строго нижней
    треугольной T, где ряд действительно конечен)."""
    n = Akk.shape[-1]
    eye = jnp.eye(n, dtype=Akk.dtype)[None]
    acc = eye
    term = eye
    for _ in range(min(iters, n)):
        term = jnp.einsum("nij,njk->nik", term, -Akk)
        acc = acc + term
    return acc


def run_part3_bc_solver_accuracy(cfg):
    """Строит ОДНУ плохо обусловленную Akk (bt=256, decay_scale=0.0 --
    худший случай из Части 1) и гоняет wy_solve_pallas с разным bc на ЭТОЙ
    ЖЕ Akk, сравнивая результат с точным эталоном (Neumann-ряд, обрывается
    точно при iters>=BT -- см. _reference_inverse_neumann). Метрика:
    ||I - (I+Akk)@A_pallas||_F -- НЕ ||A_pallas - A_exact||_F напрямую,
    т.к. при большом cond(Akk) сами элементы A могут быть огромны и
    маленькая относительная ошибка в A даёт большую абсолютную разницу
    -- невязка (I+Akk)@A относительно I содержательнее для "насколько
    решение годится downstream" (ровно то, что реально используется в
    forward: w_pseudo = A @ kb_decayed и т.п.)."""
    bsz = cfg["bsz"]
    seq_len = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    d_head = cfg["d_head"]
    bt = cfg["part3_bt"]
    bc = bt // 2  # FIX: bc жёстко = bt/2 (структурный инвариант Kernel B,
                  # см. atomic_ops/gdn2_fwd.py's assert bt==2*bc) -- реальная
                  # ось точности решателя это mb, не bc.
    mb_values = cfg["part3_mb_values"]
    seed = cfg["part3_seed"]    seed = cfg["part3_seed"]

    q, k, b, g = _make_fixed_inputs(seed, bsz, seq_len, n_heads, d_head, decay_scale=0.0)

    # Akk сама по себе НЕ зависит от bc -- строим её один раз с любым
    # валидным bc (bt должен делиться на bc; берём максимальный из
    # bc_values как "нейтральный" для построения самой Akk).
    probe_bc = max(b_ for b_ in bc_values if bt % b_ == 0)
    probe_config = KernelConfig(bt=bt, bc=bc, mb=min(16, bc), use_centering=False, wy_eps=0.0)
    _Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=probe_config, interpret=True)

    flat_Akk = Akk.astype(jnp.float32).reshape(-1, Akk.shape[-2], Akk.shape[-1])
    A_exact = _reference_inverse_neumann(flat_Akk, iters=bt)

    n = flat_Akk.shape[-1]
    eye = jnp.eye(n, dtype=jnp.float32)[None]

    resid_exact = jnp.einsum("nij,njk->nik", eye + flat_Akk, A_exact) - eye
    resid_exact_norm = float(jnp.max(jnp.linalg.norm(resid_exact.reshape(resid_exact.shape[0], -1), axis=-1)))
    print(f"[GRID-3] Эталонная невязка (Neumann, iters={bt}, sanity-check -- должна быть ~0): "
          f"{resid_exact_norm:.3e}")

    results = []
    print(f"[GRID-3] Точность wy_solve_pallas по BC при фиксированной bt={bt}, decay_scale=0.0 (худший случай)")
    for mb in mb_values:
        if bc % mb != 0:
            print(f"    bc={bc}: пропущено (bc не делится на mb={mb})")
            continue

        config = KernelConfig(bt=bt, bc=bc, mb=mb, use_centering=False, wy_eps=0.0)
        t0 = time.time()
        A_pallas = wy_solve_pallas(Akk, config)
        elapsed = time.time() - t0

        flat_A_pallas = A_pallas.astype(jnp.float32).reshape(-1, n, n)
        resid = jnp.einsum("nij,njk->nik", eye + flat_Akk, flat_A_pallas) - eye
        resid_norm_per_chunk = jnp.linalg.norm(resid.reshape(resid.shape[0], -1), axis=-1)
        max_resid = float(jnp.max(resid_norm_per_chunk))
        mean_resid = float(jnp.mean(resid_norm_per_chunk))

        diff_vs_exact = flat_A_pallas - A_exact
        max_abs_diff_vs_exact = float(jnp.max(jnp.abs(diff_vs_exact)))

        entry = {
            "bc": bc, "mb": mb,
            "max_residual_norm": max_resid, "mean_residual_norm": mean_resid,
            "max_abs_diff_vs_exact": max_abs_diff_vs_exact,
            "elapsed_s": elapsed,
        }
        results.append(entry)
        print(f"    bc={bc:4d} mb={mb:2d} -> max||I-(I+Akk)@A||={max_resid:.3e}  "
              f"mean_resid={mean_resid:.3e}  max|diff vs exact|={max_abs_diff_vs_exact:.3e}  "
              f"({elapsed:.2f}s)")

    return {"reference_sanity_residual": resid_exact_norm, "by_bc": results}


# ==========================================================================
# Часть 2 (опциональная, требует TPU): короткий train-прогон реального
# Pallas-кернела (interpret=False) для лучших кандидатов из Части 1 --
# подтверждает, что снижение cond(Akk) реально даёт лучший
# first_nonfinite_step/final_loss, а не только "число стало меньше" на
# чистой диагностике без обучения.
# ==========================================================================
def run_part2_train_check(cfg, candidates):
    """candidates: список dict {"bt":.., "bc":.., "mb":..} -- берутся из
    топа Части 1. Использует ту же TinyModel-конструкцию, что
    ablation_stability_test_real_kernel.py, но БЕЗ полного ablation --
    только сравнение bt/bc при прочих равных (без centering/xavier/fp32
    decay, чтобы изолировать именно эффект bt/bc)."""
    from flax import linen as nn
    from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
    import numpy as np

    n_heads = cfg["n_heads"]
    d_head = cfg["d_head"]
    seq_len = cfg["seq_len"]
    bsz = cfg["bsz"]
    vocab_size = cfg["vocab_size"]
    n_layers = cfg["n_layers"]
    steps = cfg["part2_steps"]
    lr = cfg["lr"]
    spike_every = cfg["spike_every"]
    seed = cfg["seed"]

    class TinyGDN2Layer(nn.Module):
        n_heads: int
        d_head: int
        config: KernelConfig

        @nn.compact
        def __call__(self, x):
            d = self.n_heads * self.d_head
            kinit = nn.initializers.lecun_normal()

            q = nn.Dense(d, use_bias=False, kernel_init=kinit, dtype=jnp.bfloat16, name="q_proj")(x)
            k = nn.Dense(d, use_bias=False, kernel_init=kinit, dtype=jnp.bfloat16, name="k_proj")(x)
            v = nn.Dense(d, use_bias=False, kernel_init=kinit, dtype=jnp.bfloat16, name="v_proj")(x)
            w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, kernel_init=kinit, dtype=jnp.bfloat16, name="write_gate")(x))
            b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, kernel_init=kinit, dtype=jnp.bfloat16, name="erase_gate")(x))

            f_proj = nn.Dense(d, use_bias=True, kernel_init=kinit, dtype=jnp.bfloat16, name="decay_proj")(x)
            f_proj = f_proj.astype(jnp.float32).reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

            a_param = self.param("decay_a", nn.initializers.zeros, (self.n_heads,)).astype(jnp.float32)
            a_safe = jnp.clip(a_param, -20.0, 20.0)
            g = -jnp.exp(a_safe)[None, None, :, None] * jax.nn.softplus(f_proj)
            g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)

            def resh(t):
                return t.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)
            q, k, v, w_gate, b_gate = map(resh, (q, k, v, w_gate, b_gate))

            eps = 1e-6
            def l2n(t):
                return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps ** 2)
            q, k = l2n(q), l2n(k)

            def sanitize(t):
                return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
            q, k, v, w_gate, b_gate, g = map(sanitize, (q, k, v, w_gate, b_gate, g))

            o, _h_final = gdn2_pallas_forward_trainable(
                q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
                w_gate.astype(jnp.float32), b_gate.astype(jnp.float32), g,
                scale=1.0, config=self.config,
            )
            o = o.reshape(x.shape[0], x.shape[1], d)
            out_proj = nn.Dense(x.shape[-1], use_bias=False, kernel_init=kinit, dtype=jnp.bfloat16, name="out_proj")
            return out_proj(o.astype(jnp.bfloat16))

    class TinyModel(nn.Module):
        n_layers: int
        n_heads: int
        d_head: int
        vocab_size: int
        config: KernelConfig

        @nn.compact
        def __call__(self, input_ids):
            d = self.n_heads * self.d_head
            embed = nn.Embed(self.vocab_size, d, name="embed")
            x = embed(input_ids).astype(jnp.float32)
            for i in range(self.n_layers):
                delta = TinyGDN2Layer(self.n_heads, self.d_head, self.config, name=f"layer_{i}")(x)
                x = jnp.clip(x + delta.astype(jnp.float32), -1e3, 1e3)
            return embed.attend(x.astype(jnp.float32))

    def make_batch(key, period=8, spike=False):
        base = jax.random.randint(key, (bsz, period), 0, vocab_size)
        reps = seq_len // period + 1
        ids = jnp.tile(base, (1, reps))[:, :seq_len]
        if spike:
            key2 = jax.random.fold_in(key, 999)
            mask = jax.random.bernoulli(key2, 0.05, ids.shape)
            spike_ids = jax.random.randint(key2, ids.shape, vocab_size // 2, vocab_size)
            ids = jnp.where(mask, spike_ids, ids)
        labels = jnp.roll(ids, -1, axis=1)
        return ids, labels

    part2_results = []

    for cand in candidates:
        config = KernelConfig(bt=cand["bt"], bc=cand["bc"], mb=cand["mb"], use_centering=False, wy_eps=0.0)
        print(f"\n--- Part2 train check: bt={cand['bt']} bc={cand['bc']} mb={cand['mb']} ---")

        model = TinyModel(n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                           vocab_size=vocab_size, config=config)

        key = jax.random.PRNGKey(seed)
        key, pkey = jax.random.split(key)
        params = model.init(pkey, jnp.zeros((bsz, seq_len), dtype=jnp.int32))["params"]

        def loss_fn(p, ids, labels):
            logits = model.apply({"params": p}, ids)
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
            return jnp.mean(nll)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        m_state = jax.tree_util.tree_map(jnp.zeros_like, params)
        v_state = jax.tree_util.tree_map(jnp.zeros_like, params)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8

        def adam_step(p, g, m, v, t):
            flat_p, treedef = jax.tree_util.tree_flatten(p)
            flat_g, _ = jax.tree_util.tree_flatten(g)
            flat_m, _ = jax.tree_util.tree_flatten(m)
            flat_v, _ = jax.tree_util.tree_flatten(v)
            new_p, new_m, new_v = [], [], []
            for pp, gg, mm, vv in zip(flat_p, flat_g, flat_m, flat_v):
                mm = b1 * mm + (1 - b1) * gg
                vv = b2 * vv + (1 - b2) * (gg * gg)
                mhat = mm / (1 - b1 ** t)
                vhat = vv / (1 - b2 ** t)
                new_p.append(pp - lr * mhat / (jnp.sqrt(vhat) + eps_adam))
                new_m.append(mm)
                new_v.append(vv)
            return (jax.tree_util.tree_unflatten(treedef, new_p),
                    jax.tree_util.tree_unflatten(treedef, new_m),
                    jax.tree_util.tree_unflatten(treedef, new_v))

        def gnorm(g):
            return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))

        first_nonfinite = None
        max_gnorm = 0.0
        t0 = time.time()

        for step in range(1, steps + 1):
            key, dkey = jax.random.split(key)
            is_spike = (spike_every > 0 and step % spike_every == 0)
            ids, labels = make_batch(dkey, period=8, spike=is_spike)

            loss, grads = grad_fn(params, ids, labels)
            gn = float(gnorm(grads))
            loss_v = float(loss)
            finite = np.isfinite(loss_v) and np.isfinite(gn)
            max_gnorm = max(max_gnorm, gn)

            if not finite and first_nonfinite is None:
                first_nonfinite = step

            if finite:
                grads = jax.tree_util.tree_map(lambda g_: jnp.clip(g_, -10.0, 10.0), grads)
                params, m_state, v_state = adam_step(params, grads, m_state, v_state, step)

        elapsed = time.time() - t0
        final_loss = loss_v
        result = {
            "bt": cand["bt"], "bc": cand["bc"], "mb": cand["mb"],
            "first_nonfinite_step": first_nonfinite,
            "max_gnorm": max_gnorm, "final_loss": final_loss,
            "elapsed_s": elapsed,
        }
        part2_results.append(result)
        print(f"    first_nonfinite={first_nonfinite} max_gnorm={max_gnorm:.3e} "
              f"final_loss={final_loss:.4f} ({elapsed:.1f}s)")

    return part2_results


# ==========================================================================
# Конфигурация запуска -- НЕТ argparse, всё через этот dict.
# ==========================================================================
RUN_CONFIG = dict(
    # ---- геометрия входа (Часть 1 и Часть 2) ----
    bsz=2,
    seq_len=1024,          # должен делиться на каждый bt из bt_values
    n_heads=4,
    d_head=128,            # жёсткое требование кернелов (MXU tile)
    vocab_size=256,

    # ---- сетка Части 1 ----
    bt_values=[256, 128, 64],
    bc_divisors=[128, 64, 32, 16],   # берутся только те, что делят bt нацело
    decay_scales=[0.0, 0.05],        # 0.0 = самое начало обучения (decay_a=0)
    seeds=[0, 1, 2],                 # усреднение/max по нескольким seed

    # ---- Часть 1b: ось decay_a_init (проверка альтернативной
    # инициализации decay_a вместо nn.initializers.zeros в model.py) ----
    decay_init_probe_bt=256,
    decay_init_probe_bc=128,
    decay_a_init_values=[0.0, -0.5, -1.0, -1.5, -2.0, -3.0],

    # ---- Часть 3: точность wy_solve_pallas по BC на фиксированной
    # плохо обусловленной Akk (bt=part3_bt, decay_scale=0.0) ----
    part3_bt=256,
    part3_bc_values=[128, 64, 32, 16],
    part3_seed=0,

    # ---- Часть 2 (короткий train check на реальном TPU-кернеле) ----
    run_part2_on_tpu=False,   # <-- поставьте True, только если есть TPU
    part2_top_n=3,            # сколько лучших конфигов из Части 1 проверить обучением
    part2_steps=300,          # короткий прогон, не 3000 -- только для сравнения
    n_layers=4,
    lr=3e-3,
    spike_every=37,
    seed=0,
)


def main(cfg=RUN_CONFIG):
    print("=== ЧАСТЬ 1: сетка (BT, BC) x decay_scale, только forward Kernel A ===")
    print("(interpret=True -- работает на CPU, TPU не требуется)\n")
    part1_results = run_grid_part1(cfg)

    with open("grid_condition_results_part1.json", "w") as f:
        json.dump(part1_results, f, indent=2)
    print("\n[GRID] Часть 1 сохранена в grid_condition_results_part1.json")

    print("\n=== ЧАСТЬ 1b: ось decay_a_init при фиксированных bt/bc ===")
    part1b_results = run_grid_part1b_decay_init(cfg)
    with open("grid_condition_results_part1b_decay_init.json", "w") as f:
        json.dump(part1b_results, f, indent=2)
    print("[GRID] Часть 1b сохранена в grid_condition_results_part1b_decay_init.json")

    print("\n=== ЧАСТЬ 3: точность wy_solve_pallas по BC на фиксированной плохой Akk ===")
    part3_results = run_part3_bc_solver_accuracy(cfg)
    with open("grid_condition_results_part3_bc_accuracy.json", "w") as f:
        json.dump(part3_results, f, indent=2)
    print("[GRID] Часть 3 сохранена в grid_condition_results_part3_bc_accuracy.json")

    if not cfg["run_part2_on_tpu"]:
        print("\n[GRID] run_part2_on_tpu=False -- Часть 2 (train-check на реальном "
              "TPU-кернеле) пропущена. Поставьте cfg['run_part2_on_tpu']=True и "
              "запустите на Kaggle TPU v5e-8, если результаты Части 1 выглядят "
              "многообещающе и нужно подтвердить эффект на реальном обучении.")
        return {"part1": part1_results, "part1b": part1b_results, "part3": part3_results, "part2": None}

    # Берём decay_scale=0.05 (более реалистичный режим после нескольких
    # сотен шагов, не голый init) для отбора топ-кандидатов в Часть 2.
    candidates_pool = [r for r in part1_results if abs(r["decay_scale"] - 0.05) < 1e-9]
    candidates_pool_sorted = sorted(candidates_pool, key=lambda r: r["agg_max_cond"])
    top_candidates = candidates_pool_sorted[: cfg["part2_top_n"]]

    print(f"\n=== ЧАСТЬ 2: train-check на реальном Pallas-кернеле (interpret=False) ===")
    print(f"Топ-{len(top_candidates)} кандидатов по agg_max_cond (decay_scale=0.05):")
    for c in top_candidates:
        print(f"  bt={c['bt']} bc={c['bc']} mb={c['mb']} agg_max_cond={c['agg_max_cond']:.3e}")

    part2_results = run_part2_train_check(cfg, top_candidates)

    with open("grid_condition_results_part2.json", "w") as f:
        json.dump(part2_results, f, indent=2)
    print("\n[GRID] Часть 2 сохранена в grid_condition_results_part2.json")

    return {"part1": part1_results, "part1b": part1b_results, "part3": part3_results, "part2": part2_results}


if __name__ == "__main__":
    main()
