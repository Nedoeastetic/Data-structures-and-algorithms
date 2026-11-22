#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 6. Линейный, бинарный и интерполяционный поиск.

Запуск:
    python lab06-search-starter.py --variant 1
    python lab06-search-starter.py --variant 1 --output-dir results

Зависимость для графиков (не обязательна для тестов):
    python -m pip install matplotlib

Метрика сравнений:
- учитываются сравнения значений массива с ключом;
- сравнения индексов, арифметика и проверки границ индексов не учитываются.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import random
import statistics
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска
# ---------------------------------------------------------------------------

def linear_search(a: list[int], key: int) -> tuple[int, int]:
    """Первое вхождение. O(1) best, O(n) average/worst, O(1) memory."""
    comparisons = 0
    for i, value in enumerate(a):
        comparisons += 1  # value == key
        if value == key:
            return i, comparisons
    return -1, comparisons


def binary_search(a: list[int], key: int) -> tuple[int, int]:
    """Любое вхождение. O(1) best, O(log n) average/worst, O(1) memory."""
    lo, hi = 0, len(a) - 1
    comparisons = 0

    while lo <= hi:
        mid = lo + (hi - lo) // 2
        comparisons += 1  # a[mid] == key
        if a[mid] == key:
            return mid, comparisons

        comparisons += 1  # a[mid] < key
        if a[mid] < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


def interpolation_search(a: list[int], key: int) -> tuple[int, int]:
    """Любое вхождение. Среднее O(log log n) на равномерных данных,
    худшее O(n); O(1) дополнительная память.

    Считаются сравнения значений массива с key в проверках границ,
    равенства и направления поиска. Проверки индексов не учитываются.
    """
    if not a:
        return -1, 0

    lo, hi = 0, len(a) - 1
    comparisons = 0

    while lo <= hi:
        comparisons += 1  # a[lo] <= key
        if a[lo] > key:
            return -1, comparisons

        comparisons += 1  # key <= a[hi]
        if key > a[hi]:
            return -1, comparisons

        # Если значения на концах равны, весь диапазон одинаков.
        if a[lo] == a[hi]:
            comparisons += 1  # a[lo] == key
            return (lo, comparisons) if a[lo] == key else (-1, comparisons)

        # Значения отсортированы, поэтому формула даёт позицию в диапазоне.
        pos = lo + ((key - a[lo]) * (hi - lo)) // (a[hi] - a[lo])
        # Защита от выхода за границы из-за округления/особых данных.
        pos = max(lo, min(pos, hi))

        comparisons += 1  # a[pos] == key
        if a[pos] == key:
            return pos, comparisons

        comparisons += 1  # a[pos] < key
        if a[pos] < key:
            lo = pos + 1
        else:
            hi = pos - 1

    return -1, comparisons


# ---------------------------------------------------------------------------
# 2. Генерация данных
# ---------------------------------------------------------------------------

def make_dataset(n: int, distribution: str, rng: random.Random) -> list[int]:
    """Отсортированный массив n целых ключей заданного распределения."""
    if n < 0:
        raise ValueError("n не может быть отрицательным")
    if distribution == "uniform":
        return sorted(rng.randint(0, 10 * n) for _ in range(n))
    if distribution == "exponential":
        # Экспоненциальные значения сгущаются у нуля; целочисленное
        # преобразование намеренно допускает дубликаты.
        return sorted(int(rng.expovariate(1.0 / 1000.0)) for _ in range(n))
    raise ValueError(f"неизвестное распределение: {distribution!r}")


def pick_keys(a: list[int], rng: random.Random, count: int,
              present: bool) -> list[int]:
    """Подбирает гарантированно присутствующие/отсутствующие ключи."""
    if count < 0:
        raise ValueError("count не может быть отрицательным")
    if present:
        if not a and count:
            raise ValueError("Нельзя выбрать присутствующие ключи из пустого массива")
        return [a[rng.randrange(len(a))] for _ in range(count)]

    have = set(a)
    hi = max(abs(a[-1]) if a else 0, abs(a[0]) if a else 0) + 2
    keys: list[int] = []
    while len(keys) < count:
        k = rng.randint(-hi, hi)
        if k not in have:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 3. Проверки корректности
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("linear", linear_search),
    ("binary", binary_search),
    ("interpolation", interpolation_search),
]


def self_check() -> None:
    """Граничные тесты и 300 случайных проверок относительно библиотечного эталона."""
    for name, fn in ALGORITHMS:
        assert fn([], 1) == (-1, 0), f"{name}: пустой массив"
        assert fn([7], 7)[0] == 0, f"{name}: один элемент, найден"
        assert fn([7], 3)[0] == -1, f"{name}: один элемент, отсутствует"
        a = [1, 3, 3, 5, 8, 13, 21]
        assert fn(a, 1)[0] == 0, f"{name}: первый"
        assert fn(a, 21)[0] == len(a) - 1, f"{name}: последний"
        assert fn(a, 4)[0] == -1, f"{name}: внутри диапазона"
        assert fn(a, 100)[0] == -1, f"{name}: за диапазоном"
        assert fn([4, 4, 4, 4], 4)[0] in range(4), f"{name}: все равны, найден"
        assert fn([4, 4, 4, 4], 6)[0] == -1, f"{name}: все равны, отсутствует"

    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.randint(0, 60) for _ in range(rng.randint(0, 40)))
        key = rng.randint(-5, 65)
        j = bisect.bisect_left(a, key)
        present = j < len(a) and a[j] == key
        assert present == (key in a)

        for name, fn in ALGORITHMS:
            idx, comparisons = fn(a, key)
            assert comparisons >= 0
            if a:
                assert comparisons >= 1, f"{name}: сравнения не считаются"
            if present:
                assert idx != -1 and a[idx] == key, f"{name}: ключ не найден"
            else:
                assert idx == -1, f"{name}: ложное совпадение"

        if present:
            assert linear_search(a, key)[0] == a.index(key)

    print("self_check: OK (границы + 300 случайных массивов)")


# ---------------------------------------------------------------------------
# 4. Бенчмарк и сохранение результатов
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
N_KEYS = 200
LINEAR_MAX_N = 30_000


def bench(fn, a: list[int], keys: list[int]) -> float:
    """Медиана времени одной серии поисков; один прогрев."""
    for k in keys:
        fn(a, k)
    measurements = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for k in keys:
            fn(a, k)
        measurements.append(time.perf_counter() - t0)
    return statistics.median(measurements)


def mean_comparisons(fn, a: list[int], keys: list[int]) -> float:
    if not keys:
        return 0.0
    return statistics.mean(fn(a, k)[1] for k in keys)


def run_benchmarks(seed: int, output_dir: Path) -> list[dict]:
    rng = random.Random(seed)
    results: list[dict] = []

    for distribution in ("uniform", "exponential"):
        print(f"\nРаспределение: {distribution}")
        for n in SIZES:
            a = make_dataset(n, distribution, rng)
            for present in (True, False):
                keys = pick_keys(a, rng, N_KEYS, present)
                label = "present" if present else "absent"
                for name, fn in ALGORITHMS:
                    if name == "linear" and n > LINEAR_MAX_N:
                        print(f"  n={n:>7} key={label:<7} {name:<13} пропуск (лимит n)")
                        continue

                    comparisons = mean_comparisons(fn, a, keys)
                    seconds = bench(fn, a, keys)
                    row = {
                        "distribution": distribution,
                        "n": n,
                        "key_status": label,
                        "algorithm": name,
                        "mean_comparisons": comparisons,
                        "median_seconds_per_200_searches": seconds,
                        "seed": seed,
                        "repeats": REPEATS,
                        "keys_per_series": N_KEYS,
                    }
                    results.append(row)
                    print(
                        f"  n={n:>7} key={label:<7} {name:<13}"
                        f" comparisons={comparisons:>10.2f}"
                        f" median_t={seconds:.6f} s"
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV сохранён: {csv_path}")

    make_plots(results, output_dir)
    print_analysis(results)
    return results


def make_plots(results: list[dict], output_dir: Path) -> None:
    """Сохраняет графики сравнений и времени отдельно для распределений."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — графики не построены. "
              "Установите: python -m pip install matplotlib")
        return

    for distribution in ("uniform", "exponential"):
        subset = [r for r in results if r["distribution"] == distribution]
        for status in ("present", "absent"):
            filtered = [r for r in subset if r["key_status"] == status]

            # График сравнений
            fig, ax = plt.subplots()
            for algorithm in ("linear", "binary", "interpolation"):
                rows = [r for r in filtered if r["algorithm"] == algorithm]
                if rows:
                    ax.plot([r["n"] for r in rows],
                            [r["mean_comparisons"] for r in rows],
                            marker="o", label=algorithm)
            ax.set_xlabel("Размер массива n")
            ax.set_ylabel("Среднее число сравнений на поиск")
            ax.set_title(f"Сравнения — {distribution}, ключ {status}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xscale("log")
            fig.tight_layout()
            path = output_dir / f"comparisons_{distribution}_{status}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)

            # График времени
            fig, ax = plt.subplots()
            for algorithm in ("linear", "binary", "interpolation"):
                rows = [r for r in filtered if r["algorithm"] == algorithm]
                if rows:
                    ax.plot([r["n"] for r in rows],
                            [r["median_seconds_per_200_searches"] for r in rows],
                            marker="o", label=algorithm)
            ax.set_xlabel("Размер массива n")
            ax.set_ylabel(f"Медианное время {N_KEYS} поисков, с")
            ax.set_title(f"Время — {distribution}, ключ {status}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.tight_layout()
            path = output_dir / f"time_{distribution}_{status}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)

    print(f"Графики сохранены в: {output_dir}")


def print_analysis(results: list[dict]) -> None:
    """Печатает краткие выводы по средним сравнениям."""
    print("\nКраткий анализ:")
    for distribution in ("uniform", "exponential"):
        for status in ("present", "absent"):
            rows = [r for r in results
                    if r["distribution"] == distribution
                    and r["key_status"] == status
                    and r["n"] == max(SIZES)]
            # На максимальном n линейный поиск намеренно пропускается.
            by_algorithm = {r["algorithm"]: r["mean_comparisons"] for r in rows}
            if "binary" in by_algorithm and "interpolation" in by_algorithm:
                b = by_algorithm["binary"]
                it = by_algorithm["interpolation"]
                relation = "меньше" if it < b else "больше/равно"
                print(f"  {distribution}, {status}: interpolation "
                      f"{relation} binary по сравнениям "
                      f"({it:.2f} против {b:.2f}).")
    print("  Интерполяционный поиск обычно эффективнее на равномерных данных,")
    print("  но при неоднородном распределении может терять преимущество.")
    print("  Бинарный поиск даёт логарифмическую гарантию независимо от распределения.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True,
                        help="номер варианта (используется для seed)")
    parser.add_argument("--output-dir", default="lab06-results",
                        help="папка для CSV и графиков")
    args = parser.parse_args()

    seed = 30 + args.variant
    output_dir = Path(args.output_dir)
    self_check()
    run_benchmarks(seed, output_dir)


if __name__ == "__main__":
    main()
