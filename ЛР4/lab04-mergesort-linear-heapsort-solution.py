#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, CountingSort, RadixSort, HeapSort и сравнение сортировок.

Запуск: python lab04-mergesort-linear-heapsort-solution.py --variant N
Для графиков рекомендуется установить matplotlib: python -m pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Сортировки
# ---------------------------------------------------------------------------

def merge(left: list, right: list, key) -> list:
    """Стабильно сливает два отсортированных списка за O(n + m)."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # <= принципиально важно: при равенстве первым берём элемент слева.
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(a: list, key=lambda x: x) -> list:
    """Возвращает новый стабильный отсортированный список: O(n log n), O(n) памяти."""
    if len(a) < 2:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key=key)
    right = merge_sort(a[mid:], key=key)
    return merge(left, right, key)


def _counting_sort_by_key(items: list, key, key_max: int) -> list:
    """Внутренняя устойчивая CountingSort для целочисленного ключа [0, key_max]."""
    if key_max < 0:
        raise ValueError("key_max должен быть неотрицательным")
    counts = [0] * (key_max + 1)
    for item in items:
        value = key(item)
        if not isinstance(value, int) or value < 0 or value > key_max:
            raise ValueError(f"ключ {value!r} вне диапазона [0, {key_max}]")
        counts[value] += 1

    # Префиксные суммы: counts[v] — конечная позиция последнего ключа v.
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]

    output = [None] * len(items)
    # Обратный проход сохраняет относительный порядок элементов с равным ключом.
    for item in reversed(items):
        value = key(item)
        counts[value] -= 1
        output[counts[value]] = item
    return output


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая CountingSort для целых ключей [0, key_max]: O(n + k) времени/памяти."""
    return _counting_sort_by_key(a, lambda x: x, key_max)


def radix_sort_lsd(a: list[int]) -> list[int]:
    """LSD RadixSort по десятичным цифрам; O(d(n + 10)) времени, O(n + 10) памяти."""
    if any(not isinstance(x, int) or x < 0 for x in a):
        raise ValueError("radix_sort_lsd принимает только неотрицательные целые числа")
    result = list(a)
    if not result:
        return result
    max_value = max(result)
    exp = 1
    while max_value // exp > 0:
        result = _counting_sort_by_key(result, lambda x, e=exp: (x // e) % 10, 9)
        exp *= 10
    return result


def sift_down(a: list[int], i: int, n: int) -> None:
    """Восстанавливает max-heap в a[0:n], просеивая узел i вниз."""
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right
        if largest == i:
            return
        a[i], a[largest] = a[largest], a[i]
        i = largest


def _build_max_heap(a: list[int]) -> None:
    """Построение max-кучи снизу вверх за O(n)."""
    for i in range(len(a) // 2 - 1, -1, -1):
        sift_down(a, i, len(a))


def heap_sort(a: list[int]) -> None:
    """HeapSort на месте, O(n log n) времени, O(1) дополнительной памяти."""
    n = len(a)
    _build_max_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


def quick_sort(a: list[int]) -> list[int]:
    """Рандомизированный трёхчастный QuickSort из ЛР 3; возвращает новую копию."""
    result = list(a)
    if len(result) < 2:
        return result

    # Глобальный random.seed(seed) задаётся в main(), поэтому вариант воспроизводим.
    stack = [(0, len(result))]
    while stack:
        lo, hi = stack.pop()
        while hi - lo > 1:
            pivot = result[random.randrange(lo, hi)]
            lt, i, gt = lo, lo, hi - 1
            while i <= gt:
                if result[i] < pivot:
                    result[lt], result[i] = result[i], result[lt]
                    lt += 1
                    i += 1
                elif result[i] > pivot:
                    result[i], result[gt] = result[gt], result[i]
                    gt -= 1
                else:
                    i += 1

            # Обрабатываем меньшую часть сразу, большую кладём в стек:
            # это ограничивает глубину стека O(log n).
            left = (lo, lt)
            right = (gt + 1, hi)
            if left[1] - left[0] < right[1] - right[0]:
                if right[1] - right[0] > 1:
                    stack.append(right)
                lo, hi = left
            else:
                if left[1] - left[0] > 1:
                    stack.append(left)
                lo, hi = right
    return result


# ---------------------------------------------------------------------------
# 2. Верификация
# ---------------------------------------------------------------------------

class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    buf = list(a)
    heap_sort(buf)
    return buf


SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": heap_sorted,
    "quick_sort": lambda a: quick_sort(a),
}


def check_sort(fn) -> None:
    boundary = [[], [7], [0], [5, 5, 5], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")

    rng = random.Random(0)
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: {res}")


def check_merge_stability() -> None:
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    got = merge_sort(pairs, key=lambda p: p[0])
    expect(got == sorted(pairs, key=lambda p: p[0]), "merge_sort нарушил стабильность")


def check_counting_stability() -> None:
    tagged = [(2, "a"), (1, "b"), (2, "c"), (1, "d"), (2, "e")]
    got = _counting_sort_by_key(tagged, lambda x: x[0], 2)
    expect(got == [(1, "b"), (1, "d"), (2, "a"), (2, "c"), (2, "e")],
           "CountingSort нарушил порядок элементов с одинаковым ключом")


def check_radix_digit_stability() -> None:
    # Проверяем один проход по цифре десятков: элементы с одинаковой цифрой
    # десятков обязаны сохранить порядок, полученный после предыдущего разряда.
    tagged = [(21, "a"), (11, "b"), (22, "c"), (12, "d"), (31, "e")]
    got = _counting_sort_by_key(tagged, lambda x: (x[0] // 10) % 10, 9)
    expected = [(11, "b"), (12, "d"), (21, "a"), (22, "c"), (31, "e")]
    expect(got == expected, f"поразрядный проход неустойчив: {got}")


def check_heap_property() -> None:
    rng = random.Random(41)
    for n in range(0, 100):
        a = [rng.randint(-1000, 1000) for _ in range(n)]
        _build_max_heap(a)
        for child in range(1, n):
            parent = (child - 1) // 2
            expect(a[parent] >= a[child],
                   f"свойство max-кучи нарушено: parent={a[parent]}, child={a[child]}")


def check_counting_range_validation() -> None:
    for bad in ([-1], [4]):
        try:
            counting_sort(bad, key_max=3)
        except ValueError:
            continue
        raise SelfCheckError(f"CountingSort не отклонил ключ вне диапазона: {bad}")


def self_check() -> None:
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.extend([
        ("стабильность merge_sort", check_merge_stability),
        ("стабильность counting_sort", check_counting_stability),
        ("устойчивость прохода radix", check_radix_digit_stability),
        ("свойство max-кучи после построения", check_heap_property),
        ("проверка диапазона counting_sort", check_counting_range_validation),
    ])
    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except Exception:
            failed += 1
            print(f"  {name}: ОШИБКА")
            traceback.print_exc(file=sys.stdout)
        else:
            print(f"  {name}: OK")
    if failed:
        raise SystemExit(f"self_check: не пройдено проверок — {failed}; замеры отменены")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк, таблица свойств и графики
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100
WIDE_KEY_MAX = 10**9


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    almost = list(range(n))
    for _ in range(max(1, n // 100)):
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Прогрев + REPEATS независимых запусков; возвращается медиана секунд."""
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


PROPERTY_ROWS = [
    ["MergeSort", "O(n log n)", "O(n log n)", "O(n log n)", "O(n)", "Да", "Произвольные сравнимые ключи; требует доп. память"],
    ["CountingSort", "O(n+k)", "O(n+k)", "O(n+k)", "O(n+k)", "Да", "Целые ключи в небольшом известном диапазоне k"],
    ["RadixSort LSD", "O(d(n+b))", "O(d(n+b))", "O(d(n+b))", "O(n+b)", "Да", "Неотрицательные целые; b=10, d — число цифр"],
    ["HeapSort", "O(n log n)", "O(n log n)", "O(n log n)", "O(1)", "Нет", "Сравнимые ключи; сортирует список на месте"],
    ["QuickSort (randomized)", "O(n log n)", "O(n log n) ожидаемо", "O(n²)", "O(log n) ожидаемо", "Нет", "Сравнимые ключи; худший случай квадратичный"],
    ["Python sorted() / Timsort", "O(n)", "O(n log n)", "O(n log n)", "O(n)", "Да", "Общие сравнимые объекты; результат — новый список"],
]


def save_properties(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sorting_properties.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Алгоритм", "Лучший случай", "Средний случай", "Худший случай",
                         "Доп. память", "Стабильность", "Ограничения"])
        writer.writerows(PROPERTY_ROWS)
    md_path = out_dir / "sorting_properties.md"
    headers = ["Алгоритм", "Лучший", "Средний", "Худший", "Память", "Стабилен", "Ограничения"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in PROPERTY_ROWS:
            f.write("| " + " | ".join(row) + " |\n")


def save_loglog_graphs(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен: графики не построены. Установите: python -m pip install matplotlib")
        return

    distributions = sorted({row["distribution"] for row in results})
    for dist in distributions:
        fig, ax = plt.subplots(figsize=(9, 6))
        subset = [r for r in results if r["distribution"] == dist]
        algorithms = sorted({r["algorithm"] for r in subset})
        for algo in algorithms:
            points = sorted((r["n"], r["seconds"]) for r in subset if r["algorithm"] == algo)
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="o", label=algo)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Размер массива n (логарифмическая шкала)")
        ax.set_ylabel("Медианное время, с (логарифмическая шкала)")
        ax.set_title(f"Сравнение сортировок: {dist}")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"loglog_{dist}.png", dpi=160)
        plt.close(fig)


def print_properties() -> None:
    print("\nСводная таблица свойств сохранена в results/sorting_properties.csv и .md")
    print("Алгоритм | Лучший | Средний | Худший | Память | Стабильность")
    for row in PROPERTY_ROWS:
        print(" | ".join(row[:6]))


def run_benchmarks(seed: int) -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort_LR3", quick_sort),
        ("sorted_Timsort", sorted),
    ]
    results: list[dict] = []
    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for dist_name, data in datasets.items():
            print(f"  распределение {dist_name}:")
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    print(f"    {algo_name:>18}: пропуск (слишком широкий диапазон ключей)")
                    continue
                elapsed = bench(fn, data)
                results.append({"n": n, "distribution": dist_name,
                                "algorithm": algo_name, "seconds": elapsed})
                print(f"    {algo_name:>18}: t = {elapsed:.6f} c")

    csv_path = out_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "distribution", "algorithm", "seconds"])
        writer.writeheader()
        writer.writerows(results)
    save_properties(out_dir)
    save_loglog_graphs(results, out_dir)
    print_properties()
    print(f"\nРезультаты сохранены в: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = parser.parse_args()
    if args.variant < 1:
        parser.error("номер варианта должен быть положительным")
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()
