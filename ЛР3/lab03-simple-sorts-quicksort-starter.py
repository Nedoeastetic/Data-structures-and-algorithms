
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback
from pathlib import Path


# ============================================================
# 1. ПРОСТЫЕ СОРТИРОВКИ
# Формат результата: (отсортированный список, сравнения, перемещения)
# Все функции сортируют копию входного списка.
# ============================================================

def bubble_sort(a: list) -> tuple[list, int, int]:
    """Сортировка пузырьком: лучший случай O(n), худший O(n^2)."""
    result = list(a)
    comparisons = 0
    swaps = 0

    for end in range(len(result) - 1, 0, -1):
        swapped = False

        for j in range(end):
            comparisons += 1

            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swaps += 1
                swapped = True

        # Если обменов не было, список уже отсортирован.
        if not swapped:
            break

    return result, comparisons, swaps


def insertion_sort(a: list) -> tuple[list, int, int]:
    """Сортировка вставками: лучший случай O(n), худший O(n^2)."""
    result = list(a)
    comparisons = 0
    moves = 0

    for i in range(1, len(result)):
        key = result[i]
        j = i - 1

        while j >= 0:
            comparisons += 1

            if result[j] > key:
                result[j + 1] = result[j]
                moves += 1
                j -= 1
            else:
                break

        # Записываем key, если он переместился на другую позицию.
        if j + 1 != i:
            result[j + 1] = key
            moves += 1

    return result, comparisons, moves


def selection_sort(a: list) -> tuple[list, int, int]:
    """Сортировка выбором: O(n^2) сравнений в любом случае."""
    result = list(a)
    comparisons = 0
    swaps = 0
    n = len(result)

    for i in range(n - 1):
        min_index = i

        for j in range(i + 1, n):
            comparisons += 1

            if result[j] < result[min_index]:
                min_index = j

        if min_index != i:
            result[i], result[min_index] = result[min_index], result[i]
            swaps += 1

    return result, comparisons, swaps


# ============================================================
# 2. QUICKSORT С РАНДОМИЗИРОВАННЫМ ОПОРНЫМ ЭЛЕМЕНТОМ
# Средняя сложность O(n log n), худшая O(n^2).
# ============================================================

def quick_sort(a: list, rng: random.Random | None = None) -> list:
    """Возвращает новый отсортированный список."""
    if rng is None:
        rng = random.Random()

    result = list(a)

    if len(result) < 2:
        return result

    # Используем явный стек, чтобы не зависеть от глубины рекурсии Python.
    stack = [(0, len(result))]

    while stack:
        lo, hi = stack.pop()

        if hi - lo < 2:
            continue

        # Опорный элемент выбирается случайно из текущего диапазона.
        pivot = result[rng.randrange(lo, hi)]

        # Трёхчастное разбиение:
        # [lo, lt)       — меньше pivot
        # [lt, i)        — равно pivot
        # [i, gt + 1)    — ещё не обработано
        # [gt + 1, hi)   — больше pivot
        lt = lo
        i = lo
        gt = hi - 1

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

        # Равные опорному элементы повторно сортировать не нужно.
        if lt - lo > 1:
            stack.append((lo, lt))

        if hi - (gt + 1) > 1:
            stack.append((gt + 1, hi))

    return result


# ============================================================
# 3. ДЕМОНСТРАЦИЯ СТАБИЛЬНОСТИ
# Сравниваем объекты только по ключу, метка не участвует.
# ============================================================

class Pair:
    def __init__(self, key: int, label: str):
        self.key = key
        self.label = label

    def __lt__(self, other: "Pair") -> bool:
        return self.key < other.key

    def __gt__(self, other: "Pair") -> bool:
        return self.key > other.key

    def __le__(self, other: "Pair") -> bool:
        return self.key <= other.key

    def __repr__(self) -> str:
        return f"({self.key}, {self.label!r})"


def stability_demo() -> None:
    original = [
        Pair(2, "a"),
        Pair(1, "b"),
        Pair(2, "c"),
        Pair(1, "d"),
    ]

    sorters = {
        "bubble_sort": lambda x: bubble_sort(x)[0],
        "insertion_sort": lambda x: insertion_sort(x)[0],
        "selection_sort": lambda x: selection_sort(x)[0],
        "quick_sort": lambda x: quick_sort(x, random.Random(17)),
    }

    print("\nДемонстрация стабильности:")
    print("Исходный список:", original)

    for name, sorter in sorters.items():
        result = sorter(original)

        stable = True
        for key in {item.key for item in original}:
            before = [
                item.label for item in original if item.key == key
            ]
            after = [
                item.label for item in result if item.key == key
            ]

            if before != after:
                stable = False
                break

        status = "стабильная" if stable else "нестабильная"
        print(f"{name}: {result} — {status}")


# ============================================================
# 4. ГЕНЕРАЦИЯ ТРЁХ КЛАССОВ ВХОДНЫХ ДАННЫХ
# ============================================================

def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed + n)
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]

    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ============================================================
# 5. САМОПРОВЕРКА
# ============================================================

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


class SelfCheckError(Exception):
    """Ошибка проверки алгоритма."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def check_algorithm(fn) -> None:
    boundary = [
        [],
        [7],
        [5, 5, 5, 5],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ]

    for a in boundary:
        result = fn(list(a))

        expect(
            isinstance(result, list),
            f"на {a} возвращён не список",
        )
        expect(
            is_sorted(result),
            f"нарушен порядок на {a}: {result}",
        )
        expect(
            sorted(result) == sorted(a),
            f"результат не является перестановкой входа: {a}",
        )

    rng = random.Random(0)

    for _ in range(300):
        a = [
            rng.randint(-100, 100)
            for _ in range(rng.randint(0, 80))
        ]

        result = fn(list(a))

        expect(
            result == sorted(a),
            f"результат отличается от sorted() на {a}: {result}",
        )


def check_counters() -> None:
    # Для selection_sort число сравнений равно n * (n - 1) // 2.
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    expect(
        comparisons == 10,
        f"selection_sort: ожидалось 10 сравнений, получено {comparisons}",
    )

    # На отсортированном списке пузырёк выполняет один проход.
    _, comparisons, swaps = bubble_sort([1, 2, 3, 4])
    expect(comparisons == 3, "неверный счётчик сравнений bubble_sort")
    expect(swaps == 0, "на отсортированном входе bubble_sort не должен менять элементы")

    # На отсортированном списке вставки не делают сдвигов.
    _, comparisons, moves = insertion_sort([1, 2, 3, 4])
    expect(comparisons == 3, "неверный счётчик сравнений insertion_sort")
    expect(moves == 0, "на отсортированном входе insertion_sort не должен делать сдвиги")

    # Проверяем, что простые сортировки не изменяют исходный список.
    for sorter in (bubble_sort, insertion_sort, selection_sort):
        original = [3, 1, 2]
        sorter(original)
        expect(
            original == [3, 1, 2],
            f"{sorter.__name__} изменил исходный список",
        )


def check_stability() -> None:
    original = [
        Pair(2, "a"),
        Pair(1, "b"),
        Pair(2, "c"),
        Pair(1, "d"),
    ]

    expected = {1: ["b", "d"], 2: ["a", "c"]}

    for name, sorter in (
        ("bubble_sort", lambda x: bubble_sort(x)[0]),
        ("insertion_sort", lambda x: insertion_sort(x)[0]),
    ):
        result = sorter(original)

        actual = {}
        for item in result:
            actual.setdefault(item.key, []).append(item.label)

        expect(
            actual == expected,
            f"{name} не сохранил порядок равных ключей: {actual}",
        )


def self_check() -> None:
    checks = [
        (name, lambda fn=fn: check_algorithm(fn))
        for name, fn in ALGORITHMS.items()
    ]

    checks.append(("счётчики", check_counters))
    checks.append(("стабильность", check_stability))

    failed = 0
    print("self_check:")

    for name, check in checks:
        try:
            check()

        except NotImplementedError as err:
            where = traceback.extract_tb(err.__traceback__)[-1].name
            print(f"  {name}: не реализовано — {where}")
            failed += 1

        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1

        except Exception:
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1

        else:
            print(f"  {name}: OK")

    if failed:
        raise SystemExit(
            f"self_check: не пройдено разделов — {failed}; "
            "замеры не выполняются."
        )

    print("self_check: OK")


# ============================================================
# 6. БЕНЧМАРК
# ============================================================

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5


def bench(fn, data: list) -> float:
    """Возвращает медианное время нескольких запусков."""
    fn(list(data))  # прогрев

    times = []

    for _ in range(REPEATS):
        arg = list(data)

        start = time.perf_counter()
        fn(arg)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return statistics.median(times)


def print_counter_table(seed: int) -> None:
    n = 100
    inputs = make_inputs(n, seed)

    print(f"\nСчётчики операций при n={n}:")

    for input_class, data in inputs.items():
        print(f"\nВход: {input_class}")

        for name, sorter in (
            ("bubble_sort", bubble_sort),
            ("insertion_sort", insertion_sort),
            ("selection_sort", selection_sort),
        ):
            _, comparisons, moves = sorter(data)

            print(
                f"  {name:<16} "
                f"сравнения={comparisons:>6}, "
                f"обмены/сдвиги={moves:>6}"
            )


def plot_benchmarks(results: dict) -> None:
    """Строит log-log графики, если установлен matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nmatplotlib не установлен — графики пропущены.\n"
            "Установка: python -m pip install matplotlib"
        )
        return

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    for input_class in ("упорядоченный", "случайный", "обратный"):
        plt.figure()

        for algorithm, classes in results.items():
            points = classes[input_class]

            if points:
                sizes = [p[0] for p in points]
                times = [p[1] for p in points]

                plt.loglog(
                    sizes,
                    times,
                    marker="o",
                    label=algorithm,
                )

        plt.xlabel("Размер массива n (log)")
        plt.ylabel("Медианное время, секунды (log)")
        plt.title(f"Сравнение сортировок: {input_class} вход")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = output_dir / f"benchmark_{input_class}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"График сохранён: {filename}")


def run_benchmarks(seed: int) -> None:
    plans = [
        (
            "bubble_sort",
            lambda a: bubble_sort(a)[0],
            SIZES_QUADRATIC,
        ),
        (
            "insertion_sort",
            lambda a: insertion_sort(a)[0],
            SIZES_QUADRATIC,
        ),
        (
            "selection_sort",
            lambda a: selection_sort(a)[0],
            SIZES_QUADRATIC,
        ),
        (
            "quick_sort",
            lambda a: quick_sort(a, random.Random(seed)),
            SIZES_QUICK,
        ),
    ]

    results = {}

    for name, fn, sizes in plans:
        print(f"\n{name}:")

        results[name] = {
            "упорядоченный": [],
            "случайный": [],
            "обратный": [],
        }

        for n in sizes:
            inputs = make_inputs(n, seed)

            for input_class, data in inputs.items():
                elapsed = bench(fn, data)

                results[name][input_class].append((n, elapsed))

                print(
                    f"  n={n:>7} "
                    f"вход={input_class:<13} "
                    f"t={elapsed:.6f} с"
                )

    print_counter_table(seed)
    plot_benchmarks(results)


# ============================================================
# 7. ЗАПУСК ПРОГРАММЫ
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--variant",
        type=int,
        required=True,
        help="номер варианта",
    )

    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="пропустить длительные замеры времени",
    )

    args = parser.parse_args()

    seed = 30 + args.variant
    random.seed(seed)

    self_check()
    stability_demo()

    if args.skip_benchmarks:
        print("\nЗамеры пропущены по параметру --skip-benchmarks.")
    else:
        run_benchmarks(seed)


if __name__ == "__main__":
    main()
