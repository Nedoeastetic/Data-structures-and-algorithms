#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 5. Бинарное дерево поиска: обходы, вставка и удаление.

Запуск:
    python lab05_bst_solution.py --variant 1
    python lab05_bst_solution.py --variant 1 --skip-bench
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback
from collections import deque


class Node:
    """Узел BST: ключ и ссылки на левое и правое поддеревья."""
    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None


class BST:
    """BST без дубликатов.

    Инвариант: ключи левого поддерева меньше ключа узла, правого — больше.
    Все основные операции поиска/вставки/удаления имеют O(h), где h — высота.
    Для случайной формы ожидается h = O(log n), в худшем случае h = O(n).
    """

    def __init__(self) -> None:
        self.root: Node | None = None

    def insert(self, key: int) -> None:
        """Вставить ключ; повторная вставка существующего ключа игнорируется."""
        if self.root is None:
            self.root = Node(key)
            return

        current = self.root
        while True:
            if key == current.key:
                return
            if key < current.key:
                if current.left is None:
                    current.left = Node(key)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(key)
                    return
                current = current.right

    def search(self, key: int) -> bool:
        """Вернуть True, если ключ присутствует."""
        current = self.root
        while current is not None:
            if key == current.key:
                return True
            current = current.left if key < current.key else current.right
        return False

    def delete(self, key: int) -> None:
        """Удалить ключ; отсутствующий ключ игнорируется.

        Для узла с двумя детьми используется in-order successor — минимум
        правого поддерева.
        """
        parent: Node | None = None
        current = self.root

        # Найти удаляемый узел и его родителя.
        while current is not None and current.key != key:
            parent = current
            current = current.left if key < current.key else current.right
        if current is None:
            return

        # Узел с двумя детьми: копируем ключ преемника и удаляем преемника.
        if current.left is not None and current.right is not None:
            successor_parent = current
            successor = current.right
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
            current.key = successor.key
            # Преемник не имеет левого ребёнка; его правый ребёнок
            # (если есть) должен занять место преемника.
            if successor_parent.left is successor:
                successor_parent.left = successor.right
            else:
                successor_parent.right = successor.right
            return

        # 0 или 1 ребёнок: ребёнок (или None) поднимается на место узла.
        child = current.left if current.left is not None else current.right
        if parent is None:
            self.root = child
        elif parent.left is current:
            parent.left = child
        else:
            parent.right = child

    def in_order(self) -> list[int]:
        """Лево — узел — право."""
        result: list[int] = []

        def visit(node: Node | None) -> None:
            if node is None:
                return
            visit(node.left)
            result.append(node.key)
            visit(node.right)

        visit(self.root)
        return result

    def pre_order(self) -> list[int]:
        """Узел — лево — право."""
        result: list[int] = []

        def visit(node: Node | None) -> None:
            if node is None:
                return
            result.append(node.key)
            visit(node.left)
            visit(node.right)

        visit(self.root)
        return result

    def post_order(self) -> list[int]:
        """Лево — право — узел."""
        result: list[int] = []

        def visit(node: Node | None) -> None:
            if node is None:
                return
            visit(node.left)
            visit(node.right)
            result.append(node.key)

        visit(self.root)
        return result

    def level_order(self) -> list[int]:
        """Обход по уровням — итеративно с очередью deque."""
        if self.root is None:
            return []
        result: list[int] = []
        queue: deque[Node] = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return result

    def height(self) -> int:
        """Высота в рёбрах: пустое дерево -1, одиночный узел 0."""
        if self.root is None:
            return -1
        # Итеративный обход предотвращает переполнение рекурсии на цепочке.
        height = -1
        queue: deque[tuple[Node, int]] = deque([(self.root, 0)])
        while queue:
            node, depth = queue.popleft()
            height = max(height, depth)
            if node.left is not None:
                queue.append((node.left, depth + 1))
            if node.right is not None:
                queue.append((node.right, depth + 1))
        return height

    def size(self) -> int:
        """Число узлов."""
        count = 0
        stack: list[Node] = []
        current = self.root
        while stack or current is not None:
            while current is not None:
                stack.append(current)
                current = current.left
            current = stack.pop()
            count += 1
            current = current.right
        return count


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def build(keys) -> BST:
    tree = BST()
    for key in keys:
        tree.insert(key)
    return tree


def expect_search(tree: BST, key: int, present: bool, where: str) -> None:
    got = tree.search(key)
    expect(got is present, f"{where}: search({key}) вернул {got!r}, ожидалось {present}")


TRAVERSALS = ("in_order", "pre_order", "post_order", "level_order")

KNOWN_TREES = {
    (4, 2, 6, 1, 3, 5, 7): {
        "in_order": [1, 2, 3, 4, 5, 6, 7],
        "pre_order": [4, 2, 1, 3, 6, 5, 7],
        "post_order": [1, 3, 2, 5, 7, 6, 4],
        "level_order": [4, 2, 6, 1, 3, 5, 7],
        "height": 2, "size": 7,
    },
    (8, 3, 10, 1, 6, 14, 4, 7, 13): {
        "in_order": [1, 3, 4, 6, 7, 8, 10, 13, 14],
        "pre_order": [8, 3, 1, 6, 4, 7, 10, 14, 13],
        "post_order": [1, 4, 7, 6, 3, 13, 14, 10, 8],
        "level_order": [8, 3, 10, 1, 6, 14, 4, 7, 13],
        "height": 3, "size": 9,
    },
}

TWO_CHILDREN_CASES = [
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 3,
     [8, 4, 1, 6, 7, 10, 14, 13], [8, 1, 6, 4, 7, 10, 14, 13]),
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 6,
     [8, 3, 1, 7, 4, 10, 14, 13], [8, 3, 1, 4, 7, 10, 14, 13]),
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 8,
     [10, 3, 1, 6, 4, 7, 14, 13], [7, 3, 1, 6, 4, 10, 14, 13]),
    ((50, 30, 70, 60, 80, 65), 50,
     [60, 30, 70, 65, 80], [30, 70, 60, 65, 80]),
    ((50, 30, 70, 60, 80, 65), 70,
     [50, 30, 80, 60, 65], [50, 30, 65, 60, 80]),
]


def check_insert_search() -> None:
    tree = BST()
    expect_search(tree, 1, False, "пустое дерево")
    tree.insert(5)
    expect_search(tree, 5, True, "дерево из одного узла")
    expect_search(tree, 4, False, "дерево из одного узла")
    keys = [5, 2, 8, 1, 3, 9, 7]
    tree = build(keys)
    for k in range(11):
        expect_search(tree, k, k in keys, f"дерево {keys}")
    expect(tree.in_order() == sorted(keys), "in_order не отсортирован после вставки")
    tree.insert(5)
    tree.insert(1)
    expect(tree.in_order() == sorted(keys), "повторная вставка добавила дубликаты")


def check_traversals() -> None:
    empty = BST()
    for name in TRAVERSALS:
        expect(getattr(empty, name)() == [], f"{name} пустого дерева должен быть []")
    for keys, answers in KNOWN_TREES.items():
        tree = build(keys)
        for name in TRAVERSALS:
            expect(getattr(tree, name)() == answers[name],
                   f"{name} для {keys}: получено {getattr(tree, name)()}, ожидалось {answers[name]}")


def check_height_size() -> None:
    cases = [("пустое", (), -1, 0), ("один узел", (5,), 0, 1)]
    cases += [(str(keys), keys, a["height"], a["size"]) for keys, a in KNOWN_TREES.items()]
    cases.append(("цепочка", tuple(range(1, 51)), 49, 50))
    for label, keys, expected_h, expected_s in cases:
        tree = build(keys)
        expect(tree.height() == expected_h, f"{label}: высота {tree.height()}, ожидалось {expected_h}")
        expect(tree.size() == expected_s, f"{label}: размер {tree.size()}, ожидалось {expected_s}")


def check_delete() -> None:
    tree = BST()
    tree.delete(1)
    expect(tree.in_order() == [], "удаление из пустого дерева изменило дерево")
    keys = [5, 2, 8, 1, 3]
    tree = build(keys)
    reference = set(keys)
    steps = [(1, "лист"), (2, "один ребёнок"), (5, "корень с двумя детьми"),
             (4, "нет в дереве"), (3, "ключ 3"), (8, "последний ключ")]
    for key, label in steps:
        tree.delete(key)
        reference.discard(key)
        expect(tree.in_order() == sorted(reference), f"delete({key}), {label}: неверный in_order")
        expect_search(tree, key, False, f"delete({key}), {label}")
        for remain in reference:
            expect_search(tree, remain, True, f"delete({key}), {label}")
    expect(tree.height() == -1 and tree.size() == 0, "после удаления всех элементов дерево не пустое")


def check_delete_two_children() -> None:
    for keys, key, successor_order, predecessor_order in TWO_CHILDREN_CASES:
        tree = build(keys)
        tree.delete(key)
        expected = sorted(set(keys) - {key})
        expect(tree.in_order() == expected, f"delete({key}): неверный набор ключей")
        expect(tree.pre_order() in (successor_order, predecessor_order),
               f"delete({key}): форма дерева не соответствует преемнику/предшественнику")


def check_bst_invariant(tree: BST) -> None:
    """Дополнительная проверка строгого BST-инварианта через границы."""
    stack: list[tuple[Node, int | None, int | None]] = []
    if tree.root is not None:
        stack.append((tree.root, None, None))
    seen: set[int] = set()
    while stack:
        node, low, high = stack.pop()
        expect((low is None or node.key > low) and (high is None or node.key < high),
               f"нарушен BST-инвариант у ключа {node.key}, границы {low}..{high}")
        ident = id(node)
        expect(ident not in seen, "обнаружен цикл или повторная ссылка на узел")
        seen.add(ident)
        if node.left is not None:
            stack.append((node.left, low, node.key))
        if node.right is not None:
            stack.append((node.right, node.key, high))


def check_against_reference() -> None:
    rng = random.Random(0)
    for run in range(1, 101):
        tree = BST()
        reference: set[int] = set()
        for step in range(1, rng.randint(1, 80) + 1):
            key = rng.randint(-30, 30)
            if rng.random() < 0.6:
                tree.insert(key)
                reference.add(key)
                op = f"insert({key})"
            else:
                tree.delete(key)
                reference.discard(key)
                op = f"delete({key})"
            expect(tree.in_order() == sorted(reference),
                   f"прогон {run}, шаг {step}, {op}: in_order не совпал с эталоном")
            check_bst_invariant(tree)
        for key in range(-30, 31):
            expect_search(tree, key, key in reference, f"прогон {run}")
        expect(tree.size() == len(reference), f"прогон {run}: неправильный size")
        for name in TRAVERSALS:
            got = getattr(tree, name)()
            expect(sorted(got) == sorted(reference), f"прогон {run}: {name} содержит неверные ключи")
        n = len(reference)
        min_height = -1 if n == 0 else n.bit_length() - 1
        max_height = -1 if n == 0 else n - 1
        expect(min_height <= tree.height() <= max_height,
               f"прогон {run}: высота {tree.height()} вне допустимого диапазона")


def self_check() -> None:
    checks = (
        ("вставка и поиск", check_insert_search),
        ("обходы", check_traversals),
        ("высота и размер", check_height_size),
        ("удаление", check_delete),
        ("удаление узла с двумя детьми", check_delete_two_children),
        ("сверка с set/sorted и BST-инвариант", check_against_reference),
    )
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
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; benchmark отменён.")
    print("self_check: OK")


SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
QUERIES = 1_000


def bench(fn) -> float:
    """Медиана повторов; один предварительный вызов — прогрев."""
    fn()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_benchmarks(seed: int) -> None:
    rng = random.Random(seed)
    print("\nBenchmark: время для 1000 запросов; медиана 5 повторов (секунды)")
    print(f"{'n':>8} {'h(BST)':>8} {'t_bst':>12} {'t_list':>12}")
    for n in SIZES:
        keys = rng.sample(range(10 * n), n)
        # Часть запросов гарантированно существующая, часть — вероятно отсутствующая.
        queries = [keys[i] for i in range(QUERIES // 2)]
        queries += [rng.randint(10 * n, 20 * n) for _ in range(QUERIES - len(queries))]
        rng.shuffle(queries)
        tree = BST()
        for key in keys:
            tree.insert(key)

        def search_bst() -> None:
            for q in queries:
                tree.search(q)

        def search_list() -> None:
            for q in queries:
                q in keys

        bst_time = bench(search_bst)
        list_time = bench(search_list)
        print(f"{n:>8} {tree.height():>8} {bst_time:>12.6f} {list_time:>12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True, help="номер варианта")
    parser.add_argument("--skip-bench", action="store_true", help="выполнить тесты без замеров")
    args = parser.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    if not args.skip_bench:
        run_benchmarks(seed)


if __name__ == "__main__":
    main()
