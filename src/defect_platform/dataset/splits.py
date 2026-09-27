"""Deterministic class-aware group splitting."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

from defect_platform.contracts import SplitSpec


def assign_splits(
    labels: list[str], groups: list[str], spec: SplitSpec
) -> list[str]:
    """Assign whole duplicate/product groups while approximately stratifying labels.

    A deterministic greedy objective balances per-class counts and total counts.
    Product or batch identifiers should be supplied as group ids where available.
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have equal lengths")
    if not labels:
        return []
    names = ("train", "validation", "test")
    fractions = (spec.train, spec.validation, spec.test)
    class_totals = Counter(labels)
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_rows[group].append(index)
    # Largest and most class-concentrated groups are assigned first. Seeded hash
    # supplies a stable tie-break without depending on Python's process hash seed.
    def order(group: str) -> tuple[float, str]:
        rows = group_rows[group]
        return (-len(rows), hashlib.sha256(f"{spec.seed}:{group}".encode()).hexdigest())

    ordered = sorted(group_rows, key=order)
    assigned_counts = {split: Counter() for split in names}
    assigned_total = Counter()
    target = {
        split: {label: class_totals[label] * fraction for label in class_totals}
        for split, fraction in zip(names, fractions)
    }
    target_total = {split: len(labels) * fraction for split, fraction in zip(names, fractions)}
    group_split: dict[str, str] = {}
    rng = random.Random(spec.seed)
    tie_order = list(names)
    rng.shuffle(tie_order)
    tie_rank = {split: i for i, split in enumerate(tie_order)}
    for group in ordered:
        group_counts = Counter(labels[i] for i in group_rows[group])
        group_size = len(group_rows[group])

        def cost(split: str) -> tuple[float, int]:
            # Penalize normalized per-class deviation more than total imbalance.
            class_cost = 0.0
            for name in names:
                for label, total in class_totals.items():
                    before = assigned_counts[name][label] - target[name][label]
                    after_count = assigned_counts[name][label]
                    if name == split:
                        after_count += group_counts[label]
                    after = after_count - target[name][label]
                    class_cost += (after * after - before * before) / max(total, 1)
            total_cost = 0.25 * sum(
                ((assigned_total[name] + (group_size if name == split else 0) - target_total[name]) ** 2
                 - (assigned_total[name] - target_total[name]) ** 2)
                / max(len(labels), 1)
                for name in names
            )
            return class_cost + total_cost, tie_rank[split]

        selected = min(names, key=cost)
        group_split[group] = selected
        assigned_total[selected] += group_size
        assigned_counts[selected].update(group_counts)
    return [group_split[group] for group in groups]
