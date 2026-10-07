#!/usr/bin/env python3
"""Compatibility signatures used when selecting hard positives / negatives."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from csencoder.mining.corrected_signals import keywords_of, pattern_edges, pattern_spd
from csencoder.mining.schema_similarity import jaccard, schema_parts, tokenize

_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_BOUND_VAR_RE = re.compile(r"(?<=[\(\[])\s*([A-Za-z_][A-Za-z0-9_]*)(?=\s*(?:[:\{\]\)\*]))")
_CYPHER_KEYWORDS = {
    "MATCH", "WHERE", "RETURN", "WITH", "CREATE", "MERGE", "DELETE", "SET", "REMOVE",
    "ORDER", "BY", "LIMIT", "SKIP", "UNWIND", "CALL", "AND", "OR", "NOT", "XOR", "IN",
    "STARTS", "ENDS", "CONTAINS", "IS", "NULL", "DISTINCT", "AS", "CASE", "WHEN",
    "THEN", "ELSE", "END", "OPTIONAL", "DETACH", "YIELD", "FOREACH", "FILTER",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "COLLECT", "EXISTS", "TRUE", "FALSE",
}


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().rstrip(";").split())


def direction_counter(cypher: str) -> Counter:
    counts: Counter = Counter()
    for _left, _right, direction in pattern_edges(cypher):
        if direction in ("in", "out"):
            counts[direction] += 1
    return counts


def spd_vector(cypher: str) -> tuple[np.ndarray, float]:
    distance = pattern_spd(cypher)
    counts = np.zeros(9, dtype=np.float64)
    finite: list[int] = []
    n_nodes = int(distance.shape[0])
    for left in range(n_nodes):
        for right in range(n_nodes):
            if left == right:
                continue
            value = int(distance[left, right])
            if value <= 0:
                continue
            counts[min(value, 9) - 1] += 1
            if value < 9:
                finite.append(value)
    return counts, (float(max(finite)) if finite else 0.0)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 1.0
    return float(left @ right / denom)


def multiset_jaccard(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    keys = set(left) | set(right)
    overlap = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return overlap / union if union else 1.0


class Signature:
    def __init__(self, cypher: str) -> None:
        self.cypher = cypher
        self.norm = normalize(cypher)
        self.tokens = tokenize(cypher)
        self.labels, self.rels, self.props = schema_parts(cypher)
        self.keywords = keywords_of(cypher)
        self.direction = direction_counter(cypher)
        self.n_directed = int(sum(self.direction.values()))
        self.spd, self.max_spd = spd_vector(cypher)

    def view(self, other: "Signature") -> dict[str, float]:
        schema = float(np.mean([
            jaccard(self.labels, other.labels),
            jaccard(self.rels, other.rels),
            jaccard(self.props, other.props),
        ]))
        return {
            "lexical": jaccard(self.tokens, other.tokens),
            "schema": schema,
            "direction": multiset_jaccard(self.direction, other.direction),
            "spd": cosine(self.spd, other.spd),
            "operator": jaccard(self.keywords, other.keywords),
            "max_spd_gap": abs(self.max_spd - other.max_spd),
        }


def shares_schema(anchor: Signature, other: Signature) -> bool:
    return bool(anchor.labels & other.labels) or bool(anchor.rels & other.rels)


def path_wrong(view: dict[str, float]) -> bool:
    return view["spd"] < 0.85 or view["max_spd_gap"] >= 1.0


def direction_wrong(view: dict[str, float]) -> bool:
    return view["direction"] < 0.999


def compatible(anchor: Signature, other: Signature, view: dict[str, float], level: str) -> bool:
    if not shares_schema(anchor, other):
        return False
    if direction_wrong(view):
        return False
    if level == "strict":
        return view["schema"] >= 0.34 and view["spd"] >= 0.90 and view["max_spd_gap"] < 1.0 and view["operator"] >= 0.5
    if level == "relaxed":
        return view["schema"] >= 0.20 and view["spd"] >= 0.80 and view["max_spd_gap"] < 1.5
    return view["schema"] >= 0.20 and view["spd"] >= 0.75


def compatible_score(view: dict[str, float]) -> float:
    return 0.50 * view["schema"] + 0.20 * view["direction"] + 0.20 * view["spd"] + 0.10 * view["operator"]
