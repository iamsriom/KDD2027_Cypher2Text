#!/usr/bin/env python3
"""Mining metrics for hard positives and hard negatives.

These scores are used only to choose training pairs. They do not change
CSEncoder's architecture or its operator-IDF input.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from csencoder.mining.corrected_signals import pattern_spd  # noqa: E402
from csencoder.mining.schema_similarity import _NODE_RE  # noqa: E402

_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_REL_GAP_RE = re.compile(r"<-|->|--|-\s*\[|\]\s*-")
_LABEL_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")
_KEYWORDS = {
    "MATCH", "WHERE", "RETURN", "WITH", "CREATE", "MERGE", "DELETE", "SET", "REMOVE",
    "ORDER", "BY", "LIMIT", "SKIP", "UNWIND", "CALL", "AND", "OR", "NOT", "XOR", "IN",
    "STARTS", "ENDS", "CONTAINS", "IS", "NULL", "DISTINCT", "AS", "CASE", "WHEN",
    "THEN", "ELSE", "END", "OPTIONAL", "DETACH", "YIELD", "ASC", "DESC", "TRUE", "FALSE",
}

_COMPOUND_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bOPTIONAL\s+MATCH\b", re.IGNORECASE), "OPTIONAL_MATCH"),
    (re.compile(r"\bSTARTS\s+WITH\b", re.IGNORECASE), "STARTS_WITH"),
    (re.compile(r"\bENDS\s+WITH\b", re.IGNORECASE), "ENDS_WITH"),
    (re.compile(r"\bIS\s+NOT\s+NULL\b", re.IGNORECASE), "IS_NOT_NULL"),
    (re.compile(r"\bIS\s+NULL\b", re.IGNORECASE), "IS_NULL"),
    (re.compile(r"\bNOT\s+IN\b", re.IGNORECASE), "NOT_IN"),
]

_SINGLE_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("CONTAINS", re.compile(r"\bCONTAINS\b", re.IGNORECASE)),
    ("WHERE", re.compile(r"\bWHERE\b", re.IGNORECASE)),
    ("RETURN", re.compile(r"\bRETURN\b", re.IGNORECASE)),
    ("WITH", re.compile(r"\bWITH\b", re.IGNORECASE)),
    ("UNWIND", re.compile(r"\bUNWIND\b", re.IGNORECASE)),
    ("EXISTS", re.compile(r"\bEXISTS\b", re.IGNORECASE)),
    ("DISTINCT", re.compile(r"\bDISTINCT\b", re.IGNORECASE)),
    ("LIMIT", re.compile(r"\bLIMIT\b", re.IGNORECASE)),
    ("MATCH", re.compile(r"\bMATCH\b", re.IGNORECASE)),
    ("IN", re.compile(r"\bIN\b", re.IGNORECASE)),
]

_AGGREGATES: list[tuple[str, re.Pattern[str]]] = [
    ("COUNT", re.compile(r"\bCOUNT\s*\(", re.IGNORECASE)),
    ("AVG", re.compile(r"\bAVG\s*\(", re.IGNORECASE)),
    ("SUM", re.compile(r"\bSUM\s*\(", re.IGNORECASE)),
    ("MIN", re.compile(r"\bMIN\s*\(", re.IGNORECASE)),
    ("MAX", re.compile(r"\bMAX\s*\(", re.IGNORECASE)),
]

_COMPARATORS = [
    ("LTE", re.compile(r"<=")),
    ("GTE", re.compile(r">=")),
    ("NEQ", re.compile(r"<>|!=")),
    ("LT", re.compile(r"(?<![<>!\-])<(?![\-=])")),
    ("GT", re.compile(r"(?<![<>!\-])>(?![\-=])")),
    ("EQ", re.compile(r"(?<![<>!])=(?!=)")),
]

_ORDER_RE = re.compile(
    r"\bORDER\s+BY\b(?P<body>.*?)(?=\b(?:RETURN|WHERE|LIMIT|SKIP|MATCH|WITH|UNWIND|CALL|UNION|OPTIONAL)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_PROP_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)|(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\b")


def sanitize_literals(cypher: str) -> str:
    """Replace quoted Cypher strings so English words inside them are not operators."""
    return _STRING_RE.sub(" '' ", cypher or "")


def _labels(node_text: str) -> str:
    names = [match.group(1).lower() for match in _LABEL_RE.finditer(node_text or "")]
    return "|".join(sorted(names))


def _rel_type(gap: str) -> str:
    names = [match.group(1).lower() for match in _LABEL_RE.finditer(gap or "")]
    return "|".join(sorted(names))


def canonical_edges(cypher: str) -> Counter:
    """Semantic relationship edges. Arrow text is rewritten into source -> target."""
    nodes = list(_NODE_RE.finditer(cypher or ""))
    edges: Counter = Counter()
    for left, right in zip(nodes, nodes[1:]):
        gap = (cypher or "")[left.end():right.start()]
        if not _REL_GAP_RE.search(_STRING_RE.sub(" ", gap)):
            continue
        left_label = _labels(left.group(0))
        right_label = _labels(right.group(0))
        rel_type = _rel_type(gap)
        outgoing = "->" in gap and "<-" not in gap
        incoming = "<-" in gap and "->" not in gap
        if outgoing:
            edges[(left_label, rel_type, right_label, "DIRECTED")] += 1
        elif incoming:
            edges[(right_label, rel_type, left_label, "DIRECTED")] += 1
        else:
            ends = tuple(sorted((left_label, right_label)))
            edges[(ends, rel_type, "UNDIRECTED")] += 1
    return edges


def _multiset_jaccard(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    keys = set(left) | set(right)
    overlap = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return overlap / union if union else 1.0


def direction_similarity(left: str, right: str) -> float:
    return _multiset_jaccard(canonical_edges(left), canonical_edges(right))


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + (" " * (end - start)) + text[end:]


def _property_name(fragment: str) -> str:
    cleaned = re.sub(r"\b(?:ASC|DESC)\b", " ", fragment or "", flags=re.IGNORECASE)
    dotted = list(re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)", cleaned))
    if dotted:
        return dotted[-1].group(2).lower()
    idents = [
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
        if match.group(0).upper() not in _KEYWORDS
    ]
    return idents[-1] if idents else ""


def semantic_operators(cypher: str) -> frozenset[str]:
    """Normalized mining signature. Compound Cypher forms stay atomic."""
    work = sanitize_literals(cypher)
    found: set[str] = set()

    for match in list(_ORDER_RE.finditer(work)):
        body = match.group("body") or ""
        pieces = [piece for piece in body.split(",") if piece.strip()] or [""]
        for piece in pieces:
            prop = _property_name(piece)
            if re.search(r"\bDESC\b", piece, flags=re.IGNORECASE):
                found.add("ORDER_BY_DESC")
                found.add(f"SORT:{prop}:DESC")
            else:
                found.add("ORDER_BY_ASC")
                found.add(f"SORT:{prop}:ASC")
        work = _blank(work, match.start(), match.end())

    for pattern, name in _COMPOUND_REPLACEMENTS:
        if pattern.search(work):
            found.add(name)
            work = pattern.sub(" ", work)

    occupied = [False] * (len(work) + 1)
    for name, pattern in _COMPARATORS:
        for match in pattern.finditer(work):
            if any(occupied[match.start():match.end()]):
                continue
            prop = _property_name(work[max(0, match.start() - 40):match.start()])
            if not prop:
                continue
            found.add(name)
            found.add(f"PRED:{prop}:{name}")
            for index in range(match.start(), match.end()):
                occupied[index] = True
    chars = list(work)
    for index, taken in enumerate(occupied[:len(chars)]):
        if taken:
            chars[index] = " "
    work = "".join(chars)

    for name, pattern in _AGGREGATES:
        if pattern.search(work):
            found.add(name)
    for name, pattern in _SINGLE_KEYWORDS:
        if pattern.search(work):
            found.add(name)
    return frozenset(found)


def operator_similarity(left: str, right: str) -> float:
    return _jaccard(semantic_operators(left), semantic_operators(right))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _spd_profile(cypher: str) -> tuple[np.ndarray, float]:
    distance = pattern_spd(cypher)
    counts = np.zeros(9, dtype=np.float64)
    finite: list[int] = []
    width = int(distance.shape[0])
    for left in range(width):
        for right in range(width):
            if left == right:
                continue
            value = int(distance[left, right])
            if value <= 0:
                continue
            counts[min(value, 9) - 1] += 1
            if value < 9:
                finite.append(value)
    return counts, (float(max(finite)) if finite else 0.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(left @ right / denom)


def path_metrics(left: str, right: str) -> dict[str, float | bool]:
    """Path agreement. Two queries with no relationships are equally trivial."""
    left_edges = sum(canonical_edges(left).values())
    right_edges = sum(canonical_edges(right).values())
    if left_edges == 0 and right_edges == 0:
        return {"path": 1.0, "path_applicable": False, "max_spd_gap": 0.0}
    if left_edges == 0 or right_edges == 0:
        nonempty = left if left_edges else right
        _counts, longest = _spd_profile(nonempty)
        return {"path": 0.0, "path_applicable": True, "max_spd_gap": float(longest)}
    left_counts, left_max = _spd_profile(left)
    right_counts, right_max = _spd_profile(right)
    return {
        "path": _cosine(left_counts, right_counts),
        "path_applicable": True,
        "max_spd_gap": abs(left_max - right_max),
    }


def graph_similarity(left: str, right: str) -> float:
    left_hops = sum(canonical_edges(left).values())
    right_hops = sum(canonical_edges(right).values())
    return 1.0 - abs(left_hops - right_hops) / max(left_hops, right_hops, 1)


def positive_score(schema: float, direction: float, path: float, operators: float) -> float:
    """Unchanged weights. Only the component definitions are corrected."""
    return 0.50 * schema + 0.20 * direction + 0.20 * path + 0.10 * operators


def qualifies_as_path_hard_negative(metrics: dict) -> bool:
    if not metrics.get("path_applicable", False):
        return False
    return (
        float(metrics["schema"]) >= 0.70
        and float(metrics["operator"]) >= 0.70
        and float(metrics["direction"]) >= 0.80
        and float(metrics["path"]) <= 0.50
    )
