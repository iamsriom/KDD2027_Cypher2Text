#!/usr/bin/env python3
"""Corrected Magnetic PE, pattern-graph SPD, and operator-only IDF.

These functions are the signal definitions. They do not read the
leakage-controlled test split.
"""

from __future__ import annotations

import math
import re

import networkx as nx
import numpy as np

from csencoder.mining.schema_similarity import _NODE_RE  # noqa: E402

Q_MAGNETIC = 0.25
PE_EIGS = 8
PE_DIM = PE_EIGS * 2
SPD_DMAX = 8
UNREACHABLE = SPD_DMAX + 1

KEYWORD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OPTIONAL_MATCH", re.compile(r"\bOPTIONAL\s+MATCH\b", re.IGNORECASE)),
    ("ORDER_BY", re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)),
    ("STARTS_WITH", re.compile(r"\bSTARTS\s+WITH\b", re.IGNORECASE)),
    ("ENDS_WITH", re.compile(r"\bENDS\s+WITH\b", re.IGNORECASE)),
    ("IS_NOT_NULL", re.compile(r"\bIS\s+NOT\s+NULL\b", re.IGNORECASE)),
    ("IS_NULL", re.compile(r"\bIS\s+NULL\b", re.IGNORECASE)),
    ("DETACH_DELETE", re.compile(r"\bDETACH\s+DELETE\b", re.IGNORECASE)),
    ("VARLEN_PATH", re.compile(r"\[[^\[\]]*\*[^\[\]]*\]")),
    ("EXISTS", re.compile(r"\bEXISTS\b", re.IGNORECASE)),
    ("WITH", re.compile(r"\bWITH\b", re.IGNORECASE)),
    ("UNWIND", re.compile(r"\bUNWIND\b", re.IGNORECASE)),
    ("CASE", re.compile(r"\bCASE\b", re.IGNORECASE)),
    ("UNION", re.compile(r"\bUNION\b", re.IGNORECASE)),
    ("CALL", re.compile(r"\bCALL\b", re.IGNORECASE)),
    ("COUNT", re.compile(r"\bCOUNT\s*\(", re.IGNORECASE)),
    ("SUM", re.compile(r"\bSUM\s*\(", re.IGNORECASE)),
    ("AVG", re.compile(r"\bAVG\s*\(", re.IGNORECASE)),
    ("MIN", re.compile(r"\bMIN\s*\(", re.IGNORECASE)),
    ("MAX", re.compile(r"\bMAX\s*\(", re.IGNORECASE)),
    ("COLLECT", re.compile(r"\bCOLLECT\s*\(", re.IGNORECASE)),
    ("STDEV", re.compile(r"\bSTDEV\s*\(", re.IGNORECASE)),
    ("STDEVP", re.compile(r"\bSTDEVP\s*\(", re.IGNORECASE)),
    ("PERCENTILECONT", re.compile(r"\bPERCENTILECONT\s*\(", re.IGNORECASE)),
    ("PERCENTILEDISC", re.compile(r"\bPERCENTILEDISC\s*\(", re.IGNORECASE)),
    ("FOREACH", re.compile(r"\bFOREACH\b", re.IGNORECASE)),
    ("REDUCE", re.compile(r"\bREDUCE\s*\(", re.IGNORECASE)),
    ("FILTER", re.compile(r"\bFILTER\s*\(", re.IGNORECASE)),
    ("LIMIT", re.compile(r"\bLIMIT\b", re.IGNORECASE)),
    ("SKIP", re.compile(r"\bSKIP\b", re.IGNORECASE)),
    ("DISTINCT", re.compile(r"\bDISTINCT\b", re.IGNORECASE)),
    ("WHERE", re.compile(r"\bWHERE\b", re.IGNORECASE)),
    ("RETURN", re.compile(r"\bRETURN\b", re.IGNORECASE)),
    ("MATCH", re.compile(r"\bMATCH\b", re.IGNORECASE)),
    ("CREATE", re.compile(r"\bCREATE\b", re.IGNORECASE)),
    ("MERGE", re.compile(r"\bMERGE\b", re.IGNORECASE)),
    ("DELETE", re.compile(r"\bDELETE\b", re.IGNORECASE)),
    ("SET", re.compile(r"\bSET\b", re.IGNORECASE)),
    ("REMOVE", re.compile(r"\bREMOVE\b", re.IGNORECASE)),
    ("YIELD", re.compile(r"\bYIELD\b", re.IGNORECASE)),
    ("XOR", re.compile(r"\bXOR\b", re.IGNORECASE)),
    ("NOT", re.compile(r"\bNOT\b", re.IGNORECASE)),
    ("AND", re.compile(r"\bAND\b", re.IGNORECASE)),
    ("OR", re.compile(r"\bOR\b", re.IGNORECASE)),
    ("CONTAINS", re.compile(r"\bCONTAINS\b", re.IGNORECASE)),
]


def keywords_of(cypher: str) -> frozenset[str]:
    text = cypher or ""
    return frozenset(name for name, pattern in KEYWORD_PATTERNS if pattern.search(text))


def idf_value(df: int, n_docs: int) -> float:
    return math.log((n_docs + 1) / (df + 1))


def pattern_edges(cypher: str) -> list[tuple[int, int, str]]:
    nodes = list(_NODE_RE.finditer(cypher or ""))
    edges: list[tuple[int, int, str]] = []
    for left, right in zip(nodes, nodes[1:]):
        gap = (cypher or "")[left.end() : right.start()]
        if "-" not in gap:
            continue
        if "->" in gap and "<-" not in gap:
            direction = "out"
        elif "<-" in gap and "->" not in gap:
            direction = "in"
        else:
            direction = "undirected"
        edges.append((nodes.index(left), nodes.index(right), direction))
    return edges


def pattern_digraph(cypher: str) -> nx.DiGraph:
    """Pattern nodes, with a directed edge for each relationship arrow."""
    edges = pattern_edges(cypher)
    count = 0
    if edges:
        count = max(max(left, right) for left, right, _ in edges) + 1
    else:
        count = len(list(_NODE_RE.finditer(cypher or "")))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(count))
    for left, right, direction in edges:
        if direction == "out":
            graph.add_edge(left, right)
        elif direction == "in":
            graph.add_edge(right, left)
        else:
            graph.add_edge(left, right)
            graph.add_edge(right, left)
    return graph


def magnetic_pe(cypher: str, q: float = Q_MAGNETIC, k: int = PE_EIGS) -> np.ndarray:
    """Real and imaginary parts of the Hermitian magnetic Laplacian eigenvectors.

    Uses the same unnormalized MagPE as ``graph_builder.compute_magnetic_laplacian``:
    ``L = D_s - A_s ⊙ exp(i 2π q (A - A^T))`` with phase canonicalization.
    Shape is ``(n_nodes, 2k)``.
    """
    from csencoder.preprocessing.graph_builder import compute_magnetic_laplacian

    graph = pattern_digraph(cypher)
    _values, pe = compute_magnetic_laplacian(graph, q=q, k=k)
    return pe


def complex_pe(pe: np.ndarray) -> np.ndarray:
    width = pe.shape[1] // 2
    return pe[:, :width] + 1j * pe[:, width:]


def phase_invariant_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute cosine of the complex PE, invariant to a global complex phase."""
    if left.shape != right.shape:
        width = max(left.shape[0], right.shape[0])
        features = left.shape[1]
        padded_left = np.zeros((width, features), dtype=np.float64)
        padded_right = np.zeros((width, features), dtype=np.float64)
        padded_left[: left.shape[0]] = left
        padded_right[: right.shape[0]] = right
        left, right = padded_left, padded_right
    a = complex_pe(left).ravel()
    b = complex_pe(right).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0
    return float(np.abs(np.vdot(a, b)) / denom)


def pattern_spd(cypher: str, dmax: int = SPD_DMAX) -> np.ndarray:
    """Directed pattern distances. Unreachable pairs use a dedicated bucket."""
    graph = pattern_digraph(cypher)
    n_nodes = graph.number_of_nodes()
    distance = np.full((n_nodes, n_nodes), UNREACHABLE, dtype=np.int16)
    if n_nodes == 0:
        return distance
    for source, targets in nx.all_pairs_shortest_path_length(graph):
        for target, length in targets.items():
            if source == target:
                distance[source, target] = 0
            else:
                distance[source, target] = min(int(length), dmax)
    return distance


def max_finite_pattern_spd(cypher: str) -> float:
    distance = pattern_spd(cypher)
    finite = distance[(distance >= 0) & (distance < UNREACHABLE) & (distance > 0)]
    if finite.size == 0:
        return 0.0
    return float(finite.max())


def direction_features(cypher: str) -> dict:
    edges = pattern_edges(cypher)
    directed = [edge for edge in edges if edge[2] in ("out", "in")]
    width = 0
    if edges:
        width = max(max(left, right) for left, right, _ in edges) + 1
    incoming = [0] * width
    outgoing = [0] * width
    for left, right, direction in directed:
        if direction == "out":
            outgoing[left] += 1
            incoming[right] += 1
        else:
            incoming[left] += 1
            outgoing[right] += 1
    mixed_nodes = sum(1 for inn, out in zip(incoming, outgoing) if inn > 0 and out > 0)
    changes = 0
    previous = None
    for _left, _right, direction in directed:
        if previous is not None and direction != previous:
            changes += 1
        previous = direction
    return {
        "pattern_hops": len(edges),
        "n_directed_edges": len(directed),
        "direction_changes": changes,
        "directional_complex": len(directed) >= 2 and (mixed_nodes > 0 or changes > 0),
        "max_pattern_spd": max_finite_pattern_spd(cypher),
    }


def operator_idf(queries: list[str]) -> dict[str, float]:
    counts = {name: 0 for name, _pattern in KEYWORD_PATTERNS}
    for query in queries:
        for name in keywords_of(query):
            counts[name] += 1
    n_docs = len(queries)
    return {name: idf_value(df, n_docs) for name, df in counts.items()}


def rarity(cypher: str, idf: dict[str, float]) -> float:
    present = keywords_of(cypher)
    if not present:
        return 0.0
    return max(idf[name] for name in present)


def operator_token_weights(cypher: str, tokens: list[str], idf: dict[str, float], alpha: float = 1.0) -> np.ndarray:
    """1 + alpha * IDF on tokens covered by an operator match; 1 elsewhere."""
    weights = np.ones(len(tokens), dtype=np.float64)
    if not tokens:
        return weights
    spans = []
    cursor = 0
    lowered = cypher.lower()
    for token in tokens:
        found = lowered.find(token.lower(), cursor)
        if found < 0:
            found = lowered.find(token.lower())
        if found < 0:
            spans.append((cursor, cursor))
            continue
        spans.append((found, found + len(token)))
        cursor = found + len(token)
    for name, pattern in KEYWORD_PATTERNS:
        for match in pattern.finditer(cypher or ""):
            boost = 1.0 + alpha * idf[name]
            for index, (start, end) in enumerate(spans):
                if start < match.end() and end > match.start():
                    weights[index] = max(weights[index], boost)
    return weights
