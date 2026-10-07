#!/usr/bin/env python3
"""Okapi BM25 helpers for Cypher hard-negative mining."""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix

_TOKEN_RE = re.compile(
    r"\b(?:MATCH|WHERE|RETURN|WITH|CREATE|MERGE|DELETE|SET|REMOVE|ORDER|BY|LIMIT|SKIP|"
    r"UNWIND|CALL|AND|OR|NOT|XOR|IN|STARTS|ENDS|CONTAINS|IS|NULL|DISTINCT|AS|CASE|"
    r"WHEN|THEN|ELSE|END)\b"
    r"|<=|>=|<>|!=|[+\-*/%<>=^]"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+\.?\d*"
    r'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    r"|[()\[\]{},:;.]",
    re.IGNORECASE,
)


def tokenize(cypher: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(cypher or "")]


def build_tf(token_lists: List[List[str]], vocab: Dict[str, int]) -> csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    counts: Dict[tuple[int, int], int] = {}
    for row, tokens in enumerate(token_lists):
        for token in tokens:
            col = vocab.get(token)
            if col is None:
                continue
            key = (row, col)
            counts[key] = counts.get(key, 0) + 1
    data = np.empty(len(counts), dtype=np.float32)
    for i, ((row, col), count) in enumerate(counts.items()):
        rows.append(row)
        cols.append(col)
        data[i] = count
    return csr_matrix((data, (rows, cols)), shape=(len(token_lists), len(vocab)))


def bm25_weights(tf: csr_matrix, k1: float, b: float) -> tuple[csr_matrix, dict]:
    """Lucene-style Okapi BM25 document-term weights."""
    n_docs = tf.shape[0]
    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float64)
    avgdl = float(doc_len.mean()) if n_docs else 0.0
    length_norm = k1 * (1.0 - b + b * doc_len / avgdl)

    coo = tf.tocoo()
    denom = coo.data.astype(np.float64) + length_norm[coo.row]
    weights = idf[coo.col] * (coo.data * (k1 + 1.0) / denom)
    matrix = csr_matrix((weights.astype(np.float32), (coo.row, coo.col)), shape=tf.shape)
    stats = {
        "k1": k1,
        "b": b,
        "n_docs": int(n_docs),
        "vocab_size": int(tf.shape[1]),
        "avg_doc_length_tokens": avgdl,
        "idf": "lucene",
    }
    return matrix, stats
