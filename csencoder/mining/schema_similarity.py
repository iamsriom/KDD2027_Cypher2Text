#!/usr/bin/env python3
"""Schema / lexical helpers used by hard-pair mining."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Set, Tuple

_NODE_RE = re.compile(r"\((?:[^()\"']|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')*\)")
_REL_TYPE_RE = re.compile(r"\[\s*(?:[A-Za-z_][A-Za-z0-9_]*)?\s*:([A-Za-z_][A-Za-z0-9_]*)")
_LABEL_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
_PROP_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(cypher: str) -> Set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(cypher or "")}


def jaccard(a: Iterable, b: Iterable) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def labels_of(node_text: str) -> Set[str]:
    return {name.lower() for name in _LABEL_RE.findall(node_text or "")}


def schema_parts(cypher: str) -> Tuple[Set[str], Set[str], Set[str]]:
    labels: Set[str] = set()
    for node in _NODE_RE.finditer(cypher or ""):
        labels |= labels_of(node.group(0))
    relations = {name.lower() for name in _REL_TYPE_RE.findall(cypher or "")}
    props = {m.group(2).lower() for m in _PROP_RE.finditer(cypher or "")}
    return labels, relations, props
