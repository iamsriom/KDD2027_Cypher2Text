#!/usr/bin/env python3
"""Hard positives and hard negatives for every training Cypher query.

Dataset: neo4j/text2cypher-2024v1, training split only.
This script does not train CSEncoder and does not read validation or test.

Artifacts are written under data/supervision/ in the KDD2027_Cypher2Text repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csencoder.mining.contrast import (  # noqa: E402
    Signature,
    compatible,
    compatible_score,
    direction_wrong,
    normalize,
    path_wrong,
    shares_schema,
    _BOUND_VAR_RE,
    _CYPHER_KEYWORDS,
    _STRING_RE,
)
from csencoder.mining.corrected_signals import operator_idf, pattern_edges  # noqa: E402
from csencoder.mining.bm25 import bm25_weights, build_tf, tokenize as bm25_tokenize  # noqa: E402
from csencoder.mining.schema_similarity import jaccard, schema_parts  # noqa: E402
from csencoder.mining.mining_metrics import (  # noqa: E402
    direction_similarity,
    graph_similarity,
    path_metrics,
    positive_score,
    semantic_operators,
)

ROOT = REPO_ROOT / "data" / "supervision"
ACTIVE_ROOT = ROOT
DATA = ROOT / "cache"
RETRIEVAL = ROOT / "retrieval"
CANDIDATES = ROOT / "candidates"
SELECTIONS = ROOT
REPORTS = ROOT / "reports"

DATASET = "neo4j/text2cypher-2024v1"
EXPECTED_TRAIN = 39554
# Optional offline datasets (only required when --dataset is not neo4j).
MIND_THE_QUERY_TRAIN = Path(
    os.environ.get("MIND_THE_QUERY_TRAIN", str(REPO_ROOT / "data" / "external" / "mind_the_query"))
)
CYPHERBENCH_DIR = Path(
    os.environ.get("CYPHERBENCH_DIR", str(REPO_ROOT / "data" / "external" / "cypherbench"))
)
# Optional cache; if missing or wrong length, mine encodes with MPNET_MODEL.
MPNET_SOURCE = ROOT / "retrieval" / "mpnet_train_cypher_embeddings_v1.npy"
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 50
K1 = 1.2
B = 0.75
MARGIN = 0.10
FALLBACK_CEILING = 0.55
BM25_HIGH_RANK = 20
MPNET_HIGH_RANK = 20
LEXICAL_FLOOR = 0.18
OPERATOR_POSITIVE_MIN = 0.5
CORE_OPERATORS = frozenset({"MATCH", "RETURN"})
LEVELS = ("strict", "relaxed", "loose")

METRIC_NAMES = [
    "lexical",
    "label",
    "relationship",
    "property",
    "schema",
    "operator",
    "rare_operator",
    "graph",
    "direction",
    "path",
    "max_spd_gap",
    "positive_score",
    "bm25_score",
    "bm25_rank",
    "mpnet_cosine",
    "mpnet_rank",
]


def rename_bound_variables(cypher: str) -> tuple[str, bool]:
    """Bound-variable rename from TrainingQueryContrast.rename_variables.

    The original replaces every whole-word occurrence, so a variable whose
    name matches its label or relationship type, ignoring case, also rewrites
    that schema token. Positions after ':' or '.' are left alone here.
    """
    # bound-var regexes imported from csencoder.mining.contrast

    masked: list[str] = []

    def hold(match: re.Match[str]) -> str:
        masked.append(match.group(0))
        return f"__STR{len(masked) - 1}__"

    protected = _STRING_RE.sub(hold, cypher or "")
    found: list[str] = []
    seen: set[str] = set()
    for match in _BOUND_VAR_RE.finditer(protected):
        name = match.group(1)
        key = name.lower()
        if name.startswith("__") or name.upper() in _CYPHER_KEYWORDS or key in seen:
            continue
        seen.add(key)
        found.append(name)
    if not found:
        return cypher, False
    alphabet = list("xyzabcdefghijklmnopqrstuvw")
    fresh = [letter for letter in alphabet if letter not in seen]
    if len(fresh) < len(found):
        fresh = [f"v{index}" for index in range(len(found))]
    mapping = {old: fresh[index] for index, old in enumerate(found)}
    def replace_variable(text: str, old: str, placeholder: str) -> str:
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", flags=re.IGNORECASE)

        def repl(match: re.Match[str]) -> str:
            start = match.start()
            previous = text[start - 1] if start else ""
            # Labels, relationship types, and property lookups stay.
            if previous in ":.":
                return match.group(0)
            # Map keys such as {capital: true} stay. A pattern variable
            # written (var:Label) or [var:TYPE] is still renamed.
            if re.match(r"\s*:", text[match.end():]) and previous not in "([":
                return match.group(0)
            return placeholder

        return pattern.sub(repl, text)

    edited = protected
    for index, old in enumerate(found):
        edited = replace_variable(edited, old, f"__VAR{index}__")
    for index, old in enumerate(found):
        edited = edited.replace(f"__VAR{index}__", mapping[old])
    for index, value in enumerate(masked):
        edited = edited.replace(f"__STR{index}__", value)
    return edited, edited != cypher


def blank_bound_variables(cypher: str) -> str:
    """Replace bound variables with one token so schema extractors ignore them."""
    # bound-var regexes imported from csencoder.mining.contrast

    protected = _STRING_RE.sub(lambda match: match.group(0), cypher or "")
    names: list[str] = []
    seen: set[str] = set()
    for match in _BOUND_VAR_RE.finditer(protected):
        name = match.group(1)
        if name.startswith("__") or name.upper() in _CYPHER_KEYWORDS or name.lower() in seen:
            continue
        seen.add(name.lower())
        names.append(name)
    edited = cypher or ""
    for name in names:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", flags=re.IGNORECASE)

        def repl(match: re.Match[str], source: str = edited) -> str:
            start = match.start()
            previous = source[start - 1] if start else ""
            if previous in ":.":
                return match.group(0)
            if re.match(r"\s*:", source[match.end():]) and previous not in "([":
                return match.group(0)
            return "qv"

        edited = pattern.sub(repl, edited)
    return edited


def score_variable_rename(anchor: Signature, renamed: str, anchor_hops: int, idf: dict[str, float]) -> dict[str, float]:
    """Score p1 after aligning variable tokens, so a variable after ':' is not a new label."""
    renamed_sig = Signature(renamed)
    view = score_pair(anchor, renamed_sig, anchor_hops, len(pattern_edges(renamed)), idf)
    original_schema = schema_parts(blank_bound_variables(anchor.cypher))
    renamed_schema = schema_parts(blank_bound_variables(renamed))
    view["label"] = jaccard(original_schema[0], renamed_schema[0])
    view["relationship"] = jaccard(original_schema[1], renamed_schema[1])
    view["property"] = jaccard(original_schema[2], renamed_schema[2])
    view["schema"] = float(np.mean([view["label"], view["relationship"], view["property"]]))
    view["positive_score"] = positive_score(view["schema"], view["direction"], view["path"], view["operator"])
    return view


def cypher_text(raw: str) -> str:
    """Same Cypher string the existing MPNet retrieval embedded."""
    text = (raw or "").strip()
    return text if text else " "


def configure(name: str) -> None:
    """Point outputs at one training set. The default keeps the Neo4j files in place."""
    global ACTIVE_ROOT, DATA, RETRIEVAL, CANDIDATES, SELECTIONS, REPORTS, DATASET, EXPECTED_TRAIN
    if name == "neo4j":
        ACTIVE_ROOT = ROOT
        DATASET = "neo4j/text2cypher-2024v1"
        EXPECTED_TRAIN = 39554
    elif name == "mind_the_query":
        ACTIVE_ROOT = ROOT / "mind_the_query"
        DATASET = "mind-the-query/automated"
        EXPECTED_TRAIN = 19245
    elif name == "cypherbench":
        ACTIVE_ROOT = ROOT / "cypherbench"
        DATASET = "megagonlabs/cypherbench"
        EXPECTED_TRAIN = 8534
    else:
        raise SystemExit(f"unknown dataset {name}")
    DATA = ACTIVE_ROOT / "data"
    RETRIEVAL = ACTIVE_ROOT / "retrieval"
    CANDIDATES = ACTIVE_ROOT / "candidates"
    SELECTIONS = ACTIVE_ROOT / "selections"
    REPORTS = ACTIVE_ROOT / "reports"


def cached_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) == EXPECTED_TRAIN:
        return rows
    return None


def write_rows(path: Path, rows: list[dict]) -> list[dict]:
    if len(rows) != EXPECTED_TRAIN:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN} training rows, found {len(rows)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def training_row(index: int, instance_id: str, question: str, cypher: str, data_source: str, alias: str, schema: str) -> dict:
    return {
        "train_index": index,
        "instance_id": instance_id,
        "question": question,
        "cypher": cypher_text(cypher),
        "data_source": data_source,
        "database_reference_alias": alias,
        "schema": schema,
    }


def load_neo4j() -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train")
    rows = []
    n = min(len(dataset), EXPECTED_TRAIN)
    for index in range(n):
        row = dataset[index]
        rows.append(training_row(
            index,
            row.get("instance_id") or "",
            row.get("question") or "",
            row.get("cypher") or "",
            row.get("data_source") or "",
            row.get("database_reference_alias") or "",
            row.get("schema") or "",
        ))
    return rows


def load_mind_the_query() -> list[dict]:
    """Automated train split only. Test files under the same trees are not read."""
    rows = []
    for dataset_dir in sorted(path for path in MIND_THE_QUERY_TRAIN.iterdir() if path.is_dir()):
        split_dir = dataset_dir / "train"
        if not split_dir.is_dir():
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for item in data:
                cypher = item.get("Cypher") or item.get("cypher") or ""
                if not isinstance(cypher, str) or not cypher.strip():
                    continue
                source = item.get("source_dataset") or dataset_dir.name
                rows.append(training_row(
                    len(rows),
                    item.get("unique_id") or f"mtq_{len(rows)}",
                    item.get("NL Question") or item.get("question") or "",
                    cypher,
                    source,
                    source,
                    "",
                ))
    return rows


def load_cypherbench() -> list[dict]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(CYPHERBENCH_DIR))["train"]
    rows = []
    for index in range(len(dataset)):
        row = dataset[index]
        graph = row.get("graph") or ""
        rows.append(training_row(
            index,
            row.get("qid") or f"cb_{index}",
            row.get("nl_question") or "",
            row.get("gold_cypher") or "",
            graph,
            graph,
            "",
        ))
    return rows


def load_train(path: Path) -> list[dict]:
    cached = cached_rows(path)
    if cached is not None:
        return cached
    if DATASET == "neo4j/text2cypher-2024v1":
        rows = load_neo4j()
    elif DATASET == "mind-the-query/automated":
        rows = load_mind_the_query()
    elif DATASET == "megagonlabs/cypherbench":
        rows = load_cypherbench()
    else:
        raise RuntimeError(f"No loader for {DATASET}")
    return write_rows(path, rows)


def build_signatures(queries: list[str], path: Path) -> list[Signature]:
    if path.exists():
        import pickle
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        if len(stored) == len(queries) and stored[0].cypher == queries[0] and stored[-1].cypher == queries[-1]:
            return stored
    signatures = [Signature(query) for query in tqdm(queries, desc="signatures")]
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return signatures


def duplicate_groups(norms: list[str]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, norm in enumerate(norms):
        grouped[norm].append(index)
    return [np.asarray(grouped[norm], dtype=np.int64) for norm in norms]


def _effective_top_k(count: int) -> int:
    """Clamp retrieval width when the smoke corpus is smaller than TOP_K."""
    return max(1, min(TOP_K, count - 1))


def topk_bm25(query_tokens: list[list[str]], weights, vocab: dict[str, int], groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = len(query_tokens)
    k = _effective_top_k(count)
    indices = np.empty((count, k), dtype=np.int32)
    values = np.empty((count, k), dtype=np.float32)
    for start in tqdm(range(0, count, 256), desc=f"BM25 top-{k}"):
        stop = min(start + 256, count)
        query_tf = build_tf(query_tokens[start:stop], vocab)
        scores = query_tf.dot(weights.T).toarray()
        for row, anchor in enumerate(range(start, stop)):
            scores[row, groups[anchor]] = -1e9
        partition = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        for row in range(scores.shape[0]):
            order = partition[row]
            order = order[np.argsort(-scores[row, order])]
            indices[start + row] = order
            values[start + row] = scores[row, order]
    return indices, values


def topk_mpnet(matrix: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    documents = torch.from_numpy(np.ascontiguousarray(matrix)).to(device)
    count = matrix.shape[0]
    k = _effective_top_k(count)
    indices = np.empty((count, k), dtype=np.int32)
    values = np.empty((count, k), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, count, 512), desc=f"MPNet top-{k} {device}"):
            stop = min(start + 512, count)
            scores = documents[start:stop] @ documents.T
            for row, anchor in enumerate(range(start, stop)):
                scores[row, torch.from_numpy(groups[anchor]).to(device)] = -1e9
            value, index = torch.topk(scores, k=k, dim=1)
            indices[start:stop] = index.cpu().numpy()
            values[start:stop] = value.cpu().numpy()
            del scores
    return indices, values


def encode_mpnet(queries: list[str], embedding_path: Path) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"encoding {len(queries)} queries with {MPNET_MODEL} on {device}", flush=True)
    model = SentenceTransformer(MPNET_MODEL, device=device)
    matrix = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] != len(queries):
        raise RuntimeError(f"MPNet rows {matrix.shape[0]} != {len(queries)}")
    np.save(embedding_path, matrix)


def stage_retrieval(queries: list[str], groups: list[np.ndarray]) -> dict[str, np.ndarray]:
    RETRIEVAL.mkdir(parents=True, exist_ok=True)
    bm25_index_path = RETRIEVAL / "bm25_top50_index_v1.npy"
    bm25_score_path = RETRIEVAL / "bm25_top50_score_v1.npy"
    mpnet_index_path = RETRIEVAL / "mpnet_top50_index_v1.npy"
    mpnet_score_path = RETRIEVAL / "mpnet_top50_score_v1.npy"
    embedding_path = RETRIEVAL / "mpnet_train_cypher_embeddings_v1.npy"
    need_encode = True
    if embedding_path.exists():
        need_encode = np.load(embedding_path, mmap_mode="r").shape[0] != len(queries)
    elif (
        DATASET == "neo4j/text2cypher-2024v1"
        and MPNET_SOURCE.exists()
        and np.load(MPNET_SOURCE, mmap_mode="r").shape[0] == len(queries)
    ):
        shutil.copyfile(MPNET_SOURCE, embedding_path)
        need_encode = False
    if need_encode:
        encode_mpnet(queries, embedding_path)
    k = _effective_top_k(len(queries))
    if all(path.exists() for path in (bm25_index_path, bm25_score_path, mpnet_index_path, mpnet_score_path)):
        stored_index = np.load(bm25_index_path)
        if stored_index.shape == (len(queries), k):
            return {
                "bm25_index": stored_index,
                "bm25_score": np.load(bm25_score_path),
                "mpnet_index": np.load(mpnet_index_path),
                "mpnet_score": np.load(mpnet_score_path),
            }
    tokens = [bm25_tokenize(query) for query in tqdm(queries, desc="BM25 tokenize")]
    vocab: dict[str, int] = {}
    for row in tokens:
        for token in row:
            if token not in vocab:
                vocab[token] = len(vocab)
    weights, stats = bm25_weights(build_tf(tokens, vocab), K1, B)
    bm25_index, bm25_score = topk_bm25(tokens, weights, vocab, groups)
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape[0] != len(queries):
        raise RuntimeError(f"MPNet rows {embeddings.shape[0]} != {len(queries)}")
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    mpnet_index, mpnet_score = topk_mpnet(matrix, groups)
    np.save(bm25_index_path, bm25_index)
    np.save(bm25_score_path, bm25_score)
    np.save(mpnet_index_path, mpnet_index)
    np.save(mpnet_score_path, mpnet_score)
    meta = {
        "dataset": DATASET,
        "split": "train",
        "n_queries": len(queries),
        "top_k": TOP_K,
        "excluded_from_retrieval": "the anchor row and every training row with the same normalized Cypher",
        "test_split_used": False,
        "bm25": {"k1": K1, "b": B, "idf": "lucene", "tokenizer": "retrieve_bm25_top5.tokenize", **stats},
        "mpnet": {
            "model": MPNET_MODEL,
            "source_embeddings": str(MPNET_SOURCE if DATASET == "neo4j/text2cypher-2024v1" else embedding_path),
            "normalized": True,
            "similarity": "cosine via normalized dot product",
            "query_text": "Cypher only",
        },
    }
    (RETRIEVAL / "retrieval_meta_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "bm25_index": bm25_index,
        "bm25_score": bm25_score,
        "mpnet_index": mpnet_index,
        "mpnet_score": mpnet_score,
    }


def rare_operator_overlap(left: frozenset[str], right: frozenset[str], idf: dict[str, float]) -> float:
    """Same weighted overlap as evaluate_dev.rare_overlap, using already extracted operators."""
    union = left | right
    if not union:
        return 1.0
    denominator = sum(idf[name] for name in union)
    if denominator == 0.0:
        return 1.0
    return sum(idf[name] for name in (left & right)) / denominator


def hop_similarity(left: int, right: int) -> float:
    return 1.0 - abs(left - right) / max(left, right, 1)


def round4(value: float) -> float:
    return round(float(value), 4)


_CLAUSE_OPERATORS = frozenset({"MATCH", "RETURN", "WHERE"})


def _specific_operators(operators: frozenset[str]) -> frozenset[str]:
    return frozenset(name for name in operators if name not in _CLAUSE_OPERATORS)


def structural_failures(view: dict[str, float]) -> list[str]:
    reasons = []
    if view["schema"] < 0.34:
        reasons.append("schema")
    if view["direction"] < 0.999:
        reasons.append("direction")
    if view.get("path_applicable", True) and (view["path"] < 0.85 or view["max_spd_gap"] >= 1.0):
        reasons.append("path")
    if view["operator"] < OPERATOR_POSITIVE_MIN:
        reasons.append("operator")
    return reasons


def score_pair(anchor: Signature, other: Signature, anchor_hops: int, other_hops: int, idf: dict[str, float]) -> dict[str, float]:
    del anchor_hops, other_hops, idf
    label = jaccard(anchor.labels, other.labels)
    relationship = jaccard(anchor.rels, other.rels)
    prop = jaccard(anchor.props, other.props)
    schema = float(np.mean([label, relationship, prop]))
    paths = path_metrics(anchor.cypher, other.cypher)
    left_ops = semantic_operators(anchor.cypher)
    right_ops = semantic_operators(other.cypher)
    operator = 1.0 if not left_ops and not right_ops else (
        0.0 if not left_ops or not right_ops else len(left_ops & right_ops) / len(left_ops | right_ops)
    )
    specific_left = _specific_operators(left_ops)
    specific_right = _specific_operators(right_ops)
    if not specific_left and not specific_right:
        rare = 1.0
    elif not specific_left or not specific_right:
        rare = 0.0
    else:
        rare = len(specific_left & specific_right) / len(specific_left | specific_right)
    direction = direction_similarity(anchor.cypher, other.cypher)
    path = float(paths["path"])
    return {
        "lexical": jaccard(anchor.tokens, other.tokens),
        "schema": schema,
        "label": label,
        "relationship": relationship,
        "property": prop,
        "direction": direction,
        "spd": path,
        "path": path,
        "path_applicable": bool(paths["path_applicable"]),
        "max_spd_gap": float(paths["max_spd_gap"]),
        "operator": operator,
        "rare_operator": rare,
        "graph": graph_similarity(anchor.cypher, other.cypher),
        "positive_score": positive_score(schema, direction, path, operator),
        "left_ops": left_ops,
        "right_ops": right_ops,
    }


def true_direction_mismatch(anchor: Signature, other: Signature) -> bool:
    """Arrow orientation differs. Extra hops in the same direction are not a reversal."""
    def ends(counter: Counter) -> tuple[int, int]:
        return int(counter.get("in", 0)), int(counter.get("out", 0))

    left_in, left_out = ends(anchor.direction)
    right_in, right_out = ends(other.direction)
    if left_in + left_out == 0 or right_in + right_out == 0:
        return False
    left_kinds = {name for name, count in (("in", left_in), ("out", left_out)) if count}
    right_kinds = {name for name, count in (("in", right_in), ("out", right_out)) if count}
    return left_kinds != right_kinds


def serious_structural_gap(cand: dict) -> bool:
    """Schema, semantic direction, path, or hop-count is off. An operator-only edit is not enough."""
    if cand["schema"] < 0.55:
        return True
    if cand["direction"] <= 0.50:
        return True
    if cand.get("path_applicable", True) and (cand["path"] < 0.85 or cand["max_spd_gap"] >= 1.0):
        return True
    if cand["graph"] <= 0.50:
        return True
    return False


def retrieval_maps(anchor: int, pools: dict[str, np.ndarray]) -> tuple[dict[int, tuple[int, float]], dict[int, tuple[int, float]]]:
    bm25 = {}
    mpnet = {}
    width = int(pools["bm25_index"].shape[1])
    for rank in range(width):
        bm25[int(pools["bm25_index"][anchor, rank])] = (rank + 1, float(pools["bm25_score"][anchor, rank]))
        mpnet[int(pools["mpnet_index"][anchor, rank])] = (rank + 1, float(pools["mpnet_score"][anchor, rank]))
    return bm25, mpnet


def candidate_pool(anchor: int, corpus: list[Signature], hops: np.ndarray, pools: dict[str, np.ndarray], idf: dict[str, float]) -> list[dict]:
    anchor_sig = corpus[anchor]
    bm25, mpnet = retrieval_maps(anchor, pools)
    ordered = list(bm25) + [index for index in mpnet if index not in bm25]
    found: list[dict] = []
    seen_norms = {anchor_sig.norm}
    for index in ordered:
        if index < 0 or index == anchor:
            continue
        other = corpus[index]
        if other.norm in seen_norms:
            continue
        seen_norms.add(other.norm)
        view = score_pair(anchor_sig, other, int(hops[anchor]), int(hops[index]), idf)
        bm25_hit = bm25.get(index)
        mpnet_hit = mpnet.get(index)
        found.append({
            "index": index,
            "signature": other,
            "lexical": view["lexical"],
            "label": view["label"],
            "relationship": view["relationship"],
            "property": view["property"],
            "schema": view["schema"],
            "operator": view["operator"],
            "rare_operator": view["rare_operator"],
            "graph": view["graph"],
            "direction": view["direction"],
            "path": view["path"],
            "path_applicable": view["path_applicable"],
            "spd": view["spd"],
            "max_spd_gap": view["max_spd_gap"],
            "positive_score": view["positive_score"],
            "ops": view["right_ops"],
            "bm25_rank": 0 if bm25_hit is None else bm25_hit[0],
            "bm25": 0.0 if bm25_hit is None else bm25_hit[1],
            "mpnet_rank": 0 if mpnet_hit is None else mpnet_hit[0],
            "mpnet_cosine": 0.0 if mpnet_hit is None else mpnet_hit[1],
            "view": view,
        })
    return found


def eligible_positive(anchor: Signature, cand: dict) -> bool:
    """Same relationship meaning, shared schema, and no large operator or path break."""
    if not shares_schema(anchor, cand["signature"]):
        return False
    if cand["schema"] < 0.34 or cand["operator"] < OPERATOR_POSITIVE_MIN or cand["direction"] < 0.999:
        return False
    if cand.get("path_applicable", True) and cand["path"] < 0.80:
        return False
    return True


def select_positives(anchor: Signature, pool: list[dict]) -> list[dict | None]:
    """P2 nearest template, P3 lexically harder structural positive, then two more."""
    eligible = [cand for cand in pool if eligible_positive(anchor, cand)]
    eligible.sort(key=lambda cand: (-cand["positive_score"], -cand["lexical"], cand["bm25_rank"] or 999))
    if not eligible:
        return [None, None, None, None]
    nearest = dict(eligible[0])
    nearest["level"] = "nearest"
    nearest["role"] = "nearest_natural_positive"
    remaining = eligible[1:]
    lexical_ceiling = nearest["lexical"] - 0.12
    mpnet_ceiling = nearest["mpnet_cosine"] - 0.05 if nearest["mpnet_rank"] else None
    structural = []
    for cand in remaining:
        lexical_gap = cand["lexical"] <= lexical_ceiling
        embedding_gap = mpnet_ceiling is not None and cand["mpnet_rank"] != 0 and cand["mpnet_cosine"] <= mpnet_ceiling
        if cand["schema"] < 0.50 or cand["graph"] < 0.50:
            continue
        if not (lexical_gap or embedding_gap):
            continue
        structural.append(cand)
    structural.sort(key=lambda cand: (-cand["positive_score"], cand["lexical"]))
    hard = None
    if structural:
        hard = dict(structural[0])
        hard["level"] = "structural_hard_positive"
        hard["role"] = "structural_hard_positive"
    used = {nearest["index"]}
    if hard is not None:
        used.add(hard["index"])
    extras = []
    for cand in remaining:
        if cand["index"] in used:
            continue
        picked = dict(cand)
        picked["level"] = "additional"
        picked["role"] = "additional_compatible"
        extras.append(picked)
        used.add(cand["index"])
        if len(extras) >= 2:
            break
    while len(extras) < 2:
        extras.append(None)
    return [nearest, hard, extras[0], extras[1]]


def protected_positive_ids(anchor: Signature, pool: list[dict]) -> set[int]:
    return {cand["index"] for cand in pool if eligible_positive(anchor, cand)}


def below_positives(score: float, positives: list[dict]) -> bool:
    if not positives:
        return score <= FALLBACK_CEILING - MARGIN
    return score <= min(item["positive_score"] for item in positives) - MARGIN


def attractive(cand: dict, rank_key: str, rank_limit: int) -> bool:
    if cand["lexical"] >= LEXICAL_FLOOR:
        return True
    rank = cand[rank_key]
    cosine = cand["mpnet_cosine"]
    return rank != 0 and rank <= rank_limit and (rank_key == "bm25_rank" or cosine >= 0.72)


def select_bm25_negative(pool: list[dict], positives: list[dict], protected: set[int], used: set[int]) -> dict | None:
    ordered = [cand for cand in pool if cand["bm25_rank"] != 0]
    ordered.sort(key=lambda cand: cand["bm25_rank"])

    def acceptable(cand: dict, rank_limit: int) -> bool:
        if cand["index"] in protected or cand["index"] in used:
            return False
        if cand["bm25_rank"] > rank_limit:
            return False
        if not attractive(cand, "bm25_rank", rank_limit):
            return False
        if not serious_structural_gap(cand):
            return False
        return below_positives(cand["positive_score"], positives)

    qualifiers = [cand for cand in ordered if acceptable(cand, BM25_HIGH_RANK)]
    if not qualifiers:
        qualifiers = [cand for cand in ordered if acceptable(cand, TOP_K)]
    if not qualifiers:
        return None
    # Highest BM25 rank among queries that are actually incompatible.
    # Rank 1 is kept only when it clears the same incompatibility gates.
    return qualifiers[0]


def select_mpnet_negative(pool: list[dict], positives: list[dict], protected: set[int], used: set[int]) -> dict | None:
    ordered = [cand for cand in pool if cand["mpnet_rank"] != 0]
    ordered.sort(key=lambda cand: cand["mpnet_rank"])
    if not ordered:
        return None
    top_cosine = ordered[0]["mpnet_cosine"]
    cosine_floor = max(0.72, top_cosine - 0.10)

    def acceptable(cand: dict, rank_limit: int) -> bool:
        if cand["index"] in protected or cand["index"] in used:
            return False
        if cand["mpnet_rank"] > rank_limit or cand["mpnet_cosine"] < cosine_floor:
            return False
        if cand["lexical"] < 0.12 and cand["mpnet_cosine"] < 0.80:
            return False
        if not serious_structural_gap(cand):
            return False
        return below_positives(cand["positive_score"], positives)

    qualifiers = [cand for cand in ordered if acceptable(cand, MPNET_HIGH_RANK)]
    if not qualifiers:
        qualifiers = [cand for cand in ordered if acceptable(cand, TOP_K)]
    return qualifiers[0] if qualifiers else None


def _path_agrees(cand: dict) -> bool:
    if not cand.get("path_applicable", True):
        return True
    return cand["path"] >= 0.85 and cand["max_spd_gap"] < 1.0


def select_operator_negative(anchor: Signature, pool: list[dict], positives: list[dict], used: set[int], idf: dict[str, float]) -> dict | None:
    del idf
    anchor_ops = semantic_operators(anchor.cypher)
    important = _specific_operators(anchor_ops)
    if not important:
        return None
    best = None
    best_strength = -1.0
    for cand in pool:
        if cand["index"] in used:
            continue
        missing = sorted(important - set(cand.get("ops") or ()))
        if not missing or cand["operator"] > 0.60:
            continue
        if cand["schema"] < 0.70 or cand["direction"] < 0.90 or cand["graph"] < 0.80 or not _path_agrees(cand):
            continue
        if positives and not below_positives(cand["positive_score"], positives):
            continue
        target = missing[0]
        closeness = max(cand["lexical"], cand["mpnet_cosine"], 0.0)
        strength = cand["schema"] * cand["direction"] * cand["graph"] * (1.0 - cand["operator"]) * (0.25 + closeness)
        if strength > best_strength:
            best_strength = strength
            picked = dict(cand)
            picked["target_operator"] = target
            picked["missing_operators"] = missing
            best = picked
    return best


def select_direction_or_path(anchor: Signature, pool: list[dict], positives: list[dict], protected: set[int], used: set[int]) -> dict | None:
    del anchor
    direction_best = None
    direction_strength = -1.0
    path_best = None
    path_strength = -1.0
    for cand in pool:
        if cand["index"] in protected or cand["index"] in used:
            continue
        if not below_positives(cand["positive_score"], positives):
            continue
        close = (
            cand["lexical"] >= 0.20
            or (cand["mpnet_rank"] != 0 and cand["mpnet_cosine"] >= 0.72)
            or (cand["bm25_rank"] != 0 and cand["bm25_rank"] <= 15)
        )
        if not close:
            continue
        closeness = max(cand["lexical"], cand["mpnet_cosine"], 0.0)
        direction_case = (
            cand["schema"] >= 0.70
            and cand["operator"] >= 0.70
            and cand["graph"] >= 0.75
            and _path_agrees(cand)
            and cand["direction"] <= 0.35
        )
        if direction_case:
            strength = cand["schema"] * cand["operator"] * cand["graph"] * (0.25 + closeness) * (1.0 - cand["direction"])
            if strength > direction_strength:
                direction_strength = strength
                direction_best = cand
            continue
        path_case = (
            cand.get("path_applicable", False)
            and cand["schema"] >= 0.70
            and cand["operator"] >= 0.70
            and cand["direction"] >= 0.80
            and cand["path"] <= 0.50
        )
        if path_case:
            strength = cand["schema"] * cand["operator"] * cand["direction"] * (0.25 + closeness) * (1.0 - cand["path"])
            if strength > path_strength:
                path_strength = strength
                path_best = cand
    if direction_best is not None:
        picked = dict(direction_best)
        picked["negative_source"] = "direction_hard_negative"
        return picked
    if path_best is not None:
        picked = dict(path_best)
        picked["negative_source"] = "path_hard_negative"
        return picked
    return None


def pack_scores(cand: dict) -> dict:
    return {
        "lexical": round4(cand["lexical"]),
        "bm25": None if cand["bm25_rank"] == 0 else round4(cand["bm25"]),
        "bm25_rank": None if cand["bm25_rank"] == 0 else int(cand["bm25_rank"]),
        "mpnet_cosine": None if cand["mpnet_rank"] == 0 else round4(cand["mpnet_cosine"]),
        "mpnet_rank": None if cand["mpnet_rank"] == 0 else int(cand["mpnet_rank"]),
        "label": round4(cand["label"]),
        "relationship": round4(cand["relationship"]),
        "property": round4(cand["property"]),
        "schema": round4(cand["schema"]),
        "operators": round4(cand["operator"]),
        "rare_operators": round4(cand["rare_operator"]),
        "graph": round4(cand["graph"]),
        "direction": round4(cand["direction"]),
        "path": round4(cand["path"]),
        "max_spd_gap": round4(cand["max_spd_gap"]),
        "positive_score": round4(cand["positive_score"]),
    }


def natural_record(cand: dict, slot: str) -> dict:
    if cand.get("role") == "structural_hard_positive":
        reason = (
            f"Structural hard positive: schema {cand['schema']:.3f}, operators {cand['operator']:.3f}, "
            f"graph {cand['graph']:.3f}, direction {cand['direction']:.3f}, path {cand['path']:.3f}, "
            f"lexical {cand['lexical']:.3f}. Lexical or embedding similarity is lower than the nearest positive."
        )
    else:
        reason = (
            f"{cand.get('role', 'compatible')} neighbour. "
            f"positive_score {cand['positive_score']:.3f} from schema {cand['schema']:.3f}, "
            f"operators {cand['operator']:.3f}, graph {cand['graph']:.3f}, "
            f"direction {cand['direction']:.3f}, path {cand['path']:.3f}."
        )
    return {
        "slot": slot,
        "id": int(cand["index"]),
        "cypher": cand["signature"].cypher,
        "type": "natural_positive",
        "role": cand.get("role"),
        "compatible_level": cand.get("level"),
        "semantic_operators": sorted(cand.get("ops") or []),
        "path_applicable": bool(cand.get("path_applicable", True)),
        "scores": pack_scores(cand),
        "selection_reason": reason,
    }


def missing_positive(slot: str, reason: str) -> dict:
    return {
        "slot": slot,
        "id": None,
        "cypher": None,
        "type": "natural_positive",
        "missing": True,
        "scores": {},
        "selection_reason": reason,
    }


def negative_record(cand: dict, slot: str, type_name: str, source: str, reason: str, extra: dict | None = None) -> dict:
    row = {
        "slot": slot,
        "id": int(cand["index"]),
        "cypher": cand["signature"].cypher,
        "type": type_name,
        "source": source,
        "semantic_operators": sorted(cand.get("ops") or []),
        "path_applicable": bool(cand.get("path_applicable", True)),
        "scores": pack_scores(cand),
        "selection_reason": reason,
    }
    if extra:
        row.update(extra)
    return row


def bm25_reason(cand: dict, positives: list[dict]) -> str:
    failures = structural_failures(cand["view"])
    floor = min(item["positive_score"] for item in positives) if positives else FALLBACK_CEILING
    return (
        f"BM25 rank {cand['bm25_rank']} (score {cand['bm25']:.3f}) with lexical similarity {cand['lexical']:.3f}, "
        f"but Cypher compatibility is {cand['positive_score']:.3f} versus a positive floor of {floor:.3f}. "
        f"Incompatible on {', '.join(failures) or 'structure'}. "
        "Not one of the schema-and-structure compatible neighbours."
    )


def mpnet_reason(cand: dict, positives: list[dict]) -> str:
    failures = structural_failures(cand["view"])
    floor = min(item["positive_score"] for item in positives) if positives else FALLBACK_CEILING
    return (
        f"MPNet cosine {cand['mpnet_cosine']:.3f} at rank {cand['mpnet_rank']}, "
        f"but Cypher compatibility is {cand['positive_score']:.3f} versus a positive floor of {floor:.3f}. "
        f"Incompatible on {', '.join(failures) or 'structure'}. "
        "This is an embedding neighbour that is not a Cypher neighbour."
    )


def operator_reason(cand: dict) -> str:
    missing = ", ".join(cand["missing_operators"])
    return (
        f"Schema similarity {cand['schema']:.3f}, direction {cand['direction']:.3f}, path {cand['path']:.3f}, "
        f"but operator similarity is only {cand['operator']:.3f}. "
        f"The anchor operator '{cand['target_operator']}' is absent"
        + (f" (also missing {missing})." if missing != cand["target_operator"] else ".")
    )


def direction_reason(cand: dict) -> str:
    if cand["negative_source"] == "direction_hard_negative":
        return (
            f"Schema similarity {cand['schema']:.3f} and operator similarity {cand['operator']:.3f}, "
            f"but relationship direction similarity is {cand['direction']:.3f}. "
            "Direction mismatch is preferred over a path-only mismatch."
        )
    return (
        f"Schema similarity {cand['schema']:.3f}, operator similarity {cand['operator']:.3f}, "
        f"direction similarity {cand['direction']:.3f}, but path similarity is {cand['path']:.3f} "
        f"(max directed-distance gap {cand['max_spd_gap']:.1f})."
    )


def missing_negative(slot: str, type_name: str, source: str, reason: str) -> dict:
    return {
        "slot": slot,
        "id": None,
        "cypher": None,
        "type": type_name,
        "source": source,
        "missing": True,
        "scores": {},
        "selection_reason": reason,
    }


def build_record(index: int, row: dict, corpus: list[Signature], hops: np.ndarray, pools: dict[str, np.ndarray], idf: dict[str, float]) -> tuple[dict, list[dict]]:
    anchor = corpus[index]
    renamed, changed = rename_bound_variables(anchor.cypher)
    renamed_view = score_variable_rename(anchor, renamed, int(hops[index]), idf)
    pool = candidate_pool(index, corpus, hops, pools, idf)
    chosen = select_positives(anchor, pool)
    protected = protected_positive_ids(anchor, pool)
    used: set[int] = set()
    missing_reasons = {
        "p2": "No training query shared the schema, semantic direction, and operators closely enough.",
        "p3": "No structural hard positive stayed compatible while being lexically or semantically less obvious than the nearest positive.",
        "p4": "No further compatible neighbour was available.",
        "p5": "No further compatible neighbour was available.",
    }
    natural = []
    positives = []
    for slot, cand in zip(("p2", "p3", "p4", "p5"), chosen):
        if cand is None:
            natural.append(missing_positive(slot, missing_reasons[slot]))
            continue
        natural.append(natural_record(cand, slot))
        positives.append(cand)
        used.add(cand["index"])

    n3 = select_operator_negative(anchor, pool, positives, used, idf)
    if n3 is not None:
        used.add(n3["index"])
    n4 = select_direction_or_path(anchor, pool, positives, protected, used)
    if n4 is not None:
        used.add(n4["index"])
    n1 = select_bm25_negative(pool, positives, protected, used)
    if n1 is not None:
        used.add(n1["index"])
    n2 = select_mpnet_negative(pool, positives, protected, used)

    positive_rows = [{
        "slot": "p1",
        "id": None,
        "cypher": renamed,
        "type": "variable_rename",
        "changed": changed,
        "path_applicable": bool(renamed_view.get("path_applicable", True)),
        "semantic_operators": sorted(semantic_operators(renamed)),
        "scores": {
            "lexical": round4(renamed_view["lexical"]),
            "schema": round4(renamed_view["schema"]),
            "operators": round4(renamed_view["operator"]),
            "rare_operators": round4(renamed_view["rare_operator"]),
            "graph": round4(renamed_view["graph"]),
            "direction": round4(renamed_view["direction"]),
            "path": round4(renamed_view["path"]),
            "label": round4(renamed_view["label"]),
            "relationship": round4(renamed_view["relationship"]),
            "property": round4(renamed_view["property"]),
            "positive_score": round4(renamed_view["positive_score"]),
        },
        "selection_reason": (
            "Bound node and relationship variables were renamed. "
            "Labels, relationship types, properties, operators, directions, and topology are unchanged."
            if changed else
            "No bound variable could be renamed. The original Cypher is stored and flagged changed=false."
        ),
    }]
    positive_rows.extend(natural)

    negatives = []
    if n1 is None:
        negatives.append(missing_negative(
            "n1", "bm25", "bm25_hard_negative",
            "No high-BM25 training query was both lexically deceptive and meaningfully less compatible than the natural positives.",
        ))
    else:
        negatives.append(negative_record(n1, "n1", "bm25", "bm25_hard_negative", bm25_reason(n1, positives)))
    if n2 is None:
        negatives.append(missing_negative(
            "n2", "mpnet", "mpnet_hard_negative",
            "No high-cosine MPNet neighbour was meaningfully less Cypher-compatible than the natural positives.",
        ))
    else:
        negatives.append(negative_record(n2, "n2", "mpnet", "mpnet_hard_negative", mpnet_reason(n2, positives)))
    if n3 is None:
        negatives.append(missing_negative(
            "n3", "operator", "operator_hard_negative",
            "No natural training query kept the schema, direction, and path while missing an important anchor operator.",
        ))
    else:
        negatives.append(negative_record(
            n3, "n3", "operator", "operator_hard_negative", operator_reason(n3),
            {"target_operator_difference": n3["target_operator"], "missing_operators": n3["missing_operators"]},
        ))
    if n4 is None:
        negatives.append(missing_negative(
            "n4", "direction_or_path", "direction_or_path_hard_negative",
            "No natural training query had high schema and operator overlap with a strong direction or path mismatch.",
        ))
    else:
        negatives.append(negative_record(
            n4, "n4", "direction_or_path", n4["negative_source"], direction_reason(n4),
        ))

    record = {
        "anchor_id": index,
        "instance_id": row["instance_id"],
        "data_source": row["data_source"],
        "database_reference_alias": row["database_reference_alias"],
        "question": row["question"],
        "anchor_cypher": anchor.cypher,
        "labels": sorted(anchor.labels),
        "relationship_types": sorted(anchor.rels),
        "properties": sorted(anchor.props),
        "operators": sorted(semantic_operators(anchor.cypher)),
        "direction": {key: int(anchor.direction[key]) for key in ("in", "out") if anchor.direction[key]},
        "hops": int(hops[index]),
        "path_histogram": [round4(value) for value in anchor.spd.tolist()],
        "natural_positive_count": sum(1 for item in natural if item.get("cypher")),
        "positives": positive_rows,
        "hard_negatives": negatives,
    }
    return record, pool


def metric_matrix(pool: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty(len(pool), dtype=np.int32)
    matrix = np.empty((len(pool), len(METRIC_NAMES)), dtype=np.float32)
    for row, cand in enumerate(pool):
        ids[row] = cand["index"]
        values = [
            cand["lexical"], cand["label"], cand["relationship"], cand["property"], cand["schema"],
            cand["operator"], cand["rare_operator"], cand["graph"], cand["direction"], cand["path"],
            cand["max_spd_gap"], cand["positive_score"], cand["bm25"], cand["bm25_rank"],
            cand["mpnet_cosine"], cand["mpnet_rank"],
        ]
        matrix[row] = values
    return ids, matrix


def write_selection(
    rows: list[dict],
    corpus: list[Signature],
    hops: np.ndarray,
    pools: dict[str, np.ndarray],
    idf: dict[str, float],
    limit: int | None,
    indices: list[int] | None = None,
    selection_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Path:
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    selection_path = selection_path or (SELECTIONS / "train_hard_examples_v1.jsonl")
    metrics_path = metrics_path or (CANDIDATES / "combined_candidate_metrics_v1.npz")
    if indices is None:
        count = len(rows) if limit is None else min(limit, len(rows))
        work = list(enumerate(rows[:count]))
    else:
        work = [(index, rows[index]) for index in indices]
    id_chunks = []
    metric_chunks = []
    offsets = [0]
    with selection_path.open("w", encoding="utf-8") as handle:
        for index, row in tqdm(work, desc="select"):
            record, pool = build_record(index, row, corpus, hops, pools, idf)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            ids, matrix = metric_matrix(pool)
            id_chunks.append(ids)
            metric_chunks.append(matrix)
            offsets.append(offsets[-1] + len(pool))
    np.savez_compressed(
        metrics_path,
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_ids=np.concatenate(id_chunks) if id_chunks else np.empty(0, dtype=np.int32),
        metrics=np.concatenate(metric_chunks) if metric_chunks else np.empty((0, len(METRIC_NAMES)), dtype=np.float32),
    )
    schema_path = CANDIDATES / ("metrics_schema_v1.json" if metrics_path.name.endswith("v1.npz") else "metrics_schema_v2.json")
    schema_path.write_text(json.dumps({
        "file": metrics_path.name,
        "one_row_per": "anchor-candidate pair in the BM25 top-50 union MPNet top-50 pool, after exact-Cypher dedupe",
        "offsets": "length n_anchors+1; pairs for anchor i are candidate_ids[offsets[i]:offsets[i+1]]",
        "metric_columns": METRIC_NAMES,
        "rank_zero": "bm25_rank or mpnet_rank 0 means that retriever did not return the candidate in its top 50",
        "positive_score": "0.50*schema + 0.20*direction + 0.20*path + 0.10*operators, the TrainingQueryContrast compatible_score",
        "graph": "hop-count similarity, the hop term of score_top5_similarity.pair_scores",
        "path": "Directed shortest-path histogram cosine when both queries have relationships. Both empty graphs score 1.0 and are not path-applicable.",
        "direction": "multiset Jaccard of semantic canonical edges (label, type, label, DIRECTED or UNDIRECTED)",
        "operators": "Jaccard of mining_metrics.semantic_operators, after string literals are removed and compound forms stay atomic",
        "rare_operators": "IDF-weighted operator overlap, evaluate_dev.rare_overlap",
        "schema": "mean Jaccard of labels, relationship types, and properties from score_top5_similarity.schema_parts",
    }, indent=2), encoding="utf-8")
    return selection_path


def filled(item: dict) -> bool:
    return bool(item.get("cypher"))


def mean_median(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": round(float(array.mean()), 4), "median": round(float(np.median(array)), 4)}


def metric_of(item: dict, key: str) -> float | None:
    if not filled(item):
        return None
    value = item.get("scores", {}).get(key)
    if value is None:
        return None
    return float(value)


def collect_metrics(items: list[dict], keys: list[str]) -> dict[str, dict]:
    columns = {key: [] for key in keys}
    for item in items:
        for key in keys:
            value = metric_of(item, key)
            if value is not None:
                columns[key].append(value)
    return {key: mean_median(values) for key, values in columns.items()}


STAT_KEYS = [
    "bm25_rank", "mpnet_rank", "lexical", "schema", "operators", "graph", "direction", "path",
    "positive_score", "mpnet_cosine", "rare_operators", "label", "relationship", "property",
]


def summarize(path: Path) -> dict:
    anchors = 0
    natural_hist = Counter()
    p1_changed = 0
    p1_schema_changed = 0
    coverage = Counter()
    by_slot: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            anchors += 1
            natural_hist[row["natural_positive_count"]] += 1
            p1 = row["positives"][0]
            if p1.get("changed"):
                p1_changed += 1
            if p1["scores"].get("schema", 1) < 0.999 or p1["scores"].get("operators", 1) < 0.999:
                p1_schema_changed += 1
            for item in row["positives"]:
                by_slot[item["slot"]].append(item)
                if filled(item):
                    coverage[item["slot"]] += 1
            for item in row["hard_negatives"]:
                by_slot[item["slot"]].append(item)
                if filled(item):
                    coverage[item["slot"]] += 1
                    if item["slot"] == "n4":
                        coverage[item["source"]] += 1
    positive_items = [item for slot in ("p2", "p3", "p4", "p5") for item in by_slot[slot] if filled(item)]
    negative_groups = {
        "bm25": [item for item in by_slot["n1"] if filled(item)],
        "mpnet": [item for item in by_slot["n2"] if filled(item)],
        "operator": [item for item in by_slot["n3"] if filled(item)],
        "direction_or_path": [item for item in by_slot["n4"] if filled(item)],
    }
    def percent(count: int) -> float:
        return round(100.0 * count / anchors, 2) if anchors else 0.0
    summary = {
        "dataset": DATASET,
        "split": "train",
        "test_split_used": False,
        "n_anchors": anchors,
        "anchors_with_4_natural_positives": natural_hist[4],
        "anchors_with_fewer_than_4_natural_positives": anchors - natural_hist[4],
        "natural_positive_count_histogram": {str(key): natural_hist[key] for key in sorted(natural_hist)},
        "p1_variable_renames_that_changed": p1_changed,
        "p1_schema_or_operator_changed": p1_schema_changed,
        "coverage_counts": {
            "p1": coverage["p1"],
            "p2": coverage["p2"],
            "p3": coverage["p3"],
            "p4": coverage["p4"],
            "p5": coverage["p5"],
            "n1_bm25": coverage["n1"],
            "n2_mpnet": coverage["n2"],
            "n3_operator": coverage["n3"],
            "n4_direction_or_path": coverage["n4"],
            "n4_direction": coverage["direction_hard_negative"],
            "n4_path": coverage["path_hard_negative"],
        },
        "coverage_percent": {
            "p1": percent(coverage["p1"]),
            "p2": percent(coverage["p2"]),
            "p3": percent(coverage["p3"]),
            "p4": percent(coverage["p4"]),
            "p5": percent(coverage["p5"]),
            "four_natural_positives": percent(natural_hist[4]),
            "n1_bm25": percent(coverage["n1"]),
            "n2_mpnet": percent(coverage["n2"]),
            "n3_operator": percent(coverage["n3"]),
            "n4_direction_or_path": percent(coverage["n4"]),
            "n4_direction": percent(coverage["direction_hard_negative"]),
            "n4_path": percent(coverage["path_hard_negative"]),
        },
        "positive_pairs": {
            "p1": coverage["p1"],
            "natural_p2_to_p5": coverage["p2"] + coverage["p3"] + coverage["p4"] + coverage["p5"],
            "total_including_p1": coverage["p1"] + coverage["p2"] + coverage["p3"] + coverage["p4"] + coverage["p5"],
        },
        "per_positive_slot": {slot: collect_metrics(by_slot[slot], STAT_KEYS) for slot in ("p2", "p3", "p4", "p5")},
        "positives_p2_to_p5": collect_metrics(positive_items, STAT_KEYS),
        "negatives": {name: collect_metrics(items, STAT_KEYS) for name, items in negative_groups.items()},
    }
    return summary


def comparison_lines(summary: dict) -> list[str]:
    keys = ["schema", "operators", "graph", "direction", "path", "lexical", "positive_score", "bm25_rank", "mpnet_cosine"]
    lines = ["Positives (p2-p5) versus hard negatives. Means are over the examples that were found.", ""]
    header = f"{'metric':<16} {'positives':>10} {'bm25_n':>10} {'mpnet_n':>10} {'operator_n':>10} {'dir_path_n':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    pos = summary["positives_p2_to_p5"]
    groups = summary["negatives"]
    for key in keys:
        cells = [pos.get(key, {}).get("mean")]
        for name in ("bm25", "mpnet", "operator", "direction_or_path"):
            cells.append(groups[name].get(key, {}).get("mean"))
        rendered = []
        for value in cells:
            rendered.append("" if value is None else f"{value:.4f}")
        lines.append(f"{key:<16} {rendered[0]:>10} {rendered[1]:>10} {rendered[2]:>10} {rendered[3]:>10} {rendered[4]:>10}")
    return lines


def format_candidate(title: str, item: dict | None) -> list[str]:
    lines = [title]
    if item is None or not filled(item):
        reason = "" if item is None else item.get("selection_reason", "")
        lines.append("  <missing>")
        if reason:
            lines.append("  why: " + reason)
        lines.append("")
        return lines
    scores = item.get("scores") or {}
    lines.append("  " + item["cypher"])
    lines.append(
        "  lexical {lexical} | BM25 {rank}/{score} | MPNet {mrank}/{cosine}".format(
            lexical=scores.get("lexical"),
            rank=scores.get("bm25_rank"),
            score=scores.get("bm25"),
            mrank=scores.get("mpnet_rank"),
            cosine=scores.get("mpnet_cosine"),
        )
    )
    lines.append(
        "  schema {schema} | operators {operators} | graph {graph} | direction {direction} | path {path} | path_applicable {applicable} | positive_score {positive}".format(
            schema=scores.get("schema"),
            operators=scores.get("operators"),
            graph=scores.get("graph"),
            direction=scores.get("direction"),
            path=scores.get("path"),
            applicable=item.get("path_applicable"),
            positive=scores.get("positive_score"),
        )
    )
    if item.get("semantic_operators"):
        lines.append("  semantic operators: " + ", ".join(item["semantic_operators"]))
    if item.get("target_operator_difference"):
        lines.append(f"  target operator: {item['target_operator_difference']}")
    if item.get("source"):
        lines.append(f"  source: {item['source']}")
    if item.get("compatible_level"):
        lines.append(f"  compatible level: {item['compatible_level']}")
    lines.append("  why: " + item.get("selection_reason", ""))
    lines.append("")
    return lines


def format_anchor(row: dict) -> list[str]:
    lines = [
        "=" * 88,
        f"ANCHOR {row['anchor_id']}  {row['data_source']}  {row['database_reference_alias']}",
        f"question: {row['question']}",
        f"semantic operators: {', '.join(row['operators'])}",
        f"labels: {', '.join(row['labels'])}",
        f"relationships: {', '.join(row['relationship_types'])}",
        row["anchor_cypher"],
        "",
    ]
    by_slot = {item["slot"]: item for item in row["positives"]}
    by_slot.update({item["slot"]: item for item in row["hard_negatives"]})
    lines.extend(format_candidate("P1 variable-renamed", by_slot.get("p1")))
    for slot, title in (
        ("p2", "P2 nearest natural positive"),
        ("p3", "P3 structural hard positive"),
        ("p4", "P4 additional compatible"),
        ("p5", "P5 additional compatible"),
    ):
        lines.extend(format_candidate(title, by_slot.get(slot)))
    lines.extend(format_candidate("N1 BM25 hard negative", by_slot.get("n1")))
    lines.extend(format_candidate("N2 MPNet hard negative", by_slot.get("n2")))
    lines.extend(format_candidate("N3 operator hard negative", by_slot.get("n3")))
    lines.extend(format_candidate("N4 direction/path hard negative", by_slot.get("n4")))
    return lines


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_case_report(path: Path, title: str, rows: list[dict], note: str) -> None:
    lines = [title, note, ""]
    if not rows:
        lines.append("No qualifying anchors.")
    for row in rows:
        lines.extend(format_anchor(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def negative_item(row: dict, slot: str) -> dict | None:
    for item in row["hard_negatives"]:
        if item["slot"] == slot and filled(item):
            return item
    return None


def write_reports(selection_path: Path, summary: dict, dataset_stats: dict, version: str = "v1") -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = dict(summary)
    summary["dataset_statistics"] = dataset_stats
    summary["sanity_check"] = comparison_lines(summary)
    (REPORTS / f"diagnostics_{version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = [
        "Training hard-positive / hard-negative diagnostics",
        f"Anchors: {summary['n_anchors']}",
        f"Unique normalized Cypher queries: {dataset_stats['unique_normalized_cypher']}",
        f"Training rows that share a normalized Cypher with another row: {dataset_stats['rows_in_duplicate_groups']}",
        f"Usable distinct training queries for neighbours: {dataset_stats['usable_distinct_queries']}",
        "",
        "Coverage",
    ]
    for key, value in summary["coverage_counts"].items():
        text.append(f"  {key}: {value} ({summary['coverage_percent'].get(key, summary['coverage_percent'].get(key.replace('_bm25','').replace('_mpnet','').replace('_operator',''), ''))})")
    text.append("")
    text.append(f"Anchors with 4 natural positives: {summary['anchors_with_4_natural_positives']}")
    text.append(f"Anchors with fewer than 4 natural positives: {summary['anchors_with_fewer_than_4_natural_positives']}")
    text.append(f"Positive pairs including p1: {summary['positive_pairs']['total_including_p1']}")
    text.append(f"Natural positive pairs p2-p5: {summary['positive_pairs']['natural_p2_to_p5']}")
    text.append(f"p1 renames that changed the text: {summary['p1_variable_renames_that_changed']}")
    text.append(f"p1 renames that changed schema or operators: {summary['p1_schema_or_operator_changed']}")
    text.append("")
    text.extend(comparison_lines(summary))
    text.append("")
    text.append(f"Per-slot positive means are in diagnostics_{version}.json under per_positive_slot.")
    (REPORTS / f"diagnostics_{version}.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    rows = load_records(selection_path)
    rng = np.random.default_rng(42)
    sample_ids = sorted(int(index) for index in rng.choice(len(rows), size=min(20, len(rows)), replace=False))
    sample_rows = [rows[index] for index in sample_ids]
    write_case_report(
        REPORTS / f"sample_anchors_{version}.txt",
        "Random sample of 20 training anchors (seed 42)",
        sample_rows,
        "Each block is one anchor and the positives and hard negatives selected for it.",
    )

    def ranked(slot: str, source: str | None, key_fn, limit: int = 20) -> list[dict]:
        scored = []
        for row in rows:
            item = negative_item(row, slot)
            if item is None:
                continue
            if source is not None and item.get("source") != source:
                continue
            scored.append((key_fn(item), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _score, row in scored[:limit]]

    write_case_report(
        REPORTS / f"strongest_direction_hard_negatives_{version}.txt",
        "20 strongest direction hard negatives",
        ranked("n4", "direction_hard_negative", lambda item: (1.0 - item["scores"]["direction"]) * item["scores"]["schema"] * (0.25 + item["scores"]["lexical"])),
        "Sorted by direction mismatch, with schema and lexical similarity kept in the ranking so the pair stays deceptive.",
    )
    write_case_report(
        REPORTS / f"strongest_operator_hard_negatives_{version}.txt",
        "20 strongest operator hard negatives",
        ranked("n3", "operator_hard_negative", lambda item: item["scores"]["schema"] * (1.0 - item["scores"]["operators"]) * (0.25 + item["scores"]["lexical"])),
        "Sorted by schema overlap times the operator gap times lexical similarity.",
    )
    write_case_report(
        REPORTS / f"strongest_path_hard_negatives_{version}.txt",
        "20 strongest path hard negatives",
        ranked("n4", "path_hard_negative", lambda item: (1.0 - item["scores"]["path"]) * item["scores"]["schema"] * (0.25 + item["scores"]["lexical"])),
        "Sorted by path mismatch, with schema and lexical similarity kept in the ranking.",
    )
    write_case_report(
        REPORTS / f"mpnet_high_cosine_low_compatibility_{version}.txt",
        "20 cases where MPNet cosine is high and Cypher compatibility is low",
        ranked("n2", "mpnet_hard_negative", lambda item: item["scores"]["mpnet_cosine"] * (1.0 - item["scores"]["positive_score"])),
        "Selected MPNet hard negatives, strongest embedding-versus-structure conflicts first.",
    )
    write_case_report(
        REPORTS / f"bm25_high_rank_low_compatibility_{version}.txt",
        "20 cases where BM25 rank is high and Cypher compatibility is low",
        ranked("n1", "bm25_hard_negative", lambda item: (1.0 / item["scores"]["bm25_rank"]) * (1.0 - item["scores"]["positive_score"]) * (0.25 + item["scores"]["lexical"])),
        "Selected BM25 hard negatives. Higher BM25 rank and lower compatibility come first.",
    )


def dataset_statistics(rows: list[dict], norms: list[str]) -> dict:
    counts = Counter(norms)
    duplicate_rows = sum(count for count in counts.values() if count > 1)
    return {
        "n_training_rows": len(rows),
        "unique_normalized_cypher": len(counts),
        "duplicate_groups": sum(1 for count in counts.values() if count > 1),
        "rows_in_duplicate_groups": duplicate_rows,
        "usable_distinct_queries": len(counts),
        "max_duplicate_group": max(counts.values()),
        "candidate_exclusion": "anchor itself, exact normalized-Cypher duplicates, and a repeated Cypher already kept in the pool",
    }


def write_config(stats: dict) -> None:
    config = {
        "dataset": DATASET,
        "split": "train",
        "expected_rows": EXPECTED_TRAIN,
        "test_split_used": False,
        "training": False,
        "positive_score": "0.50*schema + 0.20*direction + 0.20*path + 0.10*operators",
        "positive_score_source": "Same weights as TrainingQueryContrast.compatible_score, using corrected semantic direction, path, and operator components",
        "positive_gate_source": "Shared schema, semantic direction similarity 1, operator Jaccard >= 0.5, and path agreement when a relationship graph exists",
        "weights_tuned_on_test": False,
        "margin": MARGIN,
        "fallback_ceiling_when_no_natural_positive": FALLBACK_CEILING,
        "p1": "Bound-variable rename based on TrainingQueryContrast.rename_variables. Schema tokens after ':' or '.' are not rewritten, so a variable that shares its name with a label or relationship type does not change that schema token.",
        "retrieval": "BM25 top 50 union MPNet all-mpnet-base-v2 top 50, Cypher text only, training rows only",
        "synthetic_negatives": False,
        "dataset_statistics": stats,
    }
    (ACTIVE_ROOT / "selection_config_v1.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def sample_indices(rows: list[dict], size: int = 100) -> tuple[list[int], list[int]]:
    """About 100 anchors, forcing in the report cases the corrected metrics must handle."""
    checks = [
        lambda query: "ProductionCompany" in query and "PRODUCED_BY" in query,
        lambda query: ":UpdateDate" in query and "]->" in query,
        lambda query: ":UpdateDate" in query and "<-[" in query,
        lambda query: "STARTS WITH" in query.upper(),
        lambda query: "IS NOT NULL" in query.upper(),
        lambda query: query.startswith("MATCH (m:Movie)") and "]->" not in query and "<-" not in query and "-[" not in query,
        lambda query: "doi" in query.lower() and " and " in query.lower(),
        lambda query: "ORDER BY" in query.upper() and "DESC" in query.upper(),
        lambda query: "ORDER BY" in query.upper() and " ASC" in query.upper(),
    ]
    picked: list[int] = []
    targeted: list[int] = []
    seen: set[int] = set()
    for check in checks:
        for index, row in enumerate(rows):
            if index in seen:
                continue
            if check(row["cypher"]):
                picked.append(index)
                targeted.append(index)
                seen.add(index)
                break
    rng = np.random.default_rng(42)
    for index in rng.permutation(len(rows)):
        index = int(index)
        if index in seen:
            continue
        picked.append(index)
        seen.add(index)
        if len(picked) >= size:
            break
    return picked, targeted


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training hard positives and hard negatives")
    parser.add_argument("--dataset", choices=["neo4j", "mind_the_query", "cypherbench"], default="neo4j")
    parser.add_argument("--limit", type=int, default=0, help="If set, select only the first N anchors and skip reports")
    parser.add_argument("--sample", type=int, default=0, help="Mine this many anchors, including known difficult cases, without replacing the full v1 file")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="If set, load only the first N train rows (overrides EXPECTED_TRAIN; for smoke tests)",
    )
    parser.add_argument(
        "--mpnet-embeddings",
        type=Path,
        default=None,
        help="Optional cached MPNet train embeddings .npy (otherwise encodes with all-mpnet-base-v2)",
    )
    args = parser.parse_args()
    configure(args.dataset)
    global EXPECTED_TRAIN, MPNET_SOURCE
    if args.max_train_rows and args.max_train_rows > 0:
        EXPECTED_TRAIN = args.max_train_rows
    if args.mpnet_embeddings is not None:
        MPNET_SOURCE = args.mpnet_embeddings
    print(f"dataset {DATASET} output {ACTIVE_ROOT} expected_train={EXPECTED_TRAIN}", flush=True)
    for path in (DATA, RETRIEVAL, CANDIDATES, SELECTIONS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    probe, changed = rename_bound_variables("MATCH (a:Author)-[r:WROTE]->(p:Paper) RETURN a.name")
    same_name, same_changed = rename_bound_variables(
        "MATCH (singer:singer)-[:VERSION]->(version:Version) RETURN singer.name, version.name"
    )
    quoted, _quoted_changed = rename_bound_variables(
        "MATCH (capital:City)-[:IN_COUNTRY {capital: true}]->(co) WHERE date('x') = co RETURN capital"
    )
    if (
        not changed
        or probe != "MATCH (x:Author)-[y:WROTE]->(z:Paper) RETURN x.name"
        or not same_changed
        or same_name != "MATCH (x:singer)-[:VERSION]->(y:Version) RETURN x.name, y.name"
        or "{capital: true}" not in quoted
        or "date('x')" not in quoted
        or "(x:City)" not in quoted
    ):
        raise SystemExit(f"variable-rename self-check failed: {probe} || {same_name}")
    rows = load_train(DATA / "train_split.jsonl")
    queries = [row["cypher"] for row in rows]
    print(f"loaded {len(queries)} training queries", flush=True)
    norms = [normalize(query) for query in queries]
    stats = dataset_statistics(rows, norms)
    write_config(stats)
    print(
        f"unique cypher {stats['unique_normalized_cypher']} duplicate rows {stats['rows_in_duplicate_groups']}",
        flush=True,
    )
    groups = duplicate_groups(norms)
    corpus = build_signatures(queries, DATA / "signatures.pkl")
    hops = np.asarray([len(pattern_edges(query)) for query in tqdm(queries, desc="hops")], dtype=np.int16)
    idf = operator_idf(queries)
    pools = stage_retrieval(queries, groups)
    if args.sample:
        picked, targeted_ids = sample_indices(rows, args.sample)
        selection_path = write_selection(
            rows, corpus, hops, pools, idf, None,
            indices=picked,
            selection_path=SELECTIONS / "train_hard_examples_v2_sample.jsonl",
            metrics_path=CANDIDATES / "combined_candidate_metrics_v2_sample.npz",
        )
        mined = load_records(selection_path)
        by_id = {row["anchor_id"]: row for row in mined}
        rng = np.random.default_rng(42)
        draw = rng.choice(len(mined), size=min(20, len(mined)), replace=False)
        random_rows = [mined[int(index)] for index in sorted(draw)]
        write_case_report(
            REPORTS / "sample_anchors_v2.txt",
            f"Corrected mining sample: 20 of {len(mined)} anchors (seed 42)",
            random_rows,
            "Direction is semantic. Operators ignore string literals. P3 is a lexically harder structural positive.",
        )
        targeted_rows = [by_id[index] for index in targeted_ids if index in by_id]
        write_case_report(
            REPORTS / "targeted_cases_v2.txt",
            "Corrected mining on the difficult cases from the previous report",
            targeted_rows,
            "ProductionCompany arrow form, UpdateDate reversal, STARTS WITH, IS NOT NULL, single-node Movie, ASC/DESC, and a literal containing 'and'.",
        )
        print(f"sample anchors {len(mined)} targeted {len(targeted_rows)}", flush=True)
        print(selection_path, flush=True)
        return 0
    limit = args.limit or None
    selection_path = write_selection(
        rows, corpus, hops, pools, idf, limit,
        selection_path=SELECTIONS / "train_hard_examples_v2.jsonl",
        metrics_path=CANDIDATES / "combined_candidate_metrics_v2.npz",
    )
    if limit is not None:
        print(f"pilot selection wrote {limit} anchors", flush=True)
        return 0
    summary = summarize(selection_path)
    write_reports(selection_path, summary, stats, version="v2")
    print(json.dumps(summary["coverage_counts"], indent=2), flush=True)
    print("\n".join(comparison_lines(summary)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
