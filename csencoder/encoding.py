#!/usr/bin/env python3
"""Shared graph-view preparation and batch encoding for CSEncoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from csencoder.model.hybrid_cypher_graph_encoder import CSEncoder
from csencoder.preprocessing.cypher_antlr_parser import AntlrCypherASTParser
from csencoder.preprocessing.graph_builder import FEATURE_CACHE_VERSION, GraphBuilder

PE_DIM = 16
MAX_LEN = 512
ENCODE_CHUNK = 32

_PARSER: Optional[AntlrCypherASTParser] = None
_BUILDER: Optional[GraphBuilder] = None


def blank_view() -> Dict[str, Any]:
    return {
        "tokens": [],
        "pe": np.zeros((0, PE_DIM), dtype=np.float32),
        "dist": np.zeros((0, 0), dtype=np.float32),
        "feature_cache_version": FEATURE_CACHE_VERSION,
    }


def view_from_graph(graph: Dict[str, Any], max_len: int = MAX_LEN) -> Dict[str, Any]:
    tokens = list(graph["tokens"][:max_len])
    length = len(tokens)
    pe = np.zeros((length, PE_DIM), dtype=np.float32)
    raw_pe = graph.get("magnetic_pe")
    if length and raw_pe is not None and len(raw_pe):
        raw_pe = np.asarray(raw_pe, dtype=np.float32)
        if raw_pe.ndim == 2 and raw_pe.shape[1] > 0:
            rows = min(length, raw_pe.shape[0])
            cols = min(PE_DIM, raw_pe.shape[1])
            pe[:rows, :cols] = raw_pe[:rows, :cols]
    dist = np.zeros((length, length), dtype=np.float32)
    raw_dist = graph.get("dist_matrix")
    if length and raw_dist is not None and np.size(raw_dist):
        raw_dist = np.asarray(raw_dist, dtype=np.float32)
        rows = min(length, raw_dist.shape[0])
        cols = min(length, raw_dist.shape[1])
        dist[:rows, :cols] = raw_dist[:rows, :cols]
    version = graph.get("feature_cache_version", FEATURE_CACHE_VERSION)
    return {
        "tokens": tokens,
        "pe": pe,
        "dist": dist,
        "feature_cache_version": version,
    }


def validate_view_cache_version(
    views: Sequence[Dict[str, Any]],
    *,
    expected: str = FEATURE_CACHE_VERSION,
    label: str = "view cache",
) -> None:
    """Reject graph-view caches built with a previous MagPE/AST feature version."""
    if not views:
        return
    sample = views[0] if isinstance(views[0], dict) else None
    if sample is None:
        raise RuntimeError(f"{label}: expected dict views with feature_cache_version")
    version = sample.get("feature_cache_version")
    if version != expected:
        raise RuntimeError(
            f"{label}: feature_cache_version={version!r} != {expected!r}. "
            "Rebuild views with the corrected Hermitian MagPE + AST span pipeline; "
            "do not reuse old train_views.pkl / p1_views.pkl / --view-cache shards."
        )


def _init_worker() -> None:
    global _PARSER, _BUILDER
    _PARSER = AntlrCypherASTParser()
    _BUILDER = GraphBuilder(parser=_PARSER)


def _prepare_query(cypher: str) -> Dict[str, Any]:
    try:
        ast = _PARSER.parse(cypher)
    except Exception:
        ast = None
    try:
        graph = _BUILDER.build(cypher, ast=ast)
    except Exception:
        return blank_view()
    return view_from_graph(graph)


def prepare_views(queries: Sequence[str], workers: int = 0) -> List[Dict[str, Any]]:
    """Build token/PE/SPD views. workers=0 uses a single process (safer for smoke)."""
    if workers and workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        prepared: List[Dict[str, Any]] = []
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for view in tqdm(
                pool.imap(_prepare_query, queries, chunksize=8),
                total=len(queries),
                desc="graph features",
            ):
                prepared.append(view)
        return prepared

    _init_worker()
    return [_prepare_query(q) for q in tqdm(queries, desc="graph features")]


def attach_idf(view: Dict[str, Any], idf_scores: Dict[str, float]) -> Dict[str, Any]:
    unk = float(idf_scores.get("<UNK>", 0.0))
    view = dict(view)
    view["idf"] = np.asarray(
        [float(idf_scores.get(token, unk)) for token in view["tokens"]],
        dtype=np.float32,
    )
    return view


def load_vocab(path: Path) -> Dict[str, int]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "token_to_id" in payload:
        token_to_id = payload["token_to_id"]
    else:
        token_to_id = payload
    if token_to_id.get("<UNK>") != 0:
        raise RuntimeError("vocab must reserve <UNK> at id 0")
    return token_to_id


def save_vocab(path: Path, token_to_id: Dict[str, int], n_docs: int, dataset_name: str) -> None:
    import json

    payload = {
        "unk_token": "<UNK>",
        "unk_id": 0,
        "vocab_size": len(token_to_id),
        "n_train_docs": n_docs,
        "order": "ids 1..V-1 are lexicographically sorted training tokens",
        "built_from": dataset_name,
        "split": "train",
        "token_to_id": token_to_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pad_views(
    views: Sequence[Dict[str, Any]],
    token_to_id: Dict[str, int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    unk = token_to_id["<UNK>"]
    lengths = [len(view["tokens"]) for view in views]
    width = max(max(lengths), 1)
    batch = len(views)
    input_ids = torch.zeros(batch, width, dtype=torch.long)
    attention = torch.zeros(batch, width, dtype=torch.long)
    magnetic = torch.zeros(batch, width, PE_DIM, dtype=torch.float32)
    distance = torch.full((batch, width, width), 999.0, dtype=torch.float32)
    idf = torch.zeros(batch, width, dtype=torch.float32)
    for row, view in enumerate(views):
        tokens = view["tokens"]
        length = len(tokens)
        if length == 0:
            continue
        input_ids[row, :length] = torch.tensor(
            [token_to_id.get(token, unk) for token in tokens], dtype=torch.long
        )
        attention[row, :length] = 1
        magnetic[row, :length] = torch.from_numpy(np.asarray(view["pe"], dtype=np.float32))
        distance[row, :length, :length] = torch.from_numpy(
            np.asarray(view["dist"], dtype=np.float32)
        )
        if "idf" in view:
            idf[row, :length] = torch.from_numpy(np.asarray(view["idf"], dtype=np.float32))
    empty = attention.sum(dim=1) == 0
    if bool(empty.any()):
        attention[empty, 0] = 1
        input_ids[empty, 0] = unk
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "magnetic_pe": magnetic.to(device),
        "dist_matrix": distance.to(device),
        "idf_scores": idf.to(device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
    }


def encode_views(
    model: nn.Module,
    views: Sequence[Dict[str, Any]],
    token_to_id: Dict[str, int],
    device: torch.device,
    chunk: int = ENCODE_CHUNK,
) -> torch.Tensor:
    pieces = []
    for start in range(0, len(views), chunk):
        batch = pad_views(views[start : start + chunk], token_to_id, device)
        encoded = model(
            input_ids=batch["input_ids"],
            magnetic_pe=batch["magnetic_pe"],
            dist_matrix=batch["dist_matrix"],
            idf_scores=batch["idf_scores"],
            attention_mask=batch["attention_mask"],
        )
        keep = (batch["lengths"] > 0).unsqueeze(1)
        pieces.append(torch.where(keep, encoded, torch.zeros_like(encoded)))
    return torch.cat(pieces, dim=0)


def build_encoder(
    vocab_size: int,
    *,
    hidden_dim: int = 768,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> CSEncoder:
    model = CSEncoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pe_dim=PE_DIM,
        max_seq_len=MAX_LEN,
        use_structural_bias=True,
        use_rarity_bias=True,
        dropout=dropout,
    )
    return model.to(torch.device(device))
