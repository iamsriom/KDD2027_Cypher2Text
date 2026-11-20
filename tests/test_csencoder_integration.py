"""End-to-end smoke: Cypher → tokens/AST/SPD/MagPE → CSEncoder embedding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from csencoder.encoding import (
    FEATURE_CACHE_VERSION,
    attach_idf,
    blank_view,
    build_encoder,
    encode_views,
    prepare_views,
    validate_view_cache_version,
    view_from_graph,
)
from csencoder.preprocessing.cypher_antlr_parser import AntlrCypherASTParser
from csencoder.preprocessing.graph_builder import FEATURE_CACHE_VERSION as GB_VERSION
from csencoder.preprocessing.graph_builder import GraphBuilder


def test_feature_cache_version_constant_aligned():
    assert FEATURE_CACHE_VERSION == GB_VERSION == "magpe_hermitian_v2_ast_spans_v1"


def test_view_cache_rejects_old_version():
    with pytest.raises(RuntimeError, match="feature_cache_version"):
        validate_view_cache_version([{"tokens": [], "pe": np.zeros((0, 16)), "dist": np.zeros((0, 0))}])
    validate_view_cache_version(
        [{"tokens": [], "pe": np.zeros((0, 16)), "dist": np.zeros((0, 0)), "feature_cache_version": FEATURE_CACHE_VERSION}]
    )


def test_end_to_end_encoder_embedding():
    query = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = b.name RETURN a, b"
    parser = AntlrCypherASTParser()
    builder = GraphBuilder(parser=parser)
    graph = builder.build(query, ast=parser.parse(query))

    assert graph["feature_cache_version"] == FEATURE_CACHE_VERSION
    assert graph["magnetic_pe"].shape == (len(graph["tokens"]), 16)
    assert np.isfinite(graph["magnetic_pe"]).all()
    assert graph["dist_matrix"].shape == (len(graph["tokens"]), len(graph["tokens"]))

    view = view_from_graph(graph)
    view = attach_idf(view, {"MATCH": 1.0, "<UNK>": 0.1})
    assert view["feature_cache_version"] == FEATURE_CACHE_VERSION
    assert view["pe"].shape[1] == 16

    token_to_id = {"<UNK>": 0}
    for tok in view["tokens"]:
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

    model = build_encoder(len(token_to_id), hidden_dim=768, num_layers=2, num_heads=4, device="cpu")
    model.eval()
    with torch.no_grad():
        emb = encode_views(model, [view], token_to_id, torch.device("cpu"), chunk=8)
    assert emb.shape == (1, 768)
    assert torch.isfinite(emb).all()
    # L2-normalized
    assert abs(float(emb.norm(dim=1)) - 1.0) < 1e-5


def test_prepare_views_stamps_version():
    views = prepare_views(["MATCH (n) RETURN n", ""], workers=0)
    assert len(views) == 2
    assert views[0]["feature_cache_version"] == FEATURE_CACHE_VERSION
    assert views[1]["feature_cache_version"] == FEATURE_CACHE_VERSION
    assert blank_view()["feature_cache_version"] == FEATURE_CACHE_VERSION
