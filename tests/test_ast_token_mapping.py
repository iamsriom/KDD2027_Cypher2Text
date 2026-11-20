"""Tests for source-span AST → token mapping (no substring heuristics)."""

from __future__ import annotations

from typing import List, Set, Tuple

import pytest

from csencoder.preprocessing.cypher_antlr_parser import AntlrCypherASTParser
from csencoder.preprocessing.graph_builder import (
    ASTGraphBuilder,
    CypherTokenizer,
    GraphBuilder,
    intervals_overlap,
)


def _hierarchy_edges(G) -> Set[Tuple[int, int]]:
    return {
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "hierarchy"
    }


def _token_index(tokens: List[str], value: str, occurrence: int = 0) -> int:
    hits = [i for i, t in enumerate(tokens) if t == value]
    assert occurrence < len(hits), f"{value!r} occurrence {occurrence} missing in {tokens}"
    return hits[occurrence]


def test_tokenize_with_offsets_preserves_positions():
    tok = CypherTokenizer()
    query = "  MATCH (person) RETURN person"
    pairs = tok.tokenize_with_offsets(query)
    tokens = tok.tokenize(query)
    assert [t for t, _, _ in pairs] == tokens
    for token, start, end in pairs:
        assert query[start:end] == token
        assert end > start


def test_leading_whitespace_offsets_align_with_tokenizer():
    query = "   MATCH (n:Person) RETURN n"
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    builder = GraphBuilder(parser=parser)
    out = builder.build(query, ast=ast)
    assert out["token_spans"][0][0] == query.index("MATCH")
    assert out["tokens"][0] == "MATCH"
    # Root / children spans refer to the original string.
    assert ast.get("start") is not None
    assert query[ast["start"] : ast["end"]].strip().startswith("MATCH")


def test_repeated_variables_map_to_local_occurrences():
    query = "MATCH (a)-[:R]->(b) MATCH (a)-[:R]->(c) RETURN a, b, c"
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    builder = GraphBuilder(parser=parser)
    out = builder.build(query, ast=ast)
    tokens = out["tokens"]
    spans = out["token_spans"]
    # Two MATCH keywords at distinct offsets.
    match_idxs = [i for i, t in enumerate(tokens) if t.upper() == "MATCH"]
    assert len(match_idxs) >= 2
    assert spans[match_idxs[0]][0] < spans[match_idxs[1]][0]
    # Repeated 'a' tokens keep distinct positions.
    a_idxs = [i for i, t in enumerate(tokens) if t == "a"]
    assert len(a_idxs) >= 2
    assert spans[a_idxs[0]] != spans[a_idxs[1]]


def test_consecutive_match_clauses_produce_distinct_ast_nodes():
    query = "MATCH (a)-[:R]->(b)\nMATCH (a)-[:R]->(c)\nRETURN a"
    parser = AntlrCypherASTParser()
    ast = parser._parse_regex(query)
    matches = [c for c in ast["children"] if c.get("type") == "MATCH"]
    assert len(matches) == 2
    assert matches[0]["start"] < matches[1]["start"]
    assert matches[0]["end"] <= matches[1]["start"]
    assert "b" in query[matches[0]["start"] : matches[0]["end"]]
    assert "c" in query[matches[1]["start"] : matches[1]["end"]]
    assert "b" not in query[matches[1]["start"] : matches[1]["end"]]
    assert "c" not in query[matches[0]["start"] : matches[0]["end"]]


def test_optional_match_clause_boundaries():
    query = (
        "MATCH (a:Person)\n"
        "OPTIONAL MATCH (a)-[:KNOWS]->(b)\n"
        "RETURN a, b"
    )
    parser = AntlrCypherASTParser()
    ast = parser._parse_regex(query)
    types = [c.get("type") for c in ast["children"]]
    assert types.count("MATCH") == 1
    assert types.count("OPTIONAL_MATCH") == 1
    opt = next(c for c in ast["children"] if c["type"] == "OPTIONAL_MATCH")
    plain = next(c for c in ast["children"] if c["type"] == "MATCH")
    assert plain["end"] <= opt["start"]
    assert "KNOWS" in query[opt["start"] : opt["end"]]
    assert "Person" in query[plain["start"] : plain["end"]]
    assert "OPTIONAL" not in query[plain["start"] : plain["end"]]


def test_substring_identifiers_do_not_create_spurious_hierarchy():
    # 'a' is a substring of 'ab' — old heuristic linked them incorrectly.
    query = "MATCH (a:Label)-[:REL]->(ab:Label) RETURN a, ab"
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    builder = GraphBuilder(parser=parser)
    out = builder.build(query, ast=ast)
    tokens = out["tokens"]
    spans = out["token_spans"]
    a_i = _token_index(tokens, "a")
    ab_i = _token_index(tokens, "ab")
    assert not intervals_overlap(*spans[a_i], *spans[ab_i])
    # No hierarchy edge that only exists because 'a' ⊆ 'ab' text.
    # (Hierarchy may still connect via true AST ancestry; it must not
    # connect every occurrence of 'a' to every node whose text contains 'a'.)
    hier = _hierarchy_edges(out["graph"])
    # Count edges from the RETURN clause's 'a' token into the pattern 'ab' region.
    return_a = _token_index(tokens, "a", occurrence=1) if tokens.count("a") > 1 else a_i
    # Spurious old behavior: RETURN a ↔ pattern ab via substring. Forbid direct edge.
    assert (return_a, ab_i) not in hier
    assert (ab_i, return_a) not in hier


def test_multiple_match_with_aggregation_and_nested_predicates():
    query = (
        "MATCH (p:Person)-[:KNOWS]->(f:Person) "
        "WHERE p.age > 30 AND f.name CONTAINS 'Ann' "
        "WITH p, count(f) AS c "
        "MATCH (p)-[:WORKS_AT]->(c2:Company) "
        "RETURN p, c, c2"
    )
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    out = GraphBuilder(parser=parser).build(query, ast=ast)
    assert out["magnetic_pe"].shape[1] == 16
    assert len(out["tokens"]) == out["magnetic_pe"].shape[0]
    # Sequence edges always present.
    for i in range(len(out["tokens"]) - 1):
        assert out["graph"].has_edge(i, i + 1)


def test_reversed_relationship_direction_builds_graph():
    fwd = "MATCH (a)-[:R]->(b) RETURN a"
    rev = "MATCH (a)<-[:R]-(b) RETURN a"
    parser = AntlrCypherASTParser()
    b = GraphBuilder(parser=parser)
    g1 = b.build(fwd, ast=parser.parse(fwd))
    g2 = b.build(rev, ast=parser.parse(rev))
    assert g1["magnetic_pe"].shape[1] == 16
    assert g2["magnetic_pe"].shape[1] == 16
    assert np_finite(g1["magnetic_pe"]) and np_finite(g2["magnetic_pe"])


def np_finite(arr) -> bool:
    import numpy as np

    return bool(np.isfinite(arr).all())


def test_repeated_identical_relationship_labels():
    query = "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a, b, c"
    parser = AntlrCypherASTParser()
    out = GraphBuilder(parser=parser).build(query, ast=parser.parse(query))
    knows = [i for i, t in enumerate(out["tokens"]) if t == "KNOWS"]
    assert len(knows) == 2
    assert out["token_spans"][knows[0]][0] < out["token_spans"][knows[1]][0]


def test_antlr_parsing_success_attaches_spans():
    query = "MATCH (n:Person) WHERE n.age = 1 RETURN n"
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    assert ast.get("parser") == "antlr"
    assert "start" in ast and "end" in ast
    assert any("start" in c for c in ast.get("children", []))


def test_regex_fallback_when_antlr_rejects_query():
    # Single-quoted string literals are rejected by this grammar snapshot.
    query = "MATCH (n:Person) WHERE n.name = 'x' RETURN n"
    parser = AntlrCypherASTParser()
    ast = parser.parse(query)
    assert ast.get("parser") == "regex"
    assert "start" in ast and "end" in ast
    out = GraphBuilder(parser=parser).build(query, ast=ast)
    assert out["magnetic_pe"].shape[1] == 16
    assert all(d.get("edge_type") in {"sequence", "hierarchy"} for *_, d in out["graph"].edges(data=True))


def test_regex_fallback_omits_substring_hierarchy_and_keeps_spans():
    query = "MATCH (person)-[:LIKES]->(movie) RETURN person"
    parser = AntlrCypherASTParser()
    ast = parser._parse_regex(query)
    assert ast.get("start") == 0
    assert any(c.get("type") == "MATCH" and "start" in c for c in ast["children"])
    builder = ASTGraphBuilder()
    tokens_with = CypherTokenizer().tokenize_with_offsets(query)
    tokens = [t for t, _, _ in tokens_with]
    spans = [(s, e) for _, s, e in tokens_with]
    G = builder.build_graph_from_ast(ast, tokens, token_spans=spans)
    person_idxs = [i for i, t in enumerate(tokens) if t == "person"]
    assert len(person_idxs) == 2
    # Distinct occurrences keep distinct character spans.
    assert spans[person_idxs[0]] != spans[person_idxs[1]]
    # Hierarchy endpoints must themselves overlap some AST node span.
    ast_spans = []
    stack = [ast]
    while stack:
        node = stack.pop()
        if "start" in node and "end" in node:
            ast_spans.append((node["start"], node["end"]))
        for key in ("children", "patterns", "predicates"):
            stack.extend(node.get(key) or [])
    for u, v in _hierarchy_edges(G):
        assert any(intervals_overlap(*spans[u], *s) for s in ast_spans)
        assert any(intervals_overlap(*spans[v], *s) for s in ast_spans)


def test_regex_only_sequence_when_ast_lacks_spans():
    tokens = ["MATCH", "(", "n", ")", "RETURN", "n"]
    ast = {"type": "QUERY", "text": "MATCH (n) RETURN n", "children": [
        {"type": "MATCH", "text": "MATCH (n)", "children": [], "patterns": [], "predicates": []}
    ], "patterns": [], "predicates": []}
    G = ASTGraphBuilder().build_graph_from_ast(ast, tokens, token_spans=None)
    assert _hierarchy_edges(G) == set()
    for i in range(len(tokens) - 1):
        assert G.has_edge(i, i + 1)


def test_build_graph_from_regex_is_sequence_only():
    query = "MATCH (a)-[:R]->(b) RETURN a"
    G = ASTGraphBuilder().build_graph_from_regex(query)
    assert _hierarchy_edges(G) == set()
    assert all(d.get("edge_type") == "sequence" for *_, d in G.edges(data=True))
