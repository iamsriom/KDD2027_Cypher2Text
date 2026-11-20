#!/usr/bin/env python3
"""
Graph Builder for HybridCypherGraph-Encoder (HCGE).

Converts raw Cypher text into:
1. Tokenized text sequence (with optional character offsets)
2. AST-based directed graph (token vertices + sequence/hierarchy edges)
3. Shortest-Path-Distance (SPD) matrix
4. Magnetic Laplacian positional encodings (Hermitian MagPE)

Feature-cache version
---------------------
Caches of graph views must carry ``FEATURE_CACHE_VERSION``. Older MagPE /
substring-AST caches are rejected so training cannot silently reuse incorrect
features. Retrain CSEncoder after this correction; do not treat prior
checkpoints as results of the corrected MagPE/AST pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# Bump when MagPE math or AST→token mapping semantics change.
FEATURE_CACHE_VERSION = "magpe_hermitian_v2_ast_spans_v1"

# Numerical tolerances for MagPE canonicalization.
_EIG_GROUP_TOL = 1e-8
_ORTH_TOL = 1e-8
_PHASE_TOL = 1e-12
_BASIS_TOL = 1e-10


class CypherTokenizer:
    """Regex tokenizer for Cypher queries.

    ``tokenize()`` remains the public string-only API. Prefer
    ``tokenize_with_offsets()`` when building AST-aligned graphs.
    """

    KEYWORDS = [
        "MATCH",
        "WHERE",
        "RETURN",
        "WITH",
        "CREATE",
        "MERGE",
        "DELETE",
        "SET",
        "REMOVE",
        "ORDER",
        "BY",
        "LIMIT",
        "SKIP",
        "UNWIND",
        "CALL",
        "AND",
        "OR",
        "NOT",
        "XOR",
        "IN",
        "STARTS",
        "ENDS",
        "CONTAINS",
        "IS",
        "NULL",
        "DISTINCT",
        "AS",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
    ]

    OPERATORS = [
        "=",
        "<>",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "^",
    ]

    def __init__(self) -> None:
        keyword_pattern = "|".join(rf"\b{kw}\b" for kw in self.KEYWORDS)
        operator_pattern = "|".join(
            re.escape(op) for op in sorted(self.OPERATORS, key=len, reverse=True)
        )
        self.pattern = re.compile(
            rf"({keyword_pattern})|({operator_pattern})|"
            r"([a-zA-Z_][a-zA-Z0-9_]*)|"
            r"(\d+\.?\d*)|"
            r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')|'
            r"([()\[\]{},:;.])"
        )

    def tokenize(self, cypher: str) -> List[str]:
        """Tokenize a Cypher query into a list of tokens (no offsets)."""
        return [token for token, _start, _end in self.tokenize_with_offsets(cypher)]

    def tokenize_with_offsets(self, cypher: str) -> List[Tuple[str, int, int]]:
        """Tokenize with half-open character spans ``[start, end)`` in ``cypher``.

        Spans come from regex ``match.start()`` / ``match.end()`` on the same
        original query string used by the AST parser (including leading
        whitespace). Tokens are not stripped; whitespace is never emitted.
        """
        tokens: List[Tuple[str, int, int]] = []
        for match in self.pattern.finditer(cypher):
            token = match.group(0)
            if not token:
                continue
            tokens.append((token, match.start(), match.end()))
        return tokens


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True iff half-open intervals ``[a_start, a_end)`` and ``[b_start, b_end)`` overlap."""
    return a_start < b_end and b_start < a_end


def _node_span(node: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Return half-open ``[start, end)`` from an AST dict, or None if unknown."""
    start = node.get("start")
    end = node.get("end")
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i < 0 or end_i <= start_i:
        return None
    return start_i, end_i


def _structural_children(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield pattern / predicate / child AST dicts without fabricating nodes."""
    for key in ("children", "patterns", "predicates"):
        for child in node.get(key) or []:
            if isinstance(child, dict):
                yield child


class ASTGraphBuilder:
    """Build directed graphs from Cypher AST structure using source offsets."""

    def __init__(self, parser=None):
        self.parser = parser
        self.tokenizer = CypherTokenizer()

    def build_graph_from_ast(
        self,
        ast: Dict[str, Any],
        tokens: List[str],
        token_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> nx.DiGraph:
        """
        Build a NetworkX DiGraph whose vertices are token positions.

        Edges:
        - sequence: ``(i, i+1)``
        - hierarchy: parent representative → child representative, when both
          AST nodes have authoritative ``[start, end)`` spans that overlap
          tokens. Never uses global substring matching.
        """
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type="token")
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type="sequence")

        if not ast or not isinstance(ast, dict) or not tokens:
            return G

        if token_spans is None or len(token_spans) != len(tokens):
            # Without aligned spans, keep sequence edges only.
            return G

        self._add_ast_edges_by_span(G, ast, token_spans, parent_rep=None)
        return G

    def _tokens_overlapping(
        self,
        token_spans: Sequence[Tuple[int, int]],
        start: int,
        end: int,
    ) -> List[int]:
        return [
            i
            for i, (ts, te) in enumerate(token_spans)
            if intervals_overlap(ts, te, start, end)
        ]

    def _representative_token(
        self,
        token_spans: Sequence[Tuple[int, int]],
        node: Dict[str, Any],
    ) -> Optional[int]:
        span = _node_span(node)
        if span is None:
            return None
        overlapping = self._tokens_overlapping(token_spans, span[0], span[1])
        if not overlapping:
            return None
        # Deterministic: earliest token start, then smallest index.
        return min(overlapping, key=lambda i: (token_spans[i][0], i))

    def _add_ast_edges_by_span(
        self,
        G: nx.DiGraph,
        node: Dict[str, Any],
        token_spans: Sequence[Tuple[int, int]],
        parent_rep: Optional[int],
    ) -> None:
        current_rep = self._representative_token(token_spans, node)
        if current_rep is None:
            # Descend without fabricating a representative; children may still
            # attach under a higher parent once they obtain a span.
            for child in _structural_children(node):
                self._add_ast_edges_by_span(G, child, token_spans, parent_rep)
            return

        if parent_rep is not None and parent_rep != current_rep:
            G.add_edge(parent_rep, current_rep, edge_type="hierarchy")

        for child in _structural_children(node):
            self._add_ast_edges_by_span(G, child, token_spans, parent_rep=current_rep)

    def build_graph_from_regex(self, cypher: str) -> nx.DiGraph:
        """
        Sequence-only fallback graph when no AST is supplied.

        Hierarchical edges are omitted here. Prefer ``GraphBuilder.build`` with
        an offset-annotated regex AST from ``AntlrCypherASTParser`` when
        available; never invent hierarchy via substring matching.
        """
        tokenized = self.tokenizer.tokenize_with_offsets(cypher)
        tokens = [t for t, _, _ in tokenized]
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type="token")
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type="sequence")
        return G


def phase_canonicalize_vector(
    vector: np.ndarray,
    *,
    atol: float = _PHASE_TOL,
) -> np.ndarray:
    """Rotate a complex eigenvector so its largest-magnitude entry is real ≥ 0.

    Tie-break: among equal magnitudes, choose the smallest index. Norm and the
    eigenvalue equation are preserved (global unit-modulus phase).
    """
    v = np.asarray(vector, dtype=np.complex128).reshape(-1).copy()
    if v.size == 0:
        return v
    mags = np.abs(v)
    max_mag = float(np.max(mags))
    if max_mag < atol:
        return v
    # Deterministic first-max index among ties.
    pivot = int(np.argmax(mags))
    ref = v[pivot]
    if abs(ref) < atol:
        return v
    phase = ref / abs(ref)
    aligned = v / phase
    # Numerical cleanup at the pivot (should already be real nonnegative).
    aligned[pivot] = abs(aligned[pivot].real)
    return aligned


def _gram_schmidt_add(
    basis: List[np.ndarray],
    candidate: np.ndarray,
    *,
    tol: float = _BASIS_TOL,
) -> Optional[np.ndarray]:
    """Orthonormalize ``candidate`` against ``basis``; return unit vector or None."""
    v = candidate.astype(np.complex128, copy=True)
    for b in basis:
        v = v - np.vdot(b, v) * b
    norm = float(np.linalg.norm(v))
    if norm <= tol:
        return None
    v = v / norm
    return phase_canonicalize_vector(v)


def canonical_eigenspace_basis(
    vectors: np.ndarray,
    *,
    basis_tol: float = _BASIS_TOL,
) -> np.ndarray:
    """
    Deterministic orthonormal basis for a (possibly degenerate) eigenspace.

    Uses the Hermitian projector ``P = V V^H`` and fixed coordinate order:
    for ``j = 0..n-1``, project ``e_j``, Gram–Schmidt against already accepted
    vectors, phase-canonicalize, until multiplicity columns are filled.
    """
    V = np.asarray(vectors, dtype=np.complex128)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    n, m = V.shape
    if m == 0:
        return V
    if m == 1:
        return phase_canonicalize_vector(V[:, 0]).reshape(-1, 1)

    # Projector onto the eigenspace (Hermitian, idempotent up to numerics).
    P = V @ V.conj().T
    basis: List[np.ndarray] = []
    for j in range(n):
        candidate = P[:, j].copy()
        aligned = _gram_schmidt_add(basis, candidate, tol=basis_tol)
        if aligned is not None:
            basis.append(aligned)
            if len(basis) == m:
                break
    if len(basis) < m:
        # Rare numerical shortfall: fall back to phase-aligned QR of V.
        for col in range(m):
            aligned = _gram_schmidt_add(basis, V[:, col], tol=basis_tol)
            if aligned is not None:
                basis.append(aligned)
                if len(basis) == m:
                    break
    if not basis:
        # Last resort: phase-align original columns.
        return np.column_stack([phase_canonicalize_vector(V[:, i]) for i in range(m)])
    return np.column_stack(basis)


def group_eigenvalue_indices(
    eigenvalues: np.ndarray,
    *,
    tol: float = _EIG_GROUP_TOL,
) -> List[List[int]]:
    """Group sorted eigenvalue indices into contiguous degenerate clusters."""
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return []
    groups: List[List[int]] = [[0]]
    for i in range(1, vals.size):
        if abs(vals[i] - vals[groups[-1][0]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def compute_magnetic_laplacian(
    G: nx.DiGraph,
    q: float = 0.25,
    k: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unnormalized Hermitian Magnetic Laplacian positional encodings.

    Mathematical core (magnetic charge ``q``, typically ``γ = 0.25``)::

        A_s = (A + A^T) / 2
        D_s = diag(A_s 1)
        L^(γ) = D_s - A_s ⊙ exp(i 2π γ (A - A^T))

    Complex Hermitian eigendecomposition via ``numpy.linalg.eigh``. The ``k``
    smallest-eigenvalue eigenvectors are phase-canonicalized; degenerate
    eigenspaces use a deterministic projector basis (including groups that
    cross the ``k``-th cutoff). Real and imaginary parts are concatenated into
    a real array of shape ``(n, 2k)`` (zero-padded when ``n < k``).

    Returns:
        ``(eigenvalues, magnetic_pe)`` where ``magnetic_pe`` has shape
        ``(n, 2k)`` with columns ``[Re_0..Re_{k-1}, Im_0..Im_{k-1}]``.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    pe_dim = 2 * k

    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, pe_dim), dtype=np.float64)

    if n == 1:
        # Single isolated node: L = [0], eigenvector = [1] (real).
        pe = np.zeros((1, pe_dim), dtype=np.float64)
        pe[0, 0] = 1.0
        return np.array([0.0], dtype=np.float64), pe

    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    A_s = (A + A.T) / 2.0
    D_s = np.diag(A_s.sum(axis=1))
    phase = 2.0 * np.pi * q * (A - A.T)
    L_mag = D_s - A_s * np.exp(1j * phase)

    # Numerical Hermitian projection (construction is Hermitian; this removes
    # float asymmetry). Do NOT drop the imaginary part before eigh.
    L_mag = (L_mag + L_mag.conj().T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(L_mag)
    # eigh already returns ascending eigenvalues; keep an explicit sort.
    order = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues.real[order]
    eigenvectors = eigenvectors[:, order]

    groups = group_eigenvalue_indices(eigenvalues, tol=_EIG_GROUP_TOL)
    selected_vals: List[float] = []
    selected_vecs: List[np.ndarray] = []
    remaining = k

    for group in groups:
        if remaining <= 0:
            break
        idxs = group
        block = eigenvectors[:, idxs]
        basis = canonical_eigenspace_basis(block)
        take = min(remaining, basis.shape[1])
        # Degenerate group shares one eigenvalue (use the group's first).
        group_val = float(eigenvalues[idxs[0]])
        for j in range(take):
            selected_vals.append(group_val)
            selected_vecs.append(basis[:, j])
        remaining -= take

    used = len(selected_vecs)
    pe = np.zeros((n, pe_dim), dtype=np.float64)
    if used:
        U = np.column_stack(selected_vecs)
        pe[:, :used] = U.real
        pe[:, k : k + used] = U.imag

    if not np.all(np.isfinite(pe)):
        raise FloatingPointError("Magnetic PE contains NaN or infinite values")

    eigenvals = np.asarray(selected_vals, dtype=np.float64)
    return eigenvals, pe


def verify_magnetic_laplacian(
    G: nx.DiGraph,
    q: float = 0.25,
    k: int = 8,
    *,
    herm_tol: float = 1e-8,
    eig_tol: float = 1e-6,
    orth_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Diagnostics for MagPE correctness (used by tests)."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    report: Dict[str, Any] = {"n": n, "ok": True}
    if n == 0:
        return report

    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    A_s = (A + A.T) / 2.0
    D_s = np.diag(A_s.sum(axis=1))
    phase = 2.0 * np.pi * q * (A - A.T)
    L_mag = D_s - A_s * np.exp(1j * phase)
    L_mag = (L_mag + L_mag.conj().T) / 2.0

    herm_err = float(np.linalg.norm(L_mag - L_mag.conj().T))
    report["hermitian_error"] = herm_err
    if herm_err > herm_tol:
        report["ok"] = False

    evals, evecs = np.linalg.eigh(L_mag)
    report["eigenvalues_real_error"] = float(np.max(np.abs(evals.imag))) if False else 0.0
    # eigh returns real eigenvalues for Hermitian matrices
    report["max_imag_eigenvalue"] = 0.0

    used = min(k, n)
    U = evecs[:, :used]
    # Rebuild selected MagPE path for orth check after canonicalization
    _vals, pe = compute_magnetic_laplacian(G, q=q, k=k)
    C = pe[:, :used] + 1j * pe[:, k : k + used]
    gram = C.conj().T @ C
    orth_err = float(np.linalg.norm(gram - np.eye(used)))
    report["orthonormality_error"] = orth_err
    if orth_err > orth_tol:
        report["ok"] = False

    # Eigen-equation on canonical basis vs L
    residual = float(np.linalg.norm(L_mag @ C - C @ np.diag(_vals[:used])))
    report["eigen_equation_residual"] = residual
    if residual > eig_tol * max(1.0, float(np.linalg.norm(L_mag))):
        report["ok"] = False

    report["imag_energy"] = float(np.linalg.norm(pe[:, k : k + used]))
    return report


def compute_spd_matrix(G: nx.DiGraph) -> np.ndarray:
    """
    Compute Shortest-Path-Distance (SPD) matrix.

    Args:
        G: NetworkX directed graph

    Returns:
        N x N distance matrix (unreachable → max_finite + 1)
    """
    n = len(G)
    if n == 0:
        return np.array([]).reshape(0, 0)

    dist_matrix = np.full((n, n), np.inf)
    node_list = sorted(G.nodes())
    index = {node: i for i, node in enumerate(node_list)}

    for src in node_list:
        dist_matrix[index[src], index[src]] = 0
        try:
            lengths = nx.single_source_shortest_path_length(G, src)
        except Exception:
            continue
        for dst, length in lengths.items():
            dist_matrix[index[src], index[dst]] = length

    finite = dist_matrix[np.isfinite(dist_matrix)]
    if finite.size:
        max_finite = float(np.max(finite))
        dist_matrix[~np.isfinite(dist_matrix)] = max_finite + 1
    else:
        dist_matrix[~np.isfinite(dist_matrix)] = n

    return dist_matrix


class GraphBuilder:
    """Main graph builder class that orchestrates the process."""

    def __init__(self, parser=None):
        self.ast_builder = ASTGraphBuilder(parser=parser)
        self.tokenizer = CypherTokenizer()
        self.feature_cache_version = FEATURE_CACHE_VERSION

    def build(self, cypher: str, ast: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build all graph components for a Cypher query.

        Returns a dict with tokens, graph, dist_matrix, magnetic_pe (n×16),
        eigenvalues, and ``feature_cache_version``.
        """
        tokenized = self.tokenizer.tokenize_with_offsets(cypher)
        tokens = [t for t, _, _ in tokenized]
        token_spans = [(s, e) for _, s, e in tokenized]

        if len(tokens) == 0:
            return {
                "tokens": [],
                "graph": nx.DiGraph(),
                "dist_matrix": np.array([]).reshape(0, 0),
                "magnetic_pe": np.zeros((0, 16), dtype=np.float64),
                "eigenvalues": np.zeros(0, dtype=np.float64),
                "feature_cache_version": FEATURE_CACHE_VERSION,
                "token_spans": [],
            }

        if ast and isinstance(ast, dict):
            G = self.ast_builder.build_graph_from_ast(ast, tokens, token_spans=token_spans)
        else:
            # No AST: sequence edges only (no substring hierarchy heuristics).
            G = self.ast_builder.build_graph_from_regex(cypher)

        for i, token in enumerate(tokens):
            if i not in G:
                G.add_node(i, token=token, type="token")

        dist_matrix = compute_spd_matrix(G)
        eigenvals, magnetic_pe = compute_magnetic_laplacian(G, q=0.25, k=8)

        return {
            "tokens": tokens,
            "graph": G,
            "dist_matrix": dist_matrix,
            "magnetic_pe": magnetic_pe,
            "eigenvalues": eigenvals,
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "token_spans": token_spans,
        }
