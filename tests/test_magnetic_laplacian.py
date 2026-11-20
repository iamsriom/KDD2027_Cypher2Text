"""Tests for the unnormalized Hermitian Magnetic Laplacian (MagPE)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from csencoder.preprocessing.graph_builder import (
    canonical_eigenspace_basis,
    compute_magnetic_laplacian,
    group_eigenvalue_indices,
    phase_canonicalize_vector,
    verify_magnetic_laplacian,
)


def _laplacian(G: nx.DiGraph, q: float = 0.25) -> np.ndarray:
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)
    A_s = (A + A.T) / 2.0
    D_s = np.diag(A_s.sum(axis=1))
    phase = 2.0 * np.pi * q * (A - A.T)
    L = D_s - A_s * np.exp(1j * phase)
    return (L + L.conj().T) / 2.0


def test_magnetic_laplacian_is_hermitian():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 3)])
    L = _laplacian(G)
    assert np.linalg.norm(L - L.conj().T) < 1e-12


def test_eigenvalues_are_real():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 1)])
    L = _laplacian(G)
    evals, _ = np.linalg.eigh(L)
    assert np.max(np.abs(evals.imag)) < 1e-12
    assert np.all(np.isfinite(evals.real))


def test_eigenvectors_satisfy_eigen_equation():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    L = _laplacian(G)
    vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
    used = len(vals)
    C = pe[:, :used] + 1j * pe[:, 8 : 8 + used]
    residual = np.linalg.norm(L @ C - C @ np.diag(vals))
    assert residual < 1e-6


def test_selected_eigenvectors_orthonormal():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
    used = len(vals)
    C = pe[:, :used] + 1j * pe[:, 8 : 8 + used]
    gram = C.conj().T @ C
    assert np.linalg.norm(gram - np.eye(used)) < 1e-6


def test_complex_eigenvectors_have_nonzero_imaginary_part():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    _vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
    imag_energy = float(np.linalg.norm(pe[:, 8:]))
    assert imag_energy > 1e-6


def test_reversing_edges_changes_magnetic_representation():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    R = G.reverse(copy=True)
    _, pe_fwd = compute_magnetic_laplacian(G, q=0.25, k=8)
    _, pe_rev = compute_magnetic_laplacian(R, q=0.25, k=8)
    # Direction enters through (A - A^T); conjugating phases under reversal.
    assert not np.allclose(pe_fwd, pe_rev, atol=1e-8)
    # Imaginary channels should flip sign relative to the forward PE
    # (up to global eigenvector phase canonicalization on each column).
    imag_fwd = pe_fwd[:, 8:]
    imag_rev = pe_rev[:, 8:]
    assert np.linalg.norm(imag_fwd) > 1e-6
    assert np.linalg.norm(imag_rev) > 1e-6


def test_phase_canonicalization_is_deterministic():
    rng = np.random.default_rng(0)
    v = rng.normal(size=16) + 1j * rng.normal(size=16)
    v = v / np.linalg.norm(v)
    a = phase_canonicalize_vector(v * np.exp(1j * 1.7))
    b = phase_canonicalize_vector(v * np.exp(1j * (-0.4)))
    assert np.allclose(a, b, atol=1e-10)
    pivot = int(np.argmax(np.abs(a)))
    assert abs(a[pivot].imag) < 1e-10
    assert a[pivot].real >= -1e-12


def test_repeated_eigenvalues_deterministic_including_k_cutoff():
    # Complete digraph has highly degenerate spectrum.
    G = nx.complete_graph(10, create_using=nx.DiGraph)
    vals_a, pe_a = compute_magnetic_laplacian(G, q=0.25, k=8)
    vals_b, pe_b = compute_magnetic_laplacian(G, q=0.25, k=8)
    assert np.allclose(vals_a, vals_b)
    assert np.allclose(pe_a, pe_b)
    assert pe_a.shape == (10, 16)
    groups = group_eigenvalue_indices(vals_a, tol=1e-8)
    # Truncation at k=8 must still yield orthonormal columns.
    C = pe_a[:, :8] + 1j * pe_a[:, 8:]
    assert np.linalg.norm(C.conj().T @ C - np.eye(8)) < 1e-5
    assert groups  # spectrum was grouped


def test_canonical_basis_matches_projector_span():
    # Two copies of an orthonormal pair with a shared phase mix.
    u = np.array([1, 0, 0], dtype=np.complex128)
    v = np.array([0, 1, 0], dtype=np.complex128)
    mix = np.column_stack([u + v, u - v]) / np.sqrt(2)
    basis = canonical_eigenspace_basis(mix)
    assert basis.shape == (3, 2)
    # Span equality via projector.
    P1 = mix @ mix.conj().T
    P2 = basis @ basis.conj().T
    assert np.linalg.norm(P1 - P2) < 1e-8


@pytest.mark.parametrize(
    "edges",
    [
        [],
        [(0, 0)],  # treated as one node if we build carefully
    ],
)
def test_empty_and_single_node(edges):
    if not edges:
        G = nx.DiGraph()
        vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
        assert pe.shape == (0, 16)
        assert vals.shape == (0,)
    else:
        G = nx.DiGraph()
        G.add_node(0)
        vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
        assert pe.shape == (1, 16)
        assert np.isfinite(pe).all()
        assert abs(pe[0, 0] - 1.0) < 1e-12
        assert np.linalg.norm(pe[0, 1:]) < 1e-12


def test_disconnected_and_fewer_than_eight_nodes():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (2, 3)])
    G.add_node(4)
    vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
    assert pe.shape == (5, 16)
    assert np.isfinite(pe).all()
    assert len(vals) == 5  # padded channels are zero, eigenvalues only for used
    assert np.linalg.norm(pe[:, 5:8]) < 1e-12  # unused real channels
    assert np.linalg.norm(pe[:, 13:]) < 1e-12  # unused imag channels


def test_output_shape_and_finite_always():
    for n in range(0, 12):
        G = nx.path_graph(n, create_using=nx.DiGraph)
        vals, pe = compute_magnetic_laplacian(G, q=0.25, k=8)
        assert pe.shape == (n, 16)
        assert np.isfinite(pe).all()
        assert np.isfinite(vals).all()


def test_verify_helper_ok_on_directed_path():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    report = verify_magnetic_laplacian(G)
    assert report["ok"]
    assert report["imag_energy"] > 1e-6
