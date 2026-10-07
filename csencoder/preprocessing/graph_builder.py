#!/usr/bin/env python3
"""
Graph Builder for HybridCypherGraph-Encoder (HCGE).

Converts raw Cypher text into:
1. Tokenized text sequence
2. AST-based directed graph
3. Shortest-Path-Distance (SPD) matrix
4. Magnetic Laplacian positional encodings (eigenvectors)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix


class CypherTokenizer:
    """Simple tokenizer for Cypher queries using regex patterns."""
    
    # Cypher keywords and operators
    KEYWORDS = [
        'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 'DELETE',
        'SET', 'REMOVE', 'ORDER', 'BY', 'LIMIT', 'SKIP', 'UNWIND', 'CALL',
        'AND', 'OR', 'NOT', 'XOR', 'IN', 'STARTS', 'ENDS', 'CONTAINS',
        'IS', 'NULL', 'DISTINCT', 'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
    ]
    
    # Operators
    OPERATORS = ['=', '<>', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '^']
    
    def __init__(self):
        # Build regex pattern for tokenization
        keyword_pattern = '|'.join(rf'\b{kw}\b' for kw in self.KEYWORDS)
        operator_pattern = '|'.join(re.escape(op) for op in sorted(self.OPERATORS, key=len, reverse=True))
        
        # Pattern: keywords | operators | identifiers | numbers | strings | punctuation
        self.pattern = re.compile(
            rf'({keyword_pattern})|({operator_pattern})|'
            r'([a-zA-Z_][a-zA-Z0-9_]*)|'  # identifiers
            r'(\d+\.?\d*)|'  # numbers
            r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')|'  # strings
            r'([()\[\]{},:;.])'  # punctuation
        )
    
    def tokenize(self, cypher: str) -> List[str]:
        """Tokenize a Cypher query into a list of tokens."""
        tokens = []
        for match in self.pattern.finditer(cypher):
            token = match.group(0)
            if token.strip():
                tokens.append(token.strip())
        return tokens


class ASTGraphBuilder:
    """Build directed graph from Cypher AST structure."""
    
    def __init__(self, parser=None):
        """
        Initialize the graph builder.
        
        Args:
            parser: Optional AST parser (from cypher_antlr_parser)
        """
        self.parser = parser
        self.tokenizer = CypherTokenizer()
    
    def build_graph_from_ast(self, ast: Dict[str, Any], tokens: List[str]) -> nx.DiGraph:
        """
        Build NetworkX directed graph from AST.
        
        Creates edges for:
        - Hierarchy: Parent -> Child
        - Sequence: Token i -> Token i+1
        
        Args:
            ast: AST dictionary from parser
            tokens: Tokenized query
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add all tokens as nodes
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type='token')
        
        # Add sequence edges (token i -> token i+1)
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type='sequence')
        
        # Add hierarchical edges from AST
        if ast and isinstance(ast, dict):
            self._add_ast_edges(G, ast, tokens)
        
        return G
    
    def _add_ast_edges(self, G: nx.DiGraph, node: Dict[str, Any], tokens: List[str], parent_idx: int = None):
        """Recursively add AST hierarchy edges."""
        # Map AST text to token indices (simplified - match by position)
        node_text = node.get('text', '')
        if node_text:
            # Find tokens that appear in this AST node's text
            token_indices = []
            text_lower = node_text.lower()
            for i, token in enumerate(tokens):
                if token.lower() in text_lower:
                    token_indices.append(i)
            
            # Connect parent to children tokens
            if parent_idx is not None and token_indices:
                for child_idx in token_indices:
                    if parent_idx != child_idx:
                        G.add_edge(parent_idx, child_idx, edge_type='hierarchy')
            
            # Use first token as representative for this AST node
            if token_indices:
                current_rep = token_indices[0]
            else:
                return
        else:
            return
        
        # Recursively process children
        for child in node.get('children', []):
            if isinstance(child, dict):
                self._add_ast_edges(G, child, tokens, parent_idx=current_rep)
    
    def build_graph_from_regex(self, cypher: str) -> nx.DiGraph:
        """
        Build graph using regex-based parsing (fallback when AST parser unavailable).
        
        Args:
            cypher: Raw Cypher query string
            
        Returns:
            NetworkX DiGraph
        """
        tokens = self.tokenizer.tokenize(cypher)
        G = nx.DiGraph()
        
        # Add all tokens as nodes
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type='token')
        
        # Add sequence edges
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type='sequence')
        
        # Add structural edges based on Cypher patterns
        # Match patterns: MATCH (node) -[rel]-> (node)
        match_pattern = re.compile(r'MATCH\s+(.+?)(?=\b(?:WHERE|RETURN|WITH|$))', re.IGNORECASE)
        match_content = match_pattern.search(cypher)
        
        if match_content:
            # Find node and relationship tokens
            content = match_content.group(1)
            node_pattern = re.compile(r'\(([^)]+)\)')
            rel_pattern = re.compile(r'-\[([^\]]+)\]-')
            
            nodes = node_pattern.findall(content)
            rels = rel_pattern.findall(content)
            
            # Connect nodes through relationships (simplified)
            for i, token in enumerate(tokens):
                if any(node_part in token for node_part in nodes):
                    # Connect to next node token
                    for j in range(i + 1, min(i + 10, len(tokens))):
                        if any(node_part in tokens[j] for node_part in nodes):
                            G.add_edge(i, j, edge_type='structural')
                            break
        
        return G


def compute_magnetic_laplacian(G: nx.DiGraph, q: float = 0.25, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Magnetic Laplacian and its eigenvectors.
    
    Formula: L^{(q)} = D - A ⊙ exp(i 2π q (A - A^T))
    
    Args:
        G: NetworkX directed graph
        q: Magnetic charge parameter (default 0.25)
        k: Number of eigenvectors to compute
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvectors are complex
    """
    n = len(G)
    if n == 0:
        return np.array([]), np.array([])
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()), weight=None).toarray()
    A = A.astype(np.complex128)
    
    # Compute degree matrix
    D = np.diag(np.sum(A.real, axis=1))
    
    # Compute asymmetric part: (A - A^T)
    A_asym = A - A.T
    
    # Magnetic Laplacian: L^{(q)} = D - A ⊙ exp(i 2π q (A - A^T))
    # For real-valued computation, we separate real and imaginary parts
    phase = 2 * np.pi * q * A_asym
    exp_phase = np.exp(1j * phase)
    
    # Element-wise multiplication
    A_mag = A * exp_phase
    
    # Laplacian
    L_mag = D - A_mag
    
    # Convert to real symmetric matrix for eigendecomposition
    # Use (L_mag + L_mag^H) / 2 (Hermitian part)
    L_sym = (L_mag + L_mag.conj().T) / 2
    
    # Compute eigenvalues and eigenvectors
    # Use scipy for efficient computation
    L_sparse = csr_matrix(L_sym.real)  # Use real part for eigendecomposition
    
    try:
        eigenvalues, eigenvectors = eigh(L_sparse.toarray())
    except:
        # Fallback: use dense matrix
        eigenvalues, eigenvectors = eigh(L_sym.real)
    
    # Take top k eigenvectors (smallest eigenvalues)
    idx = np.argsort(eigenvalues)[:k]
    eigenvals = eigenvalues[idx]
    eigenvecs = eigenvectors[:, idx]
    
    # Convert eigenvectors to complex (they're real from symmetric matrix, but we want complex representation)
    eigenvecs_complex = eigenvecs.astype(np.complex128)
    
    # Separate real and imaginary parts for downstream processing
    # We'll treat them as 2k real-valued features
    eigenvecs_real = eigenvecs_complex.real
    eigenvecs_imag = eigenvecs_complex.imag
    
    # Stack real and imaginary parts: [real_1, ..., real_k, imag_1, ..., imag_k]
    eigenvecs_combined = np.hstack([eigenvecs_real, eigenvecs_imag])
    
    return eigenvals, eigenvecs_combined


def compute_spd_matrix(G: nx.DiGraph) -> np.ndarray:
    """
    Compute Shortest-Path-Distance (SPD) matrix.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        N x N distance matrix (inf for unreachable nodes)
    """
    n = len(G)
    if n == 0:
        return np.array([]).reshape(0, 0)
    
    # Compute shortest path lengths
    dist_matrix = np.full((n, n), np.inf)
    
    for i in sorted(G.nodes()):
        for j in sorted(G.nodes()):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                try:
                    path_length = nx.shortest_path_length(G, i, j)
                    dist_matrix[i, j] = path_length
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = np.inf
    
    # Replace inf with a large finite value (max distance + 1)
    max_finite = np.max(dist_matrix[dist_matrix != np.inf])
    if np.isfinite(max_finite):
        dist_matrix[dist_matrix == np.inf] = max_finite + 1
    else:
        dist_matrix[dist_matrix == np.inf] = n  # Fallback: use graph size
    
    return dist_matrix


class GraphBuilder:
    """Main graph builder class that orchestrates the process."""
    
    def __init__(self, parser=None):
        """
        Initialize graph builder.
        
        Args:
            parser: Optional AST parser
        """
        self.ast_builder = ASTGraphBuilder(parser=parser)
        self.tokenizer = CypherTokenizer()
    
    def build(self, cypher: str, ast: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build all graph components for a Cypher query.
        
        Args:
            cypher: Raw Cypher query string
            ast: Optional AST dictionary from parser
            
        Returns:
            Dictionary containing:
            - tokens: List of token strings
            - graph: NetworkX DiGraph
            - dist_matrix: N x N SPD matrix
            - magnetic_pe: 2k real-valued features from Magnetic Laplacian eigenvectors
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(cypher)
        
        if len(tokens) == 0:
            # Return empty structure
            return {
                'tokens': [],
                'graph': nx.DiGraph(),
                'dist_matrix': np.array([]).reshape(0, 0),
                'magnetic_pe': np.array([]).reshape(0, 0)
            }
        
        # Build graph
        if ast:
            G = self.ast_builder.build_graph_from_ast(ast, tokens)
        else:
            G = self.ast_builder.build_graph_from_regex(cypher)
        
        # Ensure graph has all token nodes (in case AST parsing missed some)
        for i in range(len(tokens)):
            if i not in G:
                G.add_node(i, token=tokens[i], type='token')
        
        # Compute SPD matrix
        dist_matrix = compute_spd_matrix(G)
        
        # Compute Magnetic Laplacian PE
        eigenvals, magnetic_pe = compute_magnetic_laplacian(G, q=0.25, k=8)
        
        return {
            'tokens': tokens,
            'graph': G,
            'dist_matrix': dist_matrix,
            'magnetic_pe': magnetic_pe,
            'eigenvalues': eigenvals
        }

