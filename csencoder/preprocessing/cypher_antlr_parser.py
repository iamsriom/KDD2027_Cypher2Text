#!/usr/bin/env python3
"""
ANTLR-based Cypher AST Parser.

Uses ANTLR4 to parse Cypher queries into AST structure.
Falls back to regex parser if ANTLR parser is not available or fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import ANTLR4 generated parser
ANTLR_AVAILABLE = False
try:
    from antlr4 import InputStream, CommonTokenStream

    parser_dir = Path(__file__).resolve().parent / "parsers" / "cypher" / "gen_cypher"
    if parser_dir.exists():
        sys.path.insert(0, str(parser_dir.parent))
        try:
            from gen_cypher.CypherLexer import CypherLexer
            from gen_cypher.CypherParser import CypherParser
            from gen_cypher.CypherVisitor import CypherVisitor
            ANTLR_AVAILABLE = True
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            try:
                from parsers.cypher.gen_cypher.CypherLexer import CypherLexer
                from parsers.cypher.gen_cypher.CypherParser import CypherParser
                from parsers.cypher.gen_cypher.CypherVisitor import CypherVisitor
                ANTLR_AVAILABLE = True
            except ImportError:
                pass
except ImportError:
    pass


class ASTBuildingVisitor:
    """
    Visitor to build AST structure from ANTLR parse tree.
    Adapts to the actual rule names in the Cypher grammar.
    """
    
    def __init__(self, original_text: str):
        self.original_text = original_text
    
    def visit(self, ctx) -> Optional[Dict[str, Any]]:
        """Generic visit method that dispatches to specific visit methods."""
        if ctx is None:
            return None
        
        # Check if this is a terminal node (leaf node, no children)
        from antlr4.tree.Tree import TerminalNodeImpl
        if isinstance(ctx, TerminalNodeImpl):
            # Terminal node - just return the text, don't try to visit children
            return {
                "type": "TERMINAL",
                "text": ctx.getText() if hasattr(ctx, 'getText') else str(ctx),
                "children": [],
                "patterns": [],
                "predicates": [],
            }
        
        # Get the class name and try to find a specific visit method
        class_name = ctx.__class__.__name__  # e.g., "OC_MatchContext"
        base_name = class_name.replace("Context", "")  # e.g., "OC_Match"
        method_name = f"visit{base_name}"  # e.g., "visitOC_Match"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(ctx)
        else:
            # Default: create a generic node
            return self._default_visit(ctx)
    
    def _default_visit(self, ctx) -> Dict[str, Any]:
        """Default visit for unknown node types."""
        # Check if this is a terminal node (leaf node, no children)
        from antlr4.tree.Tree import TerminalNodeImpl
        if isinstance(ctx, TerminalNodeImpl):
            # Terminal node - just return the text
            return {
                "type": "TERMINAL",
                "text": ctx.getText() if hasattr(ctx, 'getText') else str(ctx),
                "children": [],
                "patterns": [],
                "predicates": [],
            }
        
        # Non-terminal node - process children
        children = []
        if hasattr(ctx, 'children') and ctx.children is not None:
            for child in ctx.children:
                child_node = self.visit(child)
                if child_node:
                    children.append(child_node)
        
        return {
            "type": ctx.__class__.__name__.upper().replace("CONTEXT", ""),
            "text": ctx.getText() if hasattr(ctx, 'getText') else str(ctx),
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_Cypher(self, ctx) -> Dict[str, Any]:
        """Visit root cypher rule."""
        children = []
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                children.append(child_node)
        
        return {
            "type": "QUERY",
            "text": self.original_text,
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_Statement(self, ctx) -> Dict[str, Any]:
        """Visit statement rule."""
        children = []
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                children.append(child_node)
        
        return {
            "type": "STATEMENT",
            "text": ctx.getText(),
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_Query(self, ctx) -> Dict[str, Any]:
        """Visit query rule."""
        children = []
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                children.append(child_node)
        
        return {
            "type": "QUERY",
            "text": ctx.getText(),
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_RegularQuery(self, ctx) -> Dict[str, Any]:
        """Visit regular query rule."""
        children = []
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                children.append(child_node)
        
        return {
            "type": "QUERY",
            "text": ctx.getText(),
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_SingleQuery(self, ctx) -> Dict[str, Any]:
        """Visit single query rule."""
        children = []
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                children.append(child_node)
        
        return {
            "type": "QUERY",
            "text": ctx.getText(),
            "children": children,
            "patterns": [],
            "predicates": [],
        }
    
    def visitOC_Match(self, ctx) -> Dict[str, Any]:
        """Visit MATCH clause."""
        node = {
            "type": "MATCH",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children first
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                # Categorize children
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                elif child_node.get("type") == "PREDICATE":
                    node["predicates"].append(child_node)
                elif child_node.get("type") == "WHERE":
                    # Extract predicates from WHERE
                    if child_node.get("predicates"):
                        node["predicates"].extend(child_node["predicates"])
                else:
                    node["children"].append(child_node)
        
        # Also check for direct pattern/where attributes
        if hasattr(ctx, 'oC_Pattern') and ctx.oC_Pattern():
            pattern_node = self.visit(ctx.oC_Pattern())
            if pattern_node and pattern_node not in node["patterns"]:
                node["patterns"].append(pattern_node)
        
        if hasattr(ctx, 'oC_Where') and ctx.oC_Where():
            where_node = self.visit(ctx.oC_Where())
            if where_node and where_node.get("predicates"):
                for pred in where_node["predicates"]:
                    if pred not in node["predicates"]:
                        node["predicates"].append(pred)
        
        return node
    
    def visitOC_Where(self, ctx) -> Dict[str, Any]:
        """Visit WHERE clause."""
        node = {
            "type": "WHERE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") == "PREDICATE":
                    node["predicates"].append(child_node)
                else:
                    node["children"].append(child_node)
        
        # Also check for direct expression
        if hasattr(ctx, 'oC_Expression') and ctx.oC_Expression():
            expr_text = ctx.oC_Expression().getText()
            node["predicates"].append({
                "type": "PREDICATE",
                "text": expr_text,
            })
        
        return node
    
    def visitOC_Return(self, ctx) -> Dict[str, Any]:
        """Visit RETURN clause."""
        node = {
            "type": "RETURN",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_With(self, ctx) -> Dict[str, Any]:
        """Visit WITH clause."""
        node = {
            "type": "WITH",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Create(self, ctx) -> Dict[str, Any]:
        """Visit CREATE clause."""
        node = {
            "type": "CREATE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        
        return node
    
    def visitOC_Merge(self, ctx) -> Dict[str, Any]:
        """Visit MERGE clause."""
        node = {
            "type": "MERGE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        
        return node
    
    def visitOC_Delete(self, ctx) -> Dict[str, Any]:
        """Visit DELETE clause."""
        node = {
            "type": "DELETE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Set(self, ctx) -> Dict[str, Any]:
        """Visit SET clause."""
        node = {
            "type": "SET",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Remove(self, ctx) -> Dict[str, Any]:
        """Visit REMOVE clause."""
        node = {
            "type": "REMOVE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Order(self, ctx) -> Dict[str, Any]:
        """Visit ORDER BY clause."""
        node = {
            "type": "ORDER_BY",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Limit(self, ctx) -> Dict[str, Any]:
        """Visit LIMIT clause."""
        node = {
            "type": "LIMIT",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Skip(self, ctx) -> Dict[str, Any]:
        """Visit SKIP clause."""
        node = {
            "type": "SKIP",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Unwind(self, ctx) -> Dict[str, Any]:
        """Visit UNWIND clause."""
        node = {
            "type": "UNWIND",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Call(self, ctx) -> Dict[str, Any]:
        """Visit CALL clause."""
        node = {
            "type": "CALL",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_Pattern(self, ctx) -> Dict[str, Any]:
        """Visit pattern."""
        node = {
            "type": "PATTERN",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children (node patterns, relationship patterns)
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        
        return node
    
    def visitOC_NodePattern(self, ctx) -> Dict[str, Any]:
        """Visit node pattern."""
        node = {
            "type": "NODE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children (labels, properties, etc.)
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node
    
    def visitOC_RelationshipPattern(self, ctx) -> Dict[str, Any]:
        """Visit relationship pattern."""
        node = {
            "type": "RELATIONSHIP",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        
        # Visit all children (relationship types, properties, etc.)
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                node["children"].append(child_node)
        
        return node


class AntlrCypherASTParser:
    """
    ANTLR-based Cypher AST Parser.
    
    Uses ANTLR4 to parse Cypher queries into AST structure.
    Falls back to regex parser if ANTLR is not available.
    """
    
    def __init__(self):
        self.antlr_available = ANTLR_AVAILABLE
        self.stats = {
            "antlr_parser_success": 0,
            "antlr_parser_failed": 0,
            "regex_parser_used": 0,
        }
        
        # Initialize regex parser as fallback
        self._init_regex_parser()
        
        if not self.antlr_available:
            import logging
            logger = logging.getLogger("hybrid")
            logger.warning(
                "ANTLR parser not available. Using regex-based parser only. "
                "To enable ANTLR parser, generate parser files from Cypher.g4 grammar."
            )
    
    def _init_regex_parser(self):
        """Initialize regex-based fallback parser."""
        import re
        self.re = re
    
    def parse(self, cypher: str) -> Dict[str, Any]:
        """
        Parse a Cypher query into AST structure.
        
        Strategy:
        1. Try ANTLR parser first (if available)
        2. If it fails, fall back to regex parser
        """
        import contextlib
        from io import StringIO
        
        cypher = cypher.strip()
        
        # Try ANTLR parser first if available
        if self.antlr_available:
            try:
                # Suppress all stderr output from ANTLR parser
                with contextlib.redirect_stderr(StringIO()):
                    result = self._parse_via_antlr(cypher)
                    # Only increment success if we got a valid result
                    if result and isinstance(result, dict) and result.get('type'):
                        self.stats["antlr_parser_success"] += 1
                        return result
                    else:
                        # Invalid result, treat as failure
                        raise ValueError("ANTLR parser returned invalid result")
            except Exception as e:
                # ANTLR parser failed, use regex fallback
                # Only log occasionally to avoid spam
                self.stats["antlr_parser_failed"] += 1
                # Don't re-raise, fall through to regex parser
        
        # Use regex parser
        self.stats["regex_parser_used"] += 1
        return self._parse_regex(cypher)
    
    def _parse_via_antlr(self, cypher: str) -> Dict[str, Any]:
        """Parse Cypher using ANTLR4."""
        if not ANTLR_AVAILABLE:
            raise RuntimeError("ANTLR parser not available")
        
        import contextlib
        import io
        
        # Suppress ANTLR error messages by redirecting stderr
        error_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stderr(error_buffer):
                input_stream = InputStream(cypher)
                lexer = CypherLexer(input_stream)
                
                # Remove default error listeners to suppress error messages
                lexer.removeErrorListeners()
                
                tokens = CommonTokenStream(lexer)
                parser = CypherParser(tokens)
                
                # Remove default error listeners to suppress error messages
                parser.removeErrorListeners()
                
                # Try different entry rule names (depends on grammar)
                tree = None
                parse_exception = None
                
                for rule_name in ['oC_Cypher', 'cypher', 'oC_Statement', 'statement']:
                    if hasattr(parser, rule_name):
                        try:
                            # Create new parser instance for each attempt (reset doesn't always work)
                            tokens = CommonTokenStream(lexer)
                            parser = CypherParser(tokens)
                            parser.removeErrorListeners()
                            tree = getattr(parser, rule_name)()
                            
                            # Check if parsing was successful
                            # After removing error listeners, we need to check differently
                            if tree is not None:
                                # Check for syntax errors (this should work even without error listeners)
                                syntax_errors = 0
                                if hasattr(parser, 'getNumberOfSyntaxErrors'):
                                    syntax_errors = parser.getNumberOfSyntaxErrors()
                                
                                # If no syntax errors and tree exists, it's valid
                                if syntax_errors == 0:
                                    # Additional validation: tree should have some content
                                    if hasattr(tree, 'getChildCount'):
                                        if tree.getChildCount() > 0:
                                            break
                                    elif hasattr(tree, 'children'):
                                        if len(tree.children) > 0:
                                            break
                                    else:
                                        # Tree exists, assume valid
                                        break
                                else:
                                    # Syntax errors detected, try next rule
                                    tree = None
                                    continue
                        except Exception as e:
                            # Store exception but try next rule
                            parse_exception = e
                            tree = None
                            continue
                
                if tree is None:
                    # If all rules failed, raise the last exception or a generic error
                    if parse_exception:
                        raise parse_exception
                    raise ValueError("Could not parse query with ANTLR - no valid parse tree generated")
                
                # Build AST using visitor
                visitor = ASTBuildingVisitor(cypher)
                ast = visitor.visit(tree)
                
                # Ensure top-level schema matches expected format
                if not isinstance(ast, dict):
                    ast = {"type": "QUERY", "text": cypher, "children": [], "patterns": [], "predicates": []}
                
                ast.setdefault("type", "QUERY")
                ast.setdefault("text", cypher)
                ast.setdefault("children", [])
                ast.setdefault("patterns", [])
                ast.setdefault("predicates", [])
                
                return ast
        except Exception:
            # Any error means ANTLR parsing failed
            raise
    
    def _parse_regex(self, cypher: str) -> Dict[str, Any]:
        """Fallback regex-based parser."""
        ast = {"type": "QUERY", "children": [], "patterns": [], "predicates": [], "text": cypher}
        
        # Extract main clauses
        clauses = self._extract_clauses(cypher)
        for clause_type, clause_content in clauses:
            if clause_type == "MATCH":
                patterns = self._extract_patterns(clause_content)
                ast["patterns"].extend(patterns)
                ast["children"].append({
                    "type": "MATCH",
                    "content": clause_content,
                    "patterns": patterns,
                    "text": clause_content
                })
            elif clause_type == "WHERE":
                predicates = self._extract_predicates(clause_content)
                ast["predicates"].extend(predicates)
                ast["children"].append({
                    "type": "WHERE",
                    "content": clause_content,
                    "predicates": predicates,
                    "text": clause_content
                })
            elif clause_type == "RETURN":
                returns = self._extract_returns(clause_content)
                ast["children"].append({
                    "type": "RETURN",
                    "content": clause_content,
                    "returns": returns,
                    "text": clause_content
                })
            elif clause_type == "WITH":
                with_items = self._extract_with_items(clause_content)
                ast["children"].append({
                    "type": "WITH",
                    "content": clause_content,
                    "items": with_items,
                    "text": clause_content
                })
            elif clause_type in ["ORDER_BY", "LIMIT", "SKIP"]:
                ast["children"].append({
                    "type": clause_type,
                    "content": clause_content,
                    "text": clause_content
                })
            else:
                ast["children"].append({
                    "type": clause_type,
                    "content": clause_content,
                    "text": clause_content
                })
        
        return ast
    
    def _extract_clauses(self, cypher: str) -> List[tuple]:
        """Extract main clauses from Cypher query."""
        import re
        clauses = []
        cypher = re.sub(r"\s+", " ", cypher)
        
        clause_patterns = [
            (r"(?i)\bMATCH\s+(.+?)(?=\b(?:WHERE|RETURN|WITH|CREATE|MERGE|DELETE|SET|REMOVE|ORDER\s+BY|LIMIT|SKIP|UNWIND|CALL|$))", "MATCH"),
            (r"(?i)\bWHERE\s+(.+?)(?=\b(?:RETURN|WITH|ORDER\s+BY|LIMIT|SKIP|$))", "WHERE"),
            (r"(?i)\bRETURN\s+(.+?)(?=\b(?:ORDER\s+BY|LIMIT|SKIP|$))", "RETURN"),
            (r"(?i)\bWITH\s+(.+?)(?=\b(?:WHERE|RETURN|MATCH|ORDER\s+BY|LIMIT|SKIP|$))", "WITH"),
            (r"(?i)\bCREATE\s+(.+?)(?=\b(?:RETURN|WHERE|$))", "CREATE"),
            (r"(?i)\bMERGE\s+(.+?)(?=\b(?:RETURN|WHERE|$))", "MERGE"),
            (r"(?i)\bORDER\s+BY\s+(.+?)(?=\b(?:LIMIT|SKIP|$))", "ORDER_BY"),
            (r"(?i)\bLIMIT\s+(\d+)", "LIMIT"),
            (r"(?i)\bSKIP\s+(\d+)", "SKIP"),
        ]
        
        for pattern, clause_type in clause_patterns:
            matches = re.finditer(pattern, cypher)
            for match in matches:
                clauses.append((clause_type, match.group(1).strip()))
        
        return clauses
    
    def _extract_patterns(self, match_content: str) -> List[Dict[str, Any]]:
        """Extract node and relationship patterns from MATCH clause."""
        import re
        patterns = []
        pattern_re = r"\(([^)]+)\)(?:\s*-\s*\[([^\]]+)\]\s*-\s*>)?\s*\(([^)]+)\)?"
        matches = re.finditer(pattern_re, match_content)
        
        for match in matches:
            left_node = match.group(1) if match.group(1) else None
            rel = match.group(2) if match.group(2) else None
            right_node = match.group(3) if match.group(3) else None
            
            if left_node:
                patterns.append({
                    "type": "NODE",
                    "content": left_node,
                    "labels": re.findall(r":(\w+)", left_node),
                    "text": left_node,
                })
            if rel:
                patterns.append({
                    "type": "RELATIONSHIP",
                    "content": rel,
                    "types": re.findall(r":(\w+)", rel),
                    "text": rel,
                })
            if right_node:
                patterns.append({
                    "type": "NODE",
                    "content": right_node,
                    "labels": re.findall(r":(\w+)", right_node),
                    "text": right_node,
                })
        
        return patterns
    
    def _extract_predicates(self, where_content: str) -> List[Dict[str, Any]]:
        """Extract predicates from WHERE clause."""
        import re
        predicates = []
        parts = re.split(r"\b(?:AND|OR)\b", where_content, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if part:
                predicates.append({
                    "type": "PREDICATE",
                    "content": part,
                    "operators": re.findall(
                        r"(>=|<=|!=|<>|>|<|=|STARTS\s+WITH|ENDS\s+WITH|CONTAINS|IN|IS\s+NULL|IS\s+NOT\s+NULL)",
                        part,
                        re.IGNORECASE
                    ),
                    "text": part,
                })
        return predicates
    
    def _extract_returns(self, return_content: str) -> List[str]:
        """Extract return items."""
        items = []
        parts = return_content.split(",")
        for part in parts:
            part = part.strip()
            if part:
                items.append(part)
        return items
    
    def _extract_with_items(self, with_content: str) -> List[str]:
        """Extract WITH items."""
        items = []
        parts = with_content.split(",")
        for part in parts:
            part = part.strip()
            if part:
                items.append(part)
        return items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        total = sum(self.stats.values())
        if total > 0:
            return {
                **self.stats,
                "total": total,
                "antlr_success_rate": self.stats["antlr_parser_success"] / total if total > 0 else 0.0,
            }
        return self.stats
