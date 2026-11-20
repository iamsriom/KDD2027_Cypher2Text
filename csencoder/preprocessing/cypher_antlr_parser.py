#!/usr/bin/env python3
"""
ANTLR-based Cypher AST Parser.

Uses ANTLR4 to parse Cypher queries into AST structure with half-open
character spans ``[start, end)`` relative to the original query string
(including leading whitespace). Falls back to an offset-aware regex parser
when ANTLR is unavailable or fails.

Regex fallback never invents hierarchical edges via substring matching; it
only attaches clause/pattern spans when ``re.Match`` positions are known.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _antlr_char_span(ctx) -> Tuple[Optional[int], Optional[int]]:
    """Half-open ``[start, end)`` from an ANTLR parse-tree node, or ``(None, None)``."""
    from antlr4.tree.Tree import TerminalNodeImpl

    if isinstance(ctx, TerminalNodeImpl):
        tok = getattr(ctx, "symbol", None)
        if tok is None:
            return None, None
        start = getattr(tok, "start", None)
        stop = getattr(tok, "stop", None)
        if start is None or stop is None or start < 0 or stop < 0:
            return None, None
        return int(start), int(stop) + 1

    start_tok = getattr(ctx, "start", None)
    stop_tok = getattr(ctx, "stop", None)
    if start_tok is None or stop_tok is None:
        return None, None
    start = getattr(start_tok, "start", None)
    stop = getattr(stop_tok, "stop", None)
    if start is None or stop is None or start < 0 or stop < 0:
        return None, None
    return int(start), int(stop) + 1


def _with_span(node: Dict[str, Any], ctx) -> Dict[str, Any]:
    start, end = _antlr_char_span(ctx)
    if start is not None and end is not None:
        node["start"] = start
        node["end"] = end
    return node


def _with_explicit_span(
    node: Dict[str, Any],
    start: Optional[int],
    end: Optional[int],
) -> Dict[str, Any]:
    if start is not None and end is not None and 0 <= start < end:
        node["start"] = int(start)
        node["end"] = int(end)
    return node


class ASTBuildingVisitor:
    """
    Visitor to build AST structure from ANTLR parse tree.
    Adapts to the actual rule names in the Cypher grammar.
    Attaches half-open character spans into ``original_text``.
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
            return _with_span(
                {
                    "type": "TERMINAL",
                    "text": ctx.getText() if hasattr(ctx, "getText") else str(ctx),
                    "children": [],
                    "patterns": [],
                    "predicates": [],
                },
                ctx,
            )
        
        # Get the class name and try to find a specific visit method
        class_name = ctx.__class__.__name__  # e.g., "OC_MatchContext"
        base_name = class_name.replace("Context", "")  # e.g., "OC_Match"
        method_name = f"visit{base_name}"  # e.g., "visitOC_Match"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(ctx)
        else:
            # Default: create a generic node
            return self._default_visit(ctx)
    
    def _children(self, ctx) -> List[Dict[str, Any]]:
        children: List[Dict[str, Any]] = []
        if hasattr(ctx, "children") and ctx.children is not None:
            for child in ctx.children:
                child_node = self.visit(child)
                if child_node:
                    children.append(child_node)
        return children

    def _default_visit(self, ctx) -> Dict[str, Any]:
        """Default visit for unknown node types."""
        from antlr4.tree.Tree import TerminalNodeImpl

        if isinstance(ctx, TerminalNodeImpl):
            return _with_span(
                {
                    "type": "TERMINAL",
                    "text": ctx.getText() if hasattr(ctx, "getText") else str(ctx),
                    "children": [],
                    "patterns": [],
                    "predicates": [],
                },
                ctx,
            )

        return _with_span(
            {
                "type": ctx.__class__.__name__.upper().replace("CONTEXT", ""),
                "text": ctx.getText() if hasattr(ctx, "getText") else str(ctx),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Cypher(self, ctx) -> Dict[str, Any]:
        """Visit root cypher rule."""
        return _with_span(
            {
                "type": "QUERY",
                "text": self.original_text,
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Statement(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "STATEMENT",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Query(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "QUERY",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_RegularQuery(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "QUERY",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_SingleQuery(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "QUERY",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Match(self, ctx) -> Dict[str, Any]:
        """Visit MATCH clause."""
        node = {
            "type": "MATCH",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }

        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                elif child_node.get("type") == "PREDICATE":
                    node["predicates"].append(child_node)
                elif child_node.get("type") == "WHERE":
                    if child_node.get("predicates"):
                        node["predicates"].extend(child_node["predicates"])
                    node["children"].append(child_node)
                else:
                    node["children"].append(child_node)

        if hasattr(ctx, "oC_Pattern") and ctx.oC_Pattern():
            pattern_node = self.visit(ctx.oC_Pattern())
            if pattern_node and pattern_node not in node["patterns"]:
                node["patterns"].append(pattern_node)

        if hasattr(ctx, "oC_Where") and ctx.oC_Where():
            where_node = self.visit(ctx.oC_Where())
            if where_node and where_node.get("predicates"):
                for pred in where_node["predicates"]:
                    if pred not in node["predicates"]:
                        node["predicates"].append(pred)

        return _with_span(node, ctx)

    def visitOC_Where(self, ctx) -> Dict[str, Any]:
        """Visit WHERE clause."""
        node = {
            "type": "WHERE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }

        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") == "PREDICATE":
                    node["predicates"].append(child_node)
                else:
                    node["children"].append(child_node)

        if hasattr(ctx, "oC_Expression") and ctx.oC_Expression():
            expr_ctx = ctx.oC_Expression()
            expr_node = {
                "type": "PREDICATE",
                "text": expr_ctx.getText(),
            }
            node["predicates"].append(_with_span(expr_node, expr_ctx))

        return _with_span(node, ctx)

    def visitOC_Return(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "RETURN",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_With(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "WITH",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Create(self, ctx) -> Dict[str, Any]:
        node = {
            "type": "CREATE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        return _with_span(node, ctx)

    def visitOC_Merge(self, ctx) -> Dict[str, Any]:
        node = {
            "type": "MERGE",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["PATTERN", "NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        return _with_span(node, ctx)

    def visitOC_Delete(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "DELETE",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Set(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "SET",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Remove(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "REMOVE",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Order(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "ORDER_BY",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Limit(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "LIMIT",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Skip(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "SKIP",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Unwind(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "UNWIND",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Call(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "CALL",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_Pattern(self, ctx) -> Dict[str, Any]:
        node = {
            "type": "PATTERN",
            "text": ctx.getText(),
            "children": [],
            "patterns": [],
            "predicates": [],
        }
        for child in (ctx.children or []):
            child_node = self.visit(child)
            if child_node:
                if child_node.get("type") in ["NODE", "RELATIONSHIP"]:
                    node["patterns"].append(child_node)
                else:
                    node["children"].append(child_node)
        return _with_span(node, ctx)

    def visitOC_NodePattern(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "NODE",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )

    def visitOC_RelationshipPattern(self, ctx) -> Dict[str, Any]:
        return _with_span(
            {
                "type": "RELATIONSHIP",
                "text": ctx.getText(),
                "children": self._children(ctx),
                "patterns": [],
                "predicates": [],
            },
            ctx,
        )


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

        Offsets are relative to the original ``cypher`` string (including any
        leading/trailing whitespace). Do not strip before parsing; the
        tokenizer uses the same string for ``match.start()`` / ``match.end()``.

        Strategy:
        1. Try ANTLR parser first (if available)
        2. If it fails, fall back to offset-aware regex parser
        """
        import contextlib
        from io import StringIO

        if cypher is None or not str(cypher).strip():
            return {
                "type": "QUERY",
                "text": cypher or "",
                "children": [],
                "patterns": [],
                "predicates": [],
                "parser": "empty",
            }

        # Try ANTLR parser first if available
        if self.antlr_available:
            try:
                with contextlib.redirect_stderr(StringIO()):
                    result = self._parse_via_antlr(cypher)
                    if result and isinstance(result, dict) and result.get("type"):
                        result.setdefault("parser", "antlr")
                        self.stats["antlr_parser_success"] += 1
                        return result
                    raise ValueError("ANTLR parser returned invalid result")
            except Exception:
                self.stats["antlr_parser_failed"] += 1

        self.stats["regex_parser_used"] += 1
        result = self._parse_regex(cypher)
        result.setdefault("parser", "regex")
        return result
    
    @staticmethod
    def _shift_spans(node: Dict[str, Any], delta: int) -> Dict[str, Any]:
        """Add ``delta`` to every ``start``/``end`` in an AST tree."""
        if delta == 0 or not isinstance(node, dict):
            return node
        if "start" in node and node["start"] is not None:
            node["start"] = int(node["start"]) + delta
        if "end" in node and node["end"] is not None:
            node["end"] = int(node["end"]) + delta
        for key in ("children", "patterns", "predicates"):
            for child in node.get(key) or []:
                if isinstance(child, dict):
                    AntlrCypherASTParser._shift_spans(child, delta)
        return node

    def _parse_via_antlr(self, cypher: str) -> Dict[str, Any]:
        """Parse Cypher using ANTLR4.

        Leading/trailing whitespace is stripped only for the ANTLR input stream
        (many grammars reject it). Returned spans are shifted back so they refer
        to the original ``cypher`` string used by the tokenizer.
        """
        if not ANTLR_AVAILABLE:
            raise RuntimeError("ANTLR parser not available")

        import contextlib
        import io

        leading = len(cypher) - len(cypher.lstrip())
        trailing = len(cypher) - len(cypher.rstrip())
        core = cypher[leading : len(cypher) - trailing if trailing else len(cypher)]
        if not core:
            raise ValueError("empty Cypher after whitespace trim")

        # Suppress ANTLR error messages by redirecting stderr
        error_buffer = io.StringIO()

        try:
            with contextlib.redirect_stderr(error_buffer):
                input_stream = InputStream(core)
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
                
                # Build AST using visitor on the trimmed core, then shift spans.
                visitor = ASTBuildingVisitor(core)
                ast = visitor.visit(tree)

                if not isinstance(ast, dict):
                    ast = {
                        "type": "QUERY",
                        "text": cypher,
                        "children": [],
                        "patterns": [],
                        "predicates": [],
                    }

                ast.setdefault("type", "QUERY")
                ast["text"] = cypher
                ast.setdefault("children", [])
                ast.setdefault("patterns", [])
                ast.setdefault("predicates", [])
                self._shift_spans(ast, leading)
                if "start" not in ast:
                    ast["start"] = leading
                    ast["end"] = leading + len(core)
                return ast
        except Exception:
            raise

    def _parse_regex(self, cypher: str) -> Dict[str, Any]:
        """
        Offset-aware regex fallback.

        Clause/pattern nodes receive half-open spans from ``re.Match`` positions
        on the original query. When a sub-span cannot be determined authoritatively,
        that node omits ``start``/``end`` so the graph builder skips hierarchical
        edges for it (sequence edges only) instead of substring matching.
        """
        import re

        ast = _with_explicit_span(
            {
                "type": "QUERY",
                "children": [],
                "patterns": [],
                "predicates": [],
                "text": cypher,
            },
            0,
            len(cypher),
        )

        for clause_type, content, start, end in self._extract_clauses(cypher):
            if clause_type == "MATCH":
                patterns = self._extract_patterns(content, base_offset=start)
                ast["patterns"].extend(patterns)
                ast["children"].append(
                    _with_explicit_span(
                        {
                            "type": "MATCH",
                            "content": content,
                            "patterns": patterns,
                            "text": content,
                            "children": [],
                            "predicates": [],
                        },
                        start,
                        end,
                    )
                )
            elif clause_type == "WHERE":
                predicates = self._extract_predicates(content, base_offset=start)
                ast["predicates"].extend(predicates)
                ast["children"].append(
                    _with_explicit_span(
                        {
                            "type": "WHERE",
                            "content": content,
                            "predicates": predicates,
                            "text": content,
                            "children": [],
                            "patterns": [],
                        },
                        start,
                        end,
                    )
                )
            elif clause_type == "RETURN":
                returns = self._extract_returns(content)
                ast["children"].append(
                    _with_explicit_span(
                        {
                            "type": "RETURN",
                            "content": content,
                            "returns": returns,
                            "text": content,
                            "children": [],
                            "patterns": [],
                            "predicates": [],
                        },
                        start,
                        end,
                    )
                )
            elif clause_type == "WITH":
                with_items = self._extract_with_items(content)
                ast["children"].append(
                    _with_explicit_span(
                        {
                            "type": "WITH",
                            "content": content,
                            "items": with_items,
                            "text": content,
                            "children": [],
                            "patterns": [],
                            "predicates": [],
                        },
                        start,
                        end,
                    )
                )
            else:
                ast["children"].append(
                    _with_explicit_span(
                        {
                            "type": clause_type,
                            "content": content,
                            "text": content,
                            "children": [],
                            "patterns": [],
                            "predicates": [],
                        },
                        start,
                        end,
                    )
                )

        return ast

    def _extract_clauses(self, cypher: str) -> List[tuple]:
        """Extract main clauses with absolute ``[start, end)`` spans in ``cypher``."""
        import re

        clauses = []
        # Operate on the original string so match offsets stay aligned with the tokenizer.
        clause_patterns = [
            (
                r"(?is)\bMATCH\s+(.+?)(?=\b(?:WHERE|RETURN|WITH|CREATE|MERGE|DELETE|SET|REMOVE|ORDER\s+BY|LIMIT|SKIP|UNWIND|CALL)\b|$)",
                "MATCH",
            ),
            (
                r"(?is)\bWHERE\s+(.+?)(?=\b(?:RETURN|WITH|ORDER\s+BY|LIMIT|SKIP)\b|$)",
                "WHERE",
            ),
            (
                r"(?is)\bRETURN\s+(.+?)(?=\b(?:ORDER\s+BY|LIMIT|SKIP)\b|$)",
                "RETURN",
            ),
            (
                r"(?is)\bWITH\s+(.+?)(?=\b(?:WHERE|RETURN|MATCH|ORDER\s+BY|LIMIT|SKIP)\b|$)",
                "WITH",
            ),
            (r"(?is)\bCREATE\s+(.+?)(?=\b(?:RETURN|WHERE)\b|$)", "CREATE"),
            (r"(?is)\bMERGE\s+(.+?)(?=\b(?:RETURN|WHERE)\b|$)", "MERGE"),
            (r"(?is)\bORDER\s+BY\s+(.+?)(?=\b(?:LIMIT|SKIP)\b|$)", "ORDER_BY"),
            (r"(?i)\bLIMIT\s+(\d+)", "LIMIT"),
            (r"(?i)\bSKIP\s+(\d+)", "SKIP"),
        ]

        for pattern, clause_type in clause_patterns:
            for match in re.finditer(pattern, cypher):
                # Span covers the captured content (group 1), not the keyword itself.
                content = match.group(1)
                start = match.start(1)
                end = match.end(1)
                clauses.append((clause_type, content, start, end))

        return clauses

    def _extract_patterns(
        self, match_content: str, *, base_offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Extract node/relationship patterns with absolute character spans."""
        import re

        patterns: List[Dict[str, Any]] = []
        # Capture full node ``(...)`` and relationship ``[...]`` spans for offsets.
        node_re = re.compile(r"\(([^)]*)\)")
        rel_re = re.compile(r"\[([^\]]*)\]")

        for match in node_re.finditer(match_content):
            inner = match.group(1)
            patterns.append(
                _with_explicit_span(
                    {
                        "type": "NODE",
                        "content": inner,
                        "labels": re.findall(r":(\w+)", inner),
                        "text": match.group(0),
                        "children": [],
                        "patterns": [],
                        "predicates": [],
                    },
                    base_offset + match.start(),
                    base_offset + match.end(),
                )
            )

        for match in rel_re.finditer(match_content):
            inner = match.group(1)
            patterns.append(
                _with_explicit_span(
                    {
                        "type": "RELATIONSHIP",
                        "content": inner,
                        "types": re.findall(r":(\w+)", inner),
                        "text": match.group(0),
                        "children": [],
                        "patterns": [],
                        "predicates": [],
                    },
                    base_offset + match.start(),
                    base_offset + match.end(),
                )
            )

        return patterns

    def _extract_predicates(
        self, where_content: str, *, base_offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Extract predicates with spans relative to the original query."""
        import re

        predicates: List[Dict[str, Any]] = []
        # Split on AND/OR while retaining match positions of each segment.
        parts = re.split(r"(\b(?:AND|OR)\b)", where_content, flags=re.IGNORECASE)
        cursor = 0
        for part in parts:
            if not part:
                continue
            part_start = where_content.find(part, cursor)
            if part_start < 0:
                part_start = cursor
            part_end = part_start + len(part)
            cursor = part_end
            if re.fullmatch(r"(?i)\s*(?:AND|OR)\s*", part):
                continue
            stripped = part.strip()
            if not stripped:
                continue
            # Tighten span to non-whitespace content inside this segment.
            leading = len(part) - len(part.lstrip())
            trailing = len(part) - len(part.rstrip())
            abs_start = base_offset + part_start + leading
            abs_end = base_offset + part_end - trailing
            predicates.append(
                _with_explicit_span(
                    {
                        "type": "PREDICATE",
                        "content": stripped,
                        "operators": re.findall(
                            r"(>=|<=|!=|<>|>|<|=|STARTS\s+WITH|ENDS\s+WITH|CONTAINS|IN|IS\s+NULL|IS\s+NOT\s+NULL)",
                            stripped,
                            re.IGNORECASE,
                        ),
                        "text": stripped,
                        "children": [],
                        "patterns": [],
                        "predicates": [],
                    },
                    abs_start,
                    abs_end,
                )
            )
        return predicates

    def _extract_returns(self, return_content: str) -> List[str]:
        """Extract return items."""
        return [part.strip() for part in return_content.split(",") if part.strip()]

    def _extract_with_items(self, with_content: str) -> List[str]:
        """Extract WITH items."""
        return [part.strip() for part in with_content.split(",") if part.strip()]
    
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
