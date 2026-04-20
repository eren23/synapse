#!/usr/bin/env python3
"""Python AST tokenizer with FNV-1a hash — matches synapse-code-tokenizer (Rust).

This is a deterministic variant of the training tokenizer. Uses FNV-1a instead
of Python's PYTHONHASHSEED-randomized hash() so tokens are stable across
processes AND match the Rust port byte-for-byte.

Usage:
    from ast_tokenizer_fnv import tokenize_fnv
    tokens = tokenize_fnv("def foo(): return 1", max_len=512)

Note: this is for the BFS variant only (no OPEN/CLOSE brackets).
"""

from __future__ import annotations

import ast

import numpy as np

# ── Vocab constants (must match synapse-code-tokenizer) ────────────
IDENT_OFFSET = 100
IDENT_BUCKETS = 512
PAD = 612
BOS = 613
EOS = 614
UNK = 615
PARSE_ERROR = 616
DEPTH_OFFSET = 617
MAX_DEPTH = 15
OP_OFFSET = 633

OP_NAMES = [
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "MatMult",
    "Invert", "Not", "UAdd", "USub",
    "And", "Or",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
]
OP_MAP = {name: OP_OFFSET + i for i, name in enumerate(OP_NAMES)}

# ── Node type IDs (must match synapse-code-tokenizer exactly) ──────
# Sorted list of concrete AST node types from Python 3.12.
_NODE_NAMES = [
    "Add","And","AnnAssign","Assert","Assign","AsyncFor","AsyncFunctionDef",
    "AsyncWith","Attribute","AugAssign","AugLoad","AugStore","Await","BinOp",
    "BitAnd","BitOr","BitXor","BoolOp","Break","Bytes","Call","ClassDef",
    "Compare","Constant","Continue","Del","Delete","Dict","DictComp","Div",
    "Ellipsis","Eq","ExceptHandler","Expr","Expression","ExtSlice","FloorDiv",
    "For","FormattedValue","FunctionDef","FunctionType","GeneratorExp","Global",
    "Gt","GtE","If","IfExp","Import","ImportFrom","In","Index","Interactive",
    "Invert","Is","IsNot","JoinedStr","LShift","Lambda","List","ListComp","Load",
    "Lt","LtE","MatMult","Mod","Module","Mult","Name","NameConstant","NamedExpr",
    "Nonlocal","Not","NotEq","NotIn","Num","Or","Param","Pass","Pow","RShift",
    "Raise","Return","Set","SetComp","Slice","Starred","Store","Str","Sub",
    "Subscript","Suite","Try","Tuple","TypeIgnore","UAdd","USub","UnaryOp",
    "While","With","Yield","YieldFrom","slice",
]
NODE_TYPE_MAP = {n: i for i, n in enumerate(_NODE_NAMES)}


def fnv1a_32(s: str) -> int:
    h = 0x811c9dc5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def ident_token(name: str) -> int:
    return IDENT_OFFSET + (fnv1a_32(name) % IDENT_BUCKETS)


def _depths(tree: ast.AST) -> dict[int, int]:
    depths = {id(tree): 0}
    def visit(n, d):
        for c in ast.iter_child_nodes(n):
            depths[id(c)] = d + 1
            visit(c, d + 1)
    visit(tree, 0)
    return depths


def tokenize_fnv(source: str, max_len: int = 512) -> np.ndarray:
    if not source or not source.strip():
        toks = [BOS, EOS] + [PAD] * (max_len - 2)
        return np.array(toks[:max_len], dtype=np.uint16)

    try:
        clean = source.replace("\x00", "")
        tree = ast.parse(clean)
    except (SyntaxError, ValueError, RecursionError):
        toks = [BOS, PARSE_ERROR, EOS] + [PAD] * (max_len - 3)
        return np.array(toks[:max_len], dtype=np.uint16)

    depths = _depths(tree)
    tokens = [BOS]
    for node in ast.walk(tree):
        node_name = type(node).__name__
        tokens.append(NODE_TYPE_MAP.get(node_name, UNK))
        tokens.append(DEPTH_OFFSET + min(depths.get(id(node), 0), MAX_DEPTH))

        if isinstance(node, ast.Name):
            tokens.append(ident_token(node.id))
        elif isinstance(node, ast.Attribute):
            tokens.append(ident_token(node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tokens.append(ident_token(node.name))
        elif isinstance(node, ast.ClassDef):
            tokens.append(ident_token(node.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.append(ident_token(node.module))

        if isinstance(node, ast.BinOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.UnaryOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.BoolOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.Compare) and node.ops:
            tokens.append(OP_MAP.get(type(node.ops[0]).__name__, UNK))
        elif isinstance(node, ast.AugAssign):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))

        if isinstance(node, ast.Constant):
            val_type = type(node.value).__name__
            tokens.append(ident_token(f"__const_{val_type}__"))

        if len(tokens) >= max_len - 1:
            break

    tokens.append(EOS)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    tokens += [PAD] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.uint16)


if __name__ == "__main__":
    import sys
    src = sys.stdin.read() if not sys.stdin.isatty() else "def f(x):\n    return x + 1\n"
    toks = tokenize_fnv(src, max_len=32)
    print(list(toks))
