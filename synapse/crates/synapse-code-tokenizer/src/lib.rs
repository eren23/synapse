//! Python AST tokenizer for Code WM — matches `ast_tokenizer.ast_tokenize()`
//! from the training tap, but implemented entirely in Rust via rustpython-parser.
//!
//! This removes the Python runtime dependency from Code WM inference and
//! enables tokenization in WASM / ESP32 / any target Rust compiles to.
//!
//! ## Vocabulary layout (matches training, 662 vocab size)
//!
//! | Range      | Purpose                             |
//! |------------|-------------------------------------|
//! | 0..N       | Concrete AST node types (sorted)    |
//! | 100..612   | Identifier hash buckets (FNV-1a % 512) |
//! | 612        | PAD                                 |
//! | 613        | BOS                                 |
//! | 614        | EOS                                 |
//! | 615        | UNK                                 |
//! | 616        | PARSE_ERROR                         |
//! | 617..633   | Depth markers (DEPTH_0..DEPTH_15)   |
//! | 633..662   | Operators (Add..NotIn, 29)          |
//!
//! ## Hash choice
//!
//! Python's training pipeline used Python's built-in `hash()` which is
//! PYTHONHASHSEED-randomized per-process. The model learned to treat
//! identifier bucket assignments as noise, so any deterministic hash works
//! at inference. We use **FNV-1a** (simple, deterministic, platform-agnostic).
//!
//! Use the companion Python helper `ast_tokenizer_fnv.py` to produce
//! tokens that match byte-for-byte.
//!
//! ## Coverage (WIP)
//!
//! This initial port covers the most common ~15 node types found in typical
//! Python code. Uncommon/new nodes fall through to UNK.
//!
//! **Known divergence from reference**: Python's `ast.walk()` treats certain
//! attributes as tree children (e.g. `BinOp.op` as an `Add`/`Sub`/... node,
//! `Name.ctx` as a `Load`/`Store` node, `arguments` and `arg` as nodes).
//! Our walker currently skips these, so token counts differ. Matching exactly
//! requires a recursive-descent walker mirroring Python's `iter_child_nodes`.
//!
//! Validate against `scripts/ast_tokenizer_fnv.py` for reference output.
//! See the module-level TODOs for remaining node types to cover.

use std::collections::VecDeque;

use rustpython_parser::{ast, Parse};

// ── Vocabulary constants (must match ast_tokenizer.py) ─────────────

pub const IDENT_OFFSET: u16 = 100;
pub const IDENT_BUCKETS: u16 = 512;
pub const PAD: u16 = 612;
pub const BOS: u16 = 613;
pub const EOS: u16 = 614;
pub const UNK: u16 = 615;
pub const PARSE_ERROR: u16 = 616;
pub const DEPTH_OFFSET: u16 = 617;
pub const MAX_DEPTH: u16 = 15;
pub const OP_OFFSET: u16 = 633;

/// Ordered list of operator names (indices into OP_OFFSET..).
/// Order matches `_OP_NAMES` in ast_tokenizer.py.
pub const OP_NAMES: &[&str] = &[
    // Binary
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "MatMult",
    // Unary
    "Invert", "Not", "UAdd", "USub",
    // Boolean
    "And", "Or",
    // Comparison
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
];

pub fn op_token(name: &str) -> u16 {
    for (i, op) in OP_NAMES.iter().enumerate() {
        if *op == name {
            return OP_OFFSET + i as u16;
        }
    }
    UNK
}

// ── AST node type IDs ──────────────────────────────────────────────
//
// Sorted order matches Python's `sorted(concrete_ast_types)` as of Python 3.12.
// If your Python version differs, regenerate this list with:
//   python -c "import ast; ..." (see scripts/gen_node_type_ids.py)

/// Map from Python AST class name → integer token ID.
/// Range: 0..102 (102 concrete types in Python 3.12).
pub fn node_type_id(name: &str) -> Option<u16> {
    // Sorted list matching Python 3.12's concrete types.
    match name {
        "Add" => Some(0), "And" => Some(1), "AnnAssign" => Some(2), "Assert" => Some(3),
        "Assign" => Some(4), "AsyncFor" => Some(5), "AsyncFunctionDef" => Some(6),
        "AsyncWith" => Some(7), "Attribute" => Some(8), "AugAssign" => Some(9),
        "AugLoad" => Some(10), "AugStore" => Some(11), "Await" => Some(12),
        "BinOp" => Some(13), "BitAnd" => Some(14), "BitOr" => Some(15),
        "BitXor" => Some(16), "BoolOp" => Some(17), "Break" => Some(18),
        "Bytes" => Some(19), "Call" => Some(20), "ClassDef" => Some(21),
        "Compare" => Some(22), "Constant" => Some(23), "Continue" => Some(24),
        "Del" => Some(25), "Delete" => Some(26), "Dict" => Some(27),
        "DictComp" => Some(28), "Div" => Some(29), "Ellipsis" => Some(30),
        "Eq" => Some(31), "ExceptHandler" => Some(32), "Expr" => Some(33),
        "Expression" => Some(34), "ExtSlice" => Some(35), "FloorDiv" => Some(36),
        "For" => Some(37), "FormattedValue" => Some(38), "FunctionDef" => Some(39),
        "FunctionType" => Some(40), "GeneratorExp" => Some(41), "Global" => Some(42),
        "Gt" => Some(43), "GtE" => Some(44), "If" => Some(45),
        "IfExp" => Some(46), "Import" => Some(47), "ImportFrom" => Some(48),
        "In" => Some(49), "Index" => Some(50), "Interactive" => Some(51),
        "Invert" => Some(52), "Is" => Some(53), "IsNot" => Some(54),
        "JoinedStr" => Some(55), "LShift" => Some(56), "Lambda" => Some(57),
        "List" => Some(58), "ListComp" => Some(59), "Load" => Some(60),
        "Lt" => Some(61), "LtE" => Some(62), "MatMult" => Some(63),
        "Mod" => Some(64), "Module" => Some(65), "Mult" => Some(66),
        "Name" => Some(67), "NameConstant" => Some(68), "NamedExpr" => Some(69),
        "Nonlocal" => Some(70), "Not" => Some(71), "NotEq" => Some(72),
        "NotIn" => Some(73), "Num" => Some(74), "Or" => Some(75),
        "Param" => Some(76), "Pass" => Some(77), "Pow" => Some(78),
        "RShift" => Some(79), "Raise" => Some(80), "Return" => Some(81),
        "Set" => Some(82), "SetComp" => Some(83), "Slice" => Some(84),
        "Starred" => Some(85), "Store" => Some(86), "Str" => Some(87),
        "Sub" => Some(88), "Subscript" => Some(89), "Suite" => Some(90),
        "Try" => Some(91), "Tuple" => Some(92), "TypeIgnore" => Some(93),
        "UAdd" => Some(94), "USub" => Some(95), "UnaryOp" => Some(96),
        "While" => Some(97), "With" => Some(98), "Yield" => Some(99),
        "YieldFrom" => Some(100), "slice" => Some(101),
        _ => None,
    }
}

// ── FNV-1a 32-bit hash (deterministic identifier → bucket) ─────────

pub fn fnv1a_32(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

pub fn ident_token(name: &str) -> u16 {
    IDENT_OFFSET + (fnv1a_32(name) % IDENT_BUCKETS as u32) as u16
}

pub fn depth_token(depth: usize) -> u16 {
    DEPTH_OFFSET + depth.min(MAX_DEPTH as usize) as u16
}

// ── Traversal: walk the AST in BFS order, producing tokens ─────────

/// An AST node in our traversal — a wrapper that knows its Python class name,
/// its children, and the relevant sub-data (identifier names, operators, etc.).
#[derive(Debug)]
struct NodeRef<'a> {
    name: &'static str,
    ident: Option<&'a str>,      // Name.id, FunctionDef.name, Attribute.attr, etc.
    op: Option<&'static str>,    // For BinOp, UnaryOp, BoolOp, Compare
    const_type: Option<&'static str>, // For Constant: "int", "str", "float", "bool", etc.
    children: Vec<Box<NodeRef<'a>>>,
}

/// Tokenize a Python source file. Returns `max_len` tokens, padded with PAD.
pub fn tokenize(source: &str, max_len: usize) -> Vec<u16> {
    let trimmed: String = source.chars().filter(|c| *c != '\0').collect();
    if trimmed.trim().is_empty() {
        let mut out = vec![BOS, EOS];
        out.resize(max_len, PAD);
        return out;
    }

    let suite = match ast::Suite::parse(&trimmed, "<src>") {
        Ok(s) => s,
        Err(_) => {
            let mut out = vec![BOS, PARSE_ERROR, EOS];
            out.resize(max_len, PAD);
            return out;
        }
    };

    // Build our intermediate tree (NodeRef) from the parsed suite.
    let module = NodeRef {
        name: "Module",
        ident: None,
        op: None,
        const_type: None,
        children: suite.iter().map(|s| Box::new(stmt_to_node(s))).collect(),
    };

    // BFS walk emitting tokens.
    let mut tokens: Vec<u16> = Vec::with_capacity(max_len);
    tokens.push(BOS);
    let mut queue: VecDeque<(&NodeRef, usize)> = VecDeque::new();
    queue.push_back((&module, 0));
    while let Some((node, depth)) = queue.pop_front() {
        // Node type
        tokens.push(node_type_id(node.name).unwrap_or(UNK));
        // Depth marker
        tokens.push(depth_token(depth));
        // Identifier if present
        if let Some(id) = node.ident {
            tokens.push(ident_token(id));
        }
        // Operator
        if let Some(op) = node.op {
            tokens.push(op_token(op));
        }
        // Constant type hint
        if let Some(ct) = node.const_type {
            tokens.push(ident_token(&format!("__const_{ct}__")));
        }
        // Enqueue children
        for child in &node.children {
            queue.push_back((child, depth + 1));
        }
        if tokens.len() >= max_len.saturating_sub(1) {
            break;
        }
    }
    tokens.push(EOS);
    tokens.truncate(max_len);
    tokens.resize(max_len, PAD);
    tokens
}

// ── AST → NodeRef conversion (initial subset) ──────────────────────
// Full coverage requires handling all 100+ AST variants. This initial
// version covers the common cases needed for typical Python code.

fn stmt_to_node(stmt: &ast::Stmt) -> NodeRef {
    match stmt {
        ast::Stmt::FunctionDef(s) => NodeRef {
            name: "FunctionDef",
            ident: Some(s.name.as_str()),
            op: None, const_type: None,
            children: s.body.iter().map(|c| Box::new(stmt_to_node(c))).collect(),
        },
        ast::Stmt::ClassDef(s) => NodeRef {
            name: "ClassDef",
            ident: Some(s.name.as_str()),
            op: None, const_type: None,
            children: s.body.iter().map(|c| Box::new(stmt_to_node(c))).collect(),
        },
        ast::Stmt::Return(s) => NodeRef {
            name: "Return", ident: None, op: None, const_type: None,
            children: s.value.as_ref().map(|e| vec![Box::new(expr_to_node(e))]).unwrap_or_default(),
        },
        ast::Stmt::Assign(s) => {
            let mut children: Vec<Box<NodeRef>> = s.targets.iter().map(|t| Box::new(expr_to_node(t))).collect();
            children.push(Box::new(expr_to_node(&s.value)));
            NodeRef { name: "Assign", ident: None, op: None, const_type: None, children }
        }
        ast::Stmt::AugAssign(s) => NodeRef {
            name: "AugAssign", ident: None,
            op: Some(operator_name(&s.op)), const_type: None,
            children: vec![Box::new(expr_to_node(&s.target)), Box::new(expr_to_node(&s.value))],
        },
        ast::Stmt::If(s) => {
            let mut children: Vec<Box<NodeRef>> = vec![Box::new(expr_to_node(&s.test))];
            for b in &s.body { children.push(Box::new(stmt_to_node(b))); }
            for b in &s.orelse { children.push(Box::new(stmt_to_node(b))); }
            NodeRef { name: "If", ident: None, op: None, const_type: None, children }
        }
        ast::Stmt::For(s) => {
            let mut children: Vec<Box<NodeRef>> = vec![
                Box::new(expr_to_node(&s.target)),
                Box::new(expr_to_node(&s.iter)),
            ];
            for b in &s.body { children.push(Box::new(stmt_to_node(b))); }
            for b in &s.orelse { children.push(Box::new(stmt_to_node(b))); }
            NodeRef { name: "For", ident: None, op: None, const_type: None, children }
        }
        ast::Stmt::While(s) => {
            let mut children: Vec<Box<NodeRef>> = vec![Box::new(expr_to_node(&s.test))];
            for b in &s.body { children.push(Box::new(stmt_to_node(b))); }
            for b in &s.orelse { children.push(Box::new(stmt_to_node(b))); }
            NodeRef { name: "While", ident: None, op: None, const_type: None, children }
        }
        ast::Stmt::Expr(s) => NodeRef {
            name: "Expr", ident: None, op: None, const_type: None,
            children: vec![Box::new(expr_to_node(&s.value))],
        },
        ast::Stmt::Pass(_) => leaf("Pass"),
        ast::Stmt::Break(_) => leaf("Break"),
        ast::Stmt::Continue(_) => leaf("Continue"),
        ast::Stmt::Import(_) => leaf("Import"),
        ast::Stmt::ImportFrom(s) => NodeRef {
            name: "ImportFrom",
            ident: s.module.as_ref().map(|m| m.as_str()),
            op: None, const_type: None, children: vec![],
        },
        // TODO: AsyncFunctionDef, AsyncFor, AsyncWith, With, Raise, Try, TryStar,
        // Delete, AnnAssign, Assert, Global, Nonlocal, Match, TypeAlias
        _ => leaf("UNK"),  // Falls through to UNK token
    }
}

fn expr_to_node(expr: &ast::Expr) -> NodeRef {
    match expr {
        ast::Expr::Name(e) => NodeRef {
            name: "Name", ident: Some(e.id.as_str()), op: None, const_type: None, children: vec![],
        },
        ast::Expr::Constant(e) => {
            let ct = const_type_name(&e.value);
            NodeRef { name: "Constant", ident: None, op: None, const_type: Some(ct), children: vec![] }
        }
        ast::Expr::BinOp(e) => NodeRef {
            name: "BinOp", ident: None,
            op: Some(operator_name(&e.op)), const_type: None,
            children: vec![Box::new(expr_to_node(&e.left)), Box::new(expr_to_node(&e.right))],
        },
        ast::Expr::UnaryOp(e) => NodeRef {
            name: "UnaryOp", ident: None,
            op: Some(unaryop_name(&e.op)), const_type: None,
            children: vec![Box::new(expr_to_node(&e.operand))],
        },
        ast::Expr::BoolOp(e) => NodeRef {
            name: "BoolOp", ident: None,
            op: Some(boolop_name(&e.op)), const_type: None,
            children: e.values.iter().map(|v| Box::new(expr_to_node(v))).collect(),
        },
        ast::Expr::Compare(e) => NodeRef {
            name: "Compare", ident: None,
            op: e.ops.first().map(cmpop_name), const_type: None,
            children: std::iter::once(Box::new(expr_to_node(&e.left)))
                .chain(e.comparators.iter().map(|c| Box::new(expr_to_node(c))))
                .collect(),
        },
        ast::Expr::Call(e) => NodeRef {
            name: "Call", ident: None, op: None, const_type: None,
            children: std::iter::once(Box::new(expr_to_node(&e.func)))
                .chain(e.args.iter().map(|a| Box::new(expr_to_node(a))))
                .collect(),
        },
        ast::Expr::Attribute(e) => NodeRef {
            name: "Attribute", ident: Some(e.attr.as_str()), op: None, const_type: None,
            children: vec![Box::new(expr_to_node(&e.value))],
        },
        ast::Expr::Subscript(e) => NodeRef {
            name: "Subscript", ident: None, op: None, const_type: None,
            children: vec![Box::new(expr_to_node(&e.value)), Box::new(expr_to_node(&e.slice))],
        },
        ast::Expr::List(e) => NodeRef {
            name: "List", ident: None, op: None, const_type: None,
            children: e.elts.iter().map(|x| Box::new(expr_to_node(x))).collect(),
        },
        ast::Expr::Tuple(e) => NodeRef {
            name: "Tuple", ident: None, op: None, const_type: None,
            children: e.elts.iter().map(|x| Box::new(expr_to_node(x))).collect(),
        },
        ast::Expr::Dict(e) => {
            let mut children: Vec<Box<NodeRef>> = Vec::new();
            for k in e.keys.iter().flatten() { children.push(Box::new(expr_to_node(k))); }
            for v in &e.values { children.push(Box::new(expr_to_node(v))); }
            NodeRef { name: "Dict", ident: None, op: None, const_type: None, children }
        }
        // TODO: Lambda, IfExp, ListComp, DictComp, SetComp, GeneratorExp,
        // Starred, Await, Yield, YieldFrom, JoinedStr, FormattedValue,
        // Slice, Set, NamedExpr
        _ => leaf("UNK"),
    }
}

fn leaf(name: &'static str) -> NodeRef<'static> {
    NodeRef { name, ident: None, op: None, const_type: None, children: vec![] }
}

fn operator_name(op: &ast::Operator) -> &'static str {
    use ast::Operator::*;
    match op {
        Add => "Add", Sub => "Sub", Mult => "Mult", MatMult => "MatMult", Div => "Div",
        Mod => "Mod", Pow => "Pow", LShift => "LShift", RShift => "RShift",
        BitOr => "BitOr", BitXor => "BitXor", BitAnd => "BitAnd", FloorDiv => "FloorDiv",
    }
}

fn unaryop_name(op: &ast::UnaryOp) -> &'static str {
    use ast::UnaryOp::*;
    match op { Invert => "Invert", Not => "Not", UAdd => "UAdd", USub => "USub" }
}

fn boolop_name(op: &ast::BoolOp) -> &'static str {
    use ast::BoolOp::*;
    match op { And => "And", Or => "Or" }
}

fn cmpop_name(op: &ast::CmpOp) -> &'static str {
    use ast::CmpOp::*;
    match op {
        Eq => "Eq", NotEq => "NotEq", Lt => "Lt", LtE => "LtE",
        Gt => "Gt", GtE => "GtE", Is => "Is", IsNot => "IsNot", In => "In", NotIn => "NotIn",
    }
}

fn const_type_name(c: &ast::Constant) -> &'static str {
    use ast::Constant::*;
    match c {
        None => "NoneType",
        Bool(_) => "bool",
        Str(_) => "str",
        Bytes(_) => "bytes",
        Int(_) => "int",
        Tuple(_) => "tuple",
        Float(_) => "float",
        Complex { .. } => "complex",
        Ellipsis => "ellipsis",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_deterministic() {
        assert_eq!(fnv1a_32("hello"), fnv1a_32("hello"));
        assert_ne!(fnv1a_32("hello"), fnv1a_32("world"));
    }

    #[test]
    fn op_tokens_in_range() {
        assert_eq!(op_token("Add"), OP_OFFSET);
        assert_eq!(op_token("NotIn"), OP_OFFSET + 28);
        assert_eq!(op_token("Unknown"), UNK);
    }

    #[test]
    fn simple_function_tokenizes() {
        let src = "def f(x):\n    return x + 1\n";
        let toks = tokenize(src, 64);
        assert_eq!(toks[0], BOS);
        // First real node is Module (at depth 0), then FunctionDef (depth 1)
        assert_eq!(toks[1], node_type_id("Module").unwrap());
        // Last useful token before padding should be EOS
        assert!(toks.contains(&EOS));
    }

    #[test]
    fn empty_source() {
        let toks = tokenize("", 16);
        assert_eq!(toks[0], BOS);
        assert_eq!(toks[1], EOS);
        assert_eq!(toks[2], PAD);
    }

    #[test]
    fn parse_error_on_garbage() {
        let toks = tokenize("def f(  $$# broken", 16);
        assert_eq!(toks[0], BOS);
        assert_eq!(toks[1], PARSE_ERROR);
        assert_eq!(toks[2], EOS);
    }
}
