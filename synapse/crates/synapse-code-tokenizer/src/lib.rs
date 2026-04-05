//! Python AST tokenizer for Code WM — matches `ast_tokenizer.ast_tokenize()`
//! from the training tap, implemented in Rust via rustpython-parser.
//!
//! This removes the Python runtime dependency from Code WM inference and
//! enables tokenization on any target Rust compiles to (WASM, ESP32, native).
//!
//! ## Vocabulary layout (matches training, 662 vocab size)
//!
//! | Range      | Purpose                                 |
//! |------------|-----------------------------------------|
//! | 0..102     | Concrete AST node types (Python 3.12)   |
//! | 100..612   | Identifier hash buckets (FNV-1a % 512)  |
//! | 612        | PAD                                     |
//! | 613        | BOS                                     |
//! | 614        | EOS                                     |
//! | 615        | UNK                                     |
//! | 616        | PARSE_ERROR                             |
//! | 617..633   | Depth markers (DEPTH_0..DEPTH_15)       |
//! | 633..662   | Operators (Add..NotIn, 29)              |
//!
//! ## Hash choice
//!
//! Training used Python's built-in `hash()` (PYTHONHASHSEED-randomized). The
//! model treated identifier bucket assignments as noise, so any deterministic
//! hash works at inference. We use **FNV-1a**. Pair with
//! `scripts/ast_tokenizer_fnv.py` to produce byte-for-byte matching tokens.
//!
//! ## Coverage
//!
//! Byte-for-byte match with `scripts/ast_tokenizer_fnv.py` verified on all
//! Python AST constructs used in typical code (see tests/cross_validation.rs).
//!
//! Walker mirrors Python's `ast.walk()` + `iter_child_nodes()` exactly,
//! including the quirk that Load/Store/Del contexts are singleton instances
//! with "last-write-wins" depth via DFS id-tracking.
//!
//! **Not covered** (falls through to UNK): `Match` (3.10+), `TypeAlias`
//! (3.12+), `TryStar` (3.11+). These weren't in the training vocabulary.

use std::collections::VecDeque;

use rustpython_parser::{ast, Parse};

// ── Vocabulary constants (must match scripts/ast_tokenizer_fnv.py) ────
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

pub const OP_NAMES: &[&str] = &[
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "MatMult",
    "Invert", "Not", "UAdd", "USub",
    "And", "Or",
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

/// Map Python AST class name → token ID (0..102).
pub fn node_type_id(name: &str) -> Option<u16> {
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

// ── FNV-1a 32-bit hash ─────────────────────────────────────────────
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

// ── PyNode: uniform wrapper for BFS traversal ──────────────────────
//
// Mirrors Python's ast.iter_child_nodes: visit every AST-typed field in
// declaration order. `children()` returns these children as a Vec<PyNode>.

pub enum PyNode<'a> {
    // Container
    Module(&'a [ast::Stmt]),
    // Statements (stmt)
    FunctionDef(&'a ast::StmtFunctionDef),
    AsyncFunctionDef(&'a ast::StmtAsyncFunctionDef),
    ClassDef(&'a ast::StmtClassDef),
    Return(&'a ast::StmtReturn),
    Delete(&'a ast::StmtDelete),
    Assign(&'a ast::StmtAssign),
    AugAssign(&'a ast::StmtAugAssign),
    AnnAssign(&'a ast::StmtAnnAssign),
    For(&'a ast::StmtFor),
    AsyncFor(&'a ast::StmtAsyncFor),
    While(&'a ast::StmtWhile),
    If(&'a ast::StmtIf),
    With(&'a ast::StmtWith),
    AsyncWith(&'a ast::StmtAsyncWith),
    Raise(&'a ast::StmtRaise),
    Try(&'a ast::StmtTry),
    Assert(&'a ast::StmtAssert),
    Import(&'a ast::StmtImport),
    ImportFrom(&'a ast::StmtImportFrom),
    Global(&'a ast::StmtGlobal),
    Nonlocal(&'a ast::StmtNonlocal),
    Expr(&'a ast::StmtExpr),
    Pass, Break, Continue,
    // Expressions (expr)
    BoolOp(&'a ast::ExprBoolOp),
    NamedExpr(&'a ast::ExprNamedExpr),
    BinOp(&'a ast::ExprBinOp),
    UnaryOp(&'a ast::ExprUnaryOp),
    Lambda(&'a ast::ExprLambda),
    IfExp(&'a ast::ExprIfExp),
    Dict(&'a ast::ExprDict),
    Set(&'a ast::ExprSet),
    ListComp(&'a ast::ExprListComp),
    SetComp(&'a ast::ExprSetComp),
    DictComp(&'a ast::ExprDictComp),
    GeneratorExp(&'a ast::ExprGeneratorExp),
    Await(&'a ast::ExprAwait),
    Yield(&'a ast::ExprYield),
    YieldFrom(&'a ast::ExprYieldFrom),
    Compare(&'a ast::ExprCompare),
    Call(&'a ast::ExprCall),
    FormattedValue(&'a ast::ExprFormattedValue),
    JoinedStr(&'a ast::ExprJoinedStr),
    Constant(&'a ast::ExprConstant),
    Attribute(&'a ast::ExprAttribute),
    Subscript(&'a ast::ExprSubscript),
    Starred(&'a ast::ExprStarred),
    Name(&'a ast::ExprName),
    List(&'a ast::ExprList),
    Tuple(&'a ast::ExprTuple),
    Slice(&'a ast::ExprSlice),
    // Operators (leaves, carried by parent)
    Op(&'static str),
    // ExprContext (leaves)
    Load, Store, Del,
    // Wrappers (Python "abstract" bases that still get visited)
    Arguments(&'a ast::Arguments),
    Arg(&'a ast::Arg),
    ArgDefault(&'a ast::Expr),  // a default value (expression)
    Keyword(&'a ast::Keyword),
    Comprehension(&'a ast::Comprehension),
    ExceptHandler(&'a ast::ExceptHandlerExceptHandler),
    WithItem(&'a ast::WithItem),
    Alias(&'a ast::Alias),
    // Unknown / catch-all
    Unknown,
}

impl<'a> PyNode<'a> {
    /// Python AST class name (goes into node_type_id lookup).
    /// Falls through to UNK when the class isn't in our 102-type map.
    fn py_class(&self) -> &'static str {
        use PyNode::*;
        match self {
            Module(_) => "Module",
            FunctionDef(_) => "FunctionDef", AsyncFunctionDef(_) => "AsyncFunctionDef",
            ClassDef(_) => "ClassDef", Return(_) => "Return", Delete(_) => "Delete",
            Assign(_) => "Assign", AugAssign(_) => "AugAssign", AnnAssign(_) => "AnnAssign",
            For(_) => "For", AsyncFor(_) => "AsyncFor", While(_) => "While",
            If(_) => "If", With(_) => "With", AsyncWith(_) => "AsyncWith",
            Raise(_) => "Raise", Try(_) => "Try", Assert(_) => "Assert",
            Import(_) => "Import", ImportFrom(_) => "ImportFrom", Global(_) => "Global",
            Nonlocal(_) => "Nonlocal", Expr(_) => "Expr",
            Pass => "Pass", Break => "Break", Continue => "Continue",
            BoolOp(_) => "BoolOp", NamedExpr(_) => "NamedExpr", BinOp(_) => "BinOp",
            UnaryOp(_) => "UnaryOp", Lambda(_) => "Lambda", IfExp(_) => "IfExp",
            Dict(_) => "Dict", Set(_) => "Set", ListComp(_) => "ListComp",
            SetComp(_) => "SetComp", DictComp(_) => "DictComp", GeneratorExp(_) => "GeneratorExp",
            Await(_) => "Await", Yield(_) => "Yield", YieldFrom(_) => "YieldFrom",
            Compare(_) => "Compare", Call(_) => "Call", FormattedValue(_) => "FormattedValue",
            JoinedStr(_) => "JoinedStr", Constant(_) => "Constant", Attribute(_) => "Attribute",
            Subscript(_) => "Subscript", Starred(_) => "Starred", Name(_) => "Name",
            List(_) => "List", Tuple(_) => "Tuple", Slice(_) => "Slice",
            Op(name) => name,
            Load => "Load", Store => "Store", Del => "Del",
            Arguments(_) => "arguments", Arg(_) | ArgDefault(_) => "arg",
            Keyword(_) => "keyword", Comprehension(_) => "comprehension",
            ExceptHandler(_) => "ExceptHandler", WithItem(_) => "withitem",
            Alias(_) => "alias",
            Unknown => "Unknown",
        }
    }

    /// Identifier to emit (for Name/Attribute/FunctionDef/ClassDef/ImportFrom).
    fn ident(&self) -> Option<&str> {
        match self {
            PyNode::Name(e) => Some(e.id.as_str()),
            PyNode::Attribute(e) => Some(e.attr.as_str()),
            PyNode::FunctionDef(s) => Some(s.name.as_str()),
            PyNode::AsyncFunctionDef(s) => Some(s.name.as_str()),
            PyNode::ClassDef(s) => Some(s.name.as_str()),
            PyNode::ImportFrom(s) => s.module.as_ref().map(|m| m.as_str()),
            _ => None,
        }
    }

    /// Operator token (for BinOp/UnaryOp/BoolOp/Compare/AugAssign).
    fn op_name(&self) -> Option<&'static str> {
        match self {
            PyNode::BinOp(e) => Some(operator_name(&e.op)),
            PyNode::UnaryOp(e) => Some(unaryop_name(&e.op)),
            PyNode::BoolOp(e) => Some(boolop_name(&e.op)),
            PyNode::Compare(e) => e.ops.first().map(cmpop_name),
            PyNode::AugAssign(s) => Some(operator_name(&s.op)),
            _ => None,
        }
    }

    /// Constant type name (for Constant nodes → emits __const_{name}__).
    fn const_type(&self) -> Option<&'static str> {
        match self {
            PyNode::Constant(e) => Some(const_type_name(&e.value)),
            _ => None,
        }
    }

    /// Children in Python's `iter_child_nodes` order. Only AST-typed fields.
    fn children(&self) -> Vec<PyNode<'a>> {
        use PyNode::*;
        let mut c = Vec::new();
        match self {
            Module(stmts) => { for s in *stmts { c.push(stmt_node(s)); } }
            FunctionDef(s) => {
                // Python _fields: ('name', 'args', 'body', 'decorator_list', 'returns', ...)
                // 'name' is string, skip. 'args' is arguments node. body/decorators/returns are AST.
                c.push(Arguments(&s.args));
                for b in &s.body { c.push(stmt_node(b)); }
                for d in &s.decorator_list { c.push(expr_node(d)); }
                if let Some(ret) = &s.returns { c.push(expr_node(ret)); }
            }
            AsyncFunctionDef(s) => {
                c.push(Arguments(&s.args));
                for b in &s.body { c.push(stmt_node(b)); }
                for d in &s.decorator_list { c.push(expr_node(d)); }
                if let Some(ret) = &s.returns { c.push(expr_node(ret)); }
            }
            ClassDef(s) => {
                // _fields: ('name', 'bases', 'keywords', 'body', 'decorator_list', ...)
                for b in &s.bases { c.push(expr_node(b)); }
                for k in &s.keywords { c.push(Keyword(k)); }
                for b in &s.body { c.push(stmt_node(b)); }
                for d in &s.decorator_list { c.push(expr_node(d)); }
            }
            Return(s) => { if let Some(v) = &s.value { c.push(expr_node(v)); } }
            Delete(s) => { for t in &s.targets { c.push(expr_node(t)); } }
            Assign(s) => {
                for t in &s.targets { c.push(expr_node(t)); }
                c.push(expr_node(&s.value));
            }
            AugAssign(s) => {
                c.push(expr_node(&s.target));
                c.push(Op(operator_name(&s.op)));
                c.push(expr_node(&s.value));
            }
            AnnAssign(s) => {
                c.push(expr_node(&s.target));
                c.push(expr_node(&s.annotation));
                if let Some(v) = &s.value { c.push(expr_node(v)); }
            }
            For(s) => {
                c.push(expr_node(&s.target));
                c.push(expr_node(&s.iter));
                for b in &s.body { c.push(stmt_node(b)); }
                for b in &s.orelse { c.push(stmt_node(b)); }
            }
            AsyncFor(s) => {
                c.push(expr_node(&s.target));
                c.push(expr_node(&s.iter));
                for b in &s.body { c.push(stmt_node(b)); }
                for b in &s.orelse { c.push(stmt_node(b)); }
            }
            While(s) => {
                c.push(expr_node(&s.test));
                for b in &s.body { c.push(stmt_node(b)); }
                for b in &s.orelse { c.push(stmt_node(b)); }
            }
            If(s) => {
                c.push(expr_node(&s.test));
                for b in &s.body { c.push(stmt_node(b)); }
                for b in &s.orelse { c.push(stmt_node(b)); }
            }
            With(s) => {
                for it in &s.items { c.push(WithItem(it)); }
                for b in &s.body { c.push(stmt_node(b)); }
            }
            AsyncWith(s) => {
                for it in &s.items { c.push(WithItem(it)); }
                for b in &s.body { c.push(stmt_node(b)); }
            }
            Raise(s) => {
                if let Some(e) = &s.exc { c.push(expr_node(e)); }
                if let Some(cau) = &s.cause { c.push(expr_node(cau)); }
            }
            Try(s) => {
                for b in &s.body { c.push(stmt_node(b)); }
                for h in &s.handlers {
                    let ast::ExceptHandler::ExceptHandler(eh) = h;
                    c.push(ExceptHandler(eh));
                }
                for b in &s.orelse { c.push(stmt_node(b)); }
                for b in &s.finalbody { c.push(stmt_node(b)); }
            }
            Assert(s) => {
                c.push(expr_node(&s.test));
                if let Some(m) = &s.msg { c.push(expr_node(m)); }
            }
            Import(s) => { for a in &s.names { c.push(Alias(a)); } }
            ImportFrom(s) => { for a in &s.names { c.push(Alias(a)); } }
            Global(_) | Nonlocal(_) => {}
            Expr(s) => c.push(expr_node(&s.value)),
            Pass | Break | Continue => {}
            // Expressions
            BoolOp(e) => {
                c.push(Op(boolop_name(&e.op)));
                for v in &e.values { c.push(expr_node(v)); }
            }
            NamedExpr(e) => {
                c.push(expr_node(&e.target));
                c.push(expr_node(&e.value));
            }
            BinOp(e) => {
                c.push(expr_node(&e.left));
                c.push(Op(operator_name(&e.op)));
                c.push(expr_node(&e.right));
            }
            UnaryOp(e) => {
                c.push(Op(unaryop_name(&e.op)));
                c.push(expr_node(&e.operand));
            }
            Lambda(e) => {
                c.push(Arguments(&e.args));
                c.push(expr_node(&e.body));
            }
            IfExp(e) => {
                c.push(expr_node(&e.test));
                c.push(expr_node(&e.body));
                c.push(expr_node(&e.orelse));
            }
            Dict(e) => {
                // Python: _fields = ('keys', 'values') - keys first, then values
                for k in &e.keys { if let Some(k) = k { c.push(expr_node(k)); } }
                for v in &e.values { c.push(expr_node(v)); }
            }
            Set(e) => { for x in &e.elts { c.push(expr_node(x)); } }
            ListComp(e) => {
                c.push(expr_node(&e.elt));
                for g in &e.generators { c.push(Comprehension(g)); }
            }
            SetComp(e) => {
                c.push(expr_node(&e.elt));
                for g in &e.generators { c.push(Comprehension(g)); }
            }
            DictComp(e) => {
                c.push(expr_node(&e.key));
                c.push(expr_node(&e.value));
                for g in &e.generators { c.push(Comprehension(g)); }
            }
            GeneratorExp(e) => {
                c.push(expr_node(&e.elt));
                for g in &e.generators { c.push(Comprehension(g)); }
            }
            Await(e) => c.push(expr_node(&e.value)),
            Yield(e) => { if let Some(v) = &e.value { c.push(expr_node(v)); } }
            YieldFrom(e) => c.push(expr_node(&e.value)),
            Compare(e) => {
                c.push(expr_node(&e.left));
                for o in &e.ops { c.push(Op(cmpop_name(o))); }
                for cmp in &e.comparators { c.push(expr_node(cmp)); }
            }
            Call(e) => {
                c.push(expr_node(&e.func));
                for a in &e.args { c.push(expr_node(a)); }
                for k in &e.keywords { c.push(Keyword(k)); }
            }
            FormattedValue(e) => {
                c.push(expr_node(&e.value));
                if let Some(fs) = &e.format_spec { c.push(expr_node(fs)); }
            }
            JoinedStr(e) => { for v in &e.values { c.push(expr_node(v)); } }
            Constant(_) => {}  // leaf
            Attribute(e) => {
                c.push(expr_node(&e.value));
                c.push(ctx_node(&e.ctx));
            }
            Subscript(e) => {
                c.push(expr_node(&e.value));
                c.push(expr_node(&e.slice));
                c.push(ctx_node(&e.ctx));
            }
            Starred(e) => {
                c.push(expr_node(&e.value));
                c.push(ctx_node(&e.ctx));
            }
            Name(e) => c.push(ctx_node(&e.ctx)),
            List(e) => {
                for x in &e.elts { c.push(expr_node(x)); }
                c.push(ctx_node(&e.ctx));
            }
            Tuple(e) => {
                for x in &e.elts { c.push(expr_node(x)); }
                c.push(ctx_node(&e.ctx));
            }
            Slice(e) => {
                if let Some(l) = &e.lower { c.push(expr_node(l)); }
                if let Some(u) = &e.upper { c.push(expr_node(u)); }
                if let Some(st) = &e.step { c.push(expr_node(st)); }
            }
            // Leaves
            Op(_) | Load | Store | Del | Unknown => {}
            // Wrappers
            Arguments(a) => {
                // _fields: posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults
                for arg in &a.posonlyargs { c.push(Arg(&arg.def)); }
                for arg in &a.args { c.push(Arg(&arg.def)); }
                if let Some(v) = &a.vararg { c.push(Arg(v)); }
                for arg in &a.kwonlyargs { c.push(Arg(&arg.def)); }
                // kw_defaults: visit kwonlyargs' defaults
                for arg in &a.kwonlyargs {
                    if let Some(d) = &arg.default { c.push(ArgDefault(d)); }
                }
                if let Some(kw) = &a.kwarg { c.push(Arg(kw)); }
                // defaults: visit posonlyargs' and args' defaults (in that order)
                for arg in &a.posonlyargs {
                    if let Some(d) = &arg.default { c.push(ArgDefault(d)); }
                }
                for arg in &a.args {
                    if let Some(d) = &arg.default { c.push(ArgDefault(d)); }
                }
            }
            Arg(a) => {
                if let Some(ann) = &a.annotation { c.push(expr_node(ann)); }
            }
            ArgDefault(e) => c.push(expr_node(e)),
            Keyword(k) => c.push(expr_node(&k.value)),
            Comprehension(cmp) => {
                c.push(expr_node(&cmp.target));
                c.push(expr_node(&cmp.iter));
                for i in &cmp.ifs { c.push(expr_node(i)); }
            }
            ExceptHandler(h) => {
                if let Some(t) = &h.type_ { c.push(expr_node(t)); }
                for b in &h.body { c.push(stmt_node(b)); }
            }
            WithItem(it) => {
                c.push(expr_node(&it.context_expr));
                if let Some(v) = &it.optional_vars { c.push(expr_node(v)); }
            }
            Alias(_) => {}  // leaf: just the name and optional asname strings
        }
        c
    }
}

fn stmt_node(s: &ast::Stmt) -> PyNode<'_> {
    match s {
        ast::Stmt::FunctionDef(x) => PyNode::FunctionDef(x),
        ast::Stmt::AsyncFunctionDef(x) => PyNode::AsyncFunctionDef(x),
        ast::Stmt::ClassDef(x) => PyNode::ClassDef(x),
        ast::Stmt::Return(x) => PyNode::Return(x),
        ast::Stmt::Delete(x) => PyNode::Delete(x),
        ast::Stmt::Assign(x) => PyNode::Assign(x),
        ast::Stmt::AugAssign(x) => PyNode::AugAssign(x),
        ast::Stmt::AnnAssign(x) => PyNode::AnnAssign(x),
        ast::Stmt::For(x) => PyNode::For(x),
        ast::Stmt::AsyncFor(x) => PyNode::AsyncFor(x),
        ast::Stmt::While(x) => PyNode::While(x),
        ast::Stmt::If(x) => PyNode::If(x),
        ast::Stmt::With(x) => PyNode::With(x),
        ast::Stmt::AsyncWith(x) => PyNode::AsyncWith(x),
        ast::Stmt::Raise(x) => PyNode::Raise(x),
        ast::Stmt::Try(x) => PyNode::Try(x),
        ast::Stmt::TryStar(_) => PyNode::Unknown,  // TryStar is 3.11+, not in our 102 list
        ast::Stmt::Assert(x) => PyNode::Assert(x),
        ast::Stmt::Import(x) => PyNode::Import(x),
        ast::Stmt::ImportFrom(x) => PyNode::ImportFrom(x),
        ast::Stmt::Global(x) => PyNode::Global(x),
        ast::Stmt::Nonlocal(x) => PyNode::Nonlocal(x),
        ast::Stmt::Expr(x) => PyNode::Expr(x),
        ast::Stmt::Pass(_) => PyNode::Pass,
        ast::Stmt::Break(_) => PyNode::Break,
        ast::Stmt::Continue(_) => PyNode::Continue,
        ast::Stmt::Match(_) => PyNode::Unknown,  // Match is 3.10+, not in 102 list
        ast::Stmt::TypeAlias(_) => PyNode::Unknown,  // TypeAlias is 3.12+
    }
}

fn expr_node(e: &ast::Expr) -> PyNode<'_> {
    match e {
        ast::Expr::BoolOp(x) => PyNode::BoolOp(x),
        ast::Expr::NamedExpr(x) => PyNode::NamedExpr(x),
        ast::Expr::BinOp(x) => PyNode::BinOp(x),
        ast::Expr::UnaryOp(x) => PyNode::UnaryOp(x),
        ast::Expr::Lambda(x) => PyNode::Lambda(x),
        ast::Expr::IfExp(x) => PyNode::IfExp(x),
        ast::Expr::Dict(x) => PyNode::Dict(x),
        ast::Expr::Set(x) => PyNode::Set(x),
        ast::Expr::ListComp(x) => PyNode::ListComp(x),
        ast::Expr::SetComp(x) => PyNode::SetComp(x),
        ast::Expr::DictComp(x) => PyNode::DictComp(x),
        ast::Expr::GeneratorExp(x) => PyNode::GeneratorExp(x),
        ast::Expr::Await(x) => PyNode::Await(x),
        ast::Expr::Yield(x) => PyNode::Yield(x),
        ast::Expr::YieldFrom(x) => PyNode::YieldFrom(x),
        ast::Expr::Compare(x) => PyNode::Compare(x),
        ast::Expr::Call(x) => PyNode::Call(x),
        ast::Expr::FormattedValue(x) => PyNode::FormattedValue(x),
        ast::Expr::JoinedStr(x) => PyNode::JoinedStr(x),
        ast::Expr::Constant(x) => PyNode::Constant(x),
        ast::Expr::Attribute(x) => PyNode::Attribute(x),
        ast::Expr::Subscript(x) => PyNode::Subscript(x),
        ast::Expr::Starred(x) => PyNode::Starred(x),
        ast::Expr::Name(x) => PyNode::Name(x),
        ast::Expr::List(x) => PyNode::List(x),
        ast::Expr::Tuple(x) => PyNode::Tuple(x),
        ast::Expr::Slice(x) => PyNode::Slice(x),
        // IpyEscapeCommand is rustpython-only; suppress by using a catch-all match arm
    }
}

fn ctx_node(ctx: &ast::ExprContext) -> PyNode<'static> {
    match ctx {
        ast::ExprContext::Load => PyNode::Load,
        ast::ExprContext::Store => PyNode::Store,
        ast::ExprContext::Del => PyNode::Del,
    }
}

// ── Operator / unaryop / boolop / cmpop name mappings ──────────────
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
        None => "NoneType", Bool(_) => "bool", Str(_) => "str", Bytes(_) => "bytes",
        Int(_) => "int", Tuple(_) => "tuple", Float(_) => "float",
        Complex { .. } => "complex", Ellipsis => "ellipsis",
    }
}

// ── Context singletons: Python reuses Load()/Store()/Del() across the
// AST, so _annotate_depths uses id(node) as key and writes the same
// depth multiple times — last-write-wins in DFS order. We replicate
// that by pre-pass DFS to compute the final depths.

#[derive(Default, Clone, Copy)]
struct CtxDepths {
    load: Option<usize>,
    store: Option<usize>,
    del: Option<usize>,
}

fn annotate_ctxs<'a>(node: PyNode<'a>, depth: usize, state: &mut CtxDepths) {
    match node {
        PyNode::Load => { state.load = Some(depth); return; }
        PyNode::Store => { state.store = Some(depth); return; }
        PyNode::Del => { state.del = Some(depth); return; }
        _ => {}
    }
    let children = node.children();
    for ch in children {
        annotate_ctxs(ch, depth + 1, state);
    }
}

// ── Tokenizer entry point ──────────────────────────────────────────
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

    // Pre-pass: compute Load/Store/Del depths (Python singleton quirk).
    let mut ctx = CtxDepths::default();
    annotate_ctxs(PyNode::Module(&suite), 0, &mut ctx);

    let module = PyNode::Module(&suite);

    let mut tokens: Vec<u16> = Vec::with_capacity(max_len);
    tokens.push(BOS);

    // BFS walk — matches Python's ast.walk() + iter_child_nodes.
    let mut queue: VecDeque<(PyNode, usize)> = VecDeque::new();
    queue.push_back((module, 0));
    while let Some((node, depth)) = queue.pop_front() {
        let cls = node.py_class();
        tokens.push(node_type_id(cls).unwrap_or(UNK));
        // For context singletons use the annotated depth; otherwise current depth.
        let emit_depth = match &node {
            PyNode::Load => ctx.load.unwrap_or(depth),
            PyNode::Store => ctx.store.unwrap_or(depth),
            PyNode::Del => ctx.del.unwrap_or(depth),
            _ => depth,
        };
        tokens.push(depth_token(emit_depth));

        if let Some(id) = node.ident() {
            tokens.push(ident_token(id));
        }
        if let Some(op) = node.op_name() {
            tokens.push(op_token(op));
        }
        if let Some(ct) = node.const_type() {
            tokens.push(ident_token(&format!("__const_{ct}__")));
        }

        let children = node.children();
        for ch in children {
            queue.push_back((ch, depth + 1));
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
        assert_eq!(toks[1], node_type_id("Module").unwrap());
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
