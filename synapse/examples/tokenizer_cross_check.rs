//! Validate synapse-code-tokenizer matches the Python FNV-1a reference.
//!
//! For each test snippet, we print Rust tokens. Then the caller compares
//! against `python3 scripts/ast_tokenizer_fnv.py < snippet`.
//!
//! Usage:
//!   cargo run --release --example tokenizer_cross_check

use synapse_code_tokenizer::tokenize;

fn main() {
    let snippets = [
        ("simple_fn", "def f(x):\n    return x + 1\n"),
        ("loop", "for i in range(10):\n    print(i)\n"),
        ("assign", "x = 1\ny = x + 2\n"),
        ("compare", "def is_big(n):\n    return n > 100\n"),
        ("dict", "d = {'a': 1, 'b': 2}\nx = d['a']\n"),
    ];

    for (name, src) in snippets {
        let toks = tokenize(src, 32);
        println!("--- {name} ---");
        println!("source: {src:?}");
        print!("tokens: [");
        for (i, t) in toks.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{t}");
        }
        println!("]");
    }
}
