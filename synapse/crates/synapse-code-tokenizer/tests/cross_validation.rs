//! Byte-for-byte validation against the Python FNV-1a reference
//! (scripts/ast_tokenizer_fnv.py). If Python-produced tokens match exactly,
//! the Rust port is drop-in equivalent.

use synapse_code_tokenizer::tokenize;

/// Each case: (rust_source, expected_tokens from Python ast_tokenizer_fnv.py)
const CASES: &[(&str, &[u16])] = &[
    // "def f(x):\n    return x + 1\n"
    (
        "def f(x):\n    return x + 1\n",
        &[613, 65, 617, 39, 618, 509, 615, 619, 81, 619, 615, 620, 13, 620, 633,
          67, 621, 235, 0, 621, 23, 621, 294, 60, 622, 614, 612, 612, 612, 612,
          612, 612],
    ),
    // "for i in range(10):\n    print(i)\n"
    (
        "for i in range(10):\n    print(i)\n",
        &[613, 65, 617, 37, 618, 67, 619, 552, 20, 619, 33, 619, 86, 620, 67,
          620, 310, 23, 620, 294, 20, 620, 60, 622, 67, 621, 236, 67, 621, 552,
          60, 622],
    ),
    // "x = 1\ny = x + 2\n"
    (
        "x = 1\ny = x + 2\n",
        &[613, 65, 617, 4, 618, 4, 618, 67, 619, 235, 23, 619, 294, 67, 619, 344,
          13, 619, 633, 86, 620, 86, 620, 67, 620, 235, 0, 620, 23, 620, 294, 614],
    ),
    // "def is_big(n):\n    return n > 100\n"
    (
        "def is_big(n):\n    return n > 100\n",
        &[613, 65, 617, 39, 618, 324, 615, 619, 81, 619, 615, 620, 22, 620, 656,
          67, 621, 149, 43, 621, 23, 621, 294, 60, 622, 614, 612, 612, 612, 612,
          612, 612],
    ),
    // "d = {'a': 1, 'b': 2}\nx = d['a']\n"
    (
        "d = {'a': 1, 'b': 2}\nx = d['a']\n",
        &[613, 65, 617, 4, 618, 4, 618, 67, 619, 215, 27, 619, 67, 619, 235, 89,
          619, 86, 620, 23, 620, 292, 23, 620, 292, 23, 620, 294, 23, 620, 294,
          614],
    ),
    // quicksort (32 tokens)
    (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    return quicksort(left)",
        &[613, 65, 617, 39, 618, 588, 615, 619, 45, 619, 4, 619, 4, 619, 81, 619,
          615, 620, 22, 620, 655, 81, 620, 67, 620, 379, 89, 620, 67, 620, 212, 614],
    ),
    // ClassDef with methods
    (
        "class Foo:\n    def __init__(self, x):\n        self.x = x\n    def get(self):\n        return self.x",
        &[613, 65, 617, 21, 618, 219, 39, 619, 319, 39, 619, 443, 615, 620, 4,
          620, 615, 620, 81, 620, 615, 621, 615, 621, 8, 621, 235, 67, 621, 235,
          615, 621],
    ),
    // try/except/finally
    (
        "try:\n    x = int(\"abc\")\nexcept ValueError as e:\n    print(e)\nfinally:\n    pass",
        &[613, 65, 617, 91, 618, 4, 619, 32, 619, 77, 619, 67, 620, 235, 20, 620,
          67, 620, 592, 33, 620, 86, 621, 67, 621, 194, 23, 621, 292, 60, 623, 614],
    ),
    // import + from import + dict + list literal
    (
        "import json\nfrom os import path\ndef f():\n    return {\"a\": [1, 2, 3]}",
        &[613, 65, 617, 47, 618, 48, 618, 379, 39, 618, 509, 615, 619, 615, 619,
          615, 619, 81, 619, 27, 620, 23, 621, 292, 58, 621, 23, 622, 294, 23,
          622, 294],
    ),
];

#[test]
fn rust_matches_python_fnv_tokens() {
    for (src, expected) in CASES {
        let got = tokenize(src, expected.len());
        if got != *expected {
            eprintln!("SOURCE: {src:?}");
            eprintln!("EXPECTED: {expected:?}");
            eprintln!("GOT:      {got:?}");
            // Find first mismatch position
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                if a != b {
                    eprintln!("First mismatch at index {i}: got {a}, expected {b}");
                    break;
                }
            }
            panic!("Token mismatch (see stderr for details)");
        }
    }
}
