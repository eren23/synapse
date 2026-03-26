#!/usr/bin/env python3
"""Render shared public status blocks from a single manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "status" / "public_status.json"
WASM_CORE_PATH = ROOT / "synapse-wasm" / "pkg" / "synapse_wasm_bg.wasm"
WASM_JS_PATH = ROOT / "synapse-wasm" / "pkg" / "synapse_wasm.js"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def format_support(support: str) -> str:
    return support.replace("_", " ").title()


def format_model_status(status: str) -> str:
    return status.replace("_", " ").title()


def kb_label(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"~{round(num_bytes / 1024):d} KB"


def read_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def md_benchmark_table(manifest: dict) -> str:
    rows = [
        "| Configuration | Prefill (tok/s) | Decode (tok/s) | Support | Notes |",
        "|---------------|-----------------|----------------|---------|-------|",
    ]
    for item in manifest["benchmarks"]["configs"]:
        rows.append(
            f"| {item['label']} | {item['prefill_tps']:g} | {item['decode_tps']:g} | "
            f"{format_support(item['support'])} | {item['notes']} |"
        )
    ref = manifest["benchmarks"]["reference"]
    rows.append(
        f"| {ref['label']} | {ref['prefill_tps']:g} | {ref['decode_tps']:g} | Reference | {ref['notes']} |"
    )
    return "\n".join(rows)


def md_runtime_profile_table(manifest: dict) -> str:
    rows = [
        "| Runtime Profile | Support | Targets | Backends | Quantization |",
        "|-----------------|---------|---------|----------|--------------|",
    ]
    for profile in manifest["runtime_profiles"]:
        rows.append(
            f"| {profile['label']} | {format_support(profile['support'])} | "
            f"{', '.join(profile['targets'])} | {', '.join(profile['backends'])} | "
            f"{', '.join(profile['quantization'])} |"
        )
    return "\n".join(rows)


def md_feature_list(manifest: dict) -> str:
    return "\n".join(
        f"- **{item['label']}** ({format_support(item['support'])}) — {item['details']}"
        for item in manifest["features"]
    )


def md_model_matrix(manifest: dict) -> str:
    rows = [
        "| Model Family | Status | Notes |",
        "|--------------|--------|-------|",
    ]
    for item in manifest["model_families"]:
        rows.append(
            f"| {item['label']} | {format_model_status(item['status'])} | {item['notes']} |"
        )
    return "\n".join(rows)


def md_root_positioning(manifest: dict) -> str:
    return "\n".join(
        [
            manifest["positioning"]["headline"],
            "",
            f"- {manifest['positioning']['native_runtime']}",
            f"- {manifest['positioning']['wasm_runtime']}",
            f"- Verified benchmark baseline is {manifest['benchmarks']['model']} on {manifest['benchmarks']['device']}.",
        ]
    )


def md_status_note(manifest: dict) -> str:
    return (
        f"Measured against {manifest['benchmarks']['model']} on {manifest['benchmarks']['device']}. "
        f"Last verified: {manifest['last_verified']}."
    )


def html_subtitle(manifest: dict) -> str:
    wasm_bytes = read_size(WASM_CORE_PATH)
    return "\n".join(
        [
            '<p class="subtitle">',
            "    Real models running locally via WASM. No server, no GPU, no cloud.",
            f'    <span class="tag">{kb_label(wasm_bytes)} WASM core</span>',
            '    <span class="tag">WASM SIMD</span>',
            '    <span class="tag">Pure Rust runtime</span>',
            "</p>",
            '<p class="subtitle" style="margin-top:-8px;">',
            "    Browser demos use the pure-Rust WASM path. Native builds use Rust + Zig SIMD kernels with optional Metal acceleration.",
            "</p>",
        ]
    )


def md_artifact_budget(manifest: dict) -> str:
    size_lookup = {
        "wasm_core": read_size(WASM_CORE_PATH),
        "wasm_wrapper_js": read_size(WASM_JS_PATH),
    }
    rows = [
        "| Artifact | Current | Budget | Status |",
        "|----------|---------|--------|--------|",
    ]
    for budget in manifest["artifact_budgets"]:
        current = size_lookup.get(budget["id"])
        status = "ok" if current is None or current <= budget["max_bytes"] else "over"
        rows.append(
            f"| {budget['label']} | {kb_label(current)} | {kb_label(budget['max_bytes'])} | {status} |"
        )
    return "\n".join(rows)


def replace_block(text: str, name: str, body: str) -> str:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    replacement = f"{start}\n{body}\n{end}"
    if not pattern.search(text):
        raise RuntimeError(f"Missing markers for block '{name}'")
    return pattern.sub(replacement, text, count=1)


def rendered_files(manifest: dict) -> dict[Path, str]:
    files: dict[Path, str] = {}

    root_readme = (ROOT.parent / "README.md").read_text()
    root_readme = replace_block(root_readme, "status:root-positioning", md_root_positioning(manifest))
    root_readme = replace_block(root_readme, "status:root-benchmark", md_benchmark_table(manifest))
    root_readme = replace_block(root_readme, "status:root-profiles", md_runtime_profile_table(manifest))
    root_readme = replace_block(root_readme, "status:root-artifacts", md_artifact_budget(manifest))
    files[ROOT.parent / "README.md"] = root_readme

    synapse_readme = (ROOT / "README.md").read_text()
    synapse_readme = replace_block(synapse_readme, "status:synapse-benchmark", md_benchmark_table(manifest))
    synapse_readme = replace_block(synapse_readme, "status:synapse-features", md_feature_list(manifest))
    synapse_readme = replace_block(synapse_readme, "status:synapse-models", md_model_matrix(manifest))
    files[ROOT / "README.md"] = synapse_readme

    docs_index = (ROOT / "docs" / "src" / "index.md").read_text()
    docs_index = replace_block(docs_index, "status:docs-index-features", md_feature_list(manifest))
    docs_index = replace_block(docs_index, "status:docs-index-benchmark", md_benchmark_table(manifest))
    files[ROOT / "docs" / "src" / "index.md"] = docs_index

    docs_perf = (ROOT / "docs" / "src" / "architecture" / "performance.md").read_text()
    docs_perf = replace_block(docs_perf, "status:docs-performance-note", md_status_note(manifest))
    docs_perf = replace_block(docs_perf, "status:docs-performance-benchmark", md_benchmark_table(manifest))
    docs_perf = replace_block(docs_perf, "status:docs-performance-artifacts", md_artifact_budget(manifest))
    files[ROOT / "docs" / "src" / "architecture" / "performance.md"] = docs_perf

    docs_models = (ROOT / "docs" / "src" / "getting-started" / "models.md").read_text()
    docs_models = replace_block(docs_models, "status:docs-model-matrix", md_model_matrix(manifest))
    files[ROOT / "docs" / "src" / "getting-started" / "models.md"] = docs_models

    web_index = (ROOT / "web" / "index.html").read_text()
    web_index = replace_block(web_index, "status:web-subtitle", html_subtitle(manifest))
    files[ROOT / "web" / "index.html"] = web_index

    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if rendered files are out of sync")
    args = parser.parse_args()

    manifest = load_manifest()
    files = rendered_files(manifest)
    out_of_sync: list[Path] = []

    for path, rendered in files.items():
        current = path.read_text()
        if current != rendered:
            out_of_sync.append(path)
            if not args.check:
                path.write_text(rendered)

    if args.check and out_of_sync:
        for path in out_of_sync:
            print(f"out of sync: {path}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
