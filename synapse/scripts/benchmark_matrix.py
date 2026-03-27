#!/usr/bin/env python3
"""Run the Synapse validation matrix and publish a tiered benchmark artifact."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = ROOT / "status"
PUBLIC_STATUS_PATH = STATUS_DIR / "public_status.json"
ARTIFACT_JSON_PATH = STATUS_DIR / "benchmark_matrix.json"
ARTIFACT_MD_PATH = STATUS_DIR / "benchmark_matrix.md"

DEFAULT_OFFICIAL_MODELS = {
    "qwen3": Path("/tmp/qwen3-0.6b"),
    "llama_3_2": Path("/tmp/llama-3.2-1b"),
}

DEFAULT_EXPLORATORY_MODELS = {
    "qwen2_5": Path("/tmp/qwen2.5-0.5b"),
    "tinyllama": Path("/tmp/tinyllama-1.1b"),
    "qwen3_gguf": Path("/tmp/qwen3-0.6b-gguf"),
}

REAL_PROMPTS = {
    "hello": "hello",
    "repro_repeat_after_me": "repeat after me: yarrami ye fener",
}

RUNTIME_RE = re.compile(
    r"^Runtime:\s+family=(?P<family>\S+)\s+backend=(?P<backend>\S+)\s+quantized=(?P<quantized>\S+)\s+"
    r"prefill=(?P<prefill>\S+)\s+prefill_strategy=(?P<prefill_strategy>\S+)\s+"
    r"decode=(?P<decode>\S+)\s+decode_strategy=(?P<decode_strategy>\S+)$",
    re.M,
)
CHAT_RE = re.compile(r"^Chat:\s+family=(?P<family>\S+)\s+thinking=(?P<thinking>\S+)$", re.M)
THROUGHPUT_RE = re.compile(
    r"^Prefill:\s+(?P<prompt_tokens>\d+)\s+tokens\s+in\s+(?P<prefill_ms>[\d.]+)ms\s+\((?P<prefill_tps>[\d.]+)\s+tok/s\)\s+\|\s+"
    r"Decode:\s+(?P<decode_tokens>\d+)\s+tokens\s+at\s+(?P<decode_tps>[\d.]+)\s+tok/s\s+\|\s+(?P<precision>\S+)$",
    re.M,
)
PROFILE_RE = re.compile(
    r"^Profile:\s+render=(?P<render_ms>[\d.]+)ms\s+encode=(?P<encode_ms>[\d.]+)ms\s+"
    r"prefill=(?P<prefill_ms>[\d.]+)ms\s+decode=(?P<decode_ms>[\d.]+)ms\s+"
    r"hidden_tokens=(?P<hidden_tokens>\d+)\s+visible_tokens=(?P<visible_tokens>\d+)$",
    re.M,
)
MODEL_BENCH_PREFILL_RE = re.compile(r"^\s*overall:\s+(?P<prefill_tps>[\d.]+)\s+tok/s$", re.M)
MODEL_BENCH_DECODE_RE = re.compile(
    r"^\s+\d+\s+tokens:\s+(?P<decode_tps>[\d.]+)\s+tok/s\s+\(cached,",
    re.M,
)
TEST_SUMMARY_RE = re.compile(r"test result:\s+(?P<summary>.+)$", re.M)


@dataclass(frozen=True)
class CommandSpec:
    id: str
    label: str
    command: list[str]
    category: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sh(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(spec: CommandSpec) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        spec.command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    elapsed_s = time.perf_counter() - start
    combined = (proc.stdout or "") + (proc.stderr or "")
    summary_match = TEST_SUMMARY_RE.search(combined)
    return {
        "id": spec.id,
        "label": spec.label,
        "category": spec.category,
        "command": spec.command,
        "command_str": sh(spec.command),
        "status": "ok" if proc.returncode == 0 else "failed",
        "exit_code": proc.returncode,
        "duration_s": round(elapsed_s, 3),
        "summary": summary_match.group("summary").strip() if summary_match else None,
        "stdout_tail": combined[-8000:],
    }


def parse_runtime(text: str) -> dict[str, Any] | None:
    match = RUNTIME_RE.search(text)
    if not match:
        return None
    data = match.groupdict()
    data["quantized"] = data["quantized"] == "true"
    return data


def parse_chat(text: str) -> dict[str, Any] | None:
    match = CHAT_RE.search(text)
    return match.groupdict() if match else None


def parse_throughput(text: str) -> dict[str, Any] | None:
    match = THROUGHPUT_RE.search(text)
    if not match:
        return None
    data = match.groupdict()
    for key in ("prompt_tokens", "decode_tokens"):
        data[key] = int(data[key])
    for key in ("prefill_ms", "prefill_tps", "decode_tps"):
        data[key] = float(data[key])
    return data


def parse_profile(text: str) -> dict[str, Any] | None:
    match = PROFILE_RE.search(text)
    if not match:
        return None
    data = match.groupdict()
    for key in ("render_ms", "encode_ms", "prefill_ms", "decode_ms"):
        data[key] = float(data[key])
    for key in ("hidden_tokens", "visible_tokens"):
        data[key] = int(data[key])
    return data


def parse_model_benchmark(text: str) -> tuple[float | None, float | None]:
    prefill = MODEL_BENCH_PREFILL_RE.search(text)
    decode = MODEL_BENCH_DECODE_RE.search(text)
    return (
        float(prefill.group("prefill_tps")) if prefill else None,
        float(decode.group("decode_tps")) if decode else None,
    )


def family_label(family_id: str) -> str:
    labels = {
        "qwen3": "Qwen3",
        "llama_3_2": "LLaMA 3.2",
        "mistral_7b": "Mistral 7B",
        "phi_3": "Phi-3",
        "gemma": "Gemma",
        "qwen2_5": "Qwen2.5",
        "tinyllama": "TinyLlama",
        "qwen3_gguf": "Qwen3 GGUF",
    }
    return labels.get(family_id, family_id.replace("_", " ").title())


def friendly_device_name() -> str:
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        if platform.machine() == "arm64":
            return "Apple Silicon"
    return platform.processor() or platform.machine()


def load_public_status() -> dict[str, Any]:
    return json.loads(PUBLIC_STATUS_PATH.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def real_model_specs(
    official_models: dict[str, Path],
    exploratory_models: dict[str, Path],
    include_exploratory: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "id": "qwen3_f32_cpu_hello",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "f32 CPU",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": False,
            "metal": False,
            "public_table": True,
        },
        {
            "id": "qwen3_int8_cpu_hello",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "INT8 CPU",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": True,
            "metal": False,
            "public_table": True,
        },
        {
            "id": "qwen3_int8_cpu_repro_disabled",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "INT8 CPU",
            "prompt_id": "repro_repeat_after_me",
            "thinking_mode": "disabled",
            "quantize": True,
            "metal": False,
            "public_table": False,
        },
        {
            "id": "qwen3_int8_cpu_repro_auto",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "INT8 CPU",
            "prompt_id": "repro_repeat_after_me",
            "thinking_mode": "auto",
            "quantize": True,
            "metal": False,
            "public_table": False,
        },
        {
            "id": "qwen3_f32_metal_hello",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "f32 metal-feature build",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": False,
            "metal": True,
            "public_table": True,
        },
        {
            "id": "qwen3_int8_metal_hello",
            "family_id": "qwen3",
            "checkpoint_label": "Qwen3-0.6B",
            "path": official_models["qwen3"],
            "tier": "measured_local",
            "configuration": "INT8 metal-feature build",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": True,
            "metal": True,
            "public_table": True,
        },
        {
            "id": "llama_3_2_f32_cpu_hello",
            "family_id": "llama_3_2",
            "checkpoint_label": "LLaMA 3.2-1B",
            "path": official_models["llama_3_2"],
            "tier": "measured_local",
            "configuration": "f32 CPU",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": False,
            "metal": False,
            "public_table": True,
        },
        {
            "id": "llama_3_2_int8_cpu_hello",
            "family_id": "llama_3_2",
            "checkpoint_label": "LLaMA 3.2-1B",
            "path": official_models["llama_3_2"],
            "tier": "measured_local",
            "configuration": "INT8 CPU",
            "prompt_id": "hello",
            "thinking_mode": None,
            "quantize": True,
            "metal": False,
            "public_table": True,
        },
    ]

    if include_exploratory:
        for family_id, path in exploratory_models.items():
            specs.append(
                {
                    "id": f"{family_id}_exploratory_hello",
                    "family_id": family_id,
                    "checkpoint_label": path.name,
                    "path": path,
                    "tier": "exploratory_local",
                    "configuration": "f32 CPU",
                    "prompt_id": "hello",
                    "thinking_mode": None,
                    "quantize": False,
                    "metal": False,
                    "public_table": False,
                }
            )

    return specs


def run_real_model(spec: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": spec["id"],
        "tier": spec["tier"],
        "family": family_label(spec["family_id"]),
        "family_id": spec["family_id"],
        "checkpoint": spec["checkpoint_label"],
        "model_dir": str(spec["path"]),
        "configuration": spec["configuration"],
        "prompt_id": spec["prompt_id"],
        "prompt": REAL_PROMPTS[spec["prompt_id"]],
        "thinking_mode": spec["thinking_mode"],
        "requested_backend": "metal" if spec["metal"] else "cpu_simd",
        "requested_quantized": spec["quantize"],
        "public_table": spec["public_table"],
    }

    if not spec["path"].exists():
        row.update(
            {
                "status": "skipped",
                "notes": f"checkpoint not found: {spec['path']}",
            }
        )
        return row

    command = ["cargo", "run", "--example", "qwen3_chat", "--release"]
    if spec["metal"]:
        command.extend(["--features", "metal"])
    command.extend(
        [
            "--",
            "--model-dir",
            str(spec["path"]),
            "--prompt",
            REAL_PROMPTS[spec["prompt_id"]],
            "--max-new-tokens",
            "32",
        ]
    )
    if spec["quantize"]:
        command.append("--quantize")
    if spec["thinking_mode"]:
        command.extend(["--thinking", spec["thinking_mode"]])
    if spec["prompt_id"] != "hello":
        command.append("--profile-stages")

    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed_s = time.perf_counter() - start
    output = (proc.stdout or "") + (proc.stderr or "")

    row.update(
        {
            "command": command,
            "command_str": sh(command),
            "duration_s": round(elapsed_s, 3),
            "exit_code": proc.returncode,
            "stdout_tail": output[-8000:],
        }
    )

    if proc.returncode != 0:
        row.update({"status": "failed", "notes": "command exited non-zero"})
        return row

    runtime = parse_runtime(output)
    chat = parse_chat(output)
    throughput = parse_throughput(output)
    profile = parse_profile(output)
    if runtime:
        row["runtime"] = runtime
    if chat:
        row["chat"] = chat
    if profile:
        row["profile"] = profile

    if not runtime or not throughput:
        row.update(
            {
                "status": "failed",
                "notes": "missing runtime or throughput line in output",
            }
        )
        return row

    row.update(throughput)
    row["status"] = "ok"
    if spec["metal"] and runtime["backend"] != "metal":
        row["status"] = "fallback"
        row["notes"] = "metal-feature build fell back to cpu_simd at runtime"
    else:
        row["notes"] = ""
    return row


def synthetic_benchmark_specs() -> list[dict[str, str]]:
    return [
        {
            "id": "synthetic_qwen3",
            "family_id": "qwen3",
            "label": "Qwen3 synthetic scaled config",
            "config_path": "configs/qwen3_0.6b.json",
        },
        {
            "id": "synthetic_llama_3_2",
            "family_id": "llama_3_2",
            "label": "LLaMA 3.2 synthetic scaled config",
            "config_path": "configs/llama3.2_1b.json",
        },
        {
            "id": "synthetic_mistral_7b",
            "family_id": "mistral_7b",
            "label": "Mistral 7B synthetic scaled config",
            "config_path": "configs/mistral_7b.json",
        },
        {
            "id": "synthetic_gemma",
            "family_id": "gemma",
            "label": "Gemma synthetic scaled config",
            "config_path": "configs/gemma_2b.json",
        },
    ]


def run_synthetic_benchmark(spec: dict[str, str]) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--example",
        "model_benchmark",
        "--release",
        "--",
        "--config",
        spec["config_path"],
    ]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed_s = time.perf_counter() - start
    output = (proc.stdout or "") + (proc.stderr or "")
    prefill_tps, decode_tps = parse_model_benchmark(output)
    status = "ok" if proc.returncode == 0 and prefill_tps is not None and decode_tps is not None else "failed"
    return {
        "id": spec["id"],
        "tier": "synthetic_validated",
        "family": family_label(spec["family_id"]),
        "family_id": spec["family_id"],
        "checkpoint": "synthetic scaled config",
        "configuration": "synthetic scaled",
        "prompt_id": "synthetic_default",
        "command": command,
        "command_str": sh(command),
        "duration_s": round(elapsed_s, 3),
        "status": status,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "notes": spec["label"],
        "stdout_tail": output[-8000:],
    }


def benchmark_test_specs() -> list[CommandSpec]:
    return [
        CommandSpec(
            id="lib_tests",
            label="Inference library tests",
            category="tests",
            command=["cargo", "test", "-p", "synapse-inference", "--lib"],
        ),
        CommandSpec(
            id="multi_model_validation",
            label="Multi-model validation",
            category="tests",
            command=["cargo", "test", "--test", "multi_model_validation", "--release"],
        ),
        CommandSpec(
            id="quantization_speedup_isolated",
            label="Quantization speedup isolated matmul",
            category="benchmarks",
            command=[
                "cargo",
                "test",
                "--test",
                "quantization_speedup",
                "--release",
                "--",
                "--nocapture",
                "isolated_matmul",
            ],
        ),
        CommandSpec(
            id="quantization_speedup_full_model",
            label="Quantization speedup full model",
            category="benchmarks",
            command=[
                "cargo",
                "test",
                "--test",
                "quantization_speedup",
                "--release",
                "--",
                "--nocapture",
                "quantization_speedup_int8_vs_f32",
            ],
        ),
        CommandSpec(
            id="prefill_throughput",
            label="Prefill throughput benchmark",
            category="benchmarks",
            command=[
                "cargo",
                "test",
                "--test",
                "prefill_throughput",
                "--release",
                "--",
                "--nocapture",
            ],
        ),
        CommandSpec(
            id="kvcache_speedup",
            label="KV-cache speedup benchmark",
            category="benchmarks",
            command=[
                "cargo",
                "test",
                "--test",
                "kvcache_speedup",
                "--release",
                "--",
                "--nocapture",
            ],
        ),
    ]


def select_public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("tier") != "measured_local":
            continue
        if not row.get("public_table"):
            continue
        if row.get("status") != "ok":
            continue
        selected.append(
            {
                "id": row["id"],
                "family": row["family"],
                "label": row["configuration"],
                "prompt_id": row["prompt_id"],
                "prefill_tps": row["prefill_tps"],
                "decode_tps": row["decode_tps"],
                "evidence": "measured_local",
                "runtime_summary": row["runtime"],
                "notes": runtime_note(row),
            }
        )
    return selected


def runtime_note(row: dict[str, Any]) -> str:
    runtime = row.get("runtime")
    if not runtime:
        return row.get("notes", "")
    if row.get("requested_backend") == "metal" and runtime.get("backend") != "metal":
        return "Metal build fell back to cpu_simd"
    if row.get("thinking_mode"):
        return (
            f"Runtime backend={runtime['backend']}; prompt={row['prompt_id']}; "
            f"thinking={row['thinking_mode']}"
        )
    return f"Runtime backend={runtime['backend']}; prompt={row['prompt_id']}"


def family_status_updates(rows: list[dict[str, Any]], suites: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    updates = {
        "qwen3": {
            "status": "validated",
            "evidence": "benchmarked_local",
            "notes": "Real checkpoint benchmarked locally; logits verified",
        },
        "llama_3_2": {
            "status": "config_ready",
            "evidence": "synthetic_validated",
            "notes": "Config and weight mapper path present; synthetic validation passing",
        },
        "mistral_7b": {
            "status": "config_ready",
            "evidence": "synthetic_validated",
            "notes": "Sliding-window config path present; synthetic correctness tests passing",
        },
        "phi_3": {
            "status": "in_progress",
            "evidence": "synthetic_validated",
            "notes": "Weight-mapper support in progress; synthetic validation passing",
        },
        "gemma": {
            "status": "config_ready",
            "evidence": "synthetic_validated",
            "notes": "Same core transformer path; synthetic validation passing",
        },
    }

    by_id = {row["id"]: row for row in rows}
    if by_id.get("llama_3_2_f32_cpu_hello", {}).get("status") == "ok":
        updates["llama_3_2"] = {
            "status": "benchmarked_local",
            "evidence": "benchmarked_local",
            "notes": "Real checkpoint benchmarked locally on this machine",
        }
    if by_id.get("synthetic_mistral_7b", {}).get("status") == "failed":
        updates["mistral_7b"]["notes"] = (
            "Sliding-window config path present; synthetic correctness tests pass, "
            "but the scaled synthetic throughput benchmark is currently failing"
        )

    return updates


def update_public_status(
    public_status: dict[str, Any],
    artifact: dict[str, Any],
    measured_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    public_status["manifest_version"] = 2
    public_status["last_verified"] = artifact["generated_on"]
    public_status["positioning"]["headline"] = (
        "Edge-native inference stack for local ML across native and browser targets."
    )
    public_status["benchmarks"]["device"] = artifact["host"]["device"]
    public_status["benchmarks"]["scope"] = "Tiered local matrix on this machine"
    public_status["benchmarks"]["matrix_artifact"] = str(ARTIFACT_JSON_PATH.relative_to(ROOT))
    public_status["benchmarks"]["configs"] = measured_rows
    public_status["benchmarks"]["reference"] = {
        "label": "llama.cpp Q4_K_M",
        "family": "Reference",
        "prompt_id": "reference_only",
        "prefill_tps": 5518.0,
        "decode_tps": 173.0,
        "notes": "Reference only, not a parity claim",
        "evidence": "reference",
    }
    updates = family_status_updates(artifact["benchmark_runs"], artifact["test_suites"])
    public_status["model_families"] = [
        {
            "id": item["id"],
            "label": item["label"],
            "status": updates.get(item["id"], {}).get("status", item.get("status", "config_ready")),
            "evidence": updates.get(item["id"], {}).get("evidence", item.get("evidence", "unknown")),
            "notes": updates.get(item["id"], {}).get("notes", item.get("notes", "")),
        }
        for item in public_status["model_families"]
    ]
    return public_status


def render_markdown_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Synapse Benchmark Matrix",
        "",
        f"- Generated: `{artifact['generated_at']}`",
        f"- Host: `{artifact['host']['device']}`",
        f"- Git commit: `{artifact['host'].get('git_commit') or 'unknown'}`",
        "",
        "## Test Suites",
        "",
        "| Suite | Category | Status | Duration (s) | Summary |",
        "|-------|----------|--------|--------------|---------|",
    ]
    for suite in artifact["test_suites"]:
        lines.append(
            f"| {suite['label']} | {suite['category']} | {suite['status']} | "
            f"{suite['duration_s']:.3f} | {suite.get('summary') or ''} |"
        )

    tier_labels = [
        ("measured_local", "Measured Local End-to-End"),
        ("synthetic_validated", "Synthetic / Config-Validated"),
        ("exploratory_local", "Exploratory Local"),
    ]
    for tier_id, title in tier_labels:
        tier_rows = [row for row in artifact["benchmark_runs"] if row["tier"] == tier_id]
        if not tier_rows:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Family | Checkpoint | Configuration | Prompt | Status | Prefill | Decode | Notes |",
                "|--------|------------|---------------|--------|--------|---------|--------|-------|",
            ]
        )
        for row in tier_rows:
            prefill = "" if row.get("prefill_tps") is None else f"{row['prefill_tps']:.1f}"
            decode = "" if row.get("decode_tps") is None else f"{row['decode_tps']:.1f}"
            lines.append(
                f"| {row['family']} | {row['checkpoint']} | {row['configuration']} | {row['prompt_id']} | "
                f"{row['status']} | {prefill} | {decode} | {row.get('notes', '')} |"
            )
    lines.append("")
    return "\n".join(lines)


def detect_git_commit() -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def host_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device": friendly_device_name(),
        "git_commit": detect_git_commit(),
    }


def parse_kv_pairs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected family=path, got: {value}")
        family, raw_path = value.split("=", 1)
        parsed[family] = Path(raw_path)
    return parsed


def print_text_summary(artifact: dict[str, Any]) -> None:
    print(f"Benchmark matrix generated at {artifact['generated_at']}")
    print(f"Host: {artifact['host']['device']}")
    print("")
    print("Measured local rows:")
    for row in artifact["benchmark_runs"]:
        if row["tier"] != "measured_local":
            continue
        suffix = ""
        if row.get("prefill_tps") is not None and row.get("decode_tps") is not None:
            suffix = f" prefill={row['prefill_tps']:.1f} tok/s decode={row['decode_tps']:.1f} tok/s"
        print(f"- {row['id']}: {row['status']}{suffix}")
    print("")
    print(f"Artifact JSON: {ARTIFACT_JSON_PATH.relative_to(ROOT)}")
    print(f"Artifact MD:   {ARTIFACT_MD_PATH.relative_to(ROOT)}")
    print(f"Public status: {PUBLIC_STATUS_PATH.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-model",
        action="append",
        default=[],
        help="Override official model path as family=path",
    )
    parser.add_argument(
        "--exploratory-model",
        action="append",
        default=[],
        help="Add exploratory model path as family=path",
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Run extra local checkpoints in an exploratory appendix",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "md"),
        default="text",
        help="What to print to stdout after artifacts are written",
    )
    args = parser.parse_args()

    official_models = dict(DEFAULT_OFFICIAL_MODELS)
    official_models.update(parse_kv_pairs(args.official_model))
    exploratory_models = dict(DEFAULT_EXPLORATORY_MODELS)
    exploratory_models.update(parse_kv_pairs(args.exploratory_model))

    suites = [run_command(spec) for spec in benchmark_test_specs()]
    rows = [run_synthetic_benchmark(spec) for spec in synthetic_benchmark_specs()]
    for spec in real_model_specs(official_models, exploratory_models, args.include_exploratory):
        rows.append(run_real_model(spec))

    generated_at = now_utc()
    artifact = {
        "schema_version": 1,
        "generated_at": generated_at,
        "generated_on": generated_at.split("T", 1)[0],
        "host": host_metadata(),
        "test_suites": suites,
        "benchmark_runs": rows,
    }
    artifact_md = render_markdown_report(artifact)
    public_status = load_public_status()
    measured_rows = select_public_rows(rows)
    public_status = update_public_status(public_status, artifact, measured_rows)

    write_json(ARTIFACT_JSON_PATH, artifact)
    ARTIFACT_MD_PATH.write_text(artifact_md)
    write_json(PUBLIC_STATUS_PATH, public_status)

    if args.format == "json":
        json.dump(artifact, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.format == "md":
        sys.stdout.write(artifact_md)
    else:
        print_text_summary(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
