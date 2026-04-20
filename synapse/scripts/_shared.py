"""Shared utilities for synapse scripts."""

from __future__ import annotations

import importlib.util


def load_tokenizer_func(module_path: str, module_name: str, func_name: str):
    """Dynamically load a tokenizer function from a Python module.

    Args:
        module_path: Filesystem path to the .py file.
        module_name: Name to register the module under (for importlib).
        func_name: Name of the callable to extract from the loaded module.

    Returns:
        The callable (e.g. ``tokenize_fnv`` or ``ast_tokenize``).
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def format_label(text: str) -> str:
    """Convert underscore-separated identifiers to title case.

    Example: 'in_progress' -> 'In Progress'
    """
    return text.replace("_", " ").title()
