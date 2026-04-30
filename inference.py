"""Unified entrypoint for inference and evaluation workflows.

This file stays at the repository root so it can be used as the single command
you run for inference. The implementation for each workflow remains inside
inference/ as modular scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

WORKFLOWS = {
    "signal": "inference/run_cyclegan.py",
    "target-folders": "inference/run_target_model_folders.py",
    "compare": "inference/run_comparison.py",
    "all-checkpoints": "inference/run_all_checkpoints.py",
    "nifeadb-step": "inference/run_nifeadb_step_metrics.py",
}

DEFAULT_WORKFLOW = "target-folders"


def _print_help() -> None:
    print("Unified inference entrypoint\n")
    print("Usage:")
    print("  python inference.py [workflow] [workflow args]\n")
    print("Workflows:")
    for name, script in WORKFLOWS.items():
        print(f"  {name:<16} -> {script}")
    print("\nDefaults:")
    print(f"  If no workflow is given, {DEFAULT_WORKFLOW} is used.")
    print("\nExamples:")
    print("  python inference.py signal --input demo")
    print("  python inference.py target-folders --model-dirs models/sagan_1_New_Base --all-edf --include-arr")
    print("  python inference.py compare --db ADFECGDB --quick")
    print("  python inference.py all-checkpoints")
    print("  python inference.py nifeadb-step --step 113")


def _resolve_workflow(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return DEFAULT_WORKFLOW, []

    first = argv[0]
    if first in {"-h", "--help", "help"}:
        _print_help()
        raise SystemExit(0)

    if first in WORKFLOWS:
        return first, argv[1:]

    if first.startswith("-"):
        return DEFAULT_WORKFLOW, argv

    raise SystemExit(f"Unknown workflow: {first}\n\nRun `python inference.py --help` for available workflows.")


def _run_workflow(script_relative_path: str, workflow_args: list[str]) -> int:
    script_path = PROJECT_ROOT / script_relative_path
    if not script_path.is_file():
        raise FileNotFoundError(f"Workflow script not found: {script_path}")

    command = [sys.executable, str(script_path), *workflow_args]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        workflow = DEFAULT_WORKFLOW
        workflow_args: list[str] = []
    else:
        workflow, workflow_args = _resolve_workflow(args)

    return _run_workflow(WORKFLOWS[workflow], workflow_args)


if __name__ == "__main__":
    raise SystemExit(main())
