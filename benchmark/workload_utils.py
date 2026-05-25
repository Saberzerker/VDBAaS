"""Helpers for workload-specific collection and local storage resolution."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKLOADS_ROOT = PROJECT_ROOT / "benchmark" / "workloads"
LOCAL_WORKLOAD_DATA_ROOT = PROJECT_ROOT / "data" / "workloads"


def _workload_relative_parts(workload_dir: Path) -> tuple[str, ...]:
    workload_dir = workload_dir.resolve()
    try:
        relative = workload_dir.relative_to(WORKLOADS_ROOT.resolve())
        parts = relative.parts
    except ValueError:
        parts = (workload_dir.name,)
    return tuple(part for part in parts if part)


def derive_collection_name(workload_dir: Path, prefix: str = "hybrid_vdb") -> str:
    parts = _workload_relative_parts(workload_dir)
    slug = "_".join(part.lower().replace("-", "_") for part in parts)
    return f"{prefix}_{slug}" if slug else prefix


def derive_local_data_dirs(workload_dir: Path) -> tuple[Path, Path]:
    parts = _workload_relative_parts(workload_dir)
    root = LOCAL_WORKLOAD_DATA_ROOT.joinpath(*parts) if parts else LOCAL_WORKLOAD_DATA_ROOT
    return root / "permanent", root / "dynamic"


def resolve_workload_dir(
    *,
    manifest_file: str | None = None,
    corpus_jsonl: str | None = None,
    workload_dir: str | None = None,
) -> Path | None:
    if workload_dir:
        return Path(workload_dir).expanduser().resolve()
    if manifest_file:
        return Path(manifest_file).expanduser().resolve().parent
    if corpus_jsonl:
        return Path(corpus_jsonl).expanduser().resolve().parent
    return None
