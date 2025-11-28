from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency
    import wandb as _wandb
except ImportError:  # pragma: no cover
    _wandb = None

WANDB_AVAILABLE = _wandb is not None
wandb = _wandb

__all__ = ["WANDB_AVAILABLE", "wandb", "args_to_config", "start_wandb_run", "log_artifact"]


def args_to_config(args: Any, drop_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Convert an argparse Namespace to a JSON-serializable dict for W&B configs."""
    drop = set(drop_keys or [])
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in drop:
            continue
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def start_wandb_run(settings: dict[str, Any], config: dict[str, Any] | None = None):
    """Create a W&B run using shared CLI flags."""
    mode = settings.get("mode", "disabled") or "disabled"
    if mode == "disabled":
        os.environ.setdefault("WANDB_DISABLED", "true")
        return None
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Run `pip install wandb` to enable logging.")
    project = settings.get("project")
    if not project:
        raise ValueError("Set --wandb-project when enabling W&B logging.")
    clean_config = {k: v for k, v in (config or {}).items() if v is not None}
    run = _wandb.init(
        project=project,
        entity=settings.get("entity"),
        name=settings.get("run_name"),
        group=settings.get("group"),
        tags=settings.get("tags"),
        notes=settings.get("notes"),
        mode=mode,
        config=clean_config,
    )
    return run


def log_artifact(run, name: str, artifact_type: str, path: str | Path, description: str | None = None):
    """Upload a file or directory as a W&B artifact."""
    if run is None or not WANDB_AVAILABLE:
        return
    art = _wandb.Artifact(name=name, type=artifact_type, description=description)
    path = Path(path)
    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))
    run.log_artifact(art)
