"""
Artifact storage: plain JSON files on disk, one artifact per file.

No database, no ORM -- an artifact is small, human-reviewable, and meant to
be diffable in git. `load_artifact` re-runs full Pydantic validation on
read, not just on write: a hand-edited or corrupted artifact file fails
loudly and specifically (which field, what's wrong) before the replay
engine (Phase 6) ever tries to act on it, rather than failing confusingly
mid-replay.
"""
from __future__ import annotations

from pathlib import Path

from artifacts.schema import Artifact

DEFAULT_ARTIFACT_DIR = Path("artifacts/saved")


def save_artifact(artifact: Artifact, directory: Path | str = DEFAULT_ARTIFACT_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact.artifact_id}.json"
    path.write_text(artifact.model_dump_json(indent=2))
    return path


def load_artifact(path: Path | str) -> Artifact:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No artifact file at {path}")
    return Artifact.model_validate_json(path.read_text())
