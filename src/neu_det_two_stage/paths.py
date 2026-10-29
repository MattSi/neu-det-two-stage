"""Project paths shared by command-line scripts.

The repository can be moved without changing source code. Selected checkpoints
in ``models/`` take precedence; until they are curated, scripts fall back to the
original checkpoints preserved under ``runs/training/``.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
TRAINING_RUNS_DIR = RUNS_DIR / "training"
RESULTS_DIR = PROJECT_ROOT / "results"


def result_split_name(split: str) -> str:
    """Map the on-disk ``val`` split to the paper-facing name."""

    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    return "validation" if split == "val" else "test"


def preferred_checkpoint(
    selected_relative: Path,
    run_relative: Path,
) -> Path:
    """Prefer a curated checkpoint, otherwise use the preserved training run."""

    selected = MODELS_DIR / selected_relative
    if selected.is_file():
        return selected
    return TRAINING_RUNS_DIR / run_relative


def require_files(*paths: Path) -> None:
    """Fail early with one readable message when required inputs are missing."""

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required file(s) not found:\n{formatted}")


def require_directories(*paths: Path) -> None:
    """Fail early with one readable message when required directories are missing."""

    missing = [path for path in paths if not path.is_dir()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required directory/directories not found:\n{formatted}")

