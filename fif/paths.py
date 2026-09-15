"""Project paths and subject-level split definitions."""
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SHOW3D_ROOT = DATA_ROOT / "show3d"
FRAMES_ROOT = DATA_ROOT / "frames"
CACHE_ROOT = DATA_ROOT / "cache"
API_ROOT = PROJECT_ROOT / "third_party" / "SHOW3D-dataset-api"
TRAIN_MANIFEST = API_ROOT / "show3d" / "interaction_field" / "train_manifest_202607.jsonl"
TEST_MANIFEST = API_ROOT / "show3d" / "interaction_field" / "test_manifest_5fps_202607.jsonl"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# The 10 training subjects of the interaction-field task (train_manifest_202607.jsonl).
TRAIN_SUBJECTS: tuple[str, ...] = (
    "ASC023", "SPI102", "LWA828", "YZH016", "XXI103",
    "PCW023", "MHA016", "MMO925", "XYZ109", "LYA722",
)
# Official hidden-test subjects (labels withheld; never used for training).
TEST_SUBJECTS: tuple[str, ...] = ("BBL925", "KHE522", "SHE109")

# Subject-level folds. Fold A is the official dev split (also GOLF's).
FOLDS: dict[str, dict[str, tuple[str, ...]]] = {
    "A": {"val": ("XYZ109", "LYA722")},
    "B": {"val": ("MHA016", "MMO925")},
}
for _name, _f in FOLDS.items():
    _f["train"] = tuple(s for s in TRAIN_SUBJECTS if s not in _f["val"])


def fold_subjects(fold: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    f = FOLDS[fold]
    return f["train"], f["val"]
