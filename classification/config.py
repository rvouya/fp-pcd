"""Classification experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_thickening",
    "Pneumonia",
    "Pneumothorax",
]

TRAINING_SCENARIOS: List[str] = [
    "original",
    "spatial",
    "frequency",
    "spatial_enhanced",
    "frequency_enhanced",
]


@dataclass
class ClassificationConfig:
    """Hyperparameters and paths for a classification experiment."""

    # Model
    model_name: str = "resnet18"
    num_classes: int = 6
    pretrained: bool = True

    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5  # early stopping

    # Data
    image_size: Tuple[int, int] = (224, 224)
    val_split: float = 0.2
    num_workers: int = 4
    seed: int = 42

    # Scenario
    scenario: str = "original"

    # Paths
    output_dir: Path = PROJECT_ROOT / "output" / "05_classification"
    checkpoint_dir: Optional[Path] = None
    log_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / self.scenario / "checkpoints"
        if self.log_dir is None:
            self.log_dir = self.output_dir / self.scenario / "logs"

    def resolve_image_dir(self) -> Path:
        """Return source image directory for the configured scenario."""
        scenario_map = {
            "original": PROJECT_ROOT / "output" / "01_preprocessing" / "normalized",
            "spatial": PROJECT_ROOT / "output" / "02_spatial_filtering" / "gaussian",
            "frequency": PROJECT_ROOT / "output" / "03_frequency_filtering" / "butterworth_lpf",
            "spatial_enhanced": PROJECT_ROOT / "output" / "04_roi_enhancement" / "spatial" / "clahe",
            "frequency_enhanced": PROJECT_ROOT / "output" / "04_roi_enhancement" / "frequency" / "clahe",
        }
        return scenario_map[self.scenario]
