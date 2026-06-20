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
    "00_groundtruth",
    "01_original",
    "02_spatial",
    "03_frequency",
    "04_spatial_enhanced",
    "05_frequency_enhanced",
    "06_spatial_histeq",
    "07_frequency_histeq",
    "08_spatial_gamma",
    "09_frequency_gamma",
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
    scenario: str = "01_original"

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
        roi = PROJECT_ROOT / "output" / "04_roi_enhancement"
        scenario_map = {
            "00_groundtruth": PROJECT_ROOT / "output" / "01_preprocessing" / "groundtruth_normalized",
            "01_original": PROJECT_ROOT / "output" / "01_preprocessing" / "normalized",
            "02_spatial": PROJECT_ROOT / "output" / "02_spatial_filtering" / "gaussian",
            "03_frequency": PROJECT_ROOT / "output" / "03_frequency_filtering" / "butterworth_lpf",
            "04_spatial_enhanced": roi / "spatial" / "clahe",
            "05_frequency_enhanced": roi / "frequency" / "clahe",
            "06_spatial_histeq": roi / "spatial" / "histeq",
            "07_frequency_histeq": roi / "frequency" / "histeq",
            "08_spatial_gamma": roi / "spatial" / "gamma",
            "09_frequency_gamma": roi / "frequency" / "gamma",
        }
        return scenario_map[self.scenario]
