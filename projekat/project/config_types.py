from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    project_dir: Path
    images_dir: Path
    labels_dir: Path
    truth_file: Path
    work_dir: Path
    epochs: int
    imgsz: int
    batch: int
    device: str
    grams: float
    conf: float
    run_name: str
    mode: str
    ocr_lang: str
    exclude_truth_from_train: bool
    split_seed: int
    detector_weights: Optional[Path]
