from pathlib import Path
from typing import Optional

try:
    from .config_types import Config
except ImportError:
    from config_types import Config


def train_detector(cfg: Config, dataset_yaml: Path):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Nedostaje ultralytics. Instaliraj: pip install ultralytics") from exc

    runs_dir = cfg.work_dir / "runs"
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(runs_dir),
        name=cfg.run_name,
        pretrained=True,
    )
    best = runs_dir / cfg.run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Trening zavrsen, ali best.pt nije pronadjen: {best}")
    return best


def load_detector_model(detector_path: Optional[Path]):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    if detector_path is None:
        return None
    return YOLO(str(detector_path))
