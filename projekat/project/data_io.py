import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List

try:
    from .config_types import Config
except ImportError:
    from config_types import Config

NUTRIENTS = ["energija", "masti", "ugljeni_hidrati", "proteini"]


def parse_float(text: str):
    return float(text.strip().replace(",", "."))


def image_num(image_id: str):
    return int(re.sub(r"\D", "", image_id) or "0")


def parse_truth_line(line: str):
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^\s*(slika\d+|sliak\d+)\s*,\s*(.*)$", line, flags=re.IGNORECASE)
    if not m:
        return None
    image_id = m.group(1).lower().replace("sliak", "slika")
    rest = m.group(2)
    patterns = {
        "energija": r"energija\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
        "masti": r"masti\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
        "ugljeni_hidrati": r"ugljeni[-\s]*hidrati\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
        "proteini": r"proteini\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
    }
    values: Dict[str, float] = {}
    for k, p in patterns.items():
        pm = re.search(p, rest, flags=re.IGNORECASE)
        if not pm:
            return None
        values[k] = parse_float(pm.group(1))
    return image_id, values


def load_truth(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Ne postoji truth fajl: {path}")
    truth: Dict[str, Dict[str, float]] = {}
    bad = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parsed = parse_truth_line(line)
        if parsed is None:
            if line.strip():
                bad.append(f"Linija {i}: {line}")
            continue
        image_id, values = parsed
        truth[image_id] = values
    if bad:
        print("Upozorenje: neke truth linije nisu parsirane:")
        for row in bad:
            print(f"  - {row}")
    return truth


def save_truth_csv(truth: Dict[str, Dict[str, float]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + NUTRIENTS)
        for image_id in sorted(truth.keys(), key=image_num):
            w.writerow([image_id] + [truth[image_id][k] for k in NUTRIENTS])


def list_images(images_dir: Path):
    images = sorted(images_dir.glob("slika*.jpeg"), key=lambda p: image_num(p.stem))
    if not images:
        raise RuntimeError(f"Nema slika u: {images_dir}")
    return images


def split_ids(ids: List[str], train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    ids = ids.copy()
    random.Random(seed).shuffle(ids)
    n = len(ids)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    return {
        "train": ids[:train_n],
        "val": ids[train_n:train_n + val_n],
        "test": ids[train_n + val_n:],
    }


def prepare_detector_dataset(cfg: Config, truth: Dict[str, Dict[str, float]]):
    images = list_images(cfg.images_dir)
    print(f"Info: priprema detector skupa koristi {len(images)} slika iz {cfg.images_dir}.")
    ids = [p.stem.lower() for p in images]
    truth_ids = set(truth.keys())
    labeled_ids = [image_id for image_id in ids if (cfg.labels_dir / f"{image_id}.txt").exists()]
    excluded_truth_ids: List[str] = []
    if cfg.exclude_truth_from_train:
        excluded_truth_ids = [image_id for image_id in labeled_ids if image_id in truth_ids]
        labeled_ids = [image_id for image_id in labeled_ids if image_id not in truth_ids]
    unlabeled_ids = [image_id for image_id in ids if image_id not in set(labeled_ids)]
    if not labeled_ids:
        raise RuntimeError("Nema nijedne YOLO anotacije za trening. Dodaj labels ili ukljuci auto-label.")

    split = split_ids(labeled_ids, seed=cfg.split_seed)
    ds_root = cfg.work_dir / "detector_dataset"
    if ds_root.exists():
        shutil.rmtree(ds_root)
    for s in ["train", "val", "test"]:
        (ds_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    for s, image_ids in split.items():
        for image_id in image_ids:
            src_img = cfg.images_dir / f"{image_id}.jpeg"
            src_lbl = cfg.labels_dir / f"{image_id}.txt"
            if not src_img.exists():
                continue
            shutil.copy2(src_img, ds_root / "images" / s / src_img.name)
            shutil.copy2(src_lbl, ds_root / "labels" / s / src_lbl.name)

    (ds_root / "split_ids.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    if unlabeled_ids:
        missing_path = cfg.work_dir / "unlabeled_images.txt"
        missing_path.write_text("\n".join(unlabeled_ids), encoding="utf-8")
        print(
            f"Upozorenje: {len(unlabeled_ids)} slika nema labels i nisu ukljucene u detector trening. "
            f"Spisak: {missing_path}"
        )
    if excluded_truth_ids:
        excluded_path = cfg.work_dir / "excluded_from_train_truth_ids.txt"
        excluded_path.write_text("\n".join(excluded_truth_ids), encoding="utf-8")
        print(
            f"Info: {len(excluded_truth_ids)} labelovanih slika iz truth skupa je izbaceno iz YOLO treninga. "
            f"Spisak: {excluded_path}"
        )

    yaml_path = ds_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {ds_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: nutrition_table\n",
        encoding="utf-8",
    )
    summary = {k: len(v) for k, v in split.items()}
    summary["total_labeled"] = len(labeled_ids)
    summary["total_unlabeled"] = len(unlabeled_ids)
    summary["excluded_truth_from_train"] = len(excluded_truth_ids)
    print(f"Dataset spreman: {summary}")
    return yaml_path, summary
