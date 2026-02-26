import json
from typing import Dict, List


def print_evaluation_summary(results: List[Dict[str, object]], eval_all, correct_ids_20, correct_ids_3of4_20, round_fn):
    print("Zavrseno.")
    print("")
    print("Ukupne metrike (sve slike):")
    print(f"processed_images: {len(results)}")
    print(f"samples_with_truth: {eval_all.get('samples_with_truth', 0)}")
    metric_print_order = [
        "coverage",
        "mae",
        "found_count",
        "sum_error_signed",
        "mean_error_signed",
        "accuracy_within_20_percent",
    ]
    metric_labels = {
        "coverage": "coverage",
        "mae": "mae",
        "found_count": "found_count",
        "sum_error_signed": "sum_error_signed (zbir pred-truth)",
        "mean_error_signed": "mean_error_signed (pred - truth)",
        "accuracy_within_20_percent": "accuracy_within_20_percent",
    }
    for key in metric_print_order:
        if key in eval_all:
            print(f"{metric_labels.get(key, key)}:")
            print(json.dumps(round_fn(eval_all[key], 2), ensure_ascii=False, indent=2))

    ocr_source_counts: Dict[str, int] = {}
    for row in results:
        src = str(row.get("ocr_source") or "unknown")
        ocr_source_counts[src] = ocr_source_counts.get(src, 0) + 1
    print("ocr_source_counts:")
    print(json.dumps(round_fn(ocr_source_counts, 2), ensure_ascii=False, indent=2))
    print("")
    print(f"Broj slika tacno uradjenih (sva 4 nutrienta unutar +-20%): {len(correct_ids_20)}")
    print(f"Broj slika tacno uradjenih (3/4 nutrienta unutar +-20%): {len(correct_ids_3of4_20)}")
