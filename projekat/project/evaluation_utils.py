import statistics
from typing import Dict, List

NUTRIENTS = ["energija", "masti", "ugljeni_hidrati", "proteini"]


def evaluate_predictions(rows: List[Dict[str, object]]):
    mae = {k: [] for k in NUTRIENTS}
    signed_err = {k: [] for k in NUTRIENTS}
    found = {k: 0 for k in NUTRIENTS}
    total = 0
    within_20 = {k: 0 for k in NUTRIENTS}
    for row in rows:
        gt = row.get("truth")
        pred = row.get("pred")
        if not gt or not pred:
            continue
        total += 1
        for k in NUTRIENTS:
            pv = pred.get(k)
            gv = gt.get(k)
            if pv is not None and gv is not None:
                found[k] += 1
                diff = float(pv) - float(gv)
                err = abs(diff)
                signed_err[k].append(diff)
                mae[k].append(err)
                if float(gv) != 0.0:
                    rel = err / abs(float(gv))
                    if rel <= 0.20:
                        within_20[k] += 1
    return {
        "samples_with_truth": total,
        "found_count": {k: found[k] for k in NUTRIENTS},
        "coverage": {k: (found[k] / total if total else 0.0) for k in NUTRIENTS},
        "mae": {k: (statistics.mean(mae[k]) if mae[k] else None) for k in NUTRIENTS},
        "sum_error_signed": {k: (sum(signed_err[k]) if signed_err[k] else None) for k in NUTRIENTS},
        "mean_error_signed": {k: (statistics.mean(signed_err[k]) if signed_err[k] else None) for k in NUTRIENTS},
        "avg_diff_from_sum_div_count": {
            k: ((sum(signed_err[k]) / found[k]) if found[k] and signed_err[k] else None) for k in NUTRIENTS
        },
        "accuracy_within_20_percent": {k: (within_20[k] / found[k] if found[k] else 0.0) for k in NUTRIENTS},
    }


def _round_floats_for_print(obj, ndigits: int = 2):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats_for_print(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats_for_print(v, ndigits) for v in obj]
    return obj


def find_correct_image_ids(rows, rel_tol=0.10):
    ok_ids = []
    for row in rows:
        image_id = row.get("image_id")
        pred = row.get("pred") or {}
        gt = row.get("truth") or {}
        if not image_id or not gt:
            continue
        all_ok = True
        for k in NUTRIENTS:
            pv = pred.get(k)
            gv = gt.get(k)
            if pv is None or gv is None:
                all_ok = False
                break
            gvf = float(gv)
            pvf = float(pv)
            if gvf == 0:
                if abs(pvf - gvf) > 1e-6:
                    all_ok = False
                    break
            else:
                if abs(pvf - gvf) / abs(gvf) > rel_tol:
                    all_ok = False
                    break
        if all_ok:
            ok_ids.append(str(image_id))
    return ok_ids
