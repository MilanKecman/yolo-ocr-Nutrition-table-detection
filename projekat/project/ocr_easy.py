import threading
from typing import Dict, List, Optional, Tuple


_EASYOCR_INIT_LOCK = threading.Lock()
_EASYOCR_INFER_LOCK = threading.Lock()
_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr
    except Exception as exc:
        raise RuntimeError(
            "EasyOCR import nije uspeo (paket mozda nedostaje ili mu zavisnost puca pri importu). "
            f"Originalna greska: {type(exc).__name__}: {exc}"
        ) from exc

    with _EASYOCR_INIT_LOCK:
        if _EASYOCR_READER is not None:
            return _EASYOCR_READER
        last_exc: Optional[Exception] = None
        for langs in (["en"], ["en", "hr"], ["en", "cs"]):
            try:
                _EASYOCR_READER = easyocr.Reader(langs, gpu=False, verbose=False)
                return _EASYOCR_READER
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Ne mogu da inicijalizujem EasyOCR: {last_exc}")


def _preprocess_image_for_ocr_cv2(image_obj):
    from PIL import ImageOps, Image

    base = image_obj.convert("RGB")
    base = base.resize(
        (max(1, int(base.width * 2.0)), max(1, int(base.height * 2.0))),
        Image.Resampling.LANCZOS,
    )
    try:
        import cv2
        import numpy as np

        arr = np.array(base)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        bw_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            12,
        )
        bw_inv = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        bw = cv2.bitwise_not(bw_inv)
        return Image.fromarray(bw)
    except Exception:
        return ImageOps.grayscale(base)


def _easy_quad_to_rect(pts: List[List[float]]):
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1 = int(max(0.0, min(xs)))
    y1 = int(max(0.0, min(ys)))
    x2 = int(max(xs))
    y2 = int(max(ys))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _easyocr_lines(image_obj):
    import numpy as np

    reader = _get_easyocr_reader()
    prep = _preprocess_image_for_ocr_cv2(image_obj)
    arr = np.array(prep.convert("RGB"))
    out: List[Tuple[List[List[float]], str, float]] = []

    with _EASYOCR_INFER_LOCK:
        res = reader.readtext(arr, detail=1, paragraph=False)
    for item in res or []:
        try:
            box, txt, conf = item[0], str(item[1] or "").strip(), float(item[2] or 0.0)
        except Exception:
            continue
        if not txt:
            continue
        try:
            pts = [[float(p[0]), float(p[1])] for p in box[:4]]
        except Exception:
            continue
        out.append((pts, txt, conf))
    if out:
        return out

    raw_arr = np.array(image_obj.convert("RGB"))
    with _EASYOCR_INFER_LOCK:
        res2 = reader.readtext(raw_arr, detail=1, paragraph=False)
    for item in res2 or []:
        try:
            box, txt, conf = item[0], str(item[1] or "").strip(), float(item[2] or 0.0)
        except Exception:
            continue
        if not txt:
            continue
        try:
            pts = [[float(p[0]), float(p[1])] for p in box[:4]]
        except Exception:
            continue
        out.append((pts, txt, conf))
    return out


def _easyocr_to_data_dict(image_obj):
    lines = _easyocr_lines(image_obj)
    data: Dict[str, List[object]] = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
    }
    rows: List[Tuple[int, int, int, int, str, float]] = []
    for pts, txt, conf in lines:
        x1, y1, x2, y2 = _easy_quad_to_rect(pts)
        rows.append((x1, y1, x2, y2, txt, conf))
    rows.sort(key=lambda r: (r[1], r[0]))

    text_lines: List[str] = []
    for idx, (x1, y1, x2, y2, txt, conf) in enumerate(rows, start=1):
        data["text"].append(txt)
        data["conf"].append(round(max(0.0, min(100.0, conf * 100.0)), 2))
        data["left"].append(x1)
        data["top"].append(y1)
        data["width"].append(max(1, x2 - x1))
        data["height"].append(max(1, y2 - y1))
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(idx)
        text_lines.append(txt)
    return data, "\n".join(text_lines)
