"""annotate.py

Semi-automatic parking-spot annotation:
  1) Extract first frame from the video.
  2) Auto-propose spots via YOLO (fast pass + optional far-camera pass).
  3) Let the operator add/delete polygons in an OpenCV UI.
  4) Save polygons as JSON: {"id": i, "points": [[x,y], ...]}
"""

import cv2
import os
import json
import numpy as np
from ultralytics import YOLO

from config import (
    SPOT_MODEL_PATH,
    SPOT_CONF_CLOSE,
    SPOT_CONF_FAR,
    SPOT_NMS_IOU,
    SPOT_MAX_PROPOSALS,
    SPOT_MIN_PROPOSALS,
    SPOT_FAR_UPSCALE,
    SPOT_IMGSZ_CLOSE,
    SPOT_IMGSZ_FAR,
)

try:
    cv2.setNumThreads(0)
except Exception:
    pass

ui = {
    "boxes": [],
    "selected_idx": -1,
    "current_pts": [],
    "mouse_pos": None,
}

_WIN = "Calibration"


def _strict_non_max_suppression(boxes, scores, iou_thresh=0.15, iom_thresh=0.4):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    boxes = boxes.astype(float)
    while len(idxs) > 0:
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        iom = inter / (np.minimum(area_i, area_r) + 1e-6)
        cx_i = (boxes[i, 0] + boxes[i, 2]) / 2
        cy_i = (boxes[i, 1] + boxes[i, 3]) / 2
        cx_r = (boxes[rest, 0] + boxes[rest, 2]) / 2
        cy_r = (boxes[rest, 1] + boxes[rest, 3]) / 2
        dist = np.sqrt((cx_i - cx_r) ** 2 + (cy_i - cy_r) ** 2)
        min_dim_i = min(boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1])
        center_overlap = dist < (min_dim_i * 0.4)
        survivors = (iou < iou_thresh) & (iom < iom_thresh) & (~center_overlap)
        idxs = rest[survivors]
    return keep


def _adaptive_size_filter(boxes, scores):
    if len(boxes) < 5:
        return np.arange(len(boxes))
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    top_k = max(5, int(len(boxes) * 0.2))
    top_idxs = np.argsort(scores)[::-1][:top_k]
    med_w = float(np.median(widths[top_idxs]))
    med_h = float(np.median(heights[top_idxs]))
    keep = np.where(
        (widths >= med_w * 0.45) & (widths <= med_w * 1.8) &
        (heights >= med_h * 0.45) & (heights <= med_h * 1.8)
    )[0]
    return keep


def _predict_boxes(model, img, conf, imgsz):
    res = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    b = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    s = res[0].boxes.conf.cpu().numpy().astype(np.float32)
    return b, s


def _postprocess_spot_boxes(boxes, scores):
    if len(boxes) == 0:
        return boxes, scores
    size_keep = _adaptive_size_filter(boxes, scores)
    boxes = boxes[size_keep]
    scores = scores[size_keep]
    keep = _strict_non_max_suppression(boxes, scores, iou_thresh=SPOT_NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) > SPOT_MAX_PROPOSALS:
        top = np.argsort(scores)[::-1][:SPOT_MAX_PROPOSALS]
        boxes = boxes[top]
        scores = scores[top]
    return boxes, scores


def _auto_detect(frame, model):
    h, w = frame.shape[:2]
    boxes1, scores1 = _predict_boxes(model, frame, conf=SPOT_CONF_CLOSE, imgsz=SPOT_IMGSZ_CLOSE)
    boxes1, scores1 = _postprocess_spot_boxes(boxes1, scores1)
    need_far_pass = (len(boxes1) < SPOT_MIN_PROPOSALS)
    if need_far_pass:
        scale = float(SPOT_FAR_UPSCALE)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        boxes2, scores2 = _predict_boxes(model, resized, conf=SPOT_CONF_FAR, imgsz=SPOT_IMGSZ_FAR)
        if len(boxes2) > 0:
            boxes2 /= scale
        boxes2, scores2 = _postprocess_spot_boxes(boxes2, scores2)
        if len(boxes2) > 0:
            boxes = np.vstack([boxes1, boxes2]) if len(boxes1) else boxes2
            scores = np.concatenate([scores1, scores2]) if len(scores1) else scores2
            boxes, scores = _postprocess_spot_boxes(boxes, scores)
        else:
            boxes, scores = boxes1, scores1
    else:
        boxes, scores = boxes1, scores1
    polygons = [
        [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
        for (x1, y1, x2, y2) in boxes
    ]
    print(f"[INFO] Spot proposals: {len(polygons)} (close={len(boxes1)}{' + far-pass' if need_far_pass else ''})")
    return polygons


def find_box_at(x, y):
    for i, pts in enumerate(ui["boxes"]):
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            return i
    return -1


def on_mouse(event, x, y, flags, param):
    ui["mouse_pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if not ui["current_pts"]:
            hit = find_box_at(x, y)
            if hit >= 0:
                ui["selected_idx"] = hit
                print(f"Selected box {hit}")
            elif ui["selected_idx"] >= 0:
                ui["selected_idx"] = -1
                print("Deselected")
            else:
                ui["current_pts"].append([x, y])
        else:
            ui["current_pts"].append([x, y])
            if len(ui["current_pts"]) == 4:
                ui["boxes"].append(ui["current_pts"].copy())
                ui["selected_idx"] = len(ui["boxes"]) - 1
                ui["current_pts"] = []
    elif event == cv2.EVENT_RBUTTONDOWN:
        ui["current_pts"] = []


def render(frame):
    vis = frame.copy()
    for i, pts in enumerate(ui["boxes"]):
        color = (0, 165, 255) if i == ui["selected_idx"] else (0, 255, 0)
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly], True, color, 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(vis, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if ui["current_pts"]:
        for pt in ui["current_pts"]:
            cv2.circle(vis, tuple(pt), 5, (255, 100, 0), -1)
        for i in range(len(ui["current_pts"]) - 1):
            cv2.line(vis, tuple(ui["current_pts"][i]), tuple(ui["current_pts"][i+1]), (255, 100, 0), 2)
        if ui["mouse_pos"]:
            cv2.line(vis, tuple(ui["current_pts"][-1]), ui["mouse_pos"], (255, 100, 0), 1)
    instructions = [
        "LEFT CLICK = add/select", "RIGHT CLICK = cancel drawing",
        "DEL = delete box", "R = re-detect", "S = save", "ESC = quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(vis, text, (10, 25 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return vis


# ─── ONLY CHANGE: source=None — pour live, lit depuis le stream ──────────────
# annotate.py

def extract_frame(video_path, source=None):
    actual_source = source if source is not None else video_path

    # ── Si c'est une URL YouTube, résoudre l'URL directe du flux ──
    if actual_source.startswith("http://") or actual_source.startswith("https://") or "youtube" in actual_source:
        try:
            import yt_dlp
            ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(actual_source, download=False)
                actual_source = info["url"]  # URL directe lisible par OpenCV
        except Exception as e:
            raise RuntimeError(f"[annotate] Impossible de résoudre le stream YouTube: {e}")

    cap = cv2.VideoCapture(actual_source)
    ret, frame = cap.read()
    cap.release()
    return frame


def semi_auto_annotation(video_path, json_path, source=None):
    frame = extract_frame(video_path, source=source)
    if frame is None:
        raise RuntimeError(f"Could not read first frame from: {source or video_path}")

    print("Running auto detection (adaptive)...")
    model = YOLO(SPOT_MODEL_PATH)

    ui["boxes"] = _auto_detect(frame, model)
    ui["selected_idx"] = -1
    ui["current_pts"] = []
    ui["mouse_pos"] = None

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WIN, on_mouse)

    while True:
        cv2.imshow(_WIN, render(frame))
        key = cv2.waitKey(20) & 0xFF

        if key in (8, 127, 46):
            if ui["selected_idx"] >= 0:
                ui["boxes"].pop(ui["selected_idx"])
                ui["selected_idx"] = -1

        elif key == ord("s"):
            data = [{"id": i, "points": pts} for i, pts in enumerate(ui["boxes"])]
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved → {json_path} ({len(data)} spots)")
            break

        elif key == ord("r"):
            ui["boxes"] = _auto_detect(frame, model)
            ui["selected_idx"] = -1
            ui["current_pts"] = []
            print(f"Re-detected boxes: {len(ui['boxes'])}")

        elif key == 27:
            break

    cv2.destroyAllWindows()


# ─── ONLY CHANGE: source=None passé à semi_auto_annotation ──────────────────
def setup_annotation(paths, source=None):
    if os.path.exists(paths["json"]):
        print("⏭️ Already annotated → skipping")
        return

    semi_auto_annotation(paths["video"], paths["json"], source=source)