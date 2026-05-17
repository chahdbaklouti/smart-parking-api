import cv2
import csv
import time
import os
from datetime import datetime
from special_days import is_special_day

import threading
from ultralytics import YOLO
from config import (
    MODEL_PATH,
    SHOW_PREVIEW,
    CONFIDENCE,
    VEHICLE_CLASSES,
    DETECT_EVERY_N_FRAMES,
    OCCUPANCY_METHOD,
    DRAW_SPOT_FILL,
)

try:
    cv2.setNumThreads(0)
except Exception:
    pass
from camera import CameraStream
import json
import numpy as np
from state import state

_model_lock = threading.Lock()
_yolo_model: YOLO | None = None

COLOR_FREE     = (0, 255, 0)
COLOR_OCCUPIED = (0, 0, 255)
COLOR_BOX      = (255, 128, 0)
COLOR_TEXT_BG  = (40, 40, 40)
COLOR_TEXT     = (255, 255, 255)


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, total


def bbox_overlap(box, spot_bbox):
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = spot_bbox
    if x2 <= px1 or x1 >= px2 or y2 <= py1 or y1 >= py2:
        return False
    return True


def compute_iou_box_polygon(box, polygon, img_h, img_w):
    x1, y1, x2, y2 = map(int, box)
    poly = polygon.reshape(-1, 2)
    px1, py1 = int(poly[:, 0].min()), int(poly[:, 1].min())
    px2, py2 = int(poly[:, 0].max()), int(poly[:, 1].max())
    roi_x1 = max(0, min(x1, px1))
    roi_y1 = max(0, min(y1, py1))
    roi_x2 = min(img_w, max(x2, px2))
    roi_y2 = min(img_h, max(y2, py2))
    roi_h  = roi_y2 - roi_y1
    roi_w  = roi_x2 - roi_x1
    if roi_h <= 0 or roi_w <= 0:
        return 0.0
    mask_box  = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask_poly = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(mask_box,
                  (x1 - roi_x1, y1 - roi_y1),
                  (x2 - roi_x1, y2 - roi_y1), 1, -1)
    shifted_poly = poly.copy()
    shifted_poly[:, 0] -= roi_x1
    shifted_poly[:, 1] -= roi_y1
    cv2.fillPoly(mask_poly, [shifted_poly], 1)
    intersection = np.logical_and(mask_box, mask_poly).sum()
    union        = np.logical_or(mask_box, mask_poly).sum()
    return intersection / union if union > 0 else 0.0


def draw_visualization(frame, spots, occupied_set, boxes, occupied, available, total):
    if DRAW_SPOT_FILL:
        overlay = frame.copy()
        for i, spot in enumerate(spots):
            color = COLOR_OCCUPIED if i in occupied_set else COLOR_FREE
            cv2.fillPoly(overlay, [spot], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for i, spot in enumerate(spots):
        color = COLOR_OCCUPIED if i in occupied_set else COLOR_FREE
        cv2.polylines(frame, [spot], True, color, 2)
        centroid = spot.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(frame, str(i), (int(centroid[0]) - 8, int(centroid[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


    occupancy_pct = (occupied / total * 100) if total > 0 else 0
    stats_text = [
        f"Occupied: {occupied}",
        f"Available: {available}",
        f"Total: {total}",
        f"Occupancy: {occupancy_pct:.1f}%"
    ]
    cv2.rectangle(frame, (10, 10), (180, 110), COLOR_TEXT_BG, -1)
    cv2.rectangle(frame, (10, 10), (180, 110), COLOR_FREE, 2)
    for i, text in enumerate(stats_text):
        cv2.putText(frame, text, (20, 35 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

    bar_width, bar_height = 150, 12
    bar_x, bar_y = 15, 115
    filled_width = int(bar_width * (occupied / total)) if total > 0 else 0
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), COLOR_OCCUPIED, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

    return frame


def send_result(parking_id, frame, occupied, available, total):
    stats = {
        "parking_id"    : parking_id,
        "occupied"      : occupied,
        "available"     : available,
        "total"         : total,
        "occupancy_rate": round((occupied / total) * 100, 1) if total else 0,
        "timestamp"     : datetime.now().isoformat()
    }
    state.update(parking_id, frame, stats)


def _get_model() -> YOLO:
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            _yolo_model = YOLO(MODEL_PATH)
        return _yolo_model


def run_detection(paths, source=None):
    is_live = source is not None and (
        source.startswith("http://") or source.startswith("https://") or "youtube" in source
    )

    # ─── Camera abstraction ───────────────────────────────────────────────────
    camera = CameraStream(
        source if is_live else paths["video"],
        simulate_realtime=not is_live,
        loop=not is_live
    )

    # ─── Résolution : lire depuis la première vraie frame si live ─────────────
    if is_live:
        print("  📡 Détection de la résolution réelle du stream...")
        first_frame = None
        for _ in range(60):  # attend max ~60 tentatives
            ret, frame_probe = camera.read()
            if ret and frame_probe is not None:
                first_frame = frame_probe
                h, w = first_frame.shape[:2]
                fps, total_frames = 25, 0
                print(f"  📡 Résolution réelle : {w}x{h} @ {fps}fps")
                break
            time.sleep(0.5)
        else:
            w, h, fps, total_frames = 1280, 720, 25, 0
            first_frame = None
            print("  ⚠️  Résolution inconnue → défaut 1280x720")
    else:
        w, h, fps, total_frames = get_video_info(paths["video"])
        first_frame = None

    # ─── VideoWriter ─────────────────────────────────────────────────────────
    video_writer = cv2.VideoWriter(
        paths["output"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    yolo_model = _get_model()

    with open(paths["json"], "r") as f:
        data = json.load(f)

    spots = [np.array(s["points"], dtype=np.int32).reshape((-1, 1, 2)) for s in data]

    spot_bboxes = []
    for spot in spots:
        pts = spot.reshape(-1, 2)
        spot_bboxes.append((
            int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max())
        ))

    stats_log     = []
    start_time    = time.time()
    frame_count   = 0
    peak_occupancy = 0

    parking_id = paths.get("parking_id") or os.path.basename(paths["video"]).replace(".mp4", "")
    history_path = paths["stats"].replace("_stats.csv", "_history.csv")

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "parking_id", "free_spot_ids", "capacity",
            "free_spots", "occupancy_rate", "hour", "is_weekend", "is_special_day",
        ])

    if is_live:
        print(f"  📡 Live stream (boucle infinie — Ctrl+C pour arrêter)")
    else:
        print(f"\n  📹 Video: {w}x{h} @ {fps}fps — {total_frames} frames total")
        print(f"  🔁 Simulated camera stream (loop=True)")
    print(f"  ⏳ Processing...\n")

    last_boxes_xyxy        : list[tuple] = []
    last_occupied_spots_set: set[int]    = set()

    # ─── Injecter la première frame déjà lue (live) ───────────────────────────
    pending_frame = first_frame  # None pour les fichiers locaux

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────
    while True:

        # Utilise la première frame déjà lue, sinon lit la suivante
        if pending_frame is not None:
            im0           = pending_frame
            ret           = True
            pending_frame = None
        else:
            ret, im0 = camera.read()

        if not ret:
            if is_live:
                continue   # coupure temporaire → réessaie
            break

        # ── Resize si le stream change de résolution en cours de route ─────
        if im0.shape[1] != w or im0.shape[0] != h:
            w, h = im0.shape[1], im0.shape[0]
            video_writer.release()
            video_writer = cv2.VideoWriter(
                paths["output"],
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )

        do_infer = (DETECT_EVERY_N_FRAMES <= 1) or (frame_count % DETECT_EVERY_N_FRAMES == 0)

        if do_infer:
            with _model_lock:
                detections = yolo_model.predict(
                    source=im0,
                    conf=CONFIDENCE,
                    classes=VEHICLE_CLASSES,
                    verbose=False,
                )

            boxes_xyxy        = []
            occupied_spots_set: set[int] = set()

            if detections and len(detections) > 0 and detections[0].boxes is not None:
                for box in detections[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes_xyxy.append((x1, y1, x2, y2))

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    by = max(0.0, y2 - 2.0)

                    for i, spot_bbox in enumerate(spot_bboxes):
                        if not bbox_overlap((x1, y1, x2, y2), spot_bbox):
                            continue

                        if OCCUPANCY_METHOD in ("points", "hybrid"):
                            if (
                                cv2.pointPolygonTest(spots[i], (cx, cy), False) >= 0
                                or cv2.pointPolygonTest(spots[i], (cx, by), False) >= 0
                            ):
                                occupied_spots_set.add(i)
                                continue

                        if OCCUPANCY_METHOD == "hybrid":
                            iou = compute_iou_box_polygon((x1, y1, x2, y2), spots[i], h, w)
                            if iou > 0.2:
                                occupied_spots_set.add(i)

            last_boxes_xyxy         = boxes_xyxy
            last_occupied_spots_set = occupied_spots_set
        else:
            boxes_xyxy         = last_boxes_xyxy
            occupied_spots_set = last_occupied_spots_set

        all_spots     = set(range(len(spots)))
        free_spots    = sorted(list(all_spots - occupied_spots_set))
        occupied_spots = sorted(list(occupied_spots_set))

        occupied  = len(occupied_spots_set)
        available = len(spots) - occupied
        total     = len(spots)

        annotated_frame = draw_visualization(
            im0.copy(), spots, occupied_spots_set, boxes_xyxy,
            occupied, available, total
        )

        send_result(parking_id, annotated_frame, occupied, available, total)
        video_writer.write(annotated_frame)

        if occupied > peak_occupancy:
            peak_occupancy = occupied

        timestamp = round(frame_count / fps, 2)
        stats_log.append([timestamp, occupied, available, total])

        if frame_count % fps == 0:
            now           = datetime.now()
            occupancy_rate = occupied / total if total > 0 else 0
            with open(history_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    parking_id, str(free_spots), total,
                    available, round(occupancy_rate, 3),
                    now.hour, int(now.weekday() >= 5),
                    int(is_special_day(now)),
                ])

        frame_count += 1
        if frame_count % fps == 0:
            elapsed = time.time() - start_time
            if is_live:
                print(f"  ⏱ {elapsed:.0f}s  |  "
                      f"🔴 {occupied:3}  🟢 {available:3}  "
                      f"| 🎞 {frame_count} frames", end="\r")
            else:
                pct = (frame_count / total_frames) * 100 if total_frames else 0
                bar = ("█" * int(pct // 5)).ljust(20)
                print(f"  [{bar}] {pct:5.1f}%  |  "
                      f"🔴 {occupied:3}  🟢 {available:3}  "
                      f"| ⏱ {elapsed:.0f}s", end="\r")

        if SHOW_PREVIEW:
            cv2.imshow("Parking Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ─── Save stats CSV ───────────────────────────────────────────────────────
    stats_path = paths["stats"]
    try:
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_sec", "occupied", "available", "total_spots"])
            writer.writerows(stats_log)
    except PermissionError:
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = paths["stats"].replace(".csv", f"_{ts}.csv")
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_sec", "occupied", "available", "total_spots"])
            writer.writerows(stats_log)
        print(f"  ⚠️ File was locked → saved as {stats_path}")

    elapsed = time.time() - start_time
    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"\n\n  {'─'*45}")
    print(f"  ✅ Detection complete in {elapsed:.1f}s")
    print(f"  📊 Peak occupancy   : {peak_occupancy} / {total}")
    print(f"  🎥 Output video     : {paths['output']}")
    print(f"  📈 Stats CSV        : {paths['stats']}")
    print(f"  🗃️  History CSV     : {history_path}")
    print(f"  {'─'*45}")