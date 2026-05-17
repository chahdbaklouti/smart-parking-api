import cv2
import json
import time
import csv
import os
import numpy as np
from ultralytics import YOLO
from config import get_paths

# ─── Models to benchmark ──────────────────────────────────
BENCHMARK_MODELS = {
    "VisDrone-YOLO (ours)": r"models/visdrone-best.pt",
    "YOLOv11n (COCO)"     : "yolo11n.pt",
    "YOLOv11s (COCO)"     : "yolo11s.pt",
}

CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ──────────────────────────────────────────────────────────
def load_spots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [np.array(s["points"], dtype=np.int32) for s in data]

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def load_manual_ground_truth(gt_path):
    with open(gt_path, "r") as f:
        data = json.load(f)
    return [set(item["occupied_spots"]) for item in data]

def load_gt_frames(parking_id):
    folder = f"ground_truth/{parking_id}"
    frames, i = [], 0
    while True:
        path = f"{folder}/frame_{i}.jpg"
        if not os.path.exists(path):
            break
        frames.append(cv2.imread(path))
        i += 1
    print(f"  ✅ Loaded {len(frames)} annotated frames")
    return frames

# ──────────────────────────────────────────────────────────
def compute_metrics(predicted_list, ground_truth_list):
    if not predicted_list:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    precisions, recalls, f1s = [], [], []
    for pred, gt in zip(predicted_list, ground_truth_list):
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r / (p+r)  if (p + r)   > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "precision": round(np.mean(precisions) * 100, 1),
        "recall"   : round(np.mean(recalls)    * 100, 1),
        "f1_score" : round(np.mean(f1s)        * 100, 1),
    }

def compute_confusion_matrix(predicted_list, ground_truth_list, total_spots):
    TP = FP = FN = TN = 0
    if not predicted_list:
        for gt in ground_truth_list:
            FN += len(gt)
            TN += total_spots - len(gt)
        return TP, FP, FN, TN
    for pred, gt in zip(predicted_list, ground_truth_list):
        for spot_id in range(total_spots):
            p_occ = spot_id in pred
            g_occ = spot_id in gt
            if     p_occ and     g_occ: TP += 1
            elif   p_occ and not g_occ: FP += 1
            elif not p_occ and   g_occ: FN += 1
            else:                        TN += 1
    return TP, FP, FN, TN

def compute_stability(predicted_list):
    if not predicted_list:
        return 0.0
    counts = [len(p) for p in predicted_list]
    mean   = np.mean(counts)
    if mean < 1:
        return 0.0
    cv        = np.std(counts) / mean
    stability = max(0.0, 1.0 - cv) * 100
    return round(stability, 1)

def compute_flip_rate(predicted_list, total_spots):
    """
    Flip rate: percentage of (spot, frame) pairs where
    the spot changed state from the previous frame.
    Lower flip rate = more stable detection.
    Stability = 1 - flip_rate.
    """
    if not predicted_list or len(predicted_list) < 2:
        return 0.0

    flips = 0
    total = 0

    for i in range(1, len(predicted_list)):
        prev = predicted_list[i-1]
        curr = predicted_list[i]
        for s in range(total_spots):
            if (s in prev) != (s in curr):
                flips += 1
            total += 1

    if total == 0:
        return 0.0

    flip_stability = round((1 - flips / total) * 100, 1)
    return flip_stability

def compute_iou_box_polygon(box, polygon, image_shape):
    """
    Compute IoU between a bounding box and a polygon using masks.
    box: (x1, y1, x2, y2)
    polygon: np.array of points
    """
    h, w = image_shape[:2]

    # Create empty masks
    mask_box = np.zeros((h, w), dtype=np.uint8)
    mask_poly = np.zeros((h, w), dtype=np.uint8)

    # Draw box
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)

    # Draw polygon
    cv2.fillPoly(mask_poly, [polygon], 1)

    # Intersection and union
    intersection = np.logical_and(mask_box, mask_poly).sum()
    union        = np.logical_or(mask_box, mask_poly).sum()

    if union == 0:
        return 0.0

    return intersection / union

def print_error_analysis(results, num_frames, total_spots):
    print(f"\n  [D] Error Analysis")
    print(f"  {'─'*90}")

    for r in results:
        if r["detection_rate_%"] == 0:
            continue  # skip domain mismatch models

        total = r["TP"] + r["TN"] + r["FP"] + r["FN"]
        fp_rate = round(r["FP"] / (r["FP"] + r["TN"]) * 100, 1) if (r["FP"] + r["TN"]) > 0 else 0
        fn_rate = round(r["FN"] / (r["FN"] + r["TP"]) * 100, 1) if (r["FN"] + r["TP"]) > 0 else 0

        print(f"\n  Model: {r['model']}")
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  False Positives (FP): {r['FP']:>4}  → FP Rate: {fp_rate:>5.1f}%   │")
        print(f"  │  False Negatives (FN): {r['FN']:>4}  → FN Rate: {fn_rate:>5.1f}%   │")
        print(f"  │  True Positives  (TP): {r['TP']:>4}                       │")
        print(f"  │  True Negatives  (TN): {r['TN']:>4}                       │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"  Interpretation:")

        if r["FP"] > r["FN"]:
            print(f"  → Model over-detects: marks free spots as occupied.")
            print(f"    Likely cause: overlapping bounding boxes, shadows,")
            print(f"    or IoU threshold too low (currently 0.2).")
            print(f"    Fix: increase IoU threshold or add NMS post-processing.")
        elif r["FN"] > r["FP"]:
            print(f"  → Model under-detects: misses occupied spots.")
            print(f"    Likely cause: small/occluded vehicles from aerial view,")
            print(f"    low confidence detections filtered out.")
            print(f"    Fix: lower confidence threshold or use larger model.")
        else:
            print(f"  → Balanced error distribution. Model performs consistently.")


# ──────────────────────────────────────────────────────────
def run_model_benchmark(model_name, model_path, frames, spots, ground_truth):

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  ❌ Could not load {model_name}: {e}")
        return None

    inference_times = []
    predicted_list  = []
    confidences     = []

    for frame in frames:
        t0      = time.time()
        results = model(frame, verbose=False)[0]
        inference_times.append((time.time() - t0) * 1000)

        occupied = set()
        for box in results.boxes:
            if int(box.cls[0]) not in CAR_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # center (used only for fast filtering)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            confidences.append(conf)

            for i, spot in enumerate(spots):

                # STEP 1 — FAST PREFILTER
                if cx < np.min(spot[:,0]) or cx > np.max(spot[:,0]) or \
                cy < np.min(spot[:,1]) or cy > np.max(spot[:,1]):
                    continue

                # STEP 2 — IoU computation
                iou = compute_iou_box_polygon(
                    (x1, y1, x2, y2),
                    spot,
                    frame.shape
                )

                # STEP 3 — assignment
                if iou > 0.2: 
                    occupied.add(i)

        predicted_list.append(occupied)

    # ─── Metrics ──────────────────────────────────────────
    avg_ms    = round(np.mean(inference_times), 1)
    fps       = round(1000 / avg_ms, 1)
    avg_conf = round(np.mean(confidences) * 100, 1) if len(confidences) > 0 else 0.0
    avg_occ   = round(np.mean([len(p) for p in predicted_list]), 1)
    metrics   = compute_metrics(predicted_list, ground_truth)
    stability = compute_stability(predicted_list)
    temp_stability = compute_flip_rate(predicted_list, len(spots))
    TP, FP, FN, TN = compute_confusion_matrix(
        predicted_list, ground_truth, len(spots)
    )

    # Detection rate — % of frames where at least one car found
    det_rate = round(
        sum(1 for p in predicted_list if len(p) > 0) / len(frames) * 100, 1
    )

    return {
        "model"           : model_name,
        "avg_ms"          : avg_ms,
        "fps"             : fps,
        "avg_confidence_%" : avg_conf,
        "avg_cars_detected": avg_occ,
        "detection_rate_%" : det_rate,
        "precision_%"     : metrics["precision"],
        "recall_%"        : metrics["recall"],
        "f1_score_%"      : metrics["f1_score"],
        "stability_%"     : stability,
        "flip_stability_%": temp_stability,
        "TP"              : TP,
        "FP"              : FP,
        "FN"              : FN,
        "TN"              : TN,
    }

# ──────────────────────────────────────────────────────────
def print_results(results, total_spots, num_frames):
    W = 90
    print(f"\n  {'═'*W}")
    print(f"  {'MODEL EVALUATION RESULTS':^{W}}")
    print(f"  {'═'*W}")

    # ── [A] Speed ────────────────────────────────────────
    print(f"\n  [A] Speed & Detection")
    print(f"  {'─'*W}")
    print(f"  {'Model':<25} {'Latency':>9} {'FPS':>6} {'Avg Cars':>9} "
          f"{'Det.Rate':>10} {'Confidence':>12}")
    print(f"  {'─'*W}")
    best_fps = max(r["fps"] for r in results)
    for r in results:
        tag     = " ⚡" if r["fps"] == best_fps else ""
        avg_cars = f"{r['avg_cars_detected']:.1f}" if not np.isnan(r['avg_cars_detected']) else "N/A"
        print(f"  {r['model']:<25} "
              f"{r['avg_ms']:>8.1f}ms "
              f"{r['fps']:>5.1f}{tag:<3} "
              f"{avg_cars:>8} "
              f"{r['detection_rate_%']:>9.1f}% "
              f"{r['avg_confidence_%']:>10.1f}%")

    # ── [B] Accuracy ─────────────────────────────────────
    print(f"\n  [B] Accuracy Metrics (vs Manual Ground Truth — {num_frames} frames)")
    print(f"  {'─'*W}")
    print(f"  {'Model':<25} {'Precision':>10} {'Recall':>8} "
          f"{'F1-Score':>10} {'Stability':>11} {'Flip rate':>11}")
    print(f"  {'─'*W}")
    best_f1 = max(r["f1_score_%"] for r in results)
    for r in results:
        tag  = " 🏆" if r["f1_score_%"] == best_f1 else ""
        note = "  ← domain mismatch" if r["detection_rate_%"] == 0 else ""
        print(f"  {r['model']:<25} "
              f"{r['precision_%']:>9.1f}% "
              f"{r['recall_%']:>7.1f}% "
              f"{r['f1_score_%']:>9.1f}%{tag:<3} "
              f"{r['stability_%']:>9.1f}% "
              f"{r['flip_stability_%']:>9.1f}%"
              f"{note}")

    # ── [C] Confusion Matrix ──────────────────────────────
    print(f"\n  [C] Confusion Matrix "
          f"({total_spots} spots × {num_frames} frames = "
          f"{total_spots * num_frames} total classifications)")
    print(f"  {'─'*W}")
    print(f"  {'Model':<25} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}   "
          f"{'Accuracy':>10}")
    print(f"  {'─'*W}")
    for r in results:
        total = r["TP"] + r["TN"] + r["FP"] + r["FN"]
        acc   = round((r["TP"] + r["TN"]) / total * 100, 1) if total > 0 else 0.0
        print(f"  {r['model']:<25} "
              f"{r['TP']:>6} {r['TN']:>6} {r['FP']:>6} {r['FN']:>6}   "
              f"{acc:>9.1f}%")
    
    print_error_analysis(results, num_frames, total_spots)

    print(f"\n  {'─'*W}")
    print(f"  🏆 = Best F1   ⚡ = Fastest")

    low = [r["model"] for r in results if r["detection_rate_%"] == 0]
    if low:
        print(f"\n  ⚠️  Domain Mismatch:")
        print(f"  {', '.join(low)} are trained on street-level COCO images.")
        print(f"  Zero detections on aerial footage confirms that")
        print(f"  domain-specific training (VisDrone) is essential.")


def run_benchmark(parking_id):
    paths = get_paths(parking_id)

    print(f"\n{'═'*60}")
    print(f"  📊 BENCHMARKING — {parking_id.replace('_',' ').title()}")
    print(f"{'═'*60}")

    if not os.path.exists(paths["video"]):
        print(f"  ❌ Video not found"); return
    if not os.path.exists(paths["json"]):
        print(f"  ❌ Annotation not found — run main.py first"); return

    spots        = load_spots(paths["json"])
    gt_path      = f"ground_truth/{parking_id}/gt.json"

    if not os.path.exists(gt_path):
        print(f"  ❌ No ground truth — run prepare_gt.py first"); return

    ground_truth = load_manual_ground_truth(gt_path)
    frames       = load_gt_frames(parking_id)

    if len(frames) != len(ground_truth):
        print(f"  ❌ Mismatch: {len(frames)} frames vs {len(ground_truth)} GT entries")
        return

    print(f"  📐 {len(spots)} parking spots  |  {len(frames)} annotated frames")

    results = []
    for name, path in BENCHMARK_MODELS.items():
        r = run_model_benchmark(name, path, frames, spots, ground_truth)
        if r:
            results.append(r)

    if not results:
        print("  ❌ No models could be tested"); return

    # ── Print once ────────────────────────────────────────
    print_results(results, len(spots), len(ground_truth))

    # ── Save CSV ──────────────────────────────────────────
    os.makedirs("output/benchmark", exist_ok=True)
    bench_path = f"output/benchmark/{parking_id}_benchmark.csv"
    save_keys  = [k for k in results[0].keys()]
    with open(bench_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=save_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  💾 Saved → {bench_path}")

    best = max(results, key=lambda r: r["f1_score_%"])
    print(f"\n  ✅ Recommended model : {best['model']}")
    print(f"     Precision         : {best['precision_%']}%")
    print(f"     Recall            : {best['recall_%']}%")
    print(f"     F1-Score          : {best['f1_score_%']}%")
    print(f"     Stability         : {best['stability_%']}%")