import cv2
import json
import csv
import os
import numpy as np
from config import get_paths

def load_spots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [np.array(spot["points"], dtype=np.int32) for spot in data]

def compute_rates_from_history(history_path, total_spots):
    """
    Compute per-spot occupancy rates from history CSV.
    Uses overall occupancy_rate averaged over time as a proxy
    since history doesn't store per-spot data individually.
    Falls back to uniform distribution based on average occupancy.
    """
    if not os.path.exists(history_path):
        print(f"  ⚠️  History CSV not found: {history_path}")
        return [0.0] * total_spots

    occupied_rates = []
    try:
        with open(history_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("occupancy_rate", "0")
                try:
                    occupied_rates.append(float(val))
                except ValueError:
                    continue
    except Exception as e:
        print(f"  ⚠️  Could not read history: {e}")
        return [0.0] * total_spots

    if not occupied_rates:
        return [0.0] * total_spots

    # Average occupancy rate across all recorded timestamps
    avg_rate = np.mean(occupied_rates)
    print(f"  📊 Average occupancy from history: {round(avg_rate * 100, 1)}%")
    return [avg_rate] * total_spots

def compute_rates_from_per_spot(per_spot_path, total_spots):
    """Load real per-spot occupancy rates from per_spot CSV."""
    if not os.path.exists(per_spot_path):
        return None  # signal to fall back

    rates = []
    try:
        with open(per_spot_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rates.append(float(row["occupancy_rate"]))
    except Exception as e:
        print(f"  ⚠️  Could not read per_spot CSV: {e}")
        return None

    if len(rates) < total_spots:
        rates += [0.0] * (total_spots - len(rates))

    return rates[:total_spots]

def compute_rates_from_ground_truth(gt_path, total_spots):
    """
    Compute per-spot occupancy rates from manually annotated GT.
    This gives the most accurate heatmap if GT exists.
    """
    if not os.path.exists(gt_path):
        return None

    try:
        with open(gt_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    spot_counts = [0] * total_spots
    num_frames  = len(data)

    if num_frames == 0:
        return None

    for item in data:
        for spot_id in item.get("occupied_spots", []):
            if spot_id < total_spots:
                spot_counts[spot_id] += 1

    rates = [round(count / num_frames, 4) for count in spot_counts]
    print(f"  📊 Rates computed from {num_frames} manually annotated frames")
    return rates

def get_color_for_rate(rate):
    """Green → Yellow → Red gradient based on occupancy rate."""
    if rate < 0.5:
        r = int(255 * (rate * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (rate - 0.5) * 2))
    return (0, g, r)  # BGR

def generate_heatmap(parking_id):
    paths      = get_paths(parking_id)
    frame_path = paths["frame"]
    json_path  = paths["json"]

    os.makedirs("output/heatmaps", exist_ok=True)

    # ─── Validate required files ──────────────────────────
    for path, name in [(frame_path, "Frame"), (json_path, "Annotation JSON")]:
        if not os.path.exists(path):
            print(f"  ❌ {name} not found: {path}")
            return

    frame = cv2.imread(frame_path)
    spots = load_spots(json_path)
    total = len(spots)

    # ─── Load rates — priority order ─────────────────────
    # 1. Ground truth (most accurate — from manual annotation)
    # 2. Per-spot CSV (from detection run)
    # 3. History CSV (fallback — uniform distribution)

    gt_path       = f"ground_truth/{parking_id}/gt.json"
    history_path  = paths["history"]

    rates = None
    source = ""

    rates = compute_rates_from_ground_truth(gt_path, total)
    if rates is not None:
        source = "manual ground truth annotation"
    else:
            rates = compute_rates_from_history(history_path, total)
            source = "overall history (uniform approximation)"

    print(f"  📍 Heatmap source: {source}")
    print(f"  📊 Computing heatmap for {total} spots...")

    # ─── Build overlay ────────────────────────────────────
    heatmap_layer = np.zeros_like(frame, dtype=np.uint8)
    for spot, rate in zip(spots, rates):
        color = get_color_for_rate(rate)
        cv2.fillPoly(heatmap_layer, [spot], color)

    overlay = cv2.addWeighted(frame, 0.45, heatmap_layer, 0.55, 0)

    # ─── Draw borders + % labels ──────────────────────────
    for i, (spot, rate) in enumerate(zip(spots, rates)):
        cv2.polylines(overlay, [spot], isClosed=True,
                      color=(255, 255, 255), thickness=1)
        cx = int(np.mean(spot[:, 0]))
        cy = int(np.mean(spot[:, 1]))
        label = f"{int(rate * 100)}%"
        cv2.putText(overlay, label, (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)

    # ─── Legend ───────────────────────────────────────────
    legend = [
        ((0, 255, 0),   "Low   (0-40%)"),
        ((0, 255, 255), "Medium (40-70%)"),
        ((0, 0, 255),   "High  (70-100%)"),
    ]
    for idx, (color, text) in enumerate(legend):
        y = 30 + idx * 28
        cv2.rectangle(overlay, (12, y-14), (30, y+4), color, -1)
        cv2.putText(overlay, text, (38, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # ─── Title + source watermark ─────────────────────────
    title = f"Occupancy Heatmap — {parking_id.replace('_', ' ').title()}"
    cv2.putText(overlay, title,
                (12, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.putText(overlay, f"Source: {source}",
                (12, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (200, 200, 200), 1)

    # ─── Pure heatmap (no frame background) ──────────────
    pure = np.zeros_like(frame)
    for spot, rate in zip(spots, rates):
        cv2.fillPoly(pure, [spot], get_color_for_rate(rate))

    # ─── Save ─────────────────────────────────────────────
    overlay_path = f"output/heatmaps/{parking_id}_heatmap_overlay.jpg"
    pure_path    = f"output/heatmaps/{parking_id}_heatmap_pure.jpg"
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(pure_path, pure)

    # ─── Summary ──────────────────────────────────────────
    high   = sum(1 for r in rates if r >= 0.7)
    medium = sum(1 for r in rates if 0.4 <= r < 0.7)
    low    = sum(1 for r in rates if r < 0.4)

    print(f"\n  {'─'*45}")
    print(f"  🔴 High occupancy spots   : {high:>3} ({round(high/total*100,1)}%)")
    print(f"  🟡 Medium occupancy spots : {medium:>3} ({round(medium/total*100,1)}%)")
    print(f"  🟢 Low occupancy spots    : {low:>3} ({round(low/total*100,1)}%)")
    print(f"  🖼️  Overlay → {overlay_path}")
    print(f"  🖼️  Pure    → {pure_path}")
    print(f"  {'─'*45}")