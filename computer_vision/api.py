# api.py
import cv2
import time
import os
from database import get_connection
from flask import Flask, Response, jsonify
from flask_cors import CORS

from state import state
from config import INPUT_DIR, ANNOTATIONS_DIR

app = Flask(__name__)
CORS(app)


# ───────────────────────────────
# VIDEO STREAM
# ───────────────────────────────
def _frames(parking_id):
    while True:
        data = state.get(parking_id)

        if not data:
            time.sleep(0.05)
            continue

        frame = data["frame"]

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")

        time.sleep(0.033)


@app.route("/video/<parking_id>")
def video(parking_id):
    return Response(_frames(parking_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ───────────────────────────────
# STATS
# ───────────────────────────────
@app.route("/api/stats/<parking_id>")
def stats_one(parking_id):
    data = state.get(parking_id)

    if not data:
        return jsonify({"error": "not found"}), 404

    return jsonify(data["stats"])


@app.route("/api/stats")
def stats_all():
    stats = state.get_all_stats()

    total_occ = sum(s["occupied"] for s in stats)
    total_av = sum(s["available"] for s in stats)
    total = total_occ + total_av

    return jsonify({
        "parkings": stats,
        "global": {
            "total_occupied": total_occ,
            "total_available": total_av,
            "total_spots": total,
            "occupancy_rate": round(total_occ / total * 100, 1) if total else 0
        }
    })


# ───────────────────────────────
# CAMERAS
# ───────────────────────────────
@app.route("/api/cameras")
def cameras():
    from config import VIDEO_SOURCES, ANNOTATIONS_DIR
    result = []

    # Caméras depuis VIDEO_SOURCES (fichiers locaux + live YouTube)
    if VIDEO_SOURCES:
        for pid, src in VIDEO_SOURCES.items():
            result.append({
                "id": pid,
                "name": pid.replace("_", " ").title(),
                "hasAnnotation": os.path.exists(
                    os.path.join(ANNOTATIONS_DIR, f"{pid}.json")
                )
            })
    else:
        # Fallback original : scan des fichiers .mp4
        if os.path.exists(INPUT_DIR):
            for f in os.listdir(INPUT_DIR):
                if f.endswith(".mp4"):
                    pid = f.replace(".mp4", "")
                    result.append({
                        "id": pid,
                        "name": pid.replace("_", " ").title(),
                        "hasAnnotation": os.path.exists(
                            os.path.join(ANNOTATIONS_DIR, f"{pid}.json")
                        )
                    })

    return jsonify({"cameras": result, "total": len(result)})

@app.route("/api/history/<parking_id>")
def history(parking_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM occupancy_history
        WHERE parking_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (parking_id,))

    row = cur.fetchone()

    columns = [desc[0] for desc in cur.description]

    cur.close()
    conn.close()

    if not row:
        return {"error": "not found"}, 404

    return dict(zip(columns, row))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)