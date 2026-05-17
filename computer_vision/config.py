import os

# ─── Paths ────────────────────────────────────────────────
MODEL_PATH      = "models/visdrone-best.pt"
INPUT_DIR       = "input"
FRAMES_DIR      = "output/frames"
ANNOTATIONS_DIR = "output/annotations"
RESULTS_DIR     = "output/results"

# ─── Detection settings ───────────────────────────────────
CONFIDENCE      = 0.4
IOU_THRESHOLD   = 0.5
SHOW_PREVIEW    = False

VEHICLE_CLASSES = [2, 3, 5, 7]
DETECT_EVERY_N_FRAMES = 1
OCCUPANCY_METHOD = "points"
DRAW_SPOT_FILL = False

# ─── Spot annotation ──────────────────────────────────────
SPOT_MODEL_PATH = "models/bestbest.pt"
SPOT_CONF_CLOSE = 0.03
SPOT_CONF_FAR   = 0.0115
SPOT_NMS_IOU    = 0.1
SPOT_MAX_PROPOSALS = 220
SPOT_MIN_PROPOSALS = 60
SPOT_FAR_UPSCALE = 1
SPOT_IMGSZ_CLOSE = 960
SPOT_IMGSZ_FAR   = 1280

# ─── Sources vidéo ────────────────────────────────────────
#
#  Chaque clé = parking_id utilisé dans get_paths()
#  Valeur = chemin fichier local  OU  URL YouTube live
#
#  Exemples :
#   "parking1": "input/parking1.mp4"          ← fichier local (comportement original)
#   "parking2": "https://www.youtube.com/watch?v=LIVE_ID"   ← live YouTube
#
VIDEO_SOURCES = {
    #"parking1": "input/3551731235-preview.mp4",
    "parking2": "https://www.youtube.com/watch?v=UoqI-CdDGIs",
}

# ─── Auto-create folders ──────────────────────────────────
for folder in [INPUT_DIR, FRAMES_DIR, ANNOTATIONS_DIR, RESULTS_DIR, "models"]:
    os.makedirs(folder, exist_ok=True)

def get_paths(parking_id):
    return {
        "video"  : os.path.join(INPUT_DIR,       f"{parking_id}.mp4"),
        "frame"  : os.path.join(FRAMES_DIR,      f"{parking_id}.jpg"),
        "json"   : os.path.join(ANNOTATIONS_DIR, f"{parking_id}.json"),
        "output" : os.path.join(RESULTS_DIR,     f"{parking_id}_output.avi"),
        "stats"  : os.path.join(RESULTS_DIR,     f"{parking_id}_stats.csv"),
    }

def get_source(parking_id: str) -> str:
    """
    Retourne la source vidéo pour un parking donné.
    Priorité : VIDEO_SOURCES → sinon fallback sur le fichier local dans INPUT_DIR.
    """
    if parking_id in VIDEO_SOURCES:
        return VIDEO_SOURCES[parking_id]
    # fallback original
    return os.path.join(INPUT_DIR, f"{parking_id}.mp4")