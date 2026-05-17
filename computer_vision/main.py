import os
from threading import Thread

from config import get_paths, get_source, VIDEO_SOURCES, INPUT_DIR
from annotate import setup_annotation
from detect import run_detection


def get_all_parking_ids():
    """
    Retourne tous les parking_id :
      - depuis VIDEO_SOURCES (config) en priorité
      - fallback : fichiers .mp4 dans INPUT_DIR (comportement original)
    """
    if VIDEO_SOURCES:
        return list(VIDEO_SOURCES.keys())

    # fallback original
    return [
        f.replace(".mp4", "")
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".mp4")
    ]


def process_parking(parking_id):
    print(f"\n{'═'*50}")
    print(f"  🅿️  {parking_id.replace('_', ' ').title()}")
    print(f"{'═'*50}")

    paths = get_paths(parking_id)
    source = get_source(parking_id)  # fichier local OU URL YouTube

    is_live = source.startswith("http")

    # Pour un fichier local, vérifie qu'il existe
    if not is_live and not os.path.exists(source):
        print(f"  ❌ Video not found: {source} → skipping")
        return

    if is_live:
        print(f"  📡 Source : LIVE YouTube → {source}")
    else:
        print(f"  📂 Source : fichier local → {source}")

    # Step 1 — Annotation (only once)
    # Pour un live, on passe la source directement à setup_annotation
    setup_annotation(paths, source=source if is_live else None)

    # Step 2 — Detection (continuous stream)
    run_detection(paths, source=source)


def main():
    print("╔══════════════════════════════════════════╗")
    print("║     🚗 SMART PARKING — MULTI CAMERA      ║")
    print("╚══════════════════════════════════════════╝")

    parking_ids = get_all_parking_ids()

    if not parking_ids:
        print("\n❌ No cameras configured (VIDEO_SOURCES vide et aucun .mp4 dans input/)")
        return

    print(f"\n🎥 {len(parking_ids)} caméra(s) détectée(s) : {parking_ids}")
    print("🚀 Démarrage en parallèle...\n")

    threads = []

    for pid in parking_ids:
        t = Thread(target=process_parking, args=(pid,))
        t.start()
        threads.append(t)

        source = get_source(pid)
        kind = "📡 LIVE" if source.startswith("http") else "📂 fichier"
        print(f"▶️  [{kind}] Caméra démarrée : {pid}")

    for t in threads:
        t.join()

    print("\n🎉 Toutes les caméras ont terminé.")


if __name__ == "__main__":
    main()