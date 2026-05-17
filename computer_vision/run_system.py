from threading import Thread
import time

from main import main as run_cameras
from aggregate import HistoryAggregator
from server import start_server
import config


def start_aggregator():
    HistoryAggregator(
        input_pattern="output/results/*_history.csv",
        output_file="output/results/global_history.csv",
        refresh_interval=1800,
    ).run()


if __name__ == "__main__":
    print("SMART PARKING SYSTEM STARTING\n")

    # Affiche les sources configurées au démarrage
    for pid, src in config.VIDEO_SOURCES.items():
        kind = "LIVE YouTube" if src.startswith("http") else "fichier local"
        print(f"  [{pid}] {kind} → {src}")
    print()

    Thread(target=run_cameras, daemon=True).start()
    Thread(target=start_aggregator, daemon=True).start()

    print("API running on http://localhost:5000")
    start_server()   # blocking