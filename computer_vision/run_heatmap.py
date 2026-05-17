from heatmap import generate_heatmap
import os
from config import INPUT_DIR

def get_all_parking_ids():
    return [
        f.replace(".mp4", "")
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".mp4")
    ]

if __name__ == "__main__":
    parking_ids = get_all_parking_ids()
    print(f"🔥 Generating heatmaps for: {parking_ids}\n")
    for pid in parking_ids:
        print(f"  🅿️  {pid}")
        generate_heatmap(pid)
    print("\n✅ All heatmaps saved → output/heatmaps/")