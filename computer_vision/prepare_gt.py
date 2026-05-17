import cv2
import json
import os
import numpy as np
from config import get_paths

N_FRAMES = 20

def extract_frames(parking_id):
    paths  = get_paths(parking_id)
    cap    = cv2.VideoCapture(paths["video"])
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, N_FRAMES, dtype=int)

    output_dir = f"ground_truth/{parking_id}"
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_dir}/frame_{i}.jpg", frame)
    cap.release()
    print(f"✅ {N_FRAMES} frames saved → {output_dir}")

def load_spots(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [np.array(s["points"], dtype=np.int32) for s in data]

def annotate_frames(parking_id):
    """
    Visual tool — click on spots to mark them occupied/free.
    GREEN = free, RED = occupied.
    Press S to save and go to next frame.
    Press Q to quit early.
    """
    paths      = get_paths(parking_id)
    spots      = load_spots(paths["json"])
    folder     = f"ground_truth/{parking_id}"
    gt_output  = f"{folder}/gt.json"

    # Load existing GT if resuming
    if os.path.exists(gt_output):
        with open(gt_output, "r") as f:
            all_gt = json.load(f)
        done_frames = {item["frame_id"] for item in all_gt}
        print(f"  ℹ️  Resuming — {len(done_frames)} frames already annotated")
    else:
        all_gt      = []
        done_frames = set()

    frame_idx = 0

    while True:
        frame_path = f"{folder}/frame_{frame_idx}.jpg"
        if not os.path.exists(frame_path):
            print(f"\n✅ All frames annotated!")
            break

        if frame_idx in done_frames:
            frame_idx += 1
            continue

        frame         = cv2.imread(frame_path)
        occupied_set  = set()

        def draw_frame(frm, spots, occupied):
            display = frm.copy()
            for i, spot in enumerate(spots):
                color = (0, 0, 255) if i in occupied else (0, 255, 0)
                cv2.fillPoly(display, [spot], color)
                # Spot number label
                cx = int(np.mean(spot[:, 0]))
                cy = int(np.mean(spot[:, 1]))
                cv2.putText(display, str(i), (cx-8, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)
            # Blend overlay
            result = cv2.addWeighted(frm, 0.5, display, 0.5, 0)
            # Instructions
            cv2.putText(result,
                f"Frame {frame_idx}/{N_FRAMES-1} — Click spot to toggle | S=Save | Q=Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
            cv2.putText(result,
                f"Occupied: {sorted(occupied)}",
                (10, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            return result

        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, spot in enumerate(spots):
                    if cv2.pointPolygonTest(spot, (x, y), False) >= 0:
                        if i in occupied_set:
                            occupied_set.discard(i)
                        else:
                            occupied_set.add(i)
                        break

        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", mouse_click)

        print(f"\n  Frame {frame_idx} — click spots to mark occupied (RED)")
        print(f"  Press S to save and continue | Q to quit")

        while True:
            display = draw_frame(frame, spots, occupied_set)
            cv2.imshow("Annotate", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                all_gt.append({
                    "frame_id"      : frame_idx,
                    "occupied_spots": sorted(list(occupied_set))
                })
                with open(gt_output, "w") as f:
                    json.dump(all_gt, f, indent=2)
                print(f"  ✅ Frame {frame_idx} saved — occupied: {sorted(occupied_set)}")
                frame_idx += 1
                break

            elif key == ord("q"):
                print(f"\n⏸️  Paused at frame {frame_idx}. Run again to resume.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    print(f"\n🎉 Annotation complete → {gt_output}")
    print(f"   Total frames annotated: {len(all_gt)}")

if __name__ == "__main__":
    parking_id = "parking_lot_A"     # ← change this per parking lot
    extract_frames(parking_id)
    annotate_frames(parking_id)