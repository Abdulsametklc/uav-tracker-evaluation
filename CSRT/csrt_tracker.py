#env: tracker_SORT
##tracker_kore enviromentta çalışmaktadır. 
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ========== KULLANICI AYARLARI ==========
VIDEO_PATH = "uav.mp4"
MODEL_PATH = "uav.pt"
OUTPUT_PATH = "output.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECT_EVERY_N_FRAMES = 15
CONF_THRESH = 0.4
# ========================================


# --- MODELİ YÜKLE ---
def load_model(path):
    model = YOLO(path)  # Ultralytics formatı için
    return model


# --- YOLO MODELİYLE TESPİT ---
def detect(model, frame):
    """YOLOv8 modeliyle tek karede tespit yapar."""
    results = model.predict(frame, device=DEVICE, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    if len(scores) == 0 or max(scores) < CONF_THRESH:
        return None

    idx = np.argmax(scores)
    x1, y1, x2, y2 = boxes[idx]
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    return (x, y, w, h)


# --- CSRT TRACKER OLUŞTUR ---
def create_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    else:
        return cv2.legacy.TrackerCSRT_create()


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = load_model(MODEL_PATH)
    tracker = None
    bbox = None

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Belirli aralıklarla yeniden tespit yap
        if frame_count % DETECT_EVERY_N_FRAMES == 1 or tracker is None:
            bbox = detect(model, frame)
            if bbox:
                tracker = create_csrt()
                tracker.init(frame, bbox)
                print(f"[{frame_count}] Yeni tespit ve CSRT başlatıldı:", bbox)
            else:
                print(f"[{frame_count}] Tespit yok!")
                tracker = None

        elif tracker is not None:
            success, box = tracker.update(frame)
            if success:
                bbox = [int(v) for v in box]
            else:
                print(f"[{frame_count}] Takip kaybedildi.")
                tracker = None

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow("CSRT Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı. Çıktı kaydedildi:", OUTPUT_PATH)


if __name__ == "__main__":
    main()