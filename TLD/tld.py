#env: tracker_SORT
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "uav.pt"
VIDEO_PATH = "uav.mp4"
OUTPUT_PATH = "tld_tracked.mp4"
CONF_THRES = 0.4

# YOLO modeli
yolo = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# VideoWriter ayarları
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:  # NaN kontrolü
    fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

tracker = None
initBB = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracker is None:
        # YOLO tespiti
        results = yolo.predict(source=frame, conf=CONF_THRES, verbose=False)
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            initBB = (x1, y1, x2 - x1, y2 - y1)
            
            # TLD tracker başlat
            tracker = cv2.legacy.TrackerTLD_create()
            tracker.init(frame, initBB)
            print("TLD tracker başlatıldı.")

    else:
        success, box = tracker.update(frame)
        if success:
            x, y, w_box, h_box = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        else:
            print("Takip kaybedildi, YOLO ile yeniden tespit yapılacak.")
            tracker = None

    # Frame'i hem göster hem kaydet
    cv2.imshow("YOLO + TLD Tracker", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Takip edilen video kaydedildi:", OUTPUT_PATH)
