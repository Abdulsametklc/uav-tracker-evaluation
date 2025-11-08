#env: tracker_SORT
import cv2
import numpy as np
from norfair import Detection, Tracker
from ultralytics import YOLO

# ----------------------------
# YOLOv8 modelini yükle
# ----------------------------
model_path = "uav.pt"  # senin YOLOv8 modelin
model = YOLO(model_path)

# ----------------------------
# Norfair tracker ayarları
# ----------------------------
tracker = Tracker(
    distance_function=lambda det, trk: np.linalg.norm(det.points - trk.estimate),
    distance_threshold=30
)

# ----------------------------
# Video giriş ve çıkış
# ----------------------------
video_input = "uav.mp4"
video_output = "track_uav.mp4"

cap = cv2.VideoCapture(video_input)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

# ----------------------------
# Frame frame işle
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detections
    results = model(frame)[0]
    detections = []

    if results.boxes is not None:
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(Detection(np.array([cx, cy])))

            # Kutu ve label çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{int(cls)}:{conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Norfair takibi
    tracked_objects = tracker.update(detections=detections)

    for trk in tracked_objects:
        x, y = trk.estimate[0]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(frame, f'ID: {trk.id}', (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Çıktı videoya yaz
    out.write(frame)

    # Ekranda göster
    cv2.imshow("YOLOv8 + Norfair Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
