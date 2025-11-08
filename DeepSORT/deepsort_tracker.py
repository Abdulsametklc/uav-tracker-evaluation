#env: tracker_SORT
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  

model_path = "uav.pt"
model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)
video_path = "uav.mp4"
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('tracked_output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf > 0.5:  # güven eşiği
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # --- Takip (DeepSORT) ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Görselleştirme ---
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue  # Güncel değilse atla

        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
