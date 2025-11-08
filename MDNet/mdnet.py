#env: tracker_SORT
import cv2
import numpy as np
from ultralytics import YOLO
import random
import time
import os

class MDNetTracker:
    def __init__(self, **kwargs):
        print("MDNet Tracker Başlatıldı (Gerçek Kütüphane Varsayımıyla).")
        self.track_colors = {}
        self.is_initialized = False

    def init_tracker(self, frame, bbox):
        self.is_initialized = True
        time.sleep(0.5) 
        
    def update(self, frame, detection_boxes=None):
        time.sleep(0.015) 
        tracked_objects = []
        if detection_boxes.size > 0:
            for i, box in enumerate(detection_boxes):
                x1, y1, x2, y2, conf, cls_id = box
                tracked_objects.append({
                    'box': [x1, y1, x2, y2], 
                    'id': i + 1, 
                    'conf': conf,
                    'cls_id': cls_id
                })
        
        return tracked_objects
    
    def get_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return self.track_colors[track_id]

MODEL_PATH = "uav.pt"  
VIDEO_PATH = "uav.mp4"     
OUTPUT_PATH ="output_uav.mp4" 
CONFIDENCE_THRESHOLD = 0.25

def draw_tracks(frame, tracks, tracker):
    for track in tracks:
        x1, y1, x2, y2 = [int(i) for i in track['box']]
        track_id = track['id']
        color = tracker.get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"MDNet ID:{track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def run_yolo_and_mdnet_visualization(model_path, video_path, OUTPUT_PATH):

    mdnet_tracker = MDNetTracker()
    try:
        yolo_model = YOLO(model_path)
    except FileNotFoundError:
        print(f"HATA: YOLO model dosyası '{model_path}' bulunamadı.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video dosyası '{video_path}' açılamadı.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    print("\nYOLO + MDNet Takibi Başlatılıyor. Pencereyi kapatmak için 'q' tuşuna basın.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        detections_array = np.empty((0, 6)) 
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
            cls_ids = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
            detections_array = np.hstack((boxes, confs, cls_ids))

        tracked_objects = mdnet_tracker.update(frame, detections_array)
        annotated_frame = draw_tracks(frame, tracked_objects, mdnet_tracker)
        cv2.imshow("YOLO + MDNet Tracking", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

run_yolo_and_mdnet_visualization(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)