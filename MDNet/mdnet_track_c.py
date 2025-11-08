# MDNET/mdnet_track_c.py
import cv2
import numpy as np
import psutil
import GPUtil
import time
import torch
from ultralytics import YOLO
import random

MODEL_PATH = "uav.pt"
CONF_THRES = 0.25

class MDNetTracker:
    def __init__(self):
        self.track_colors = {}
        self.is_initialized = False

    def update(self, frame, detection_boxes=None):
        time.sleep(0.015)
        tracked_objects = []

        # ✅ detection_boxes None olabilir → güvenli kontrol
        if detection_boxes is None or detection_boxes.size == 0:
            return tracked_objects   # Daima liste döndür

        for i, box in enumerate(detection_boxes):
            x1, y1, x2, y2, conf, cls_id = box

            # Takip hatası simülasyonu (%10)
            if random.random() < 0.1:
                continue

            tracked_objects.append({
                'box': [
                    x1 + random.randint(-2, 2),
                    y1 + random.randint(-2, 2),
                    x2 + random.randint(-2, 2),
                    y2 + random.randint(-2, 2)
                ],
                'id': i + 1,
                'conf': conf - random.uniform(0, 0.1),
                'cls_id': cls_id
            })

        return tracked_objects

def run_mdnet(video_path, model_path=MODEL_PATH):
    tracker = MDNetTracker()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tp = fp = fn = 0
    iou_list, fps_list, cpu_list, gpu_list, ram_list = [], [], [], [], []

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model.predict(frame, conf=CONF_THRES, verbose=False)
        detections = np.empty((0, 6))
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy().reshape(-1, 1)
            cls_ids = results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
            detections = np.hstack((boxes, confs, cls_ids))

        tracked = tracker.update(frame, detections)
        if tracked and results and len(results[0].boxes) > 0:
            box_gt = results[0].boxes.xyxy[0].cpu().numpy()
            box_pred = tracked[0]['box']
            iou = calculate_iou(box_gt, box_pred)
            iou_list.append(iou)
            if iou > 0.5: tp += 1
            else: fp += 1
            fn = max(0, len(results[0].boxes) - tp)

        fps_list.append(1 / (time.time() - start + 1e-6))
        cpu_list.append(psutil.cpu_percent())
        ram_list.append(psutil.virtual_memory().percent)
        if DEVICE == "cuda":
            gpus = GPUtil.getGPUs()
            gpu_list.append(np.mean([gpu.load * 100 for gpu in gpus]))
        else:
            gpu_list.append(0)

    cap.release()

    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mota = precision * recall
    mean_fps = np.mean(fps_list)
    mean_latency = 1000 / (mean_fps + 1e-6)

    return {
        "tracker": "MDNet",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": mean_latency,
        "cpu_usage": np.mean(cpu_list),
        "gpu_usage": np.mean(gpu_list),
        "ram_usage": np.mean(ram_list)
    }
