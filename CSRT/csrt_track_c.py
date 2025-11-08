# csrt_tracker.py
import cv2
import torch
import numpy as np
import time
import psutil
import GPUtil
from ultralytics import YOLO

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def run_csrt(video_path, model_path, detect_interval=15, conf_thresh=0.4, iou_thresh=0.5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    tracker, bbox = None, None
    frame_count = 0
    fps_list, iou_list, latencies = [], [], []

    # Metrik sayaçları
    TP, FP, FN = 0, 0, 0
    last_yolo_box = None

    # Sistem ölçümü başlangıcı
    cpu_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().used / (1024 ** 2)
    gpu_start = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    def detect(model, frame):
        results = model.predict(frame, device=DEVICE, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        if len(scores) == 0 or max(scores) < conf_thresh:
            return None
        idx = np.argmax(scores)
        x1, y1, x2, y2 = boxes[idx]
        return [int(x1), int(y1), int(x2), int(y2)]

    def create_csrt():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        else:
            return cv2.legacy.TrackerCSRT_create()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        start_t = time.time()

        if frame_count % detect_interval == 1 or tracker is None:
            yolo_box = detect(model, frame)
            if yolo_box:
                x1, y1, x2, y2 = yolo_box
                tracker = create_csrt()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                last_yolo_box = yolo_box
        elif tracker:
            success, box = tracker.update(frame)
            if success:
                tx, ty, tw, th = [int(v) for v in box]
                tracker_box = [tx, ty, tx + tw, ty + th]
                if last_yolo_box:
                    iou = calculate_iou(tracker_box, last_yolo_box)
                    iou_list.append(iou)

                    # TP / FP / FN
                    if iou >= iou_thresh:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FP += 1
            else:
                FN += 1
                tracker = None

        end_t = time.time()
        latencies.append((end_t - start_t) * 1000)
        fps_list.append(1 / (end_t - start_t))

    cap.release()

    # Doğruluk metrikleri
    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    mota = 1 - ((FP + FN) / (TP + FP + FN + 1e-6))

    # Performans metrikleri
    mean_fps = np.mean(fps_list) if fps_list else 0
    mean_latency = np.mean(latencies) if latencies else 0

    # Sistem metrikleri
    cpu_end = psutil.cpu_percent(interval=None)
    ram_end = psutil.virtual_memory().used / (1024 ** 2)
    gpu_end = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    metrics = {
        "tracker": "CSRT",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": mean_latency,
        "cpu_usage": abs(cpu_end - cpu_start),
        "gpu_usage": abs(gpu_end - gpu_start),
        "ram_usage": abs(ram_end - ram_start),
    }

    return metrics
