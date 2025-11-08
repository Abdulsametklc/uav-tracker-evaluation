# deepsort_tracker.py

import cv2
import torch
import numpy as np
import time
import psutil
import GPUtil
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def run_deepsort(video_path, model_path, conf_thresh=0.5, iou_thresh=0.5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(video_path)

    fps_list, iou_list, latencies = [], [], []
    TP, FP, FN = 0, 0, 0
    last_yolo_box = None

    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

    # Sistem ölçümü başlangıcı
    cpu_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().used / (1024 ** 2)
    gpu_start = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.time()

        results = model.predict(frame, device=DEVICE, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf > conf_thresh:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        if len(detections) == 0:
            FN += 1
            continue

        # YOLO kutusu (tek objeyi temsil edecek şekilde)
        yolo_box = [int(x1), int(y1), int(x2), int(y2)]
        tracks = tracker.update_tracks(detections, frame=frame)

        matched = False
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue

            x1, y1, x2, y2 = map(int, t.to_ltrb())
            tracker_box = [x1, y1, x2, y2]

            iou = calculate_iou(tracker_box, yolo_box)
            iou_list.append(iou)

            if iou >= iou_thresh:
                TP += 1
                matched = True
            else:
                FP += 1

        if not matched:
            FN += 1

        end = time.time()
        latencies.append((end - start) * 1000)
        fps_list.append(1 / (end - start))

    cap.release()

    # === METRİKLER ===
    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    mota = 1 - ((FP + FN) / (TP + FP + FN + 1e-6))

    mean_fps = np.mean(fps_list) if fps_list else 0
    mean_latency = np.mean(latencies) if latencies else 0

    # Sistem metrikleri
    cpu_end = psutil.cpu_percent(interval=None)
    ram_end = psutil.virtual_memory().used / (1024 ** 2)
    gpu_end = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0

    metrics = {
        "tracker": "DeepSORT",
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
