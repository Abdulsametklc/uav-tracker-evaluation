# MOSSE/mosse_track_c.py
import cv2
import time
import psutil
import GPUtil
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "uav.pt"
CONF_THRES = 0.45
IHA_CLASS_ID = 0
RE_DETECTION_INTERVAL = 25   # Her 25 karede YOLO yeniden kontrol

def run_mosse(video_path, model_path=MODEL_PATH):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tracker = None
    frame_count = 0
    current_bbox = None

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
        return inter / (areaA + areaB - inter + 1e-6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # YOLO yeniden tespit
        if tracker is None or frame_count % RE_DETECTION_INTERVAL == 0:
            results = model(frame, conf=CONF_THRES, verbose=False)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls = results[0].boxes.cls.cpu().numpy()

                best = -1
                best_box = None
                for i in range(len(boxes)):
                    if cls[i] == IHA_CLASS_ID and confs[i] > best:
                        best = confs[i]
                        best_box = boxes[i]

                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    w, h = x2 - x1, y2 - y1
                    current_bbox = (int(x1), int(y1), int(w), int(h))

                    # MOSSE tracker oluştur
                    try:
                        tracker = cv2.TrackerMOSSE_create()
                    except:
                        tracker = cv2.legacy.TrackerMOSSE_create()

                    tracker.init(frame, current_bbox)

        # MOSSE Takip
        if tracker:
            ok, current_bbox = tracker.update(frame)
            if not ok:
                tracker = None

        # IoU hesapla (takip + yolo karşılaştırması)
        results = model(frame, conf=CONF_THRES, verbose=False)
        if len(results[0].boxes) > 0 and current_bbox:
            gt_box = results[0].boxes.xyxy[0].cpu().numpy()
            pred_box = [current_bbox[0], current_bbox[1], current_bbox[0] + current_bbox[2], current_bbox[1] + current_bbox[3]]
            iou = calculate_iou(gt_box, pred_box)
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

        frame_count += 1

    cap.release()

    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mota = precision * recall
    mean_fps = np.mean(fps_list)
    latency = 1000 / (mean_fps + 1e-6)

    return {
        "tracker": "MOSSE",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": latency,
        "cpu_usage": np.mean(cpu_list),
        "gpu_usage": np.mean(gpu_list),
        "ram_usage": np.mean(ram_list)
    }
