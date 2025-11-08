import cv2
import numpy as np
from norfair import Detection, Tracker
from ultralytics import YOLO
import time
import psutil
import GPUtil


def measure_usage():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().used / (1024 ** 2)
    gpus = GPUtil.getGPUs()
    gpu = gpus[0].load * 100 if gpus else 0
    return cpu, gpu, ram


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)


def run_norfair(video_path, model_path):
    model = YOLO(model_path)

    tracker = Tracker(
        distance_function=lambda det, trk: np.linalg.norm(det.points - trk.estimate),
        distance_threshold=35
    )

    cap = cv2.VideoCapture(video_path)

    tp = fp = fn = 0
    iou_list = []
    fps_list = []
    latency_list = []
    cpu_list, gpu_list, ram_list = [], [], []

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # BBOX bilgisini detection data'ya ekliyoruz ✅
                detections.append(
                    Detection(np.array([cx, cy]), data={"bbox": [x1, y1, x2, y2]})
                )

        tracked_objects = tracker.update(detections)

        # Değerlendirme: sadece ilk hedef -> tek drone senaryosu için
        if len(results.boxes) > 0 and len(tracked_objects) > 0:
            gt_box = results.boxes.xyxy[0].cpu().numpy()
            pred_box = tracked_objects[0].last_detection.data["bbox"]

            iou = calculate_iou(gt_box, pred_box)
            iou_list.append(iou)

            if iou > 0.5:
                tp += 1
            else:
                fp += 1

            fn = max(0, len(results.boxes) - tp)

        # FPS & Latency
        latency = (time.time() - start_time) * 1000
        fps = 1000 / latency if latency > 0 else 0

        latency_list.append(latency)
        fps_list.append(fps)

        # Sistem kaynakları
        cpu, gpu, ram = measure_usage()
        cpu_list.append(cpu)
        gpu_list.append(gpu)
        ram_list.append(ram)

    cap.release()

    # Metrikler
    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mota = precision * recall

    return {
        "tracker": "Norfair",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mota": mota,
        "mean_fps": float(np.mean(fps_list)),
        "mean_latency_ms": float(np.mean(latency_list)),
        "cpu_usage": float(np.mean(cpu_list)),
        "gpu_usage": float(np.mean(gpu_list)),
        "ram_usage": float(np.mean(ram_list))
    }
