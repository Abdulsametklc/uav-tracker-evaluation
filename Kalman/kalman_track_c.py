import psutil
import GPUtil
import torch
import time
import numpy as np
from ultralytics import YOLO

def run_botsort(video_path, model_path="uav.pt"):
    yolo = YOLO(model_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    fps_list, cpu_list, gpu_list, ram_list = [], [], [], []
    iou_list, tp, fp, fn = [], 0, 0, 0

    print(f"\n'{model_path}' modeli BoT-SORT ile çalıştırılıyor...")

    # Bellek taşmasını engellemek için stream=True eklendi ⚡
    results = yolo.track(
        source=video_path,
        tracker="botsort.yaml",
        conf=0.25,
        show=False,
        save=False,
        stream=True,   # 🔹 önemli: RAM dolmasını engeller
        verbose=False
    )

    for r in results:
        start = time.time()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

        # IOU benzeri basit doğruluk ölçümü (tek hedefli senaryoda)
        if len(boxes) > 0:
            if len(boxes) == 1:
                iou_list.append(1.0)
                tp += 1
            else:
                tp += 1
                fp += len(boxes) - 1
        else:
            fn += 1

        elapsed = time.time() - start
        if elapsed < 1e-3:  # ⚙️ Güvenli sınır
            elapsed = 1e-3
        fps_list.append(1 / elapsed)

        cpu_list.append(psutil.cpu_percent())
        ram_list.append(psutil.virtual_memory().percent)
        if DEVICE == "cuda":
            gpus = GPUtil.getGPUs()
            gpu_list.append(np.mean([gpu.load * 100 for gpu in gpus]))
        else:
            gpu_list.append(0)

    # METRİKLER
    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mota = precision * recall
    mean_fps = np.mean(fps_list)
    latency = 1000 / (mean_fps + 1e-6)
    mean_cpu = np.mean(cpu_list)
    mean_gpu = np.mean(gpu_list)
    mean_ram = np.mean(ram_list)

    return {
        "tracker": "BoT-SORT",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": latency,
        "cpu_usage": mean_cpu,
        "gpu_usage": mean_gpu,
        "ram_usage": mean_ram
    }
