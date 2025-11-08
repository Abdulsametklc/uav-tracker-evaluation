#env: tracker_SORT
import cv2
import torch
import numpy as np
import time
import psutil
import GPUtil
from ultralytics import YOLO
from .sort.sort import Sort  # ← Senin mevcut SORT implementasyonun

MODEL_PATH = "uav.pt"
VIDEO_PATH = "uav.mp4"
OUTPUT_PATH = "tracked_sort.mp4"
CONF_THRESH = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_system_usage():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().used / (1024**2)
    gpus = GPUtil.getGPUs()
    gpu = gpus[0].load * 100 if gpus else 0
    return cpu, gpu, ram


def run_sort(model_path=MODEL_PATH, video_path=VIDEO_PATH, out_path=OUTPUT_PATH):
    print(f"[INFO] Cihaz: {DEVICE}")

    # ✅ Model önce, video sonra (compare.py uyumlu)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    frame_count = 0
    start_time = time.time()

    # === METRİK HAZIRLIĞI ===
    gt_file = "test_data/Video_01/groundtruth.txt"
    groundtruth = np.loadtxt(gt_file, delimiter=",")

    iou_list = []
    tp = 0
    fp = 0
    fn = 0

    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = boxA_area + boxB_area - inter
        return inter / union if union > 0 else 0

    print("[INFO] Takip başlıyor...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, device=DEVICE, verbose=False)
        detections = []

        if results and results[0].boxes:
            for box in results[0].boxes:
                if box.conf[0] > CONF_THRESH:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    detections.append([x1, y1, x2, y2, score])

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracked = tracker.update(detections)

        # --- METRİK KAYDI ---
        if frame_count < len(groundtruth):
            gt_box = groundtruth[frame_count]  #  [x1,y1,x2,y2]
            if len(tracked) > 0:
                best_iou = 0
                for x1,y1,x2,y2,tid in tracked:
                    iou = compute_iou([x1,y1,x2,y2], gt_box)
                    best_iou = max(best_iou, iou)
                iou_list.append(best_iou)

                if best_iou >= 0.5: tp += 1
                else: fp += 1
            else:
                fn += 1

        for x1, y1, x2, y2, tid in tracked:
            x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)
        frame_count += 1

        # İstersen overlay göstermeyi kaldırabiliriz test performans için:
        # cv2.imshow("SORT Tracker", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    mean_fps = frame_count / total_time if total_time > 0 else 0

    cpu, gpu, ram = measure_system_usage()

    # ✅ compare.py'nin beklediği formatta metrik döndür
    mean_iou = np.mean(iou_list) if len(iou_list) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mota = recall  # Basitleştirilmiş MOTA

    metrics = {
        "tracker": "SORT",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": (1/mean_fps)*1000 if mean_fps > 0 else 0,
        "cpu_usage": cpu,
        "gpu_usage": gpu,
        "ram_usage": ram,
    }


    print("[BİTTİ] Çıktı kaydedildi →", out_path)
    return metrics


if __name__ == "__main__":
    run_sort()
