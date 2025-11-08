# fairmot_track_c.py
import time
import cv2
import numpy as np
import psutil
import GPUtil
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


class FairMOT_Tracker_Placeholder:
    def __init__(self):
        print("FairMOT Tracker (Yer Tutucu) Başlatıldı.")
        self.tracks = {}
        self.next_id = 1

    def update(self, detections_xyxy_conf_cls):
        time.sleep(0.008)
        new_tracks = []
        for det in detections_xyxy_conf_cls:
            new_tracks.append({
                'box': det[:4],
                'id': self.next_id,
                'conf': det[4]
            })
            self.next_id += 1
        return new_tracks


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = [int(i) for i in track['box']]
        conf = track['conf']
        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Conf:{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run_fairmot(video_path, model_path, show_video=False, show_plot=False):
    CONFIDENCE_THRESHOLD = 0.25
    OUTPUT_VIDEO_PATH = "output_tracked_fairmot.mp4"

    try:
        yolo_model = YOLO(model_path)
    except FileNotFoundError:
        print(f"HATA: YOLO model dosyası '{model_path}' bulunamadı.")
        return {}

    fairmot_tracker = FairMOT_Tracker_Placeholder()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video dosyası '{video_path}' açılamadı.")
        return {}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    yolo_times, fairmot_times, fps_list = [], [], []
    iou_list, cpu_list, gpu_list, ram_list = [], [], [], []
    tp, fp, fn = 0, 0, 0
    frame_counter = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nFairMOT Entegre Takip Başlatılıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        # YOLO detection
        det_start = time.time()
        results = yolo_model.predict(source=frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        det_end = time.time()
        yolo_times.append(det_end - det_start)

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append(np.append(box, [conf, cls_id]))

        # Tracking
        trk_start = time.time()
        tracked_objects = fairmot_tracker.update(detections)
        trk_end = time.time()
        fairmot_times.append(trk_end - trk_start)

        # IOU & Accuracy metrics
        if tracked_objects and detections:
            for t in tracked_objects:
                for d in detections:
                    iou = calculate_iou(t["box"], d[:4])
                    if iou > 0.5:
                        tp += 1
                    else:
                        fp += 1
                    iou_list.append(iou)
            fn += max(0, len(detections) - tp)

        end_time = time.time()
        fps_list.append(1 / (end_time - start_time))
        frame_counter += 1

        # Donanım metrikleri
        cpu_list.append(psutil.cpu_percent())
        ram_list.append(psutil.virtual_memory().percent)
        if DEVICE == "cuda":
            gpus = GPUtil.getGPUs()
            gpu_list.append(np.mean([gpu.load * 100 for gpu in gpus]))
        else:
            gpu_list.append(0)

        frame = draw_tracks(frame, tracked_objects)
        out.write(frame)

        if show_video:
            cv2.imshow("YOLO + FairMOT Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ------------------- METRİKLER -------------------
    mean_iou = np.mean(iou_list) if iou_list else 0
    mean_fps = np.mean(fps_list)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    mota = precision * recall
    mean_cpu = np.mean(cpu_list)
    mean_gpu = np.mean(gpu_list)
    mean_ram = np.mean(ram_list)
    mean_latency_ms = 1000 / (mean_fps + 1e-6)

    if show_plot:
        labels = ['YOLO Tespiti', 'FairMOT Takip (Sim.)']
        sizes = [sum(yolo_times), sum(fairmot_times)]
        colors = ['#ff9999', '#66b3ff']
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title('YOLO + FairMOT Süre Dağılımı')
        plt.axis('equal')
        plt.show()

    return {
        "tracker": "FairMOT",
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mota": mota,
        "mean_fps": mean_fps,
        "mean_latency_ms": mean_latency_ms,
        "cpu_usage": mean_cpu,
        "gpu_usage": mean_gpu,
        "ram_usage": mean_ram
    }
