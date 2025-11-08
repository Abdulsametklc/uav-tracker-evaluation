#env: tracker_SORT
import cv2
import numpy as np
import psutil
import GPUtil
import torch
import time
from ultralytics import YOLO

MODEL_PATH = "uav.pt"
CONF_THRES = 0.4

def run_flowtracker(video_path, model_path=MODEL_PATH):
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("flow_tracked.mp4", fourcc, fps, (w, h))

    tracker_initialized = False
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    tp, fp, fn = 0, 0, 0
    fps_list, iou_list, cpu_list, gpu_list, ram_list = [], [], [], [], []
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    old_gray = None
    bbox = None
    p0 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracker_initialized:
            results = yolo.predict(source=frame, conf=CONF_THRES, verbose=False)
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                mask = np.zeros_like(frame_gray)
                mask[y1:y2, x1:x2] = 255
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=200,
                                             qualityLevel=0.01, minDistance=3, blockSize=3)
                old_gray = frame_gray.copy()
                tracker_initialized = True
            else:
                continue
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            else:
                good_new, good_old = [], []

            if len(good_new) > 0:
                dx = np.mean(good_new[:, 0] - good_old[:, 0])
                dy = np.mean(good_new[:, 1] - good_old[:, 1])
                x, y, w_box, h_box = bbox
                x += int(dx)
                y += int(dy)
                bbox = (x, y, w_box, h_box)
            else:
                tracker_initialized = False
                continue

            # IoU hesapla (YOLO tespitiyle)
            results = yolo.predict(source=frame, conf=CONF_THRES, verbose=False)
            if results and len(results[0].boxes) > 0:
                box_gt = results[0].boxes.xyxy[0].cpu().numpy()
                box_pred = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                iou = calculate_iou(box_gt, box_pred)
                iou_list.append(iou)
                if iou > 0.5:
                    tp += 1
                else:
                    fp += 1
                fn = max(0, len(results[0].boxes) - tp)

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        fps_list.append(1 / (time.time() - start))
        cpu_list.append(psutil.cpu_percent())
        ram_list.append(psutil.virtual_memory().percent)
        if DEVICE == "cuda":
            gpus = GPUtil.getGPUs()
            gpu_list.append(np.mean([gpu.load * 100 for gpu in gpus]))
        else:
            gpu_list.append(0)

        out.write(frame)
        cv2.imshow("YOLO + Optical Flow Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

    return {
        "tracker": "FlowTracker",
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
