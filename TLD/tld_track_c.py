import cv2
from ultralytics import YOLO
import time
import psutil
import GPUtil
import numpy as np

def bbox_iou(boxA, boxB):
    # IoU hesaplama
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def run_tld(model_path, video_path, conf_thres=0.4):
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    tracker = None
    initBB = None

    frame_count = 0
    iou_list = []
    tp = 0
    fp = 0
    fn = 0

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # YOLO tespiti
        results = yolo.predict(source=frame, conf=conf_thres, verbose=False)
        gt_boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy():
                gt_boxes.append([int(x) for x in box])

        # Tracker başlat / güncelle
        if tracker is None and len(gt_boxes) > 0:
            x1, y1, x2, y2 = gt_boxes[0]
            initBB = (x1, y1, x2-x1, y2-y1)
            tracker = cv2.legacy.TrackerTLD_create()
            tracker.init(frame, initBB)
        elif tracker is not None:
            success, box = tracker.update(frame)
            if success:
                x, y, w_box, h_box = [int(v) for v in box]
                pred_box = [x, y, x+w_box, y+h_box]
                # Metric hesaplama
                max_iou = 0
                for gt in gt_boxes:
                    iou = bbox_iou(pred_box, gt)
                    if iou > max_iou:
                        max_iou = iou
                iou_list.append(max_iou)
                if max_iou >= 0.5:
                    tp += 1
                else:
                    fp += 1
                fn += max(0, len(gt_boxes) - 1)
            else:
                tracker = None
                fp += 1
                fn += len(gt_boxes)

    end_time = time.time()
    cap.release()

    mean_iou = np.mean(iou_list) if iou_list else 0
    precision = tp / (tp+fp+1e-6)
    recall = tp / (tp+fn+1e-6)
    f1_score = 2*precision*recall/(precision+recall+1e-6)

    cpu, gpu, ram = psutil.cpu_percent(), GPUtil.getGPUs()[0].load*100 if GPUtil.getGPUs() else 0, psutil.virtual_memory().used/(1024**2)
    metrics = {
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mota": recall,  # basit MOTA = recall
        "mean_fps": frame_count / (end_time-start_time),
        "mean_latency_ms": (end_time-start_time)/frame_count*1000,
        "cpu_usage": cpu,
        "gpu_usage": gpu,
        "ram_usage": ram
    }
    return metrics
