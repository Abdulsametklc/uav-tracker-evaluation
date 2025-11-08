#env: tracker_SORT
import time
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt 

MODEL_PATH = "uav.pt"  
VIDEO_PATH = "uav.mp4"      
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_VIDEO_PATH = "output_tracked.mp4" 

class FairMOT_Tracker_Placeholder:
    def __init__(self):
        print("FairMOT Tracker (Yer Tutucu) Başlatıldı.")
        self.tracks = {}
        self.next_id = 1
        
    def update(self, detections_xyxy_conf_cls):
        """
        Gelen YOLO tespitlerini FairMOT'un takip motoruna besler.
        """
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

def run_yolo_and_fairmot_analysis_external(model_path, video_path):
    try:
        yolo_model = YOLO(model_path)
    except FileNotFoundError:
        print(f"HATA: YOLO model dosyası '{model_path}' bulunamadı.")
        return

    fairmot_tracker = FairMOT_Tracker_Placeholder()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"HATA: Video dosyası '{video_path}' açılamadı.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))


    yolo_detection_times = []
    fairmot_tracking_times = []
    
    frame_counter = 0

    print("\nFairMOT Entegre Takip Başlatılıyor. 'q' ile çıkış yapabilirsiniz.")

    while True:
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        detection_start_time = time.time()
        results = yolo_model.predict(source=frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detection_end_time = time.time()
        yolo_detection_times.append(detection_end_time - detection_start_time)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append(np.append(box, [conf, cls_id])) 

        tracking_start_time = time.time()
        tracked_objects = fairmot_tracker.update(detections)
        
        tracking_end_time = time.time()
        fairmot_tracking_times.append(tracking_end_time - tracking_start_time)
        frame = draw_tracks(frame, tracked_objects)

        out.write(frame)
        cv2.imshow("YOLO + FairMOT Tracking", frame)

        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    
    cap.release()
    out.release() 
    cv2.destroyAllWindows()

    total_yolo_time = sum(yolo_detection_times)
    total_fairmot_time = sum(fairmot_tracking_times)
    total_runtime = time.time() - frame_start_time 
    overhead_time = total_runtime - total_yolo_time - total_fairmot_time
    overhead_time = max(0, overhead_time) 
    total_analyzed_time = total_runtime
    
    if total_analyzed_time == 0:
        print("Video işlenmediği için analiz yapılamadı.")
        return

    yolo_ratio = total_yolo_time / total_analyzed_time
    fairmot_ratio = total_fairmot_time / total_analyzed_time
    overhead_ratio = overhead_time / total_analyzed_time

    print("\n" + "=" * 60)
    print("YOLO + FAIRMOT ANALİZ SONUÇLARI")
    print("=" * 60)
    print(f"Toplam İşlenen Kare Sayısı: {frame_counter}")
    print(f"Toplam Çalışma Süresi: {total_runtime:.4f} saniye")
    print("-" * 60)
    print(f"1. YOLO Tespiti (Detection) Süresi: {total_yolo_time:.4f} saniye ({yolo_ratio * 100:.2f} %)")
    print(f"2. FairMOT Takip (Simülasyon) Süresi: {total_fairmot_time:.4f} saniye ({fairmot_ratio * 100:.2f} %)")
    print(f"3. Diğer Maliyetler (Çizim, I/O, Döngü): {overhead_time:.4f} saniye ({overhead_ratio * 100:.2f} %)")
    print("=" * 60)

    labels = ['YOLO Tespiti', 'FairMOT Takip (Sim.)', 'Diğer Maliyetler (Çizim/I/O)']
    sizes = [total_yolo_time, total_fairmot_time, overhead_time]
    colors = ['#ff9999','#66b3ff', '#99ff99']
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, colors=colors, 
            autopct=lambda p: f'{p:.1f}%\n({p*total_analyzed_time/100:.2f}s)', 
            shadow=True, startangle=90)
    plt.title(f'YOLO vs. FairMOT (Simülasyon) Süresi Dağılımı\n(Toplam Süre: {total_analyzed_time:.2f} s)')
    plt.axis('equal') 
    plt.show()

def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = [int(i) for i in track['box']]
        track_id = track['id']
        conf = track['conf']
        
        color = (255, 0, 0) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Conf:{conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

run_yolo_and_fairmot_analysis_external(MODEL_PATH, VIDEO_PATH)