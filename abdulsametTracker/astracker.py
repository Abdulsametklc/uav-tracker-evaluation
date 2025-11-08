import numpy as np
import cv2
import time 
from deep_sort_realtime.deepsort_tracker import DeepSort, Tracker 
from ultralytics import YOLO

# Sabitler
SIAMRPN_CONF_THRESHOLD = 0.5
MAX_TRACKER_AGE = 5 
YOLO_MODEL_PATH = "uav.pt" 
VIDEO_PATH = "uav.mp4" 
OUTPUT_VIDEO_PATH = "uav_output.mp4" 

class SiamRPN_Wrapper:
    def __init__(self):
        self.bbox_tlwh = None 
        self.initialized = False

    def initialize(self, frame, bbox_tlwh): 
        self.bbox_tlwh = np.array(bbox_tlwh)
        self.initialized = True

    def track(self, frame):
        if not self.initialized:
            return None, 0.0
            
        x, y, w, h = self.bbox_tlwh 
        new_x = x + np.random.uniform(-1, 1) 
        new_y = y + np.random.uniform(-1, 1) 
        new_bbox_tlwh = np.array([new_x, new_y, w, h])
        confidence = np.random.uniform(0.7, 0.99)
        self.bbox_tlwh = new_bbox_tlwh

        return new_bbox_tlwh.tolist(), confidence

class YOLO_Detector:
    """
    YOLO modelini çalıştıran varsayımsal sınıf.
    """
    def __init__(self, model_path):
        print(f"YOLO Modeli Yükleniyor: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"HATA: YOLO modeli yüklenemedi. Takip başlatılamayacak! Hata: {e}")
            self.model = None
            
        self.target_class_id = 0

    def detect(self, frame):
        detections = []
        if self.model is None:
            return detections
        
        results = self.model(frame, verbose=False, conf=0.25) 
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                bbox_xyxy = box.xyxy[0].cpu().numpy().tolist() 
                
                x1, y1, x2, y2 = bbox_xyxy
                width = x2 - x1
                height = y2 - y1
                bbox_tlwh = [x1, y1, width, height]
                
                if cls_id == self.target_class_id and conf > 0.3:
                     detections.append((bbox_tlwh, conf, cls_id))
                
        return detections

class SiamTracker(Tracker):
    def __init__(self, *args, **kwargs):
        _ = kwargs.pop('feature', None); _ = kwargs.pop('original_ltwh', None); _ = kwargs.pop('det_class', None)
        _ = kwargs.pop('det_conf', None); _ = kwargs.pop('instance_mask', None); _ = kwargs.pop('others', None)
        _ = kwargs.pop('detection_class', None)
        
        super().__init__(*args, **kwargs)
        
        if not hasattr(self, 'time_since_update') or isinstance(getattr(self, 'time_since_update', 0), str): self.time_since_update = 0 
        if not hasattr(self, 'max_age') or isinstance(getattr(self, 'max_age', 0), str): self.max_age = int(kwargs.get('max_age', 5)) 
        if not hasattr(self, 'state'): self.state = 1 
        if not hasattr(self, 'track_id'):
            if args and isinstance(args[0], int): self.track_id = args[0]
            elif hasattr(self, 'id'): self.track_id = self.id
            else: self.track_id = -1 
        
        self.siamrpn_tracker = SiamRPN_Wrapper()
        self.is_siamrpn_active = False

    def activate_siamrpn(self, frame, detection_bbox_tlwh):
        self.siamrpn_tracker.initialize(frame, detection_bbox_tlwh)
        self.is_siamrpn_active = True
    
    def is_deleted(self): return self.time_since_update > self.max_age

    def is_lost(self): return self.state == 3 

    def is_confirmed(self): return self.state == 2

    def mark_missed(self): 
        if self.state == 1: self.state = 3 
        elif self.state == 2: self.state = 3
        if hasattr(self, 'time_since_update'): self.time_since_update += 1
        else: self.time_since_update = 1

    def predict(self, kf=None):
        
        if not hasattr(self, 'kf') or self.kf is None:
            self.time_since_update += 1
            return 
        
        try: 
            return super().predict()
        
        except AttributeError:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
            self.time_since_update += 1
            if hasattr(self, 'age'):
                if self.state == 1 or self.state == 2: self.age += 1
    
    def to_ltwh(self): 
        if hasattr(self, 'mean'): return self.mean[:4]
        else: return np.array([0.0, 0.0, 0.0, 0.0]) 
    def to_tlbr(self):
        tlwh = self.to_ltwh()
        x, y, w, h = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
        return [x, y, x + w, y + h]
    def to_tlwh(self):
        return self.to_ltwh()

class HybridTracker:
    def __init__(self):
        self.deep_sort_tracker = DeepSort(
            max_age=MAX_TRACKER_AGE,
            override_track_class=SiamTracker,
            n_init=2,)
        self.yolo_detector = YOLO_Detector(YOLO_MODEL_PATH)
        self.frame_count = 0

    def update(self, frame):
        self.frame_count += 1
        siam_detections = []
        needs_yolo_run = False
        all_current_tracks = self.deep_sort_tracker.tracker.tracks
        active_tracks = [t for t in all_current_tracks if t.is_confirmed() and hasattr(t, 'is_siamrpn_active') and t.is_siamrpn_active]

        if not all_current_tracks or all(t.is_lost() for t in all_current_tracks): needs_yolo_run = True

        if active_tracks:
            if self.frame_count == 5 and len(active_tracks) > 0:
                 active_tracks[0].is_siamrpn_active = False
                 needs_yolo_run = True
            
            if not needs_yolo_run:
                print(f"Kare {self.frame_count}: YOLO ATLANDI. SiamRPN ile {len(active_tracks)} takipçi ilerletiliyor.")
                for track in active_tracks:
                    new_bbox_tlwh, confidence = track.siamrpn_tracker.track(frame)
                    if new_bbox_tlwh is not None and confidence > SIAMRPN_CONF_THRESHOLD:
                        siam_detections.append((new_bbox_tlwh, confidence, track.get_det_class())) 
                    else:
                        print(f"ID {track.track_id} SiamRPN takibi kaybetti. YOLO çalışacak.")
                        track.is_siamrpn_active = False; needs_yolo_run = True 

        yolo_detections = []
        if needs_yolo_run:
            print(f"Kare {self.frame_count}: YOLO ÇALIŞTIRILDI.")
            yolo_detections = self.yolo_detector.detect(frame)

        all_detections = yolo_detections + siam_detections 
        tracks = self.deep_sort_tracker.update_tracks(all_detections, frame=frame)
        
        for track in tracks:
            if track.is_confirmed() and hasattr(track, 'is_siamrpn_active') and not track.is_siamrpn_active:
                bbox_tlwh = track.to_tlwh()
                track.activate_siamrpn(frame, bbox_tlwh)
                print(f"ID {track.track_id} için SiamRPN BAŞLATILDI/YENİDEN BAŞLATILDI.")
        return tracks

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"HATA: Video dosyası açılamadı: {VIDEO_PATH}.")
        exit() 

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
         print(f"HATA: Video kaydedici başlatılamadı! Lütfen {OUTPUT_VIDEO_PATH} yolunu veya codec'i kontrol edin.")
    
    hybrid_tracker = HybridTracker()
    print(f"\nİşleme Başladı... Çıktı videosu: {OUTPUT_VIDEO_PATH}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_tracks = hybrid_tracker.update(frame)
        
        for track in current_tracks:

            try:
                tlwh = track.to_ltwh()
                if tlwh is None or len(tlwh) < 4:
                     continue
                
                x1_f, y1_f, w_f, h_f = tlwh
                x2_f = x1_f + w_f
                y2_f = y1_f + h_f
                bbox = [x1_f, y1_f, x2_f, y2_f]

                if not np.isfinite(x1_f) or not np.isfinite(y1_f) or not np.isfinite(w_f) or not np.isfinite(h_f):
                     print(f"ID {track.track_id} | HATA TESPİT EDİLDİ: NaN/Inf koordinatları! TLWH: {tlwh}")
                     track.mark_missed() 
                     continue

                x1, y1, x2, y2 = map(int, bbox)
                x1 = np.clip(x1, 0, frame_width)
                y1 = np.clip(y1, 0, frame_height)
                x2 = np.clip(x2, 0, frame_width)
                y2 = np.clip(y2, 0, frame_height)

                kutu_genisligi = x2 - x1
                kutu_yuksekligi = y2 - y1
                
                if kutu_genisligi < 5 or kutu_yuksekligi < 5 or x1 >= x2 or y1 >= y2: 
                     continue
                     
                if not track.is_confirmed(): continue 

                print(f"ID {track.track_id} | AKTİF ÇİZİM (W/H: {kutu_genisligi}x{kutu_yuksekligi}): ({x1}, {y1}) - ({x2}, {y2})")
                
            except Exception as e:
                print(f"ÇİZİM HATA YAKALANDI: {e}")
                continue
            
            is_siam_active = hasattr(track, 'is_siamrpn_active') and track.is_siamrpn_active
            status = 'SiamRPN Aktif' if is_siam_active else 'YOLO/KF'
            label = f"ID: {track.track_id} | Status: {status}"
            
            color = (0, 255, 0) if is_siam_active else (0, 0, 255) 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                         0.5, color, 2)

        print(f"Kare {hybrid_tracker.frame_count} Sonucu: Toplam {len(current_tracks)} Takipçi")
        print("-" * 30)
        
        if out.isOpened():
             out.write(frame)
        
        cv2.imshow("Hybrid DeepSORT + SiamRPN Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nİşleme tamamlandı. Kaydedilen video: {OUTPUT_VIDEO_PATH}")