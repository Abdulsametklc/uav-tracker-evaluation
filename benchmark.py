"""
CSRT vs DeepSORT Tracker Karşılaştırma Sistemi
Tracker'ların tahmin doğruluğunu YOLO tespitleri ile karşılaştırır
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import json
from datetime import datetime
from pathlib import Path


class BaseTrackerWrapper:
    """Tüm tracker'lar için ortak interface"""
    def __init__(self):
        self.initialized = False
        self.name = "Base"
    
    def init(self, frame, bbox):
        """Tracker'ı başlat
        Args:
            frame: numpy array (H, W, 3)
            bbox: (x, y, w, h) formatında tuple
        """
        raise NotImplementedError
    
    def update(self, frame):
        """Frame'i güncelle ve tahmin al
        Args:
            frame: numpy array (H, W, 3)
        Returns:
            success: bool - tracking başarılı mı
            bbox: (x, y, w, h) formatında tuple veya None
        """
        raise NotImplementedError
    
    def reset(self):
        """Tracker'ı sıfırla"""
        self.initialized = False


class CSRTTrackerWrapper(BaseTrackerWrapper):
    """CSRT Tracker Wrapper"""
    def __init__(self):
        super().__init__()
        self.name = "CSRT"
        self.tracker = None
    
    def init(self, frame, bbox):
        """CSRT tracker'ı başlat"""
        try:
            if hasattr(cv2, "TrackerCSRT_create"):
                self.tracker = cv2.TrackerCSRT_create()
            else:
                self.tracker = cv2.legacy.TrackerCSRT_create()
            
            self.tracker.init(frame, bbox)
            self.initialized = True
            return True
        except Exception as e:
            print(f"CSRT init hatası: {e}")
            return False
    
    def update(self, frame):
        """CSRT ile tahmin yap"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return True, tuple([int(v) for v in bbox])
            return False, None
        except Exception as e:
            print(f"CSRT update hatası: {e}")
            return False, None
    
    def reset(self):
        """CSRT'yi sıfırla"""
        super().reset()
        self.tracker = None


class KCFTrackerWrapper(BaseTrackerWrapper):
    """KCF Tracker Wrapper"""
    def __init__(self):
        super().__init__()
        self.name = "KCF"
        self.tracker = None
    
    def init(self, frame, bbox):
        """KCF tracker'ı başlat"""
        try:
            if hasattr(cv2, "TrackerKCF_create"):
                self.tracker = cv2.TrackerKCF_create()
            else:
                self.tracker = cv2.legacy.TrackerKCF_create()
            
            self.tracker.init(frame, bbox)
            self.initialized = True
            return True
        except Exception as e:
            print(f"KCF init hatası: {e}")
            return False
    
    def update(self, frame):
        """KCF ile tahmin yap"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return True, tuple([int(v) for v in bbox])
            return False, None
        except Exception as e:
            print(f"KCF update hatası: {e}")
            return False, None
    
    def reset(self):
        """KCF'yi sıfırla"""
        super().reset()
        self.tracker = None


class MOSSETrackerWrapper(BaseTrackerWrapper):
    """MOSSE Tracker Wrapper"""
    def __init__(self):
        super().__init__()
        self.name = "MOSSE"
        self.tracker = None
    
    def init(self, frame, bbox):
        """MOSSE tracker'ı başlat"""
        try:
            if hasattr(cv2, "TrackerMOSSE_create"):
                self.tracker = cv2.TrackerMOSSE_create()
            else:
                self.tracker = cv2.legacy.TrackerMOSSE_create()
            
            self.tracker.init(frame, bbox)
            self.initialized = True
            return True
        except Exception as e:
            print(f"MOSSE init hatası: {e}")
            return False
    
    def update(self, frame):
        """MOSSE ile tahmin yap"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return True, tuple([int(v) for v in bbox])
            return False, None
        except Exception as e:
            print(f"MOSSE update hatası: {e}")
            return False, None
    
    def reset(self):
        """MOSSE'yi sıfırla"""
        super().reset()
        self.tracker = None


class MedianFlowTrackerWrapper(BaseTrackerWrapper):
    """MedianFlow Tracker Wrapper"""
    def __init__(self):
        super().__init__()
        self.name = "MedianFlow"
        self.tracker = None
    
    def init(self, frame, bbox):
        """MedianFlow tracker'ı başlat"""
        try:
            if hasattr(cv2, "TrackerMedianFlow_create"):
                self.tracker = cv2.TrackerMedianFlow_create()
            else:
                self.tracker = cv2.legacy.TrackerMedianFlow_create()
            
            self.tracker.init(frame, bbox)
            self.initialized = True
            return True
        except Exception as e:
            print(f"MedianFlow init hatası: {e}")
            return False
    
    def update(self, frame):
        """MedianFlow ile tahmin yap"""
        if not self.initialized or self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return True, tuple([int(v) for v in bbox])
            return False, None
        except Exception as e:
            print(f"MedianFlow update hatası: {e}")
            return False, None
    
    def reset(self):
        """MedianFlow'u sıfırla"""
        super().reset()
        self.tracker = None
    """DeepSORT Tracker Wrapper"""
    def __init__(self, max_age=30, n_init=1, max_cosine_distance=0.3):
        super().__init__()
        self.name = "DeepSORT"
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,  # n_init=1 yaparak ilk frame'den itibaren confirm ediyoruz
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=1.0  # Tek nesne tracking'de NMS'i devre dışı bırak
        )
        self.current_track_id = None
        self.last_bbox = None
        self.frame_count = 0
    
    def init(self, frame, bbox):
        """DeepSORT tracker'ı başlat"""
        try:
            x, y, w, h = bbox
            # DeepSORT format: ([x, y, w, h], confidence, class)
            detections = [([x, y, w, h], 1.0, 0)]
            
            # İlk başlatma - track oluştur
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            if len(tracks) > 0:
                # İlk track'i al (en yüksek confidence'lı)
                self.current_track_id = tracks[0].track_id
                x1, y1, x2, y2 = tracks[0].to_ltrb()
                self.last_bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                self.initialized = True
                self.frame_count = 0
                return True
            return False
        except Exception as e:
            print(f"DeepSORT init hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update(self, frame):
        """DeepSORT ile tahmin yap - tracking mantığı"""
        if not self.initialized:
            return False, None
        
        try:
            self.frame_count += 1
            
            # YÖNTEM 1: Boş detection ile (sadece tahmin)
            # Problem: DeepSORT track'i hızlıca kaybediyor
            
            # YÖNTEM 2: Son bilinen pozisyonu düşük confidence ile gönder
            # Bu, track'in canlı kalmasını sağlar ama "tahmin" yerine "güncelleme" yapar
            
            if self.last_bbox and self.frame_count % 3 != 0:  # Her 3 frame'den 2'sinde boş gönder
                # Tamamen boş - sadece tahmin
                tracks = self.tracker.update_tracks([], frame=frame)
            else:
                # Arada bir son pozisyonu düşük confidence ile gönder (track'i canlı tut)
                x, y, w, h = self.last_bbox if self.last_bbox else (0, 0, 1, 1)
                detections = [([x, y, w, h], 0.3, 0)]  # Düşük confidence
                tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Bizim track'imizi bul
            for track in tracks:
                if track.track_id == self.current_track_id:
                    # Track'i kontrol et
                    x1, y1, x2, y2 = track.to_ltrb()
                    bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    self.last_bbox = bbox
                    
                    # Track confirmed ve aktif mi?
                    if track.is_confirmed():
                        return True, bbox
                    else:
                        return False, bbox
            
            # Track bulunamadı
            return False, self.last_bbox
            
        except Exception as e:
            print(f"DeepSORT update hatası (frame {self.frame_count}): {e}")
            return False, self.last_bbox
    
    def reset(self):
        """DeepSORT'u sıfırla"""
        super().reset()
        self.tracker = DeepSort(
            max_age=30, 
            n_init=1,  # n_init=1 için güncellendi
            max_cosine_distance=0.3,
            nms_max_overlap=1.0
        )
        self.current_track_id = None
        self.last_bbox = None
        self.frame_count = 0


class TrackerComparator:
    """Tracker'ları karşılaştıran ana sınıf"""
    
    def __init__(self, model_path, device="cuda"):
        """
        Args:
            model_path: YOLO model yolu
            device: 'cuda' veya 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        print(f"YOLO model yüklendi: {model_path}")
        print(f"Cihaz: {self.device}")
    
    def calculate_iou(self, box1, box2):
        """IoU hesapla (x, y, w, h formatında)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # (x,y,w,h) -> (x1,y1,x2,y2)
        box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
        box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
        
        # Kesişim
        x_left = max(box1_xyxy[0], box2_xyxy[0])
        y_top = max(box1_xyxy[1], box2_xyxy[1])
        x_right = min(box1_xyxy[2], box2_xyxy[2])
        y_bottom = min(box1_xyxy[3], box2_xyxy[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Birleşim
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_center_error(self, box1, box2):
        """Merkez noktası hatası (piksel cinsinden)"""
        center1 = [box1[0] + box1[2]/2, box1[1] + box1[3]/2]
        center2 = [box2[0] + box2[2]/2, box2[1] + box2[3]/2]
        
        error = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return error
    
    def detect_yolo(self, frame, conf_thresh=0.4):
        """YOLO ile tespit yap ve en yüksek confidence'lı nesneyi döndür"""
        results = self.model.predict(frame, device=self.device, verbose=False, conf=conf_thresh)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        if len(scores) == 0:
            return None
        
        # En yüksek confidence'lı nesne
        idx = np.argmax(scores)
        x1, y1, x2, y2 = boxes[idx]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        return (x, y, w, h)
    
    def evaluate_tracker(self, tracker_wrapper, video_path, conf_thresh=0.4, 
                        reinit_interval=15, max_frames=None):
        """
        Tek bir tracker'ı değerlendir
        
        Args:
            tracker_wrapper: BaseTrackerWrapper türevi
            video_path: Test video yolu
            conf_thresh: YOLO confidence threshold
            reinit_interval: Kaç frame'de bir YOLO ile yeniden başlat
            max_frames: Test edilecek maksimum frame sayısı (None = tümü)
        
        Returns:
            dict: Metrikler
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Video açılamadı: {video_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"{tracker_wrapper.name} Tracker Test Ediliyor...")
        print(f"{'='*60}")
        
        metrics = {
            'tracker_name': tracker_wrapper.name,
            'iou_scores': [],
            'center_errors': [],
            'processing_times': [],
            'tracking_success_count': 0,
            'tracking_fail_count': 0,
            'reinit_count': 0,
            'total_frames': 0,
            'frames_with_detection': 0
        }
        
        frame_count = 0
        last_yolo_bbox = None
        is_deepsort = tracker_wrapper.name == "DeepSORT"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            
            # Maksimum frame kontrolü
            if max_frames and frame_count > max_frames:
                break
            
            # Belirli aralıklarla veya tracker başlatılmadıysa YOLO tespit
            if frame_count % reinit_interval == 1 or not tracker_wrapper.initialized:
                yolo_bbox = self.detect_yolo(frame, conf_thresh)
                
                if yolo_bbox:
                    tracker_wrapper.init(frame, yolo_bbox)
                    last_yolo_bbox = yolo_bbox
                    metrics['reinit_count'] += 1
                    metrics['frames_with_detection'] += 1
                    print(f"  Frame {frame_count}: YOLO tespit - {yolo_bbox}")
                else:
                    print(f"  Frame {frame_count}: YOLO tespit YOK")
                    tracker_wrapper.reset()
                continue
            
            # YOLO ile gerçek tespiti al (karşılaştırma için)
            yolo_bbox = self.detect_yolo(frame, conf_thresh)
            
            # Tracker ile tahmin yap
            start_time = time.time()
            
            if is_deepsort:
                # DeepSORT için özel: YOLO detection'ını ver ama önceki tahmini al
                success, predicted_bbox = tracker_wrapper.update(frame, yolo_bbox)
            else:
                # Diğer tracker'lar için normal
                success, predicted_bbox = tracker_wrapper.update(frame)
            
            process_time = time.time() - start_time
            metrics['processing_times'].append(process_time)
            
            if yolo_bbox:
                metrics['frames_with_detection'] += 1
                last_yolo_bbox = yolo_bbox
                
                if success and predicted_bbox:
                    # Tracker tahmini vs YOLO tespiti
                    iou = self.calculate_iou(predicted_bbox, yolo_bbox)
                    center_error = self.calculate_center_error(predicted_bbox, yolo_bbox)
                    
                    metrics['iou_scores'].append(iou)
                    metrics['center_errors'].append(center_error)
                    metrics['tracking_success_count'] += 1
                    
                    if frame_count % 50 == 0:
                        print(f"  Frame {frame_count}: IoU={iou:.3f}, Merkez Hata={center_error:.1f}px")
                else:
                    metrics['tracking_fail_count'] += 1
                    if frame_count % 50 == 0:
                        print(f"  Frame {frame_count}: Tracker BAŞARISIZ")
            else:
                if success:
                    metrics['tracking_success_count'] += 1
        
        cap.release()
        
        # Ortalama metrikleri hesapla
        if metrics['processing_times']:
            avg_time = np.mean(metrics['processing_times'])
            metrics['avg_fps'] = 1.0 / avg_time if avg_time > 0 else 0
            metrics['avg_processing_time_ms'] = avg_time * 1000
        
        if metrics['iou_scores']:
            metrics['avg_iou'] = np.mean(metrics['iou_scores'])
            metrics['std_iou'] = np.std(metrics['iou_scores'])
            metrics['min_iou'] = np.min(metrics['iou_scores'])
            metrics['max_iou'] = np.max(metrics['iou_scores'])
            metrics['median_iou'] = np.median(metrics['iou_scores'])
        
        if metrics['center_errors']:
            metrics['avg_center_error'] = np.mean(metrics['center_errors'])
            metrics['std_center_error'] = np.std(metrics['center_errors'])
            metrics['max_center_error'] = np.max(metrics['center_errors'])
        
        if metrics['total_frames'] > 0:
            metrics['success_rate'] = metrics['tracking_success_count'] / metrics['total_frames']
            metrics['fail_rate'] = metrics['tracking_fail_count'] / metrics['total_frames']
        
        print(f"\n{tracker_wrapper.name} Testi Tamamlandı!")
        print(f"  Toplam Frame: {metrics['total_frames']}")
        print(f"  Başarılı: {metrics['tracking_success_count']}")
        print(f"  Başarısız: {metrics['tracking_fail_count']}")
        if metrics['iou_scores']:
            print(f"  Ortalama IoU: {metrics['avg_iou']:.4f}")
        
        return metrics
    
    def compare_all(self, video_path, conf_thresh=0.4, reinit_interval=15, max_frames=None):
        """
        CSRT ve DeepSORT'u karşılaştır
        
        Args:
            video_path: Test video yolu
            conf_thresh: YOLO confidence threshold
            reinit_interval: Kaç frame'de bir yeniden başlat
            max_frames: Test edilecek maksimum frame (None = tümü)
        
        Returns:
            dict: Her tracker için metrikler
        """
        trackers = [
            CSRTTrackerWrapper(),
            DeepSORTTrackerWrapper(),
        ]
        
        all_results = {}
        
        for tracker in trackers:
            results = self.evaluate_tracker(
                tracker,
                video_path,
                conf_thresh=conf_thresh,
                reinit_interval=reinit_interval,
                max_frames=max_frames
            )
            
            if results:
                all_results[tracker.name] = results
        
        return all_results
    
    def print_comparison(self, results):
        """Karşılaştırma sonuçlarını güzel formatta yazdır"""
        print("\n" + "="*100)
        print("TRACKER KARŞILAŞTIRMA SONUÇLARI")
        print("="*100)
        
        # Ana metrikler tablosu
        print(f"\n{'Metrik':<25} {'CSRT':<20} {'DeepSORT':<20} {'Kazanan':<15}")
        print("-"*80)
        
        csrt = results.get('CSRT', {})
        deepsort = results.get('DeepSORT', {})
        
        comparisons = [
            ('Ortalama IoU', 'avg_iou', 'higher'),
            ('IoU Std Sapma', 'std_iou', 'lower'),
            ('Medyan IoU', 'median_iou', 'higher'),
            ('Min IoU', 'min_iou', 'higher'),
            ('Ort. Merkez Hatası (px)', 'avg_center_error', 'lower'),
            ('Max Merkez Hatası (px)', 'max_center_error', 'lower'),
            ('Başarı Oranı (%)', 'success_rate', 'higher'),
            ('Ortalama FPS', 'avg_fps', 'higher'),
            ('İşlem Süresi (ms)', 'avg_processing_time_ms', 'lower'),
        ]
        
        winners = {'CSRT': 0, 'DeepSORT': 0}
        
        for metric_name, metric_key, better in comparisons:
            csrt_val = csrt.get(metric_key, 0)
            deepsort_val = deepsort.get(metric_key, 0)
            
            # Yüzde formatı için
            if 'rate' in metric_key:
                csrt_str = f"{csrt_val*100:.2f}%"
                deepsort_str = f"{deepsort_val*100:.2f}%"
            else:
                csrt_str = f"{csrt_val:.4f}"
                deepsort_str = f"{deepsort_val:.4f}"
            
            # Kazananı belirle (0 değerler varsa onları atla)
            if csrt_val == 0 and deepsort_val == 0:
                winner = "-"
            elif csrt_val == 0:
                winner = 'DeepSORT'
            elif deepsort_val == 0:
                winner = 'CSRT'
            else:
                if better == 'higher':
                    winner = 'CSRT' if csrt_val > deepsort_val else 'DeepSORT'
                else:
                    winner = 'CSRT' if csrt_val < deepsort_val else 'DeepSORT'
            
            if winner != "-":
                winners[winner] += 1
            
            print(f"{metric_name:<25} {csrt_str:<20} {deepsort_str:<20} {winner:<15}")
        
        print("-"*80)
        print(f"\n🏆 GENEL KAZANAN: {max(winners, key=winners.get)} "
              f"({winners[max(winners, key=winners.get)]}/{len(comparisons)} metrikte üstün)")
        print("="*100)
    
    def save_results(self, results, output_path='comparison_results.json'):
        """Sonuçları JSON'a kaydet"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Sonuçlar kaydedildi: {output_path}")


def main():
    """Ana test fonksiyonu"""
    
    # ========== AYARLAR ==========
    VIDEO_PATH = "uav.mp4"
    MODEL_PATH = "uav.pt"
    CONF_THRESH = 0.4
    REINIT_INTERVAL = 15  # Kaç frame'de bir YOLO ile yeniden başlat
    MAX_FRAMES = None  # None = tüm video, veya örn: 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ============================
    
    # Dosya kontrolü
    if not Path(MODEL_PATH).exists():
        print(f"❌ HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        print(f"   Mevcut dizin: {Path.cwd()}")
        print(f"\n   Lütfen MODEL_PATH değişkenini doğru yol ile güncelleyin.")
        print(f"   Örnek: MODEL_PATH = 'path/to/your/uav.pt'")
        return
    
    if not Path(VIDEO_PATH).exists():
        print(f"❌ HATA: Video dosyası bulunamadı: {VIDEO_PATH}")
        print(f"   Mevcut dizin: {Path.cwd()}")
        print(f"\n   Lütfen VIDEO_PATH değişkenini doğru yol ile güncelleyin.")
        return
    
    print("CSRT vs DeepSORT Karşılaştırma Sistemi")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Cihaz: {DEVICE}")
    print(f"Yeniden başlatma aralığı: {REINIT_INTERVAL} frame")
    print(f"Confidence threshold: {CONF_THRESH}")
    print("\n⚠️  NOT: DeepSORT her frame'de YOLO detection alır ama")
    print("   karşılaştırma için önceki frame'deki tahmini kullanır.")
    
    # Comparator oluştur
    comparator = TrackerComparator(MODEL_PATH, device=DEVICE)
    
    # Karşılaştırmayı çalıştır
    results = comparator.compare_all(
        video_path=VIDEO_PATH,
        conf_thresh=CONF_THRESH,
        reinit_interval=REINIT_INTERVAL,
        max_frames=MAX_FRAMES
    )
    
    # Sonuçları göster
    comparator.print_comparison(results)
    
    # Sonuçları kaydet
    comparator.save_results(results, 'csrt_vs_deepsort_results.json')
    
    print("\n✅ Test tamamlandı!")


if __name__ == "__main__":
    main()