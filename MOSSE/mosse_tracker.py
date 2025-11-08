#env: tracker_SORT
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- YAPILANDIRMA PARAMETRELERİ ---

# 1. MODEL VE VİDEO YOLLARI
YOLO_MODEL_PATH = "uav.pt"  # Kendi .pt dosyanızın yolunu buraya girin
VIDEO_SOURCE = "uav.mp4"   # Takip edilecek video dosyasının yolu (veya 0 webcam için)

# 2. İHA SINIFI (COCO'ya göre değil, kendi modelinizin çıktısına göre ayarlayın)
# Kendi modelinizde İHA hangi indekste (örneğin: 0, 1, 5 vb.)
IHA_CLASS_ID = 0 

# 3. YENİDEN TESPİT (RE-DETECTION) AYARLARI
# MOSSE takipçisinin ne zaman başarısız olduğunu anlamak için kullanılan eşik
# 'success' değişkeni (True/False) sadece başlangıçta kullanılır, ancak 
# pratik uygulamada PSR (Peak-to-Sidelobe Ratio) kullanılır. Bu örnekte basitleştirilmiştir.
# Takip başarısız olursa (False dönerse) veya belirli aralıklarla YOLO çalışır.
RE_DETECTION_INTERVAL = 30  # Her 30 karede bir YOLO ile doğrulama yap.
YOLO_CONF_THRESHOLD = 0.6   # YOLO tespiti için minimum güven skoru.

# --- GLOBAL DEĞİŞKENLER ---
tracker_initialized = False
tracker = None
frame_counter = 0
current_bbox = None 

def initialize_mosse_tracker(frame, bbox):
    """MOSSE takipçisini verilen kare ve sınırlayıcı kutu ile başlatır."""
    global tracker
    
    # MOSSE takipçisi oluşturulur
    # mosse_tracker.py, initialize_mosse_tracker fonksiyonu içinde

    # cv2.legacy'den çağırmayı deneyin, bazı sürümlerde buraya taşınmıştır
    # Eğer hata verirse bu satırı silip sadece cv2.TrackerMOSSE.create() kalsın.

    if hasattr(cv2, 'TrackerMOSSE_create'):
        tracker = cv2.TrackerMOSSE_create()
    elif hasattr(cv2, 'TrackerMOSSE'):
        tracker = cv2.TrackerMOSSE.create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE'):
        tracker = cv2.legacy.TrackerMOSSE.create()
    else:
        raise Exception("Hata: MOSSE takipçisi OpenCV'de bulunamadı. Lütfen 'opencv-contrib-python'ın doğru yüklendiğinden emin olun.")
    
    # Bbox'ı tamsayılara yuvarlayın (tracker'lar int değerler bekler)
    bbox = tuple(map(int, bbox))
    x, y, w, h = map(int, bbox)
    bbox = (x, y, w, h)
    # Takipçiyi başlatın
    success = tracker.init(frame, bbox)
    return success, tracker

def detect_uav_with_yolo(model, frame, target_class_id, conf_threshold):
    """YOLO modelini kullanarak İHA'yı tespit eder ve en iyi kutuyu döndürür."""
    results = model(frame, verbose=False) # verbose=False ile çıktıları gizler

    best_conf = -1
    best_bbox = None

    for r in results:
        # Pytorch'un tensor çıktısını al ve numpy'a çevir
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            cls = int(classes[i])
            conf = confs[i]

            if cls == target_class_id and conf > conf_threshold:
                # Sınıf İHA ve güvenilir ise
                if conf > best_conf:
                    best_conf = conf
                    # YOLO çıktı formatı [x1, y1, x2, y2]
                    x1, y1, x2, y2 = boxes[i]
                    # OpenCV tracker formatı: (x, y, w, h)
                    w = x2 - x1
                    h = y2 - y1
                    best_bbox = (x1, y1, w, h)
    
    return best_bbox

# --- ANA İŞLEM DÖNGÜSÜ ---

# YOLO Modelini Yükle
try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO modeli ({YOLO_MODEL_PATH}) başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: YOLO modeli yüklenemedi. {e}")
    exit()

# Video Akışını Başlat
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Hata: Video kaynağı ({VIDEO_SOURCE}) açılamadı.")
    exit()

# --- Video Kayıt Ayarları ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
print("Video kaydı: output.mp4 olarak başlatıldı.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    if not tracker_initialized or frame_counter % RE_DETECTION_INTERVAL == 0:
        print(f"--- Kare {frame_counter}: YOLO ile İHA tespiti yapılıyor...")
        
        # YOLO'yu DOĞRU ARGÜMANLARLA çalıştır
        yolo_bbox = detect_uav_with_yolo(model, frame, IHA_CLASS_ID, YOLO_CONF_THRESHOLD)
        
        if yolo_bbox:
            # İHA tespit edildi. MOSSE'yi bu kutuyla yeniden başlat.
            # NOT: Bu kısım önceki yanıtta eklenen hata kontrolünü içerir.
            success_mosse, tracker = initialize_mosse_tracker(frame, yolo_bbox)
            
            if success_mosse:
                tracker_initialized = True
                current_bbox = yolo_bbox
                print("MOSSE takipçisi başarıyla başlatıldı/yeniden başlatıldı.")
            else:
                tracker_initialized = False
                print("MOSSE başlatma hatası!")
        else:
            # İHA tespit edilemedi. Takip durdurulur.
            tracker_initialized = False
            print("YOLO İHA'yı tespit edemedi.")
    
    # Eğer takipçi başlatıldıysa, YOLO'yu çalıştırmadan hızlı takip yap
    elif tracker_initialized:
        # Takipçiyi güncelle (burada asıl MOSSE algoritması çalışır)
        (success_track, mosse_bbox) = tracker.update(frame)

        if success_track:
            # Takip başarılı
            current_bbox = mosse_bbox
            # print(f"Kare {frame_counter}: MOSSE Takip Başarılı.")
        else:
            # Takip başarısız (nesne kayboldu/hata oluştu), bir sonraki karede YOLO çalıştırılacak
            tracker_initialized = False
            print(f"Kare {frame_counter}: MOSSE Takip Kaybı. YOLO'yu bekliyor...")
    
    if current_bbox:
        p1 = (int(current_bbox[0]), int(current_bbox[1]))
        p2 = (int(current_bbox[0] + current_bbox[2]), int(current_bbox[1] + current_bbox[3]))
        
        # Kırmızı renk (BGR), kalınlık 2
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2) 
        cv2.putText(frame, "TAKIP: IHA (MOSSE)", (p1[0], p1[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TAKIP YOK", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    out.write(frame)
    cv2.imshow("YOLO + MOSSE IHA Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme
cap.release()
cv2.destroyAllWindows()
print("Takip tamamlandı ve sistem kapatıldı.")