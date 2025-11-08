#env: tracker_SORT
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort # SORT kütüphanesini içe aktar

# ========== KULLANICI AYARLARI ==========
VIDEO_PATH = "uav.mp4"       # Kaynak video dosyasının yolu
MODEL_PATH = "uav.pt"        # Eğitilmiş YOLOv8 modelinin (.pt) yolu
OUTPUT_PATH = "output_sort.mp4" # Takip sonuçlarının kaydedileceği dosya yolu

# Cihazı ayarla (varsa CUDA, yoksa CPU kullan)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {DEVICE}")

# Tespit için güven eşiği (confidence threshold)
CONF_THRESH = 0.4
# ========================================

def main():
    # --- 1. Model ve Video Kurulumu ---
    
    # YOLOv8 modelini yükle
    model = YOLO(MODEL_PATH)
    
    # Video dosyasını aç
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"HATA: Video açılamadı! Yol: {VIDEO_PATH}")
        return

    # Video özelliklerini al (genişlik, yükseklik, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video yazıcısını (writer) oluştur
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- 2. SORT Takipçisini Başlatma ---
    
    # max_age: Bir nesnenin kaç kare boyunca tespit edilemezse takipten çıkarılacağını belirtir.
    # min_hits: Bir nesnenin takibe alınması için kaç kare üst üste tespit edilmesi gerektiğini belirtir.
    # iou_threshold: Nesneleri eşleştirmek için kullanılan IoU (Intersection over Union) eşiği.
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    frame_count = 0

    # --- 3. Ana Video İşleme Döngüsü ---
    while True:
        # Videodan bir kare (frame) oku
        ok, frame = cap.read()
        if not ok:
            print("Video akışı sona erdi.")
            break
            
        frame_count += 1

        # --- A. Nesne Tespiti (YOLOv8) ---
        # Modeli kullanarak karedeki nesneleri tespit et
        results = model.predict(frame, device=DEVICE, verbose=False)
        
        # Tespit sonuçlarını SORT'un istediği formata dönüştür
        detections_for_sort = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                # Sadece güven skoru eşikten yüksek olanları al
                if box.conf[0] > CONF_THRESH:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].cpu().numpy()
                    
                    # SORT algoritması [x1, y1, x2, y2, score] formatında bir liste bekler
                    detections_for_sort.append([x1, y1, x2, y2, score])

        # Eğer herhangi bir tespit varsa, numpy dizisine çevir
        if len(detections_for_sort) > 0:
            detections_np = np.array(detections_for_sort)
        else:
            detections_np = np.empty((0, 5))

        # --- B. Nesne Takibi (SORT) ---
        # Tespit edilen nesneleri SORT takipçisine gönder ve takip sonuçlarını al
        # tracker.update() metodu [x1, y1, x2, y2, track_id] formatında bir dizi döndürür.
        tracked_objects = tracker.update(detections_np)

        # --- C. Sonuçları Görselleştirme ---
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            
            # Koordinatları ve ID'yi tam sayıya çevir
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            # Nesnenin etrafına bir kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Nesnenin takip ID'sini kutunun üzerine yaz
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # İşlenen kareyi çıktı videosuna yaz
        out.write(frame)

        # İşlenen kareyi ekranda göster
        cv2.imshow("YOLOv8 + SORT Tracker", frame)

        # Çıkış için 'ESC' tuşuna basılmasını bekle
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --- 4. Temizleme ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"İşlem tamamlandı. Çıktı şu dosyaya kaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
