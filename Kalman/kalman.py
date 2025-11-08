#env: tracker_SORT
from ultralytics import YOLO

MODEL_PATH = "uav.pt" 
VIDEO_PATH = "uav.mp4" 
OUTPUT_FOLDER = "my_output"

try:
    model = YOLO(MODEL_PATH)
except FileNotFoundError:
    print(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı.")
    exit()

print(f"'{MODEL_PATH}' modeli başarıyla yüklendi. Takip başlatılıyor...")

results = model.track(
    source=VIDEO_PATH,
    save=True,      
    show=True,       
    tracker="botsort.yaml",
    conf=0.25,
    project=OUTPUT_FOLDER,  
    name="run1"             
)

print(f"Kaydedilen sonuçlar '{OUTPUT_FOLDER}/run1' klasöründe bulunur.")
