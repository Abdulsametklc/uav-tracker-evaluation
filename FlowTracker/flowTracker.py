#env: tracker_SORT
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "uav.pt"
VIDEO_PATH = "uav.mp4"
OUTPUT_PATH = "flow_tracked.mp4"
CONF_THRES = 0.4

# YOLO modeli
yolo = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# VideoWriter ayarları
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

tracker_initialized = False
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracker_initialized:
        # YOLO tespiti
        results = yolo.predict(source=frame, conf=CONF_THRES, verbose=False)
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            bbox = (x1, y1, x2-x1, y2-y1)
            
            # Feature points çıkar
            mask = np.zeros_like(frame_gray)
            mask[y1:y2, x1:x2] = 255
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=200, qualityLevel=0.01, minDistance=3, blockSize=3)
            tracker_initialized = True
            old_gray = frame_gray.copy()
    else:
        # Optical Flow ile takip
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]

        if len(good_new) > 0:
            dx = np.mean(good_new[:,0] - good_old[:,0])
            dy = np.mean(good_new[:,1] - good_old[:,1])
            x, y, w_box, h_box = bbox
            x += int(dx)
            y += int(dy)
            bbox = (x, y, w_box, h_box)

        else:
            # Eğer hiç point kalmadıysa YOLO ile yeniden tespit yap
            tracker_initialized = False
            print("Feature point kalmadı, YOLO ile yeniden tespit yapılacak.")
            continue


        # draw
        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0,255,0), 2)

        # update for next frame
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    out.write(frame)
    cv2.imshow("YOLO + Flow Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Takip edilen video kaydedildi:", OUTPUT_PATH)
