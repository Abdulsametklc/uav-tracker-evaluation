#env: tracker_SORT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-based UAV Target Tracking -> PX4 Offboard AttitudeRate (MAVSDK)
Parametreler kod içinde ayarlanmıştır.
Video kaydı opsiyonel.
"""

import os
import cv2
import time
import math
import asyncio
import numpy as np
from typing import Optional, Dict

# ----------------------- PID Controller -----------------------
class PID:
    def __init__(self, kp=0.12, ki=0.0, kd=0.02, integ_limit=1000.0, out_limit=45.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integ = 0.0
        self.prev_err = None
        self.integ_limit = float(integ_limit)
        self.out_limit = float(out_limit)

    def reset(self):
        self.integ = 0.0
        self.prev_err = None

    def step(self, err, dt):
        if dt <= 0:
            dt = 1e-6
        p = self.kp * err
        self.integ += err * dt
        self.integ = max(min(self.integ, self.integ_limit), -self.integ_limit)
        i = self.ki * self.integ
        d = 0.0
        if self.prev_err is not None:
            d = self.kd * (err - self.prev_err) / dt
        self.prev_err = err
        out = p + i + d
        out = max(min(out, self.out_limit), -self.out_limit)
        return out

# ----------------------- MAVSDK sender ------------------------
latest_rates: Dict[str, float] = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
rates_lock = asyncio.Lock()

async def mavsdk_sender(connection_url: str,
                        send_hz: float = 30.0,
                        thrust_value: float = 0.0,
                        invert = {"roll": False, "pitch": False, "yaw": False}):
    from mavsdk import System
    from mavsdk.offboard import AttitudeRate

    drone = System()
    print(f"[MAVSDK] Connecting {connection_url} ...")
    await drone.connect(system_address=connection_url)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] Connected to vehicle.")
            break

    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, thrust_value))
    try:
        await drone.offboard.start()
        print("[MAVSDK] Offboard started.")
    except Exception as e:
        print("[MAVSDK] Offboard start failed:", e)
        return

    period = 1.0 / max(1.0, send_hz)
    try:
        while True:
            async with rates_lock:
                r = latest_rates["roll"]
                p = latest_rates["pitch"]
                y = latest_rates["yaw"]

            if invert.get("roll", False):  r = -r
            if invert.get("pitch", False): p = -p
            if invert.get("yaw", False):   y = -y

            def clamp(v, lo, hi): return max(min(v, hi), lo)
            r = clamp(r, -120.0, 120.0)
            p = clamp(p, -120.0, 120.0)
            y = clamp(y, -120.0, 120.0)

            await drone.offboard.set_attitude_rate(
                AttitudeRate(roll_deg_s=r, pitch_deg_s=p, yaw_deg_s=y, thrust_value=thrust_value)
            )
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await drone.offboard.stop()
            print("[MAVSDK] Offboard stopped.")
        except Exception as e:
            print("[MAVSDK] Offboard stop err:", e)

# -------------------- Yardımcı/Overlay -----------------------
def direction_text(dx, dy, dead_px=8):
    horiz = "center"
    vert = "center"
    if dx > dead_px:
        horiz = "go RIGHT"
    elif dx < -dead_px:
        horiz = "go LEFT"
    if dy > dead_px:
        vert = "go DOWN/forward"
    elif dy < -dead_px:
        vert = "go UP/back"
    if horiz == "center" and vert == "center":
        return "ON TARGET"
    return f"{horiz} + {vert}" if horiz != "center" and vert != "center" else (horiz if vert == "center" else vert)

def draw_hud(frame, center, target_pt, cmd, det_txt=None, bbox=None):
    h, w = frame.shape[:2]
    cx, cy = center
    cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if target_pt is not None:
        cv2.circle(frame, target_pt, 8, (0, 0, 255), 2)
        cv2.line(frame, (cx, cy), target_pt, (255, 255, 0), 1)
    lines = [
        f"dir: {cmd['dir']}",
        f"yaw:  {cmd['yaw']:+6.2f}",
        f"pitch:{cmd['pitch']:+6.2f}",
        f"roll: {cmd['roll']:+6.2f}",
    ]
    if det_txt:
        lines.append(det_txt)
    y0 = 24
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

# -------------------------- Main ------------------------------
async def main_async():
    # ---------------- Parametreler ----------------
    MODEL_PATH   = "uav.pt"            # YOLOv8 model
    VIDEO_INPUT  = "uav.mp4"         # "" ise webcam(0)
    VIDEO_OUTPUT = "output.mp4"        # "" ise kaydetme
    SCALE        = 1.0
    CONF         = 0.25
    CLASS_NAME   = "drone"             # takip edilecek sınıf
    DEAD_PX      = 10

    # PID
    YAW_KP, YAW_KD     = 0.14, 0.03
    PITCH_KP, PITCH_KD = 0.16, 0.03
    ROLL_KP, ROLL_KD   = 0.10, 0.02
    OUT_LIMIT          = 45.0

    # MAVSDK
    CONNECTION_URL = "serial://COM5:115200"
    SEND_HZ        = 30.0
    THRUST         = 0.0
    INVERT = {"roll": False, "pitch": False, "yaw": False}

    # ---------------- YOLO ----------------
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    names = model.names

    # sınıf eşleştirme
    prefer_cls = None
    if CLASS_NAME is not None:
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == CLASS_NAME.lower():
                    prefer_cls = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).lower() == CLASS_NAME.lower():
                    prefer_cls = int(i)
                    break

    # video
    cap = cv2.VideoCapture(0 if VIDEO_INPUT == "" else VIDEO_INPUT)
    if not cap.isOpened():
        print("ERROR: Could not open video/camera.")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1.0 or np.isnan(src_fps):
        src_fps = 30.0
    writer = None
    def ensure_writer(frame):
        nonlocal writer
        if VIDEO_OUTPUT and writer is None:
            h, w = frame.shape[:2]
            ext = os.path.splitext(VIDEO_OUTPUT)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))
            writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, src_fps, (w, h))
            if not writer.isOpened():
                print("[WARN] Could not open writer; recording disabled.")
                writer = None
            else:
                print(f"[REC] Saving processed video to: {VIDEO_OUTPUT} @ {src_fps:.1f} FPS, size=({w}x{h})")

    # PID'ler
    pid_yaw   = PID(kp=YAW_KP,   ki=0.0, kd=YAW_KD,   out_limit=OUT_LIMIT)
    pid_pitch = PID(kp=PITCH_KP, ki=0.0, kd=PITCH_KD, out_limit=OUT_LIMIT)
    pid_roll  = PID(kp=ROLL_KP,  ki=0.0, kd=ROLL_KD,  out_limit=OUT_LIMIT)

    # MAVSDK sender
    sender_task = asyncio.create_task(
        mavsdk_sender(connection_url=CONNECTION_URL, send_hz=SEND_HZ, thrust_value=THRUST, invert=INVERT)
    )

    prev_time = time.time()
    prev_center: Optional[tuple] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)

            ensure_writer(frame)

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            now = time.time()
            dt = now - prev_time
            prev_time = now

            # YOLO inference
            results = model(frame, conf=CONF, verbose=False)
            dets = []
            if results and len(results) > 0:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    cls_ids  = r.boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        dets.append((xyxy[i].tolist(), float(confs[i]), int(cls_ids[i])))

            # en iyi hedef
            best = None
            if dets:
                cand = [d for d in dets if (prefer_cls is None or d[2] == prefer_cls)]
                if not cand:
                    cand = dets
                if prev_center is not None:
                    px, py = prev_center
                    cand.sort(key=lambda d: math.hypot(((d[0][0]+d[0][2])//2)-px,
                                                       ((d[0][1]+d[0][3])//2)-py))
                else:
                    cand.sort(key=lambda d: -d[1])
                best = cand[0]

            bbox = None
            target_pt = None
            det_txt = None
            yaw_cmd = pitch_cmd = roll_cmd = 0.0
            dir_txt = "NO TARGET"

            if best is not None:
                (x1, y1, x2, y2), conf_val, cls_id = best
                bbox = (x1, y1, x2, y2)
                if isinstance(names, dict) and cls_id in names:
                    cname = names[cls_id]
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    cname = names[cls_id]
                else:
                    cname = f"id:{cls_id}"
                det_txt = f"det: {cname} {conf_val:.2f}"

                tx = (x1 + x2) // 2
                ty = (y1 + y2) // 2
                target_pt = (tx, ty)
                prev_center = (tx, ty)
                dx_px = tx - cx
                dy_px = ty - cy

                nx = max(min(dx_px / max(1.0, (w/2.0)), 1.0), -1.0)
                ny = max(min(dy_px / max(1.0, (h/2.0)), 1.0), -1.0)

                yaw_cmd   = pid_yaw.step(nx, dt)
                pitch_cmd = pid_pitch.step(ny, dt)
                roll_cmd  = pid_roll.step(nx * 0.6, dt)

                if abs(dx_px) < DEAD_PX:
                    yaw_cmd = 0.0
                    roll_cmd = 0.0
                if abs(dy_px) < DEAD_PX:
                    pitch_cmd = 0.0

                dir_txt = direction_text(dx_px, dy_px, dead_px=DEAD_PX)

            cmd = {"dir": dir_txt, "yaw": yaw_cmd, "pitch": pitch_cmd, "roll": roll_cmd}
            draw_hud(frame, (cx, cy), target_pt, cmd, det_txt=det_txt, bbox=bbox)

            async with rates_lock:
                latest_rates["yaw"] = float(yaw_cmd)
                latest_rates["pitch"] = float(pitch_cmd)
                latest_rates["roll"] = float(roll_cmd)

            cv2.imshow("YOLO -> PX4 Offboard (AttitudeRate)", frame)
            if writer is not None:
                writer.write(frame)

            print(f"[{time.strftime('%H:%M:%S')}] dir={dir_txt:>14s} | yaw={yaw_cmd:+6.2f}  pitch={pitch_cmd:+6.2f}  roll={roll_cmd:+6.2f}  {det_txt or ''}", end="\r")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if writer is not None:
            writer.release()
            print("\n[REC] Saved video to:", VIDEO_OUTPUT if VIDEO_OUTPUT else "(none)")
        cap.release()
        cv2.destroyAllWindows()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        await asyncio.sleep(0)

def run():
    asyncio.run(main_async())

if __name__ == "__main__":
    run()
