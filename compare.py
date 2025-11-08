import psutil
import torch
import time
import GPUtil
import numpy as np


# Tracker modülleri
from CSRT.csrt_track_c import run_csrt
from DeepSORT.deepsort_track_c import run_deepsort
from FlowTracker.flow_track_c import run_flowtracker
from Kalman.kalman_track_c import run_botsort
from MDNet.mdnet_track_c import run_mdnet
from MOSSE.mosse_track_c import run_mosse
from Norfair.norfair_track_c import run_norfair
#from SORT.sort_track_c import run_sort
from TLD.tld_track_c import run_tld


video_path = "uav.mp4"
model_path = "uav.pt"

# ÇALIŞTIRMA MODLARI
RUN_ALL = True       # Hepsini çalıştır
TEST_SINGLE = False     # Tek tracker testi
TEST_TRACKER = "" # CSRT, DeepSORT, FlowTrack, BoT-SORT, MDNet, MOSSE, Norfair, SORT


def measure_system_usage(duration=1.0):
    cpu_usage = psutil.cpu_percent(interval=duration)
    ram_usage = psutil.virtual_memory().used / (1024 ** 2)
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0
    return cpu_usage, gpu_usage, ram_usage


def print_header():
    print("\n=== TRACKER KARŞILAŞTIRMA SONUÇLARI ===")
    print(
        f" {'Metrik':<20} | {'CSRT':<10} | {'DeepSORT':<10} | {'FlowTrack':<10} | "
        f"{'BoT-SORT':<10} | {'MDNet':<10} | {'MOSSE':<10} | {'NORFAIR':<10} | {'TLD':<10}"
    )
    print("-" * 130)


def show(metric, *metrics_list):
    values = [m.get(metric, None) for m in metrics_list]
    if all(v is not None for v in values):
        print(f"{metric:<20} | " + " | ".join(f"{v:<10.3f}" for v in values))


def main():
    base_cpu, base_gpu, base_ram = measure_system_usage(0.5)
    print(f"\nSistem başlangıç durumu -> CPU: {base_cpu:.1f}%, GPU: {base_gpu:.1f}%, RAM: {base_ram:.1f} MB")

    # ---- TEK TRACKER TEST MODU ----
    if TEST_SINGLE:
        print(f"\n>>> TEK TRACKER TEST MODU AKTİF: {TEST_TRACKER}\n")

        tracker_functions = {
            "CSRT": run_csrt,
            "DeepSORT": run_deepsort,
            "FlowTrack": run_flowtracker,
            "BoT-SORT": run_botsort,
            "MDNet": run_mdnet,
            "MOSSE": run_mosse,
            "Norfair": run_norfair,
            #"SORT": run_sort
            "TLD": run_tld
        }

        if TEST_TRACKER not in tracker_functions:
            print("HATA: Tanınmayan tracker adı!")
            return

        metrics = tracker_functions[TEST_TRACKER](model_path, video_path)

        print("\n--- TEK TRACKER SONUÇLARI ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<20} : {value:.4f}")
            else:
                print(f"{key:<20} : {value}")
        return


    # ---- TÜM TRACKER KARŞILAŞTIRMA MODU ----
    print("\nCSRT Tracker çalıştırılıyor...")
    csrt_metrics = run_csrt(video_path, model_path)
    time.sleep(1)

    print("DeepSORT Tracker çalıştırılıyor...")
    deepsort_metrics = run_deepsort(video_path, model_path)
    time.sleep(1)

    print("FlowTrack Tracker çalıştırılıyor...")
    flowtrack_metrics = run_flowtracker(video_path, model_path)
    time.sleep(1)

    print("BoT-SORT Tracker çalıştırılıyor...")
    botsort_metrics = run_botsort(video_path, model_path)
    time.sleep(1)

    print("MDNet Tracker çalıştırılıyor...")
    mdnet_metrics = run_mdnet(video_path, model_path)
    time.sleep(1)

    print("MOSSE Tracker çalıştırılıyor...")
    mosse_metrics = run_mosse(video_path, model_path)
    time.sleep(1)

    print("Norfair Tracker çalıştırılıyor...")
    norfair_metrics = run_norfair(video_path, model_path)
    time.sleep(1)

    print("SORT Tracker çalıştırılıyor...")
    tld_metrics = run_tld(model_path, video_path)  # ✅ düzeltilmiş
    time.sleep(1)

    print_header()

    metrics = [
        "mean_iou", "precision", "recall", "f1_score", "mota",
        "mean_fps", "mean_latency_ms", "cpu_usage", "gpu_usage", "ram_usage"
    ]

    for m in metrics:
        show(m, csrt_metrics, deepsort_metrics, flowtrack_metrics, botsort_metrics, mdnet_metrics, mosse_metrics, norfair_metrics, tld_metrics)

    print("\nKarşılaştırma tamamlandı ✅")


if __name__ == "__main__":
    main()
