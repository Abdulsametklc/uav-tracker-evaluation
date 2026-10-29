<div align="center">

# 🛰️ Multi-Tracker Performance Benchmarking Suite
**Object Tracking algoritmalarını tarafsız ve tek çatı altında karşılaştıran benchmark framework**

---

### 🧪 Desteklenen Tracker'lar
`CSRT` · `DeepSORT` · `FlowTracker` · `BoT-SORT` · `MOSSE` · `NORFAIR` · `TLD`

---

<img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/OpenCV-Enabled-green?style=for-the-badge">
<img src="https://img.shields.io/badge/RealTime-Yes-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Platform-Linux/Windows-purple?style=for-the-badge">

</div>

---

## 🎯 Amaç
Gerçek senaryolarda **görev ihtiyaçlarına göre ideal takip algoritmasını seçmeyi** kolaylaştırmak.

---

## 📊 Performans Karşılaştırma Sonuçları

| Metrik | CSRT | DeepSORT | FlowTrack | BoT-SORT | MOSSE | NORFAIR | TLD |
|-------|------|---------|-----------|-----------|--------|---------|-----|
| **mean IoU** | 0.469 | 0.899 | 0.256 | **1.000** | 0.775 | 0.967 | 0.216 |
| **Precision** | 0.476 | **1.000** | 0.179 | 0.992 | 0.882 | 0.990 | 0.129 |
| **Recall** | **1.000** | 0.874 | **1.000** | 0.950 | **1.000** | **1.000** | 0.958 |
| **F1 Score** | 0.645 | 0.933 | 0.304 | **0.971** | 0.937 | **0.995** | 0.228 |
| **MOTA** | 0.476 | 0.874 | 0.179 | **0.943** | 0.882 | **0.990** | 0.958 |
| **FPS** | 8.66 | 7.77 | 7.68 | **985.26** | 9.96 | 9.96 | 4.20 |

---

## 🔍 Kısa Değerlendirme

| Sonuç | Açıklama |
|------|----------|
| **BoT-SORT** | Hız + doğruluk dengesi en iyi |
| **DeepSORT** | ID koruma güçlü, stabil |
| **MOSSE / NORFAIR** | Çok hızlı, düşük donanım gereksinimi |
| **TLD / FlowTracker** | Modern görevlerde önerilmez |

---

## 🧩 Tracker Açıklamaları

| Tracker | Yöntem | Özellik |
|--------|--------|---------|
| **CSRT** | DCF Tabanlı | Hedef değişimlerine dayanıklı |
| **DeepSORT** | Appearance + Motion | Güçlü ID koruma |
| **FlowTracker** | Optical Flow | Düşük doğruluk, yüksek FPS |
| **BoT-SORT** | ReID + Kalman | Düşük ID Switch, stabil |
| **MOSSE** | Hafif DCF | Çok hızlı |
| **NORFAIR** | Motion-only | GPU gerektirmez |
| **TLD** | Track-Learn-Detect | Eski nesil |

---

## ⚙️ Ölçülen Metrikler

| Metrik | Açıklama |
|-------|----------|
| **IoU** | BBox doğruluk değerlendirmesi |
| **F1 Score** | Tespit doğruluk dengesi |
| **MOTA** | Genel izleme başarımı |
| **FPS / Latency** | Gerçek zamanlılık ölçütü |
| **CPU / GPU / RAM** | Sistem kaynak tüketimi |

---

## 🚀 Kurulum & Çalıştırma

```bash
git clone https://github.com/Abdulsametklc/uav-tracker-evaluation.git
cd uav-tracker-evaluation
pip install -r requirements.txt
python compare.py
```

---

## ⚠️ Bilinen Sınırlamalar (Ekim 2026)

Bu ilk sürümde metrik hesapları tracker'lar arasında tutarlı değil:

- Bazı tracker'larda IoU, elle etiketlenmiş referans (`test_data/Video_01/groundtruth.txt`) yerine dedektörün kendi kutusuyla hesaplanıyor.
- BoT-SORT'ta FPS ölçümü tespit süresini kapsamıyor.

Bu yüzden tablodaki değerler kesin sonuç değil, **kaba bir eleme** olarak yorumlanmalı.

**Planlanan v2:** ortak tespit çıktıları, ground truth tabanlı değerlendirme (TrackEval: MOTA, IDF1, HOTA), ByteTrack / OC-SORT eklenmesi ve Jetson üzerinde uçtan uca FPS ölçümü.
