# 🛰️ Multi-Tracker Performance Benchmarking Suite

Bu proje; farklı **Object Tracking** algoritmalarını ortak bir veri seti ve ortak değerlendirme metrikleri altında karşılaştırmak için geliştirilmiş bir kıyaslama (benchmark) altyapısı sunar.  
Tracking modelleri; tespit (detection) sonrası **ID koruma, kayıp takibi, kararlılık ve hız** gibi performans kriterleri üzerinden değerlendirilir.

---

## 🎯 Amaç

Farklı takip algoritmalarının gerçek senaryolardaki güçlü ve zayıf yönlerini karşılaştırmak ve görev ihtiyaçlarına göre en uygun takipçiyi seçmeyi kolaylaştırmak.

---

## 🧩 Desteklenen Tracker’lar

| Tracker | Yöntem | Not |
|--------|--------|-----|
| CSRT | DCF Tabanlı | Hedef değişimlerine dayanıklı |
| DeepSORT | Appearance + Motion | ID koruma konusunda güçlü |
| FlowTracker | Optical Flow | Yüksek kare hızı avantajı |
| BoT-SORT | ReID + Kalman | ID Switch düşük, stabil |
| MOSSE | Çok hafif DCF | Çok hızlı, düşük doğruluk |
| NORFAIR | Motion-only | GPU kullanmayan hafif yapı |
| TLD | Track-Learn-Detect | Eski nesil, düşük kararlılık |

---

## 📊 Sonuçlar (Özet Tablo)

| Metrik | CSRT | DeepSORT | FlowTrack | BoT-SORT | MOSSE | NORFAIR | TLD |
|-------|------|----------|-----------|----------|-------|-------|---------|-----|
| **mean IoU** | 0.469 | 0.899 | 0.256 | 1.000 |  0.775 | 0.967 | 0.216 |
| **Precision** | 0.476 | 1.000 | 0.179 | 0.992 | 0.882 | 0.990 | 0.129 |
| **Recall** | 1.000 | 0.874 | 1.000 | 0.950 | 1.000 | 1.000 | 0.958 |
| **F1 Score** | 0.645 | 0.933 | 0.304 | 0.971 | 0.937 | 0.995 | 0.228 |
| **MOTA** | 0.476 | 0.874 | 0.179 | 0.943 | 0.882 | 0.990 | 0.958 |
| **FPS** | 8.66 | 7.77 | 7.68 | **985.26** | 9.96 | 9.96 | 4.20 |

> **BoT-SORT** hız + doğruluk dengesinde en ideal sonuçları verdi.  
> **DeepSORT** ID koruma açısından güçlü.  
> **MOSSE** ve **NORFAIR** oldukça hızlı, düşük sistem maliyetli.  
> **TLD** ve **FlowTracker** modern görevlerde önerilmez.

---

## ⚙️ Ölçülen Performans Göstergeleri

| Metrik | Açıklama |
|-------|----------|
| **mean IoU** | Takip kutusunun doğruluğu |
| **Precision / Recall / F1** | Tespit kalitesi |
| **MOTA** | Toplam takip doğruluk metriği |
| **FPS** | Gerçek zamanlılık ölçütü |
| **Latency (ms)** | Çerçeve gecikmesi |
| **CPU / GPU / RAM Usage** | Sistem kaynak tüketimi |
