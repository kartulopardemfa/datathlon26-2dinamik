# Datathon 2026 — Career Success Score Tahmini

Öğrenci profillerinden (akademik, teknik, proje, portfolyo, mülakat, sosyal beceri
verileri + mentör geri bildirim metni) 0-100 aralığındaki `career_success_score`
hedefini tahmin eden çözüm. Metrik: **MSE**.

## Yaklaşım

### 1. Özellik Mühendisliği (`fe.py`)
- **Beceri agregatları:** 9 teknik beceri skorunun ortalama / min / max / std değerleri.
- **Rol-beceri eşleşmesi:** Her `target_role` için role en uygun beceri alt kümesinin
  ortalaması (`role_skill_mean`) ve genel beceri ortalamasından sapması
  (`role_skill_gap`). Örn. DevOps Engineer için devops + cloud + backend skorları.
- **Deneyim özellikleri:** toplam proje, staj süresi × sayısı, GitHub aktivitesi
  (repo × yıldız), mülakat/başvuru oranı.
- **Etkileşimler:** `project_quality_score` (hedefle en yüksek korelasyonlu özellik,
  r=0.54) × teknik mülakat / beceri ortalaması / iletişim çarpımları.
- **Eksik veri:** Eksik değer sayısı özelliği; GBDT'ler NaN'ı doğal işler.

### 2. Metin Modellemesi (`mentor_feedback_text`)
- **TF-IDF (kelime 1-3 gram + karakter 3-5 gram) → Ridge:** 5-fold OOF tahmini,
  meta-özellik olarak ana modele girer (tek başına korelasyon ≈ 0.61).
- **Çok dilli sentence-transformer embeddingleri**
  (`paraphrase-multilingual-MiniLM-L12-v2`) → Ridge OOF tahmini (ikinci meta-özellik).
- TF-IDF SVD (64 bileşen) ve embedding PCA (32 bileşen) bileşenleri doğrudan
  özellik olarak eklendi.
- Tüm OOF üretimi ana CV ile aynı fold'ları kullanır → **sızıntı yok**.

### 3. Model Topluluğu
Ortak 5-fold CV (seed=42) üzerinde:

| Model | CV MSE |
|---|---|
| LightGBM (2 seed) | ~78.1–78.4 |
| XGBoost (2 seed) | ~78.3 |
| CatBoost (2 seed) | ~76.9–77.1 |
| MLP (çeşitlilik için) | ~103 |
| **Ağırlıklı harman (Nelder-Mead, OOF üzerinde)** | **~76.15** |

- Hedef 100'de kırpılmış (train'de 773 örnek tam 100) → nihai tahminler
  `[0, 100]` aralığına kırpılır.
- Harman ağırlıkları OOF tahminleri üzerinde Nelder-Mead ile optimize edilir.

## Çalıştırma

```bash
pip install scikit-learn pandas numpy lightgbm xgboost catboost sentence-transformers scipy
# data/train.csv, data/test.csv, data/sample_submission.csv dosyalarını yerleştirin
python train_ensemble.py   # submission.csv üretir
```

Yarışma verisi kurallar gereği depoya dahil edilmemiştir (`.gitignore`).

## Süreçteki Ara Sonuçlar (5-fold CV MSE)

| Adım | CV MSE |
|---|---|
| LightGBM baseline (ham özellikler) | 83.40 |
| + özellik mühendisliği + TF-IDF SVD + metin OOF | 79.43 |
| + kelime+karakter TF-IDF, CatBoost | 77.93 |
| 6 GBDT harmanı | 76.31 |
| + MLP ile nihai harman | **76.15** |
