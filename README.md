# Datathon 2026 — Career Success Score: Takım Çalışma Raporu (Kapsamlı)

Bu doküman üç şeyi açıklar: (1) sistemin nasıl çalıştığı — özellikle train/test ayrımı ve
doğrulama mimarisi, (2) tüm denemelerin sonuçları ve başarısızların NEDEN başarısız olduğu,
(3) kalan işlerin adım adım listesi.

---

## 1. Train/Test Ayrımı ve Doğrulama Mimarisi (en çok karışan konu)

**İki ayrı "test" kavramı var, karıştırmayın:**

### a) Yarışmanın test seti (`test.csv`)
- 10.000 satır, `career_success_score` YOK. Bunu hiçbir zaman "böleriz" — tamamı tahmin edilir.
- `colab_allinone.ipynb` içinde `te` değişkenidir. Modeller her zaman train'in tamamı
  (veya fold'ları) üzerinde eğitilir, `te` üzerinde sadece TAHMİN üretilir.

### b) Bizim iç doğrulamamız: 5-Fold Cross-Validation (train.csv İÇİNDE)
- `KFold(5, shuffle=True, random_state=42)` — **her notebook'ta, her modelde AYNI**.
  Bu standartlık kritik: tüm modellerin OOF tahminleri aynı satır-fold eşleşmesini
  kullandığı için harman (blend) ağırlıkları sızıntısız öğrenilebiliyor.
- **OOF (out-of-fold) nedir?** Train 5 parçaya bölünür. Model 4 parçada eğitilir,
  5.'yi tahmin eder; bu 5 kez döner. Sonuçta train'deki HER satır için, o satırı
  hiç görmemiş bir modelin tahmini elde edilir. Modelin gerçek genelleme gücü budur.
- Metin meta-özellikleri (TF-IDF→Ridge tahmini, BERT tahmini, Gemini puanı vb.) de
  aynı fold'larla OOF üretilir → ana modele özellik olarak girerken sızıntı oluşmaz.
- Test tarafı için: 5 fold-modelinin test tahminlerinin ortalaması alınır.

### c) LB_est: Kaggle skorunu önceden bilme sistemimiz
- **Sorun:** Test seti yıllara göre train'den farklı dağılıyor (test'in %62'si 2024-2026;
  train her yıldan eşit). Geç yıllarda hedefin gürültüsü de yüksek (std 12.6 → 18.3).
  Bu yüzden düz CV MSE (≈75) Kaggle skorunu (≈85) TAHMİN EDEMEZ.
- **Çözüm:** OOF hatalarını test setinin yıl oranlarıyla ağırlıklandırıyoruz:
  `LB_est = Σ_yıl w(yıl) × MSE(yıl)`, `w(yıl) = test_oranı(yıl)/train_oranı(yıl)`.
- **Kanıtlanmış isabet:** tahmin 86.4-87.2 → gerçek 87.045; tahmin 85.571 → gerçek
  85.567; tahmin 85.02 → gerçek 85.12. **±0.3'ten iyi.** Artık submission hakkı
  yakmadan her adayın skorunu biliyoruz.

---

## 2. Skor Gelişimi

| Submission | LB_est (tahmin) | Public (gerçek) | İçerik |
|---|---|---|---|
| v2 (ilk) | 86.4-87.2 | **87.045** | 7 model + TF-IDF/embedding meta |
| Kaggle v1 nb | 85.571 | **85.567** | + BERT FT + doldurma bayrakları + dönem-bazlı harman |
| v7 | 85.02 | **85.12** | + AutoGluon + büyük füzyon (28 bileşen) |
| Sıradaki | ~84.5? | — | Hepsi-bir-arada (Optuna + füzyon×2 + Gemini?) |

Mevcut hedef çizgisi: Top-10 barajı public ~82.3, lider 80.76.

---

## 3. Neden Bazı Denemelerin MSE'si Yüksek Çıktı? (başarısızlık analizi)

Yüksek MSE'li denemeler hata değildi — her biri bir hipotezi test etti ve öldürdü.
Mekanizmalarıyla birlikte:

| Deneme | LB_est | Neden yüksek çıktı |
|---|---|---|
| **Büyük füzyon (ham)** | 96.6 | Model bozuk değil (corr=0.80 her yılda). İki sebep: (1) Sinir ağı tablo verisinde GBDT'den doğal olarak zayıf (corr 0.80 vs 0.83); (2) MSE, korelasyon aynıyken hedef varyansıyla büyür — geç yıllarda varyans 2 kat olduğundan aynı corr'lu modelin MSE'si uçar. Yıl-kalibrasyonla 95.0'a indi; harmanda küçük ağırlıkla katkı verdi. **Ders: corr ve MSE farklı şeyler ölçer; gürültülü segmentte MSE tek başına "model kötü" demek değildir.** |
| **İki aşamalı kırpma modeli** | 89.2 | y=100'e yapışık kütleyi ayrı sınıflandırıcıyla modellemek, regresörü y<100 satırlarına mahkûm etti → ana model zayıfladı. Tek model + tahminleri [0,100]'e kırpmak zaten optimuma yakın (y=100 satırlarında ortalama tahminimiz 95.7 — gayet iyi). |
| **CatBoost yerleşik text_features** | 89.3 | CatBoost'un iç metin işleyişi (sözlük/BoW) bizim OOF meta-özellik yaklaşımımızdan zayıf; ham metin kolonu eklemek SVD+meta özellikleri DEĞİŞTİRİNCE bilgi kaybı oldu. |
| **Pseudo-labeling (ilk hali)** | "84.1" → gerçek 87.3 | Klasik sızıntı: test pseudo-etiketleri, doğrulama fold'unu görmüş modellerden üretilmişti; model kendi doğrulama cevabını pseudo-etiketlerin içinden geri okudu. Fold-temiz kurulumda kazanç kayboldu, hatta zarar (-0.4). **En öğretici vaka: CV'de mucize görürseniz önce sızıntı arayın.** |
| **Örnek ağırlıklandırma (yıl bazlı)** | 88.3 (pw=1) / 87.4 (pw=0.3) | Kavram kayması yok, sadece gürültü artışı var. Gürültülü (geç yıl) örneklere ağırlık vermek modele daha çok gürültü fit ettirdi. Optimum: ağırlıksız eğitim + yıl özelliği. |
| **Yıl-normalizasyonu, aspect-sentiment, kNN-hedef, duygu-kapılı çarpımlar** | 87.1-87.8 | Hepsi mevcut özelliklerin taşıdığı bilgiyi farklı biçimde tekrar etti; GBDT zaten öğrenmişti. Yeni bilgi yok → küçük gürültü artışı. |

**Çapraz bulgu:** 4 farklı model ailesinin (LGB/XGB/CatBoost/sinir ağı füzyonu) hata
korelasyonu 0.93. Hepsi aynı sinyal sınırına dayandı. Public 80-82'lerin bir kısmının
istatistiksel şans (winner's curse: 6000 satırlık public örnekleminde binlerce
submission'ın en şanslıları) olduğu hipotezimiz buna dayanıyor — gerçek sıralamayı
private %40 belirleyecek ve bizim skorumuz kalibrasyon gereği orada sapmaz.

---

## 4. Çalışan Bileşenler (mevcut en iyi: 85.12)

1. **Doldurma bayrakları:** 6 beceri kolonunda 10+ ondalıklı değerler = organizatör
   doldurması. Bayrak olarak eklemek -0.34. (Ham CSV'yi string okuyarak bulundu.)
2. **BERT fine-tune meta-özelliği:** BERTurk metin→skor OOF tahmini (corr 0.66) → -0.4.
3. **Dönem-bazlı harman:** Erken (2019-23) ve geç (2024-26) yıllar için AYRI blend
   ağırlıkları, Nelder-Mead ile OOF üzerinde optimize → -0.2.
4. **AutoGluon:** Çok katmanlı stacking, tek başına 85.13'lük bileşen → harmanda -0.4.
5. **Çeşitlilik havuzu:** 28 bileşen (Kaggle 9 + yerel 16 + AG + füzyon ×2).

## 5. `colab_allinone.ipynb` Mimarisi (hücre hücre)

1. **setup:** Drive bağlama, veri okuma, `folds` (KFold 5, seed 42) ve `lb_est` tanımı.
   Checkpoint klasörü: `Drive/datathon_ckpt_final/`.
2. **füzyon ×2 seed:** BERTurk + tablo vektörü tek ağda; fold başına Drive checkpoint
   (kopma olursa kaldığı fold'dan devam). OOF + test tahmini üretir.
3. **metin özellikleri:** TF-IDF (kelime+karakter) Ridge OOF, mpnet embedding Ridge OOF,
   SVD-64, PCA-32, kNN-hedef (embedding uzayında komşu skorları, OOF-temiz).
4. **özellik mühendisliği:** rol-beceri eşleşmesi, etkileşimler, doldurma bayrakları,
   persona ("siz" hitabı), **+ Drive'da `datathon_gemini/` varsa Gemini özellikleri otomatik eklenir.**
5. **Optuna:** CatBoost için 30 denemelik ayar taraması, yıl-ağırlıklı hedefle, 3 fold.
6. **modeller:** Ayarlı CatBoost ×3 seed + LGB ×2 + XGB ×2, hepsi aynı 5 fold, OOF + test.
7. **harman + rapor:** Global ve dönem-bazlı ağırlık optimizasyonu, iyi olan seçilir,
   `submission_allinone.csv` + `O_allinone.npy`/`T_allinone.npy` (Drive'a da) + rapor.

## 6. YAPILACAKLAR (sırayla — buradan takip edin)

- [ ] **Gemini koşusu** (Colab, GPU'suz): düzeltilmiş tek-hücre kod çalışıyor olmalı.
      `>>> ERKEN SINYAL corr=...` satırını Claude'a bildir.
      - corr ≥ 0.70 → altın; 0.50-0.70 → faydalı; < 0.50 → katkı sınırlı ama dosyaları yine de al.
      - Bitince: `Drive/datathon_gemini/llm_feats_train.csv` + `llm_feats_test.csv` → Claude'a yükle.
- [ ] **HEPSİ BİR ARADA koşusu** (Colab A100, ~3 saat): Gemini bittikten SONRA başlat
      (özellikleri otomatik alır). Kopma olursa Run All ile devam.
- [ ] Bitince üç şey: (1) `submission_allinone.csv` → Kaggle'a yükle, skoru bildir;
      (2) `O_allinone.npy` + `T_allinone.npy` → Claude'a yükle (son mega-harman için);
      (3) rapor bloğunu yapıştır.
- [ ] **13 Haziran akşamı:** Claude son mega-harmanı kurar → son aday dosya Kaggle'a yüklenir.
- [ ] **14 Haziran (son gün):** Kaggle'da "Submit Predictions → My Submissions" üzerinden
      **2 final submission seç**: (1) en düşük LB_est'li dosya, (2) ona en az benzeyen
      ikinci en iyi (çeşitlilik sigortası). Seçimi LB_est'e göre yapacağız, public skora değil.

## 7. Dosya Rehberi

| Dosya | Ne işe yarar |
|---|---|
| `colab_allinone.ipynb` | **ANA NOTEBOOK** — her şey tek koşuda (bölüm 5) |
| `colab_gemini.ipynb` / chat'teki tek-hücre kod | Gemini metin anotasyonu |
| `colab_bigfusion_v2.ipynb` | xlm-r-large füzyon (koşuldu; çıktıları v7'de) |
| `kaggle_gpu_pipeline.ipynb` (v1) | İlk GPU pipeline (koşuldu → 85.567) |
| `kaggle_full_v2.ipynb` / `kaggle_fusion_v3.ipynb` / `kaggle_final.ipynb` | Ara sürümler (referans) |
| `fe.py`, `train_ensemble*.py`, `breakthrough_test.py`, `pseudo_clean.py`, `noise_leak_test.py`, `ag_train.py` | Yerel deney kodları (bölüm 3'teki kanıtlar) |
| `submission_v7_fusion.csv` | Şu anki en iyi yüklenmiş dosya (85.12) |
