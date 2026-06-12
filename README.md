# Datathon 2026 — Career Success Score: Takım Çalışma Raporu

Bu doküman, yarışmadaki tüm denemelerimizi, neyin işe yarayıp neyin yaramadığını ve
liderle aramızdaki farkın olası kaynaklarını **öğrenme amaçlı** anlatır.

---

## 1. Problemin Yapısı (önce bunu anlayın)

- Hedef: `career_success_score` (0-100, sürekli). Metrik: **MSE** (düşük = iyi).
- 10k train / 10k test. Sayısal + kategorik + Türkçe mentor metni.
- **Kritik keşif #1 — zamansal kayma:** Train her yıldan eşit (~1250/yıl, 2019-2026)
  örnek içerirken, test 2024-2026'ya yığılmış (%62). Üstelik geç yıllarda hedefin
  standart sapması 12.6'dan 18.3'e çıkıyor (daha gürültülü).
- **Sonuç:** Rastgele K-Fold CV puanı (76) public LB'yi (87) TAHMİN EDEMEZ. CV
  hatalarını test setinin yıl dağılımıyla ağırlıklayınca LB'yi ±0.3 isabetle tahmin
  eden bir gösterge kurduk (`LB_est`). İlk submission bunu doğruladı:
  tahmin 86.4-87.2 → gerçek 87.045.
  **Ders: CV şemanız test dağılımını yansıtmıyorsa, CV puanınız bir hayal ürünüdür.**

- **Kritik keşif #2 — kırpılmış hedef:** 773 örnek tam 100.00 (tavana yapışık).
  Tahminleri [0,100]'e kırpmak bedava kazanç.

- **Kritik keşif #3 — organizatör doldurması:** 6 beceri kolonunda değerlerin bir
  kısmı 2 ondalık yerine 10+ ondalık basamaklı (örn. `63.0418967196188`). Bunlar
  sonradan doldurulmuş sentetik değerler. Ağaç modelleri ondalık uzunluğunu göremez;
  "bu değer doldurulmuş" bayrağı eklemek **-0.34 LB** kazandırdı.
  **Ders: Ham CSV'ye string olarak bakın. Sayıların yazım biçimi bile bilgidir.**

## 2. Pipeline (mevcut en iyi: LB_est ≈ 86.2)

```
özellik müh. (rol-beceri eşleşmesi, etkileşimler, eksiklik sayısı)
+ doldurma bayrakları
+ metin: TF-IDF(kelime+karakter)→Ridge OOF, mpnet embedding→Ridge OOF,
         fine-tune MiniLM→OOF, SVD-64, PCA-32
→ CatBoost/LGB/XGB (2 seed) + MLP
→ harman ağırlıkları OOF üzerinde Nelder-Mead ile, YIL-AĞIRLIKLI MSE hedefine göre
→ erken/geç dönem için AYRI harman ağırlıkları
→ tahminler [0,100] kırpılır
```

## 3. Denenip BAŞARISIZ Olanlar (ve nedenleri — en öğretici kısım)

| Deneme | Sonuç | Neden işe yaramadı |
|---|---|---|
| Örnek ağırlıklandırma (test yıl dağılımına göre) | 87.2→88.3 (kötüleşti) | Kavram kayması yok, sadece gürültü artışı var. Gürültülü örneklere ağırlık vermek modeli BOZAR. |
| Yıl-bazlı hedef normalizasyonu | etkisiz | Model zaten yıl özelliğiyle koşullu ortalamayı öğreniyor. |
| İki aşamalı model (P(y=100) sınıflandırıcı + regresör) | 89.2 (kötü) | Kırpma zaten tek modelle + clip ile daha iyi yakalanıyor. |
| Aspect-sentiment özellikleri (metinden beceri-bazlı ± çıkarımı) | etkisiz | TF-IDF n-gramları aynı bilgiyi zaten taşıyor. |
| **Pseudo-labeling** | sahte -2.8 kazanç! | **EN ÖNEMLİ DERS:** Test tahminlerini eğitime katınca CV 84.1 gösterdi. Ama pseudo-etiketler, doğrulama fold'unu görmüş modellerden üretilmişti → sızıntı. Fold-temiz kurulumda gerçek etki: 87.26 (kötüleşme). CV'de "mucize" görürseniz önce sızıntı arayın. |
| MiniLM fine-tune (metin→skor) | tek başına iyi (corr 0.64), harmanda ~0 katkı | Metnin sinyali tablo özellikleriyle büyük ölçüde örtüşüyor. |
| Türkçe BERT (GPU) | fold0: 134.7 (MiniLM 137.7) | Marjinal — standart fine-tune ile metin tavanı corr ≈ 0.65. |
| ID/sıra/şablon sızıntı taramaları | temiz | Sentetik veri düzgün karılmış. |

## 4. Liderle Fark Nereden Geliyor? (analiz)

Lider 80.76, biz ~86.2 (LB_est). Korkutucu görünen bu fark, açıklanan varyans dilinde
**sadece ~%2**: public varyans ~270 → lider %70.1'ini, biz %68'ini açıklıyoruz.
Yani aradaki şey tek bir "sihirli formül" değil; bizim sıkamadığımız bir bilgi
kaynağı. Eleme sonrası kalan adaylar:

1. **Metin × tablo ORTAK modelleme** (en güçlü aday): Mentor metni muhtemelen
   hangi özelliklerin gizli formülde ağır bastığını İMA ediyor ("SQL'i dikkat
   çekici" → SQL ağırlığı yüksek). Bunu yakalamak metni ve tabloyu AYNI modelde
   birleştirmeyi gerektirir. Test edilenler: CatBoost `text_features` (H1),
   duygu-kapılı çarpımlar (H2). Kaggle notebook v2 bunları ölçüyor.
2. **kNN-hedef özellikleri** (H3): embedding uzayında komşu skorları — sentetik
   veride lokal yapıyı ağaçlardan iyi yakalayabilir.
3. **Derin stacking** (AutoGluon tarzı çok katman) — kümülatif küçük kazançlar.

## 5. Durum ve Sonraki Adımlar

- [x] LB tahmin göstergesi kuruldu (±0.3 doğrulukta) → submission israfı yok
- [x] v3 submission: LB_est ~86.4 (gönderilmeye hazır/gönderildi)
- [ ] Kaggle GPU notebook v1 raporu (BERT + temel pipeline) — **bekleniyor**
- [ ] Kaggle notebook v2 raporu (H1/H2/H3 testleri) — **bekleniyor**
- [ ] AutoGluon sonucu — eğitimde
- [ ] Rapor sonuçlarına göre: en iyi bileşenlerle final submission (son gün: 14 Haziran)

## 6. Takım İçin Pratik Dersler

1. **Önce metriği ve test dağılımını anlayın, sonra model kurun.** Bizim ilk 24
   saatlik "CV 76!" sevincimiz, yanlış dağılımda ölçüm yapmaktandı.
2. **Her iyileştirmeyi TEK değişkenle test edin** ve LB_est ile ölçün; LB hakkı yakmayın.
3. **Sızıntı her yerde:** pseudo-labeling vakası ders kitaplık örnek.
4. **Ham veriye string olarak bakın:** ondalık uzunluğu bile özellik olabilir.
5. **Negatif sonuç da sonuçtur:** Yukarıdaki başarısızlar tablosu, neyi DENEMEYECEĞİMİZİ
   söyleyerek zaman kazandırır.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `fe.py` | Özellik mühendisliği |
| `train_ensemble.py` / `train_ensemble_v2.py` | Uçtan uca ensemble pipeline'ları |
| `finetune_text.py` / `finetune_text_tr.py` | Transformer fine-tune (MiniLM / BERTurk) |
| `breakthrough_test.py` | H1/H2/H3 hipotez testleri |
| `ag_train.py` | AutoGluon deneyi |
| `pseudo_clean.py` | Fold-temiz pseudo-labeling (sızıntı kanıtı) |
| `kaggle_gpu_pipeline.ipynb` | Kaggle GPU notebook v1 (BERT + pipeline + submission) |
| `kaggle_full_v2.ipynb` | Kaggle notebook v2 (tam deney seti + submission) |
| `solution.ipynb` | İlk sürüm jüri notebook'u |
