# CV - İş İlanı Eşleştirme Sistemi

BLM4514 Özel Konular dersi dönem projesi.

## Proje Hakkında

CV'ler ile iş ilanları arasındaki anlamsal benzerliği farklı NLP yöntemleriyle ölçen bir eşleştirme sistemi geliştirildi. Anahtar kelime tabanlı arama yerine vektör uzayı modellerine dayalı bir yaklaşım benimsendi.

## Kullanılan Yöntemler

- **TF-IDF + Kosinüs Benzerliği** — sklearn ile klasik vektör uzayı modeli
- **PPMI + SVD** — ortak-oluşum matrisinden kelime gömmeleri (sıfırdan yazıldı)
- **Sentence-BERT** — all-MiniLM-L6-v2 modeli ile derin öğrenme tabanlı gömme

## Veri Seti

- `resume_data.csv` — Kaggle'dan alınan 962 CV kaydı, 50 tanesi kullanıldı
- `alldata.csv` — 6.964 veri bilimi iş ilanı, 200 tanesi kullanıldı
- `fulltime*.csv` — 15 şehre ait tam zamanlı iş ilanları, toplam 400 tanesi kullanıldı

Kaggle kaynakları:
- saugataroyarghya/resume-dataset
- sl6149/data-scientist-job-market-in-the-us

## Dosya Yapısı

```
cv-matching/
├── kod/
│   ├── main.py            # Ana çalıştırma dosyası
│   ├── data_loader.py     # Veri yükleme
│   ├── preprocess.py      # Metin ön işleme
│   ├── tfidf_model.py     # TF-IDF modeli
│   ├── word2vec_model.py  # PPMI+SVD modeli
│   ├── sbert_model.py     # Sentence-BERT modeli
│   └── evaluation.py     # Değerlendirme metrikleri
├── veri/                  # Ham veri setleri
├── sonuclar/              # Model çıktıları
└── requirements.txt
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python kod/main.py
```

## Sonuçlar

| Yöntem | Cosine Sim Avg | MRR | P@1 | P@3 | P@5 |
|---|---|---|---|---|---|
| TF-IDF | 0.847 | 0.742 | %72 | %78 | %80 |
| PPMI+SVD | 0.412 | 0.301 | %28 | %34 | %38 |
| SBERT | 0.823 | 0.798 | %76 | %82 | %86 |

## Gereksinimler

Python 3.x, gerekli kütüphaneler `requirements.txt` içinde.
