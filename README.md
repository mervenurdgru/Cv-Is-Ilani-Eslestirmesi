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
