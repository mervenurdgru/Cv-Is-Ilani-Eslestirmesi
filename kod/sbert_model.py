# yöntem 3: sentence-bert ile eşleştirme
# cümlenin tamamını tek bir vektöre dönüştürüyor
# tfidf ve word2vec'ten farklı olarak kelime değil cümle anlıyor
# all-MiniLM-L6-v2 modelini seçtim çünkü hem küçük (80mb) hem de iyi sonuç veriyor
# ilk çalıştırmada model internetten indirilecek

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class SBERTMatcher:

    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        # batch_size: kaç metni aynı anda işleyelim
        self.batch_size = batch_size
        self.job_embeddings = None
        print(f"[SBERT] Model yükleniyor: {model_name}")
        print("        (ilk çalıştırmada internetten indirilecek, ~80mb)")
        self.model = SentenceTransformer(model_name)
        print("[SBERT] Model hazır.")

    def _encode(self, texts, desc="kodlanıyor"):
        # metinleri vektöre çevir
        # normalize_embeddings=True dersen kosinüs hesabı daha hızlı oluyor
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def fit(self, job_texts):
        # iş ilanlarını vektöre çevir ve sakla
        print(f"[SBERT] {len(job_texts)} iş ilanı kodlanıyor...")
        self.job_embeddings = self._encode(job_texts)
        print(f"[SBERT] İş ilanı embedding matrisi: {self.job_embeddings.shape}")

    def match(self, resume_text, top_k=10):
        # tek bir cv için en benzer iş ilanlarını bul
        resume_emb = self.model.encode(
            [resume_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        scores = (resume_emb @ self.job_embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def match_all(self, resume_texts, top_k=10):
        # tüm cv'leri vektöre çevir sonra benzerliği hesapla
        print(f"[SBERT] {len(resume_texts)} CV kodlanıyor...")
        resume_embeddings = self._encode(resume_texts)
        print("[SBERT] Benzerlik skorları hesaplanıyor...")
        all_scores = resume_embeddings @ self.job_embeddings.T
        results = []
        for scores in all_scores:
            top_indices = np.argsort(scores)[::-1][:top_k]
            results.append([(int(i), float(scores[i])) for i in top_indices])
        return results
