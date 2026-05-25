# yöntem 1: tfidf + kosinüs benzerliği
# en basit yöntem, diğerleriyle karşılaştırmak için baseline olarak kullandım
# sklearn kütüphanesinden TfidfVectorizer kullandım

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:

    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.job_matrix = None

    def fit(self, job_texts):
        print("[TF-IDF] İş ilanları vektörleştiriliyor...")
        self.job_matrix = self.vectorizer.fit_transform(job_texts)
        print(f"[TF-IDF] İş ilanı matrisi: {self.job_matrix.shape}")

    def match(self, resume_text, top_k=10):
        resume_vec = self.vectorizer.transform([resume_text])
        scores = cosine_similarity(resume_vec, self.job_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def match_all(self, resume_texts, top_k=10):
        print(f"[TF-IDF] {len(resume_texts)} CV eşleştiriliyor...")
        resume_matrix = self.vectorizer.transform(resume_texts)
        all_scores = cosine_similarity(resume_matrix, self.job_matrix)
        results = []
        for scores in all_scores:
            top_indices = np.argsort(scores)[::-1][:top_k]
            results.append([(int(i), float(scores[i])) for i in top_indices])
        return results
