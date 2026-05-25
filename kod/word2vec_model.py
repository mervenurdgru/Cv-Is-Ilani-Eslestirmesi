# yöntem 2: Word2Vec ile eşleştirme
# gensim python 3.14 ile import bile etmiyor
# temel mantik: hangi kelimeler birlikte geciyor -> matris yap -> svd ile kucult

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix


class Word2VecMatcher:

    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word2id = {}
        self.word_vecs = None
        self.tfidf = None
        self.tfidf_vocab = {}
        self.job_vectors = None

    def _build_vocab(self, tokenized):
        kelime_sayilari = defaultdict(int)
        for token_listesi in tokenized:
            for kelime in token_listesi:
                kelime_sayilari[kelime] += 1
        kelimeler = [k for k, sayi in kelime_sayilari.items() if sayi >= self.min_count]
        self.word2id = {k: idx for idx, k in enumerate(kelimeler)}

    def _build_cooccurrence(self, tokenized):
        V = len(self.word2id)
        birlikte_gecme = defaultdict(float)

        for token_listesi in tokenized:
            id_listesi = [self.word2id[k] for k in token_listesi if k in self.word2id]

            for i, kelime_id in enumerate(id_listesi):
                pencere_bas = max(0, i - self.window)
                pencere_son = min(len(id_listesi), i + self.window + 1)

                for j in range(pencere_bas, pencere_son):
                    if i == j:
                        continue
                    komsu_id = id_listesi[j]
                    uzaklik = abs(i - j)
                    birlikte_gecme[(kelime_id, komsu_id)] += 1.0 / uzaklik

        satirlar, sutunlar, degerler = [], [], []
        for (k1, k2), val in birlikte_gecme.items():
            satirlar.append(k1)
            sutunlar.append(k2)
            degerler.append(val)

        M = csr_matrix((degerler, (satirlar, sutunlar)), shape=(V, V), dtype=np.float32)
        return M

    def _apply_ppmi(self, M):
        M = M.astype(np.float64)
        toplam = M.sum()
        satir_toplami = np.array(M.sum(axis=1)).flatten()
        sutun_toplami = np.array(M.sum(axis=0)).flatten()

        satirlar, sutunlar = M.nonzero()
        degerler = np.array(M[satirlar, sutunlar]).flatten()

        pmi = np.log2(
            (degerler * toplam) /
            (satir_toplami[satirlar] * sutun_toplami[sutunlar] + 1e-9)
        )
        ppmi = np.maximum(pmi, 0)  
        V = M.shape[0]
        return csr_matrix((ppmi, (satirlar, sutunlar)), shape=(V, V), dtype=np.float32)

    def _svd(self, M):
        k = min(self.vector_size, M.shape[0] - 1, M.shape[1] - 1)
        U, S, Vt = randomized_svd(M, n_components=k, random_state=42)
        W = U * np.sqrt(S)
        normlar = np.linalg.norm(W, axis=1, keepdims=True)
        normlar[normlar == 0] = 1
        return W / normlar

    def _metni_vektore_cevir(self, metin):
        kelimeler = metin.split()
        vektor_listesi = []
        agirlik_listesi = []

        for kelime in kelimeler:
            if kelime not in self.word2id:
                continue
            idx = self.word2id[kelime]
            if idx >= len(self.word_vecs):
                continue

            vektor = self.word_vecs[idx]
            agirlik = 1.0
            if kelime in self.tfidf_vocab:
                tfidf_idx = self.tfidf_vocab[kelime]
                agirlik = float(self.tfidf.idf_[tfidf_idx])

            vektor_listesi.append(vektor * agirlik)
            agirlik_listesi.append(agirlik)

        if not vektor_listesi:
            return np.zeros(self.vector_size)

        return np.sum(vektor_listesi, axis=0) / (sum(agirlik_listesi) + 1e-9)

    def fit(self, all_texts, job_texts):
        print(f"[Word2Vec] egitim baslıyor, {len(all_texts)} metin var...")

        # 1. tokenize
        tokenized = [t.split() for t in all_texts]

        # 2. kelime sozlugu olustur
        self._build_vocab(tokenized)
        print(f"[Word2Vec] sozlukte {len(self.word2id)} kelime var")

        # 3. hangi kelimeler yanyana geciyor
        print("[Word2Vec] esgesme matrisi olusturuluyor...")
        M = self._build_cooccurrence(tokenized)

        # 4. ppmi uygula
        print("[Word2Vec] ppmi uygulanıyor...")
        M_ppmi = self._apply_ppmi(M)

        # 5. boyut indir
        print(f"[Word2Vec] svd ile {self.vector_size} boyuta indiriliyor...")
        self.word_vecs = self._svd(M_ppmi)
        print(f"[Word2Vec] kelime vektoru boyutu: {self.word_vecs.shape}")

        # 6. tfidf agirliklarini hesapla
        print("[Word2Vec] tfidf agirliklari hesaplaniyor...")
        self.tfidf = TfidfVectorizer(max_features=30000)
        self.tfidf.fit(all_texts)
        self.tfidf_vocab = self.tfidf.vocabulary_

        # 7. is ilanlarini vektorlestir
        print(f"[Word2Vec] {len(job_texts)} is ilani vektorlestiriliyor...")
        is_vektorleri = np.array([self._metni_vektore_cevir(t) for t in job_texts])
        normlar = np.linalg.norm(is_vektorleri, axis=1, keepdims=True)
        normlar[normlar == 0] = 1
        self.job_vectors = is_vektorleri / normlar
        print(f"[Word2Vec] bitti. is ilanı matrisi: {self.job_vectors.shape}")

    def match(self, resume_text, top_k=10):
        vec = self._metni_vektore_cevir(resume_text)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        skorlar = vec @ self.job_vectors.T
        en_iyi = np.argsort(skorlar)[::-1][:top_k]
        return [(int(i), float(skorlar[i])) for i in en_iyi]

    def match_all(self, resume_texts, top_k=10):
        print(f"[Word2Vec] {len(resume_texts)} cv vektorlestiriliyor...")
        cv_vektorleri = np.array([self._metni_vektore_cevir(t) for t in resume_texts])
        normlar = np.linalg.norm(cv_vektorleri, axis=1, keepdims=True)
        normlar[normlar == 0] = 1
        cv_vektorleri = cv_vektorleri / normlar

        print("[Word2Vec] benzerlik hesaplaniyor...")
        tum_skorlar = cv_vektorleri @ self.job_vectors.T

        sonuclar = []
        for skorlar in tum_skorlar:
            en_iyi = np.argsort(skorlar)[::-1][:top_k]
            sonuclar.append([(int(i), float(skorlar[i])) for i in en_iyi])
        return sonuclar
