# sonuçları değerlendirme
# üç metrik kullandım: kosinüs benzerliği, mrr ve precision@k
# ground truth otomatik oluşturuluyor, cv'nin hedef pozisyonu ile iş ilanı başlığı karşılaştırılıyor

import re
import numpy as np
import pandas as pd
def _normalize_position(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _position_keywords(text):
    noise = {"senior","junior","sr","jr","lead","manager","executive",
             "officer","specialist","associate","assistant","head",
             "director","principal","chief","staff","and","of","the",
             "in","for","ii","iii","iv","i","ml","ai"}
    tokens = _normalize_position(text).split()
    return {t for t in tokens if len(t) > 2 and t not in noise}


def build_ground_truth(resume_df, job_df, min_overlap=1):
    pos_col = None
    for c in resume_df.columns:
        if "job_position" in c.lower():
            pos_col = c
            break

    if pos_col is None:
        print("[!] Ground truth: job_position_name sütunu bulunamadı.")
        return [None] * len(resume_df)

    job_kws = [_position_keywords(p) for p in job_df.get("position", pd.Series(dtype=str))]

    ground_truth = []
    for _, row in resume_df.iterrows():
        cv_kws = _position_keywords(row.get(pos_col, ""))
        if not cv_kws:
            ground_truth.append(None)
            continue
        relevant = [j for j, jkw in enumerate(job_kws) if len(cv_kws & jkw) >= min_overlap]
        ground_truth.append(relevant if relevant else None)

    matched = sum(1 for g in ground_truth if g is not None)
    print(f"[Ground Truth] {matched}/{len(resume_df)} CV için eşleşen ilan bulundu")
    return ground_truth



def average_cosine_similarity(match_results):
    top_scores = [r[0][1] for r in match_results if r]
    return float(np.mean(top_scores)) if top_scores else 0.0


def mean_reciprocal_rank(match_results, relevant_indices):
    # doğru eşleşme kaçıncı sırada çıktı? 1. sırada çıksa 1.0, 2. sırada 0.5 gibi
    if not relevant_indices or all(r is None for r in relevant_indices):
        return 0.0
    rrs = []
    for results, rel in zip(match_results, relevant_indices):
        if rel is None:
            continue
        rel_set = set(rel) if isinstance(rel, list) else {rel}
        rr = 0.0
        for rank, (job_idx, _) in enumerate(results, start=1):
            if job_idx in rel_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def precision_at_k(match_results, relevant_indices, k=5):
    if not relevant_indices or all(r is None for r in relevant_indices):
        return 0.0
    precs = []
    for results, rel in zip(match_results, relevant_indices):
        if rel is None:
            continue
        rel_set = set(rel) if isinstance(rel, list) else {rel}
        top_k = {job_idx for job_idx, _ in results[:k]}
        precs.append(len(top_k & rel_set) / k)
    return float(np.mean(precs)) if precs else 0.0


def score_distribution_stats(match_results):
    top_scores = [r[0][1] for r in match_results if r]
    arr = np.array(top_scores)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


def print_evaluation_report(method_name, match_results, relevant_indices=None, k_values=None):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print(f"\n{'='*60}")
    print(f"  Yöntem: {method_name}")
    print(f"{'='*60}")

    stats = score_distribution_stats(match_results)
    print(f"  Kosinüs Benzerliği (Top-1):")
    print(f"    Ortalama : {stats['mean']:.4f}")
    print(f"    Medyan   : {stats['median']:.4f}")
    print(f"    Std Dev  : {stats['std']:.4f}")
    print(f"    Min/Max  : {stats['min']:.4f} / {stats['max']:.4f}")

    metrics = {"avg_cosine": stats["mean"], "median_cosine": stats["median"]}

    if relevant_indices and not all(r is None for r in relevant_indices):
        mrr = mean_reciprocal_rank(match_results, relevant_indices)
        print(f"\n  MRR: {mrr:.4f}")
        metrics["mrr"] = mrr
        for k in k_values:
            pk = precision_at_k(match_results, relevant_indices, k=k)
            print(f"  Precision@{k:<3}: {pk:.4f}")
            metrics[f"p@{k}"] = pk
    else:
        print("\n  [!] Ground truth yok, MRR ve P@K hesaplanamadı.")
        metrics["mrr"] = 0.0
        for k in k_values:
            metrics[f"p@{k}"] = 0.0

    print(f"{'='*60}\n")
    return metrics
