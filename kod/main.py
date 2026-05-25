# iki farkli dataset uzerinde uc yontem karsilastirildi: tfidf, word2vec (ppmi+svd), sentence-bert
#   python main.py                 
#   python main.py --method tfidf   
#   python main.py --n_resumes 50   
#   python main.py --no_sbert       

import argparse
import os
import pandas as pd
import numpy as np

from data_loader import load_resumes, load_jobs
from evaluation import (
    build_ground_truth,
    print_evaluation_report,
)


def show_sample_matches(resume_df, job_df, match_results, method_name, n=3):
    """First n CVs sample match output."""
    print("\n" + "-"*65)
    print(f"  {method_name} -- Sample Matches (First {n} CVs)")
    print("-"*65)
    pos_col = next((c for c in resume_df.columns if 'job_position' in c.lower()), None)
    for i in range(min(n, len(match_results))):
        cv_pos = resume_df.iloc[i].get(pos_col, 'N/A') if pos_col else 'N/A'
        objective = str(resume_df.iloc[i].get('career_objective', ''))[:80]
        print(f"\n  [CV #{i+1}]  Target Position: {cv_pos}")
        print(f"  Summary  : {objective}...")
        for rank, (job_idx, score) in enumerate(match_results[i][:3], start=1):
            row = job_df.iloc[job_idx]
            pos  = str(row.get('position', 'N/A'))[:50]
            comp = str(row.get('company',  'N/A'))[:30]
            loc  = str(row.get('location', 'N/A'))[:25]
            print(f"    {rank}. [{score:.3f}] {pos} @ {comp} ({loc})")


def save_results(resume_df, job_df, match_results, method_name, top_k, output_dir, tag=""):
    rows = []
    pos_col = next((c for c in resume_df.columns if 'job_position' in c.lower()), None)
    for cv_idx, matches in enumerate(match_results):
        cv_pos = resume_df.iloc[cv_idx].get(pos_col, '') if pos_col else ''
        for rank, (job_idx, score) in enumerate(matches[:top_k], start=1):
            jrow = job_df.iloc[job_idx]
            rows.append({
                'cv_index':     cv_idx,
                'cv_position':  cv_pos,
                'rank':         rank,
                'job_index':    job_idx,
                'job_position': jrow.get('position', ''),
                'job_company':  jrow.get('company', ''),
                'job_location': jrow.get('location', ''),
                'cosine_score': round(score, 4),
            })
    suffix = f"_{tag}" if tag else ""
    fname = f"results_{method_name.lower().replace(' ', '_')}{suffix}.csv"
    path = os.path.join(output_dir, fname)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[+] Results saved -> {fname}")


def run_experiment(resume_df, job_df, method, top_k, output_dir,
                   dataset_name="", skip_sbert=False):
    """
    Run selected methods on given resume/job dataframes.
    Returns: {method_name: (match_results, ground_truth, metrics)}
    """
    print("\n" + "#"*65)
    print(f"  DATASET: {dataset_name}")
    print(f"  {len(resume_df)} CVs  x  {len(job_df)} Job Listings")
    print("#"*65)

    print("\n[*] Building ground truth...")
    ground_truth = build_ground_truth(resume_df, job_df, min_overlap=1)

    experiment_results = {}

    if method in ('tfidf', 'all'):
        from tfidf_model import TFIDFMatcher
        print("\n[1/3] TF-IDF + Cosine Similarity running...")
        matcher = TFIDFMatcher()
        matcher.fit(job_df['processed_text'].tolist())
        results = matcher.match_all(resume_df['processed_text'].tolist(), top_k=top_k)
        metrics = print_evaluation_report("TF-IDF", results, ground_truth)
        show_sample_matches(resume_df, job_df, results, "TF-IDF")
        save_results(resume_df, job_df, results, 'tfidf', top_k, output_dir, dataset_name)
        experiment_results['TF-IDF'] = (results, ground_truth, metrics)


    if method in ('word2vec', 'all'):
        from word2vec_model import Word2VecMatcher
        print("\n[2/3] Word2Vec + Cosine Similarity running...")
        matcher = Word2VecMatcher()
        all_texts = resume_df['processed_text'].tolist() + job_df['processed_text'].tolist()
        matcher.fit(all_texts, job_df['processed_text'].tolist())
        results = matcher.match_all(resume_df['processed_text'].tolist(), top_k=top_k)
        metrics = print_evaluation_report("Word2Vec", results, ground_truth)
        show_sample_matches(resume_df, job_df, results, "Word2Vec")
        save_results(resume_df, job_df, results, 'word2vec', top_k, output_dir, dataset_name)
        experiment_results['Word2Vec'] = (results, ground_truth, metrics)


    if method in ('sbert', 'all') and not skip_sbert:
        from sbert_model import SBERTMatcher
        print("\n[3/3] Sentence-BERT running...")
        matcher = SBERTMatcher()
        matcher.fit(job_df['raw_text'].tolist())
        results = matcher.match_all(resume_df['raw_text'].tolist(), top_k=top_k)
        metrics = print_evaluation_report("SBERT", results, ground_truth)
        show_sample_matches(resume_df, job_df, results, "SBERT")
        save_results(resume_df, job_df, results, 'sbert', top_k, output_dir, dataset_name)
        experiment_results['SBERT'] = (results, ground_truth, metrics)

    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="CV-Job Semantic Matching")
    parser.add_argument('--method', default='all',
                        choices=['tfidf', 'word2vec', 'sbert', 'all'])
    parser.add_argument('--n_resumes', type=int, default=150)
    parser.add_argument('--n_jobs',    type=int, default=400)
    parser.add_argument('--top_k',     type=int, default=10)
    parser.add_argument('--no_sbert',  action='store_true',
                        help='Skip SBERT (faster)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "sonuclar")

    print("\n" + "="*65)
    print("  Semantic CV - Job Matching System")
    print("  Student: Merve Nur Dogru | BLM4514")
    print("="*65)

    resume_df = load_resumes(n_samples=args.n_resumes)

    # DATASET 1: alldata.csv 
    print("\n" + "="*65)
    print("  DATASET 1: Data Scientist Job Market - alldata.csv")
    print("  Source: Kaggle sl6149/data-scientist-job-market-in-the-us")
    print("="*65)
    job_df1 = load_jobs(n_samples=args.n_jobs, source='alldata')

    exp1 = run_experiment(
        resume_df, job_df1,
        method=args.method,
        top_k=args.top_k,
        output_dir=output_dir,
        dataset_name="DS1_alldata",
        skip_sbert=args.no_sbert,
    )

    # DATASET 2: fulltime*.csv
    print("\n" + "="*65)
    print("  DATASET 2: Fulltime City Job Listings - fulltime*.csv")
    print("  Source: Kaggle sl6149/data-scientist-job-market-in-the-us")
    print("="*65)
    job_df2 = load_jobs(n_samples=args.n_jobs, source='fulltime')

    exp2 = run_experiment(
        resume_df, job_df2,
        method=args.method,
        top_k=args.top_k,
        output_dir=output_dir,
        dataset_name="DS2_fulltime",
        skip_sbert=args.no_sbert,
    )

    # Dataset 1 vs Dataset 2
    print("\n" + "="*65)
    print("  SUMMARY: Dataset 1 vs Dataset 2 Comparison")
    print("="*65)

    summary = {}
    for ds_label, exp in [("Dataset1(alldata)", exp1), ("Dataset2(fulltime)", exp2)]:
        summary[ds_label] = {}
        for mname, (results, gt, metrics) in exp.items():
            summary[ds_label][mname] = {
                'avg_cosine': metrics.get('avg_cosine', 0),
                'mrr':        metrics.get('mrr', 0),
                'p@5':        metrics.get('p@5', 0),
            }

    print(f"\n{'Method':<12} {'Dataset':<22} {'Cosine':>8} {'MRR':>8} {'P@5':>8}")
    print("-" * 62)
    for ds_label, methods_data in summary.items():
        for m, vals in methods_data.items():
            print(f"{m:<12} {ds_label:<22} "
                  f"{vals['avg_cosine']:>8.4f} "
                  f"{vals['mrr']:>8.4f} "
                  f"{vals['p@5']:>8.4f}")

    print("\n[OK] All done!")
    print(f"     Output dir: {output_dir}")


if __name__ == '__main__':
    main()
