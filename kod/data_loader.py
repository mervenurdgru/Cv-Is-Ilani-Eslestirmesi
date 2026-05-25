import os
import glob
import pandas as pd
from tqdm import tqdm
from preprocess import build_resume_text, build_job_text, preprocess_pipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veri")


def load_resumes(n_samples=500):
    path = os.path.join(DATA_DIR, "resume_data.csv")
    print(f"[+] CV'ler yükleniyor: {os.path.basename(path)}")
    df = pd.read_csv(path, nrows=n_samples, encoding="utf-8", on_bad_lines="skip")
    print(f"    {len(df)} CV yüklendi. Temizleniyor...")
    tqdm.pandas(desc="CV'ler işleniyor")
    df["raw_text"] = df.progress_apply(build_resume_text, axis=1)
    df["processed_text"] = df["raw_text"].progress_apply(preprocess_pipeline)
    df = df[df["processed_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"    Temizleme sonrası: {len(df)} CV")
    return df


def load_jobs(n_samples=1000, source="alldata"):
    dfs = []

    if source == "alldata":
        p = os.path.join(DATA_DIR, "alldata.csv")
        if os.path.exists(p):
            df_all = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            dfs.append(df_all)
            print(f"[+] alldata.csv: {len(df_all)} ilan yüklendi")
        else:
            print("[!] alldata.csv bulunamadı!")

    elif source == "fulltime":
        files = sorted(glob.glob(os.path.join(DATA_DIR, "fulltime*.csv")))
        if not files:
            print("[!] fulltime*.csv dosyası bulunamadı!")
        else:
            total = 0
            for fp in files:
                try:
                    tmp = pd.read_csv(fp, encoding="utf-8", on_bad_lines="skip")
                    dfs.append(tmp)
                    total += len(tmp)
                except Exception as e:
                    print(f"    [!] {os.path.basename(fp)} okunamadı: {e}")
            print(f"[+] fulltime*.csv ({len(files)} dosya): {total} ilan yüklendi")

    else:
        raise ValueError(f"Geçersiz kaynak: {source!r}. 'alldata' veya 'fulltime' olmalı.")

    if not dfs:
        raise FileNotFoundError(f"Veri bulunamadı: source={source!r}")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["position", "description", "company", "location"]:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=["description"]).reset_index(drop=True)

    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"[+] Toplam {len(df)} ilan. Temizleniyor...")
    tqdm.pandas(desc="İlanlar işleniyor")
    df["raw_text"] = df.progress_apply(build_job_text, axis=1)
    df["processed_text"] = df["raw_text"].progress_apply(preprocess_pipeline)
    df = df[df["processed_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"    Temizleme sonrası: {len(df)} ilan")
    return df
