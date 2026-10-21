"""
MediAssist AI — Comprehensive Dataset Audit & Leakage Analysis Script
Audits raw vs processed datasets, class distributions, sample origins,
exact duplicates, template near-duplicates, and split leakage.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_S2D = PROJECT_ROOT / "data" / "raw" / "symptoms" / "Symptom2Disease.csv"
RAW_SYNTH = PROJECT_ROOT / "data" / "raw" / "synthetic" / "outpatient_indian_symptoms.csv"
BERT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"

def jaccard_similarity(str1: str, str2: str) -> float:
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))

def run_audit():
    print("=" * 80)
    print("MEDPASSIST AI — DATASET COMPOSITION & LEAKAGE AUDIT")
    print("=" * 80)

    # 1. Inspect Raw Files
    df_s2d = pd.read_csv(RAW_S2D)
    df_synth = pd.read_csv(RAW_SYNTH)
    
    print("\n[1. RAW DATASET INSPECTION]")
    print(f"Symptom2Disease Raw Rows: {len(df_s2d)}")
    print(f"Symptom2Disease Raw Columns: {df_s2d.columns.tolist()}")
    s2d_classes = sorted(df_s2d["label"].unique())
    print(f"Symptom2Disease Unique Classes ({len(s2d_classes)}): {s2d_classes}")
    
    # Internal duplicates in S2D
    s2d_dup_count = df_s2d["text"].duplicated().sum()
    print(f"Symptom2Disease Internal Duplicate Texts: {s2d_dup_count} (Unique texts: {len(df_s2d) - s2d_dup_count})")
    
    print(f"\nSynthetic Raw Rows: {len(df_synth)}")
    synth_disease_col = "disease" if "disease" in df_synth.columns else "label"
    synth_text_col = "symptom_text" if "symptom_text" in df_synth.columns else "text"
    synth_classes = sorted(df_synth[synth_disease_col].unique())
    print(f"Synthetic Unique Classes ({len(synth_classes)}): {synth_classes}")
    synth_dup_count = df_synth[synth_text_col].duplicated().sum()
    print(f"Synthetic Internal Duplicate Texts: {synth_dup_count}")

    # 2. Inspect Processed Dataset
    proc_df = pd.read_csv(BERT_DATA_DIR / "dataset.csv")
    train_df = pd.read_csv(BERT_DATA_DIR / "train.csv")
    val_df = pd.read_csv(BERT_DATA_DIR / "val.csv")
    test_df = pd.read_csv(BERT_DATA_DIR / "test.csv")
    
    print("\n[2. PROCESSED DATASET INSPECTION]")
    print(f"Processed Full Dataset Rows: {len(proc_df)} across {proc_df['disease'].nunique()} conditions")
    print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} | Total: {len(train_df)+len(val_df)+len(test_df)}")

    # 3. Source Provenance Reconciliation
    s2d_texts = set(df_s2d["text"].str.strip())
    synth_texts = set(df_synth[synth_text_col].str.strip())
    
    s2d_in_proc = sum(1 for t in proc_df["symptom_text"].str.strip() if t in s2d_texts)
    synth_in_proc = sum(1 for t in proc_df["symptom_text"].str.strip() if t in synth_texts)
    supplementary_in_proc = [t for t in proc_df["symptom_text"].str.strip() if t not in s2d_texts and t not in synth_texts]
    
    print("\n[3. PROVENANCE RECONCILIATION]")
    print(f"  - Rows originating from Symptom2Disease: {s2d_in_proc} (across 24 classes)")
    print(f"  - Rows originating from Synthetic Outpatient: {synth_in_proc} (across 8 classes)")
    print(f"  - Rows originating from Hardcoded Supplementary: {len(supplementary_in_proc)} (across 3 classes)")
    print(f"  - Sum: {s2d_in_proc + synth_in_proc + len(supplementary_in_proc)} == {len(proc_df)}")
    
    supp_df = proc_df[proc_df["symptom_text"].str.strip().isin(supplementary_in_proc)]
    print(f"  - Supplementary conditions break-down: {supp_df['disease'].value_counts().to_dict()}")

    # 4. Class Distribution & Imbalance Analysis
    print("\n[4. PER-CLASS SAMPLE DISTRIBUTION]")
    class_counts = proc_df["disease"].value_counts().sort_index()
    distribution_records = []
    
    for disease in sorted(proc_df["disease"].unique()):
        total_c = (proc_df["disease"] == disease).sum()
        train_c = (train_df["disease"] == disease).sum()
        val_c = (val_df["disease"] == disease).sum()
        test_c = (test_df["disease"] == disease).sum()
        
        # Origin
        if disease in ["Influenza (Flu)", "Tuberculosis", "Gastroenteritis"]:
            origin = "Supplementary (Code Injection)"
        elif disease in synth_classes:
            origin = "Synthetic Outpatient"
        else:
            origin = "Symptom2Disease"
            
        distribution_records.append({
            "Disease": disease,
            "Origin": origin,
            "Total": total_c,
            "Train": train_c,
            "Val": val_c,
            "Test": test_c,
            "Share (%)": round((total_c / len(proc_df)) * 100, 2)
        })
        
    dist_df = pd.DataFrame(distribution_records)
    print(dist_df.to_string(index=False))

    max_c = class_counts.max()
    min_c = class_counts.min()
    print(f"\nClass Imbalance: Max = {max_c} ({class_counts.idxmax()}), Min = {min_c} ({class_counts.idxmin()}), Ratio = {max_c / min_c:.2f}:1")

    # 5. Exact and Near-Duplicate Data Leakage Check
    print("\n[5. DATA LEAKAGE & OVERLAP ANALYSIS]")
    train_texts = set(train_df["symptom_text"].str.strip())
    val_texts = set(val_df["symptom_text"].str.strip())
    test_texts = set(test_df["symptom_text"].str.strip())

    exact_train_val = len(train_texts.intersection(val_texts))
    exact_train_test = len(train_texts.intersection(test_texts))
    exact_val_test = len(val_texts.intersection(test_texts))

    print(f"Exact Duplicate Text Overlap:")
    print(f"  - Train vs Val: {exact_train_val}")
    print(f"  - Train vs Test: {exact_train_test}")
    print(f"  - Val vs Test: {exact_val_test}")

    # Near-duplicate / template variation check between train and test
    print("\nTemplate & Near-Duplicate Check between Train and Test (Jaccard > 0.60):")
    high_sim_pairs = []
    
    # Check synthetic classes specifically
    for _, test_row in test_df.iterrows():
        t_text = test_row["symptom_text"].strip()
        t_dis = test_row["disease"]
        
        # compare with train samples in the same disease
        matching_train = train_df[train_df["disease"] == t_dis]
        for _, tr_row in matching_train.iterrows():
            tr_text = tr_row["symptom_text"].strip()
            sim = jaccard_similarity(t_text, tr_text)
            if sim >= 0.60 and t_text.lower() != tr_text.lower():
                high_sim_pairs.append({
                    "disease": t_dis,
                    "test_text": t_text,
                    "train_text": tr_text,
                    "similarity": round(sim, 2)
                })
                
    print(f"Found {len(high_sim_pairs)} near-duplicate / template variation pairs (Jaccard >= 0.60) across Train and Test!")
    print("Sample near-duplicate pairs:")
    for pair in high_sim_pairs[:8]:
        print(f"  [{pair['disease']}] Sim={pair['similarity']}")
        print(f"    Test : {pair['test_text']}")
        print(f"    Train: {pair['train_text']}")

    # Save summary report
    report_path = PROJECT_ROOT / "reports" / "dataset_audit_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_samples": len(proc_df),
            "total_classes": proc_df["disease"].nunique(),
            "s2d_samples": s2d_in_proc,
            "s2d_classes": 24,
            "synthetic_samples": synth_in_proc,
            "synthetic_classes": 8,
            "supplementary_samples": len(supplementary_in_proc),
            "supplementary_classes": 3,
            "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "exact_overlap_train_test": exact_train_test,
            "near_duplicate_pairs_ge_0_6": len(high_sim_pairs),
            "class_imbalance": {"max": int(max_c), "min": int(min_c), "ratio": round(max_c / min_c, 2)},
            "per_class_counts": dist_df.to_dict(orient="records")
        }, f, indent=2)
    print(f"\n[Saved full audit report to {report_path}]")

if __name__ == "__main__":
    run_audit()
