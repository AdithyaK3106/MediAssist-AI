import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "transformers" / "pubmedbert"
REPORTS_DIR = PROJECT_ROOT / "reports" / "model_benchmarks"
RESEARCH_DIR = PROJECT_ROOT / "research" / "benchmarks"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

class SymptomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    
    top1_acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_macro = f1_score(labels, preds, average="macro")
    
    # Top-3 Accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])
    
    return {
        "accuracy_top1": float(top1_acc),
        "accuracy_top3": float(top3_acc),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro)
    }

def train_pubmedbert(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    epochs=3,
    batch_size=32,
    lr=3e-5,
    output_dir=DEFAULT_OUTPUT_DIR
):
    print("=" * 60)
    print(f"Fine-Tuning Medical BERT: {model_name}")
    print("=" * 60)
    
    # Load label mapping
    mapping_path = DATA_DIR / "label_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)
    print(f"Loaded {num_classes} conditions.")

    # Load data splits
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"Dataset split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Load Tokenizer & Model
    print(f"Downloading/Loading tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={int(k): v for k, v in label_mapping.items()},
        label2id={v: int(k) for k, v in label_mapping.items()}
    )

    train_dataset = SymptomDataset(train_df["symptom_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = SymptomDataset(val_df["symptom_text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_dataset = SymptomDataset(test_df["symptom_text"].tolist(), test_df["label"].tolist(), tokenizer)

    def is_cuda_usable():
        if not torch.cuda.is_available():
            return False
        try:
            t = torch.tensor([1.0], device="cuda")
            _ = t + 1.0
            return True
        except Exception:
            return False

    cuda_ok = is_cuda_usable()
    device = "cuda" if cuda_ok else "cpu"
    print(f"Active training device: {device}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_top3",
        logging_steps=20,
        save_total_limit=1,
        fp16=cuda_ok,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", json.dumps(test_results, indent=2))

    # Save final model & tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save benchmark metrics for reports and research paper
    metrics_summary = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_accuracy_top1": test_results.get("eval_accuracy_top1"),
        "test_accuracy_top3": test_results.get("eval_accuracy_top3"),
        "test_f1_weighted": test_results.get("eval_f1_weighted"),
        "test_loss": test_results.get("eval_loss")
    }
    
    with open(REPORTS_DIR / "pubmedbert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(RESEARCH_DIR / "pubmedbert_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nModel and metrics successfully saved to {output_dir}")
    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    
    train_pubmedbert(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
