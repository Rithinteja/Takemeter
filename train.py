"""TakeMeter training pipeline: fine-tune DistilBERT, run Groq baseline, export results."""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    ID_TO_LABEL,
    LABEL_MAP,
    MODEL_DIR,
    MODEL_NAME,
    NUM_LABELS,
    RESULTS_PATH,
    SYSTEM_PROMPT,
)

load_dotenv()


def load_data():
    df = pd.read_csv(DATASET_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df


def split_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_datasets(train_df, val_df, test_df, tokenizer):
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    def to_ds(frame):
        ds = Dataset.from_pandas(
            frame[["text", "label_id"]].rename(columns={"label_id": "labels"})
        )
        return ds.map(tokenize, batched=True)

    return to_ds(train_df), to_ds(val_df), to_ds(test_df)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def fine_tune(train_ds, val_ds, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    print("Starting fine-tuning...")
    trainer.train()
    return trainer


def evaluate_finetuned(trainer, test_ds, test_df):
    output = trainer.predict(test_ds)
    pred_ids = np.argmax(output.predictions, axis=-1)
    true_ids = output.label_ids
    probs = torch.nn.functional.softmax(
        torch.tensor(output.predictions), dim=-1
    ).numpy()

    acc = accuracy_score(true_ids, pred_ids)
    names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        true_ids, pred_ids, target_names=names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(true_ids, pred_ids)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    wrong = []
    for idx in np.where(pred_ids != true_ids)[0]:
        wrong.append({
            "text": test_df.iloc[idx]["text"],
            "true_label": ID_TO_LABEL[true_ids[idx]],
            "predicted_label": ID_TO_LABEL[pred_ids[idx]],
            "confidence": float(probs[idx][pred_ids[idx]]),
        })

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "label_names": names,
        "wrong_predictions": wrong,
        "pred_ids": pred_ids,
        "true_ids": true_ids,
        "probs": probs,
    }


def classify_with_groq(client, text):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this post:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        for label in sorted(LABEL_MAP, key=len, reverse=True):
            if raw == label or label in raw:
                return label
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def run_baseline(test_df, api_key):
    if not api_key:
        print("GROQ_API_KEY not set — skipping baseline.")
        return None

    from groq import Groq

    client = Groq(api_key=api_key)
    preds = []
    print(f"Running Groq baseline on {len(test_df)} examples...")
    for i, (_, row) in enumerate(test_df.iterrows()):
        preds.append(classify_with_groq(client, row["text"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_df)} complete...")
        time.sleep(0.1)

    valid = [(p, t) for p, t in zip(preds, test_df["label_id"]) if p is not None]
    if not valid:
        return None

    bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
    bl_true_ids = [t for _, t in valid]
    names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    acc = accuracy_score(bl_true_ids, bl_pred_ids)
    report = classification_report(
        bl_true_ids, bl_pred_ids, target_names=names, zero_division=0, output_dict=True
    )
    none_count = preds.count(None)
    if none_count:
        print(f"Warning: {none_count} unparseable baseline responses.")

    return {"accuracy": acc, "report": report, "parseable": len(valid)}


def main():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} examples")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds = make_datasets(train_df, val_df, test_df, tokenizer)

    trainer = fine_tune(train_ds, val_ds, tokenizer)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    ft = evaluate_finetuned(trainer, test_ds, test_df)
    print(f"\nFine-tuned accuracy: {ft['accuracy']:.3f}")

    baseline = run_baseline(test_df, os.getenv("GROQ_API_KEY"))
    if baseline:
        print(f"Baseline accuracy: {baseline['accuracy']:.3f}")

    results = {
        "finetuned_accuracy": round(ft["accuracy"], 4),
        "finetuned_report": ft["report"],
        "confusion_matrix": ft["confusion_matrix"],
        "label_names": ft["label_names"],
        "wrong_predictions": ft["wrong_predictions"][:10],
        "test_set_size": len(test_df),
        "label_map": LABEL_MAP,
        "model": MODEL_NAME,
        "hyperparameters": {
            "epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 16,
        },
    }
    if baseline:
        results["baseline_accuracy"] = round(baseline["accuracy"], 4)
        results["baseline_report"] = baseline["report"]
        results["improvement"] = round(ft["accuracy"] - baseline["accuracy"], 4)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {RESULTS_PATH} and {CONFUSION_MATRIX_PATH}")
    print("=" * 50)
    print(f"{'Fine-tuned DistilBERT':<30} {ft['accuracy']:.3f}")
    if baseline:
        print(f"{'Zero-shot baseline (Groq)':<30} {baseline['accuracy']:.3f}")


if __name__ == "__main__":
    main()
