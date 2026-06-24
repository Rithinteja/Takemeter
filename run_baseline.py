"""Run Groq baseline only and merge results into evaluation_results.json."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from config import ID_TO_LABEL, LABEL_MAP, NUM_LABELS, RESULTS_PATH
from train import classify_with_groq, load_data, split_data

load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("Set GROQ_API_KEY in .env first.")
        return

    from groq import Groq
    client = Groq(api_key=api_key)

    df = load_data()
    _, _, test_df = split_data(df)

    print(f"Running baseline on {len(test_df)} test examples...")
    preds = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        preds.append(classify_with_groq(client, row["text"]))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(test_df)}...")
        time.sleep(0.1)

    valid = [(p, t) for p, t in zip(preds, test_df["label_id"]) if p is not None]
    bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
    bl_true_ids = [t for _, t in valid]
    names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    acc = accuracy_score(bl_true_ids, bl_pred_ids)
    report = classification_report(
        bl_true_ids, bl_pred_ids, target_names=names, zero_division=0, output_dict=True
    )

    results = {}
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    results["baseline_accuracy"] = round(acc, 4)
    results["baseline_report"] = report
    if "finetuned_accuracy" in results:
        results["improvement"] = round(results["finetuned_accuracy"] - acc, 4)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline accuracy: {acc:.3f}")
    if "finetuned_accuracy" in results:
        print(f"Fine-tuned accuracy: {results['finetuned_accuracy']:.3f}")
        print(f"Improvement: {results.get('improvement', 'N/A')}")


if __name__ == "__main__":
    main()
