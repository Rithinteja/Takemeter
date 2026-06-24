"""Demo script for TakeMeter — run sample classifications for video recording."""

import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from app import classify_once

SAMPLES = [
    {
        "text": "The Celtics' defensive rating drops 8.2 points per 100 possessions when Horford sits. Their switch scheme falls apart without his IQ at the nail.",
        "note": "Expected: analysis — specific stat with tactical explanation",
    },
    {
        "text": "Luka is already a top-5 player all-time and it's not even close. People who disagree just haven't been watching.",
        "note": "Expected: hot_take — bold claim without evidence",
    },
    {
        "text": "That block by Wemby at the end had me jumping off my couch. Absolutely unreal sequence.",
        "note": "Expected: reaction — emotional response to a play",
    },
    {
        "text": "LeBron would average 40 in today's era. The league is too soft.",
        "note": "Expected: hot_take — opinion without supporting data",
    },
    {
        "text": "Jokic's assist rate on short roll passes (under 8 feet) is 47% — highest among centers since tracking began.",
        "note": "Expected: analysis — verifiable statistic with context",
    },
]


def main():
    model_dir = Path("models/takemeter-model")
    if not model_dir.exists():
        print("Model not found. Run first:")
        print("  python generate_dataset.py")
        print("  python train.py")
        sys.exit(1)

    print("=" * 60)
    print("TakeMeter Demo — Sample Classifications")
    print("=" * 60)

    for i, sample in enumerate(SAMPLES, 1):
        result = classify_once(sample["text"])
        print(f"\n--- Sample {i} ---")
        print(f"Post: {sample['text'][:120]}...")
        print(f"Note: {sample['note']}")
        print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
        print("Probabilities:")
        for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {label}: {prob:.1%}")

    results_path = Path("evaluation_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print("\n" + "=" * 60)
        print("Evaluation Summary (from evaluation_results.json)")
        print("=" * 60)
        print(f"Fine-tuned accuracy: {results.get('finetuned_accuracy', 'N/A')}")
        if "baseline_accuracy" in results:
            print(f"Baseline accuracy:   {results['baseline_accuracy']}")
            print(f"Improvement:         {results.get('improvement', 'N/A')}")
        print(f"Test set size:       {results.get('test_set_size', 'N/A')}")

    print("\n" + "=" * 60)
    print("To launch the web interface: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
