"""TakeMeter — deployed interface for classifying r/nba discourse quality."""

import json
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ID_TO_LABEL, MODEL_DIR, VALID_LABELS

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        model_path = Path(MODEL_DIR)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_DIR}. Run `python train.py` first."
            )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _model, _tokenizer


def classify_post(text: str):
    if not text or not text.strip():
        return "Enter a post to classify.", 0.0, {}

    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(probs.argmax())
    label = ID_TO_LABEL[pred_id]
    confidence = float(probs[pred_id])
    all_probs = {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(VALID_LABELS))}

    return label, confidence, all_probs


def format_probs(probs: dict) -> str:
    if not probs:
        return ""
    lines = [f"**{k}**: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])]
    return "\n".join(lines)


EXAMPLES = [
    ["The Celtics' defensive rating drops 8.2 points per 100 possessions when Horford sits."],
    ["Luka is already a top-5 player all-time and it's not even close."],
    ["That block by Wemby at the end had me jumping off my couch. Absolutely unreal."],
    ["LeBron would average 40 in today's era. The league is too soft."],
    ["Curry's off-ball gravity forces 1.3 extra defenders within 6 feet on every drive."],
]


def build_ui():
    with gr.Blocks(title="TakeMeter — r/nba Discourse Classifier") as demo:
        gr.Markdown(
            "# TakeMeter\n"
            "Classify r/nba posts as **analysis**, **hot_take**, or **reaction**.\n\n"
            "Fine-tuned DistilBERT model trained on 210 labeled examples."
        )
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Post text",
                    placeholder="Paste an r/nba post or comment here...",
                    lines=5,
                )
                classify_btn = gr.Button("Classify", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try these")
            with gr.Column():
                label_output = gr.Textbox(label="Predicted label")
                confidence_output = gr.Number(label="Confidence", precision=2)
                probs_output = gr.Markdown(label="All label probabilities")

        def run_classification(text):
            label, confidence, probs = classify_post(text)
            return label, confidence, format_probs(probs)

        classify_btn.click(
            fn=run_classification,
            inputs=text_input,
            outputs=[label_output, confidence_output, probs_output],
        )

    return demo


def classify_once(text: str):
    """Single classification for demo script / CLI use."""
    label, conf, probs = classify_post(text)
    return {"label": label, "confidence": conf, "probabilities": probs}


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
