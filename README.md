# TakeMeter — r/nba Discourse Quality Classifier

TakeMeter is a fine-tuned text classifier that evaluates discourse quality in r/nba posts. It assigns each post to one of three labels — **analysis**, **hot_take**, or **reaction** — based on whether the post offers structured evidence, asserts an unsupported opinion, or expresses an immediate emotional response.

---

## Community and Labels

**Community:** r/nba (Reddit's primary NBA discussion subreddit)

**Why r/nba:** The community produces varied discourse daily — detailed statistical breakdowns sit alongside bold unsupported claims and in-the-moment game reactions. Regulars routinely distinguish "quality analysis" from "lazy hot takes," making it a natural fit for a discourse-quality classifier.

### Label Taxonomy

| Label | Definition |
|-------|------------|
| **analysis** | The post makes a structured argument backed by specific statistics, film observations, or tactical reasoning. Evidence is concrete and verifiable. |
| **hot_take** | A bold, confident opinion stated without substantive supporting evidence. The claim may be debatable but the post asserts rather than argues. |
| **reaction** | An immediate emotional response to a specific game event or news item. Little to no argument — the post expresses a feeling in the moment. |

**Example — analysis:** "The Celtics' defensive rating drops 8.2 points per 100 possessions when Horford sits. Their switch scheme falls apart without his IQ at the nail."

**Example — hot_take:** "Luka is already a top-5 player all-time and it's not even close. People who disagree just haven't been watching."

**Example — reaction:** "That block by Wemby at the end had me jumping off my couch. Absolutely unreal sequence."

---

## Dataset

**Source:** Public r/nba-style posts written to reflect typical community discourse patterns.

**Size:** 216 labeled examples (72 per label)

**Label distribution:**

| Label | Count | Percentage |
|-------|-------|------------|
| analysis | 72 | 33.3% |
| hot_take | 72 | 33.3% |
| reaction | 72 | 33.3% |

**Split:** 70% train (151) / 15% validation (32) / 15% test (33), stratified by label.

**Labeling process:** Each post was read and labeled using the definitions in `planning.md`. Edge cases were resolved using explicit decision rules (e.g., single-stat posts with accusatory framing → hot_take).

**Difficult examples:**

1. *"LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."* — Could be analysis ( cites stat) or hot_take (decorative stat). **Decision: hot_take** — stat is cherry-picked, framing is accusatory.

2. *"Oh yeah, the refs definitely didn't decide that game."* — Could be reaction (emotional) or hot_take (opinion). **Decision: hot_take** — sarcasm expressing an opinion, not a genuine in-the-moment reaction.

3. *"The 47% assist rate on short rolls is insane — Jokic is on another level."* — Could be analysis or reaction (excited tone). **Decision: analysis** — verifiable stat is the core content.

Dataset file: `data/takemeter_dataset.csv`

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Platform:** Local CPU training (Google Colab T4 GPU also supported via the starter notebook)

**Training setup:**
- 3 epochs
- Learning rate: 2e-5
- Batch size: 16
- Max sequence length: 256 tokens
- Optimizer: AdamW with weight decay 0.01
- Warmup steps: 50

**Key hyperparameter decision:** Kept 3 epochs rather than increasing to 5, because with only 151 training examples, additional epochs risk overfitting. Validation accuracy plateaued by epoch 3 in initial runs.

**To train locally:**
```bash
pip install -r requirements.txt
python generate_dataset.py   # if dataset not yet created
python train.py
```

**Colab:** Use `ai201_project3_takemeter_starter_clean.py` (export as notebook) with T4 GPU runtime.

---

## Baseline Comparison

**Baseline model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt approach:** System prompt defines each label with a one-sentence definition and example post. Model instructed to respond with ONLY the label name.

**How to run baseline:** Set `GROQ_API_KEY` in `.env`, then run `python train.py`. Baseline runs automatically on the test set after fine-tuning.

```bash
cp .env.example .env
# Edit .env and add your Groq API key
python train.py
```

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Fine-tuned DistilBERT | **0.848** |
| Zero-shot baseline (Groq) | Run with `GROQ_API_KEY` in `.env` |

Fine-tuning on 216 r/nba-style posts achieved **84.8%** test accuracy. Set your Groq API key and re-run `python train.py` to populate baseline comparison numbers.

### Per-Class Metrics (Fine-Tuned)

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| analysis | 0.846 | 1.000 | 0.917 | 11 |
| hot_take | 0.800 | 0.727 | 0.762 | 11 |
| reaction | 0.900 | 0.818 | 0.857 | 11 |
| **Macro avg** | 0.849 | 0.848 | 0.845 | 33 |

**Lowest-performing class:** hot_take (F1 = 0.762). Hot takes that lack typical opinion markers ("overrated," "fraud") or use team-focused phrasing ("The Knicks are...") get confused with analysis or reaction.

### Confusion Matrix (Fine-Tuned)

|  | analysis | hot_take | reaction |
|--|----------|----------|----------|
| **analysis** | 11 | 0 | 0 |
| **hot_take** | 2 | 8 | 1 |
| **reaction** | 0 | 2 | 9 |

Image: `confusion_matrix.png`

Directional errors: hot_take → analysis (2), reaction → hot_take (2), hot_take → reaction (1).

### Error Analysis — Three Wrong Predictions

**1. hot_take → analysis:** "The Knicks are the most overrated team in the East."
- **Why:** Team-focused declarative sentence without emotional markers. Lacks stats but reads like an analytical claim. Model weights "The [Team] are..." structure toward analysis.

**2. reaction → hot_take:** "The arena erupted. Best moment of the season."
- **Why:** "Best moment of the season" is an assertive superlative — overlaps with hot_take language. Emotional scene-setting ("arena erupted") wasn't enough to override the evaluative claim.

**3. hot_take → reaction:** "The Jazz should have kept Mitchell. Trading stars never works."
- **Why:** Short, punchy phrasing with a generalization ("never works") resembles reaction-style venting. Missing the bold unsupported claim markers the model learned for hot_take.

### Sample Classifications

| Post | Predicted | Confidence | Notes |
|------|-----------|------------|-------|
| "The Celtics' defensive rating drops 8.2 points per 100 when Horford sits." | analysis | 0.95+ | Stat + tactical reasoning — model correctly picks up percentage and defensive terminology |
| "Luka is already a top-5 player all-time." | hot_take | 0.92+ | Bold superlative without evidence |
| "That block by Wemby had me jumping off my couch." | reaction | 0.98+ | First-person emotional response |
| "Curry's off-ball gravity forces 1.3 extra defenders within 6 feet." | analysis | 0.94+ | Correct — specific stat with tactical explanation |
| "The NBA is rigged for big markets." | hot_take | 0.88+ | Unsupported systemic claim |

Run `python demo.py` for live predictions with confidence scores.

### Reflection: Intended vs. Learned Behavior

**Intended:** The model should learn format/structure — whether a post argues with evidence, asserts without evidence, or reacts emotionally.

**Learned:** The model heavily weights lexical cues: numbers and percentages → analysis; first-person emotional verbs ("crying," "screamed," "devastated") → reaction; superlatives and absolutes ("best ever," "fraud," "rigged") → hot_take.

**Gap:** Borderline cases where a single stat supports a hot take, or sarcastic reactions that are actually opinions, remain the main failure mode. The model learned surface patterns more reliably than the nuanced taxonomy rules in `planning.md`. This matches the course insight: models learn what the labels teach, and adjacent classes with overlapping language are hardest to separate.

---

## Deployed Interface

**Run the web UI:**
```bash
python app.py
```

Opens a Gradio interface at `http://127.0.0.1:7860`. Paste any r/nba-style post to get a predicted label and confidence score.

**CLI demo:**
```bash
python demo.py
```

---

## Spec Reflection

**How the spec helped:** The project rubric's requirement to define labels with complete sentences and explicit edge-case rules forced precision before annotation. Writing the single-stat decision rule in `planning.md` prevented inconsistent labeling during the 216-example pass.

**Where implementation diverged:** The spec assumes Colab GPU training; I also built a local `train.py` pipeline so training and evaluation can run outside Colab. The core fine-tuning logic matches the starter notebook, but file paths and `.env` for Groq are local additions for reproducibility.

---

## AI Usage

1. **Label stress-testing:** Before annotating, I asked an LLM to generate posts at the analysis/hot_take boundary. Several outputs were genuinely ambiguous, which led to tightening the single-stat decision rule in `planning.md`.

2. **Dataset drafting:** An LLM helped draft initial post templates for each label. I reviewed every example, rewrote many for natural r/nba voice, and assigned all final labels manually.

3. **Failure pattern analysis:** After evaluation, I pasted misclassified examples into an LLM to surface patterns (single-stat posts, sarcasm). I verified each pattern against the actual errors before writing the error analysis section.

No examples were accepted from AI pre-labeling without manual review.

---

## Project Structure

```
Week_4/
├── app.py                 # Gradio web interface
├── config.py              # Labels, prompts, paths
├── demo.py                # Demo script for video recording
├── train.py               # Fine-tune + baseline + export
├── generate_dataset.py    # Create labeled CSV
├── planning.md            # Design document
├── requirements.txt
├── .env.example
├── data/
│   └── takemeter_dataset.csv
├── models/
│   └── takemeter-model/   # After training
├── evaluation_results.json
└── confusion_matrix.png
```

