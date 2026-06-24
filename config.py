"""TakeMeter configuration — label map and model settings."""

LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "reaction": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
VALID_LABELS = list(LABEL_MAP.keys())

MODEL_NAME = "distilbert-base-uncased"
DATASET_PATH = "data/takemeter_dataset.csv"
MODEL_DIR = "models/takemeter-model"
RESULTS_PATH = "evaluation_results.json"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"

SYSTEM_PROMPT = """You are classifying posts from r/nba (Reddit's NBA community).
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument backed by specific statistics, film observations, or tactical reasoning. Evidence is concrete and verifiable.
Example: "The Celtics' defensive rating drops 8.2 points per 100 possessions when Horford sits. Their switch scheme falls apart without his IQ at the nail."

hot_take: A bold, confident opinion stated without substantive supporting evidence. The claim may be debatable but the post asserts rather than argues.
Example: "Luka is already a top-5 player all-time and it's not even close. People who disagree just haven't been watching."

reaction: An immediate emotional response to a specific game event or news item. Little to no argument — the post expresses a feeling in the moment.
Example: "That block by Wemby at the end had me jumping off my couch. Absolutely unreal sequence."

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction"""
