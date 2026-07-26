"""
Comments for better understanding of this project - Developer Sarawin

Real-time ticket classification.
Usage:
    python predict.py                          -> interactive CLI demo
    python predict.py "some ticket text"        -> classify one ticket and exit
"""
import sys
import joblib
from train import clean_text

MODEL_PATH = "model.joblib"
CONFIDENCE_THRESHOLD = 0.60  # below this -> route to manual review queue

URGENT_KEYWORDS = {
    "urgent", "asap", "immediately", "down", "outage", "not working",
    "crash", "crashes", "crashed", "broken", "critical", "emergency",
    "right now", "everyone", "production",
}


def priority_tag(raw_text: str) -> str:
    """Simple keyword-rule urgency tagger. Runs on raw (uncleaned) text so multi-word
    phrases like 'not working' or 'right now' are still matched intact."""
    text = raw_text.lower()
    return "URGENT" if any(kw in text for kw in URGENT_KEYWORDS) else "Normal"


def load_pipeline(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["vectorizer"], bundle["model_name"]


def classify(raw_text: str, model, vectorizer):
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = probs.argmax()
    category = classes[best_idx]
    confidence = probs[best_idx]

    result = {
        "category": category,
        "confidence": round(float(confidence), 3),
        "priority": priority_tag(raw_text),
        "needs_human_review": confidence < CONFIDENCE_THRESHOLD,
    }
    return result


def format_result(text: str, result: dict) -> str:
    lines = [
        f"Ticket: {text[:80]}{'...' if len(text) > 80 else ''}",
        f"  -> Category   : {result['category']}"
        + ("  [LOW CONFIDENCE - ROUTED TO MANUAL REVIEW]" if result["needs_human_review"] else ""),
        f"  -> Confidence : {result['confidence'] * 100:.1f}%",
        f"  -> Priority   : {result['priority']}",
    ]
    return "\n".join(lines)


def run_cli():
    model, vectorizer, model_name = load_pipeline()
    print(f"Loaded model: {model_name}  (auto-assign threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)")
    print("Type a support ticket (subject + body) and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        result = classify(text, model, vectorizer)
        print(format_result(text, result))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticket_text = " ".join(sys.argv[1:])
        model, vectorizer, model_name = load_pipeline()
        result = classify(ticket_text, model, vectorizer)
        print(format_result(ticket_text, result))
    else:
        run_cli()
