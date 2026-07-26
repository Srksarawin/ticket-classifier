# Auto Email / Ticket Categorizer

Routes an incoming support ticket to **Billing / Technical / HR / General** using TF-IDF + a classic sklearn classifier, with confidence scoring, a low-confidence human-review fallback, and keyword-based urgency tagging.

## Files
- `data/tickets.csv` — labeled training data (40 tickets, 4 classes)
- `train.py` — cleans text, builds TF-IDF features, trains Naive Bayes + Logistic Regression, prints accuracy/precision/recall/confusion matrix, saves the better model to `model.joblib`
- `predict.py` — loads the saved model and classifies new tickets in real time (CLI demo or one-shot arg)
- `REFLECTION.md` — what I'd improve with more time/data

## Run it
```bash
pip install scikit-learn pandas joblib
python train.py              # trains + evaluates, saves model.joblib
python predict.py            # interactive CLI — type a ticket, get category/confidence/priority
python predict.py "The app crashes every time I open it"   # one-shot classification
```

## Design choices
- **TF-IDF (uni+bigrams)** over raw Bag-of-Words so short phrases like "not working" or "still have" keep meaning instead of being reduced to isolated words.
- **Naive Bayes vs Logistic Regression**, both trained; whichever scores higher on the held-out split is auto-selected and retrained on the full dataset for deployment.
- **Confidence threshold (60%)**: any prediction below this is flagged `needs_human_review` instead of auto-routed — mirrors real triage tools that won't blindly auto-assign an uncertain ticket.
- **Priority tagging** is a separate keyword-rule layer (`urgent`, `down`, `crash`, `outage`, etc.) run on raw text, independent of the ML category prediction.
- **Edge cases**: a ticket that doesn't match any category cleanly should land in `needs_human_review` rather than being forced into a wrong bucket — see `REFLECTION.md` for how I'd extend this to a real "unclassified" class.
