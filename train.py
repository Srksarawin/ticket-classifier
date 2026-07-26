"""
Train a ticket/email category classifier.
Pipeline: clean text -> TF-IDF -> Multinomial Naive Bayes -> evaluate -> save model.
"""
import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/tickets.csv"
MODEL_PATH = "model.joblib"

# Minimal stopword list (kept small on purpose — TF-IDF already downweights common words,
# and for a small dataset over-aggressive stopword removal can strip useful signal like "not").
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "with", "at", "this", "that", "it", "my", "i", "you",
    "your", "please", "have", "has", "had", "will", "would", "can", "could", "do",
    "does", "did", "as", "by", "from", "about",
}


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits noise, collapse whitespace, drop stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # strip punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=0.25, random_state=42, stratify=df["category"]
    )

    # TF-IDF: unigrams + bigrams capture short phrases like "not working" / "still have".
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Naive Bayes: strong baseline for short, sparse text with distinct keyword vocab per class
    # (billing vs technical vs HR vocab barely overlaps) and trains instantly on small data.
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_preds = nb.predict(X_test_vec)

    # Logistic Regression: compare against NB since it models feature interactions better
    # and gives well-calibrated probabilities (useful for the confidence score bonus).
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_preds = lr.predict(X_test_vec)

    print("=" * 60)
    print("Naive Bayes")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, nb_preds):.2f}")
    print(classification_report(y_test, nb_preds, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(df["category"].unique())
    print(pd.DataFrame(confusion_matrix(y_test, nb_preds, labels=labels), index=labels, columns=labels))

    print("\n" + "=" * 60)
    print("Logistic Regression")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.2f}")
    print(classification_report(y_test, lr_preds, zero_division=0))

    # Pick the better model on this split; ties go to LR for better-calibrated confidence scores.
    nb_acc = accuracy_score(y_test, nb_preds)
    lr_acc = accuracy_score(y_test, lr_preds)
    best_model, best_name = (lr, "LogisticRegression") if lr_acc >= nb_acc else (nb, "MultinomialNB")
    print(f"\nSelected model for deployment: {best_name}")

    # Retrain the chosen model on ALL data before saving (max signal for production use).
    X_all_vec = vectorizer.fit_transform(df["clean_text"])
    best_model.fit(X_all_vec, df["category"])

    joblib.dump({"model": best_model, "vectorizer": vectorizer, "model_name": best_name}, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train()
