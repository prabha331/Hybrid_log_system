import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from backend.bert_classifier import BERTClassifier

MIN_BERT_SAMPLES = 20

# 1. Load data
df = pd.read_csv("data/synthetic_logs.csv")

# 2. Split by complexity
regex_df = df[df["complexity"] == "regex"]
bert_df  = df[df["complexity"] == "bert"]
llm_df   = df[df["complexity"] == "llm"]

print(f"Dataset split — regex: {len(regex_df)}, bert: {len(bert_df)}, llm: {len(llm_df)}")

# 3. Train BERTClassifier on bert_df
X = bert_df["log_message"].tolist()
y = bert_df["target_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining BERTClassifier on {len(X_train)} samples, evaluating on {len(X_test)}...")

classifier = BERTClassifier(model_dir="models/")
classifier.fit(X_train, y_train)

# 4. Evaluate on test split
predictions = [classifier.predict(msg)["label"] for msg in X_test]
print("\nClassification Report:")
print(classification_report(y_test, predictions))

test_accuracy = accuracy_score(y_test, predictions)

# 5-fold cross-validation on training embeddings
print("Running 5-fold cross-validation...")
train_embeddings = classifier._get_embeddings(X_train)
train_embeddings = classifier.scaler.transform(train_embeddings)
encoded_labels = classifier.label_encoder.transform(y_train)

cv_scores = cross_val_score(
    classifier.logreg, train_embeddings, encoded_labels, cv=5, scoring="accuracy"
)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"Cross-validation accuracy: {cv_mean:.2f} (+/- {cv_std:.2f})")

# 5. Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nTotal samples per label (full dataset):")
label_counts = df["target_label"].value_counts()
for label, count in label_counts.items():
    print(f"  {label:<25} {count}")

llm_labels = [label for label, count in label_counts.items() if count < MIN_BERT_SAMPLES]
print(f"\nLabels with < {MIN_BERT_SAMPLES} samples (will fall back to LLM):")
if llm_labels:
    for label in llm_labels:
        print(f"  {label}")
else:
    print("  None")

print(f"\nCross-val accuracy (realistic estimate): {cv_mean:.4f} (+/- {cv_std:.4f})")
print(f"Test set accuracy:                        {test_accuracy:.4f}")

if test_accuracy > 0.98:
    print("\nWarning: possible overfitting detected")

print("\nModel saved to models/logreg.pkl, models/label_encoder.pkl, models/scaler.pkl")
