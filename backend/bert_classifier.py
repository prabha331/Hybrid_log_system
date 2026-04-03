import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


class BERTClassifier:
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        self.dropout = nn.Dropout(p=0.3)
        self.logreg = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                outputs = self.bert(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                cls_embedding = self.dropout(cls_embedding)
                embeddings.append(cls_embedding.squeeze().numpy())
        return np.array(embeddings)

    def fit(self, X: list[str], y: list[str]):
        embeddings = self._get_embeddings(X)
        embeddings = self.scaler.fit_transform(embeddings)
        encoded_labels = self.label_encoder.fit_transform(y)
        self.logreg = LogisticRegression(C=0.1, max_iter=1000)
        self.logreg.fit(embeddings, encoded_labels)
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.logreg, os.path.join(self.model_dir, "logreg.pkl"))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, "label_encoder.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))

    def predict(self, log_message: str) -> dict:
        if self.logreg is None:
            raise RuntimeError("Model not loaded. Call fit() or load() first.")
        embedding = self._get_embeddings([log_message])
        embedding = self.scaler.transform(embedding)
        encoded = self.logreg.predict(embedding)[0]
        proba = self.logreg.predict_proba(embedding)[0]
        label = self.label_encoder.inverse_transform([encoded])[0]
        confidence = float(proba[encoded])
        return {"label": label, "confidence": round(confidence, 4)}

    def load(self, model_dir="models/"):
        self.model_dir = model_dir
        self.logreg = joblib.load(os.path.join(model_dir, "logreg.pkl"))
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
