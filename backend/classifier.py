import time

from backend.patterns import classify_with_regex
from backend.bert_classifier import BERTClassifier
from backend.llm_classifier import LLMClassifier


LLM_ONLY_LABELS = ["Workflow Error", "Deprecation Warning"]


class HybridClassifier:
    def __init__(self, bert_threshold=0.6, min_bert_samples=20):
        self.bert_threshold = bert_threshold
        self.min_bert_samples = min_bert_samples
        self._bert: BERTClassifier | None = None
        self._llm: LLMClassifier | None = None

    @property
    def bert(self) -> BERTClassifier:
        if self._bert is None:
            self._bert = BERTClassifier()
            self._bert.load()
        return self._bert

    @property
    def llm(self) -> LLMClassifier:
        if self._llm is None:
            self._llm = LLMClassifier()
        return self._llm

    def classify(self, log_message: str) -> dict:
        # STEP 0 — LLM-only keyword shortcut
        if any(keyword in log_message.lower() for keyword in
               ["escalation", "deprecat", "undefined escalation", "priority level"]):
            result = self.llm.classify(log_message)
            return {"label": result["label"], "layer": "llm",
                    "confidence": result["confidence"], "latency_ms": 0}

        # STEP 1 — Regex
        t0 = time.time()
        label = classify_with_regex(log_message)
        latency_ms = round((time.time() - t0) * 1000, 3)
        if label is not None:
            return {
                "label": label,
                "layer": "regex",
                "confidence": 1.0,
                "latency_ms": latency_ms,
            }

        # STEP 2 — BERT
        t0 = time.time()
        try:
            bert_result = self.bert.predict(log_message)
            latency_ms = round((time.time() - t0) * 1000, 3)
            if bert_result["label"] in LLM_ONLY_LABELS:
                pass  # skip to LLM — too few training samples for this label
            elif bert_result["confidence"] >= self.bert_threshold:
                return {
                    "label": bert_result["label"],
                    "layer": "bert",
                    "confidence": bert_result["confidence"],
                    "latency_ms": latency_ms,
                }

        except Exception:
            pass

        # STEP 3 — LLM fallback
        t0 = time.time()
        llm_result = self.llm.classify(log_message)
        latency_ms = round((time.time() - t0) * 1000, 3)
        return {
            "label": llm_result["label"],
            "layer": "llm",
            "confidence": llm_result["confidence"],
            "latency_ms": latency_ms,
        }
