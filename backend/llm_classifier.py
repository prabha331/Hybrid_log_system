import json
import os
import re

import groq
from dotenv import load_dotenv

load_dotenv()

VALID_LABELS = {
    "HTTP Status",
    "Security Alert",
    "System Notification",
    "Error",
    "Resource Usage",
    "Critical Error",
    "User Action",
    "Workflow Error",
    "Deprecation Warning",
}

SYSTEM_PROMPT = """You are a log classifier. Classify the given log message into exactly one of these labels:
HTTP Status, Security Alert, System Notification, Error, Resource Usage,
Critical Error, User Action, Workflow Error, Deprecation Warning

Respond with ONLY a JSON object in this exact format, no explanation:
{"label": "<one of the labels above>", "confidence": <float between 0.0 and 1.0>}"""


class LLMClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = groq.Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def classify(self, log_message: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": log_message},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Extract JSON from response (handles cases where model adds extra text)
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                return {"label": "Unknown", "confidence": 0.0}

            result = json.loads(match.group())
            label = result.get("label", "")
            confidence = float(result.get("confidence", 0.0))

            if label not in VALID_LABELS:
                return {"label": "Unknown", "confidence": 0.0}

            return {"label": label, "confidence": round(confidence, 4)}

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return {"label": "Unknown", "confidence": 0.0}
