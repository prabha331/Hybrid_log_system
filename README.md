# Hybrid Log Classifier

A production-style text classification system that routes programming logs
through a 3-layer hybrid pipeline — Regex, BERT, and LLM — using the
cheapest and fastest method for each log type.

## Architecture
<img width="900" height="820" alt="architecture" src="https://github.com/user-attachments/assets/b84a7edc-d977-40e5-af03-2b8bee2788ff" />

## Tech Stack

| Layer | Technology |
|---|---|
| Regex | Python `re` module |

| ML Classifier | HuggingFace BERT + Sklearn LogisticRegression |
| LLM | Groq API — llama-3.3-70b-versatile |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data | Pandas |

## Project Structure
```
hybrid_log_classifier/
├── data/
│   └── synthetic_logs.csv       # 2410 labeled log samples
├── models/                       # saved after training
│   ├── logreg.pkl
│   └── label_encoder.pkl
├── backend/
│   ├── classifier.py             # main hybrid pipeline
│   ├── patterns.py               # regex rules
│   ├── bert_classifier.py        # BERT + LogReg
│   ├── llm_classifier.py         # Groq LLM
│   └── main.py                   # FastAPI app
├── frontend/
│   └── app.py                    # Streamlit UI
├── train.py                      # training script
├── requirements.txt
└── .env                          # GROQ_API_KEY (not committed)
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/hybrid-log-classifier
cd hybrid-log-classifier
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at https://console.groq.com

### 4. Train the model
```bash
python train.py
```
This downloads BERT and trains LogisticRegression. Takes ~20 mins on CPU.

## Running the App

Open two terminals:

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## Dataset

2,410 synthetic programming logs across 9 classes:

| Label | Samples | Layer |
|---|---|---|
| HTTP Status | 1017 | regex + bert |
| Security Alert | 371 | bert |
| System Notification | 356 | regex |
| Error | 177 | bert |
| Resource Usage | 177 | bert |
| Critical Error | 161 | bert |
| User Action | 144 | regex |
| Workflow Error | 4 | llm |
| Deprecation Warning | 3 | llm |

## How the Hybrid Pipeline Works

1. **Regex layer** — checks for fixed patterns (file uploads, backups,
   HTTP status codes). Instant, zero ML cost. Handles ~21% of logs.

2. **BERT layer** — extracts CLS embeddings from bert-base-uncased,
   classifies with LogisticRegression. Handles ~79% of logs.

3. **LLM layer** — sends rare/complex logs to Groq API. Only used for
   Workflow Error and Deprecation Warning which have too few samples
   to train a reliable ML model.

## Key Design Decision

> Using LLM only when truly needed reduces API cost by ~79% compared
> to an LLM-only approach, while maintaining accuracy across all classes.
