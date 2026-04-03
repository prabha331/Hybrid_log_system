from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.classifier import HybridClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = HybridClassifier()
    yield


app = FastAPI(title="Hybrid Log Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LogRequest(BaseModel):
    log_message: str


class ClassifyResponse(BaseModel):
    log_message: str
    label: str
    layer: str
    confidence: float
    latency_ms: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Hybrid Log Classifier API"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, body: LogRequest):
    result = request.app.state.classifier.classify(body.log_message)
    return ClassifyResponse(log_message=body.log_message, **result)


@app.get("/health")
def health(request: Request):
    classifier = request.app.state.classifier
    bert_loaded = classifier._bert is not None
    llm_loaded = classifier._llm is not None
    return {
        "status": "ok",
        "bert_loaded": bert_loaded,
        "llm_loaded": llm_loaded,
    }
