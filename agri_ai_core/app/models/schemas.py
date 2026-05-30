from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str

class DiagnosisResponse(BaseModel):
    status: str
    detected_disease: str
    confidence: float
    treatment_plan: str