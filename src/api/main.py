from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.conversational.triage_orchestrator import MediAssistTriageOrchestrator

app = FastAPI(
    title="MediAssist AI — Hybrid Healthcare Triage API",
    description="Safety-Aware Bilingual AI Healthcare Triage Assistant API",
    version="2.0.0"
)

# Global orchestrator instance (or per-session in future)
orchestrator = MediAssistTriageOrchestrator()

class TriageRequest(BaseModel):
    message: str
    latitude: Optional[float] = 12.9716
    longitude: Optional[float] = 77.5946

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "MediAssist AI",
        "architecture": "Hybrid (Qwen 2.5 7B + Emergency Firewall + PubMedBERT + Clinical Decision Layer)",
        "version": "2.0.0",
        "supported_languages": ["en", "hi"]
    }

@app.post("/triage")
async def process_triage(request: TriageRequest):
    """
    Execute 5-Layer Healthcare Triage Pipeline:
    Parser -> Emergency Veto -> Medical BERT -> Decision Layer -> Bilingual Response
    """
    try:
        result = orchestrator.process_message(
            user_message=request.message,
            user_lat=request.latitude,
            user_lon=request.longitude
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_session():
    orchestrator.reset()
    return {"status": "session_reset_successful"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
