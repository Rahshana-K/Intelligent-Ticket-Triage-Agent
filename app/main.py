"""
FastAPI wrapper around TriagePipeline.

Run with:
    uvicorn app.main:app --reload --port 8000

Then:
    POST /triage
    {
      "subject": "Unable to connect DataBridge Pro to Connectors",
      "body": "..."
    }

or the full ticket record shape from tickets.json (extra fields are ignored).
"""
from functools import lru_cache
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm_client import LLMError
from app.schemas import TriageResult
from app.triage import TriagePipeline

app = FastAPI(
    title="Support Ticket Triage Agent",
    description="Classifies, routes, and drafts first responses for incoming support tickets.",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    ticket_id: Optional[str] = None
    subject: str = ""
    body: str
    company: Optional[str] = None
    plan_tier: Optional[str] = None
    product: Optional[str] = None


@lru_cache(maxsize=1)
def get_pipeline() -> TriagePipeline:
    # Cached: the LLM client and KB index (which parses every markdown file
    # in the knowledge base) are both cheap to build but no reason to redo
    # it per-request.
    return TriagePipeline()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResult)
def triage_ticket(payload: Union[TicketRequest, Dict[str, Any]]) -> TriageResult:
    pipeline = get_pipeline()
    raw = payload.model_dump() if isinstance(payload, TicketRequest) else payload
    try:
        return pipeline.run(raw)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:  # noqa: BLE001 - surface unexpected errors clearly to the caller
        raise HTTPException(status_code=500, detail=f"Triage pipeline error: {e}")
