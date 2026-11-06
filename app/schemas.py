from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Urgency(str, Enum):
    P1 = "P1"  # Critical: production down / widespread outage / security / data loss, no workaround
    P2 = "P2"  # High: major functionality broken or badly degraded, partial workaround at best
    P3 = "P3"  # Medium: limited impact, single user or minor bug, workaround available
    P4 = "P4"  # Low: cosmetic issue, question, feature request, documentation


class TicketInput(BaseModel):
    """Normalized ticket, regardless of what shape it arrived in."""

    ticket_id: Optional[str] = None
    subject: str
    body: str
    company: Optional[str] = None
    plan_tier: Optional[str] = None
    product_hint: Optional[str] = None  # e.g. a 'product' field the ticketing system already set

    @classmethod
    def from_raw(cls, raw: Any) -> "TicketInput":
        """
        Accepts:
          - a plain string (treated as the body, subject left blank)
          - a dict with at least 'subject' + 'body' (extra fields ignored/mapped)
        """
        if isinstance(raw, str):
            return cls(subject="", body=raw)
        if isinstance(raw, dict):
            return cls(
                ticket_id=raw.get("ticket_id"),
                subject=raw.get("subject", ""),
                body=raw.get("body", ""),
                company=raw.get("company"),
                plan_tier=raw.get("plan_tier"),
                product_hint=raw.get("product"),
            )
        raise TypeError(f"Unsupported ticket input type: {type(raw)}")

    def as_prompt_text(self) -> str:
        meta_lines = []
        if self.company:
            meta_lines.append(f"Company: {self.company}")
        if self.plan_tier:
            meta_lines.append(f"Plan tier: {self.plan_tier}")
        if self.product_hint:
            meta_lines.append(f"Product field (as set by requester/portal): {self.product_hint}")
        meta = ("\n".join(meta_lines) + "\n") if meta_lines else ""
        return f"{meta}Subject: {self.subject}\n\nBody:\n{self.body}"


class Classification(BaseModel):
    product_area: str = Field(..., description="The product or product area the issue belongs to")
    issue_category: str = Field(
        ..., description="One of: Bug, Outage, Question, Feature Request, Billing, Access/Auth, Configuration, Data, Other"
    )
    urgency: Urgency
    reasoning: str = Field(..., description="Short explanation of why this urgency/category was chosen")
    impact_summary: str = Field(..., description="One-sentence summary of who/what is affected")
    affected_user_count_estimate: Optional[int] = Field(
        default=None, description="Best-effort number of impacted users, if stated or implied"
    )
    search_keywords: List[str] = Field(
        default_factory=list, description="3-6 keywords/phrases to use for knowledge-base retrieval"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's self-reported confidence, 0-1")

    @field_validator("confidence")
    @classmethod
    def _clip_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class KBMatch(BaseModel):
    doc_path: str
    heading_path: str
    snippet: str
    score: float


class RoutingAndResponse(BaseModel):
    recommended_team: str = Field(..., description="Team best suited to own this ticket")
    routing_reasoning: str = Field(..., description="Why this team, in one or two sentences")
    draft_response: str = Field(..., description="Ready-to-edit first-response message to the customer")
    escalation_flag: bool = Field(
        default=False, description="True if this should be escalated / paged beyond normal queueing"
    )


class TriageResult(BaseModel):
    ticket_id: Optional[str]
    classification: Classification
    kb_matches: List[KBMatch]
    routing: RoutingAndResponse
    needs_human_review: bool = Field(
        description="Deterministic safety flag set by the pipeline (not the model) when confidence is low "
        "or no KB match was found for a Bug/Outage ticket"
    )
    model_used: str
    pipeline_latency_ms: float

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact view for logging / dashboards."""
        return {
            "ticket_id": self.ticket_id,
            "product_area": self.classification.product_area,
            "issue_category": self.classification.issue_category,
            "urgency": self.classification.urgency.value,
            "recommended_team": self.routing.recommended_team,
            "needs_human_review": self.needs_human_review,
            "kb_match_count": len(self.kb_matches),
        }
