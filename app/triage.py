"""
Orchestrates the end-to-end triage pipeline:

  raw ticket -> normalize -> [LLM] classify -> KB retrieve (BM25, no LLM)
             -> [LLM] route + draft response -> TriageResult

Two LLM calls per ticket, kept separate on purpose:
  1. Classification is a narrow, structured task -- easy to eval/grade on
     its own against a held-out set if you later want to.
  2. Routing + drafting depends on KB retrieval output, which depends on
     (1)'s search_keywords -- so it has to happen after anyway.
They could be merged into one call for lower latency/cost once you're
happy with quality; see README "Optimizations" section.
"""
import time
from typing import Any, Optional

from app.config import KB_MATCH_SCORE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD, DEFAULT_KB_ROOT
from app.kb_index import KBIndex
from app.llm_client import LLMClient
from app.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    ROUTING_SYSTEM_PROMPT,
    build_classification_user_prompt,
    build_routing_user_prompt,
)
from app.schemas import (
    Classification,
    KBMatch,
    RoutingAndResponse,
    TicketInput,
    TriageResult,
)


class TriagePipeline:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        kb_index: Optional[KBIndex] = None,
        kb_root: str = DEFAULT_KB_ROOT,
        kb_top_k: int = 3,
    ):
        self.llm = llm_client or LLMClient()
        self.kb = kb_index or KBIndex(kb_root)
        self.kb_top_k = kb_top_k
        self._known_products = self.kb.known_product_names()

    def run(self, raw_ticket: Any) -> TriageResult:
        start = time.perf_counter()
        raw_ticket_id = raw_ticket.get("ticket_id") if isinstance(raw_ticket, dict) else None
        self._trace(f"start ticket_id={raw_ticket_id or '<generated>'}")

        try:
            ticket = TicketInput.from_raw(raw_ticket)
            self._trace(f"normalized ticket_id={ticket.ticket_id}")

            classification = self._classify(ticket)
            self._trace(
                "classified "
                f"category={classification.issue_category} "
                f"product_area={classification.product_area} "
                f"urgency={classification.urgency} "
                f"confidence={classification.confidence:.2f}"
            )

            kb_matches = self._retrieve_kb(ticket, classification)
            self._trace(f"retrieved kb_matches={len(kb_matches)}")

            routing = self._route_and_draft(ticket, classification, kb_matches)
            self._trace(f"routed recommended_team={routing.recommended_team}")

            needs_human_review = self._compute_needs_review(classification, kb_matches)
            latency_ms = (time.perf_counter() - start) * 1000
            self._trace(
                f"complete needs_human_review={needs_human_review} "
                f"latency_ms={latency_ms:.1f}"
            )

            return TriageResult(
                ticket_id=ticket.ticket_id,
                classification=classification,
                kb_matches=kb_matches,
                routing=routing,
                needs_human_review=needs_human_review,
                model_used=self.llm.config.model,
                pipeline_latency_ms=round(latency_ms, 1),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self._trace(
                f"failed error={type(exc).__name__}: {exc} latency_ms={latency_ms:.1f}"
            )
            raise

    @staticmethod
    def _trace(message: str) -> None:
        print(f"[triage-trace] {message}", flush=True)

    # ---------- steps ----------

    def _classify(self, ticket: TicketInput) -> Classification:
        user_prompt = build_classification_user_prompt(ticket, self._known_products)
        return self.llm.chat_json(CLASSIFICATION_SYSTEM_PROMPT, user_prompt, Classification)

    def _retrieve_kb(self, ticket: TicketInput, classification: Classification) -> list:
        query_parts = list(classification.search_keywords) + [
            classification.product_area,
            ticket.subject,
        ]
        query = " ".join(p for p in query_parts if p)
        raw_matches = self.kb.search(query, top_k=self.kb_top_k)
        return [KBMatch(**m) for m in raw_matches if m["score"] >= 0]

    def _route_and_draft(
        self, ticket: TicketInput, classification: Classification, kb_matches: list
    ) -> RoutingAndResponse:
        user_prompt = build_routing_user_prompt(ticket, classification, kb_matches)
        return self.llm.chat_json(ROUTING_SYSTEM_PROMPT, user_prompt, RoutingAndResponse)

    def _compute_needs_review(self, classification: Classification, kb_matches: list) -> bool:
        """
        Deterministic, code-level safety net -- not something the model
        decides about itself. Flags for a human to double check when:
          - the model's own confidence is low, or
          - it's a Bug/Outage with no supporting KB match at all (higher
            risk of a fabricated or generic fix in the draft response).
        """
        if classification.confidence < LOW_CONFIDENCE_THRESHOLD:
            return True
        strong_match_exists = any(m.score >= KB_MATCH_SCORE_THRESHOLD for m in kb_matches)
        if classification.issue_category in ("Bug", "Outage") and not strong_match_exists:
            return True
        return False
