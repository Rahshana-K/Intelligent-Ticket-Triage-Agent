import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import DEFAULT_KB_ROOT  # noqa: E402
from app.kb_index import KBIndex  # noqa: E402
from app.schemas import Classification, RoutingAndResponse, Urgency  # noqa: E402
from app.triage import TriagePipeline  # noqa: E402


class FakeLLMClient:
    """
    Stands in for LLMClient in tests -- returns canned, schema-valid
    objects instead of calling a real model, so the pipeline's own logic
    (KB retrieval wiring, needs_human_review, latency measurement) can be
    tested without network access or an API key.
    """

    class _Config:
        model = "fake-model"

    def __init__(self, classification: Classification, routing: RoutingAndResponse):
        self.config = self._Config()
        self._classification = classification
        self._routing = routing
        self.calls = []

    def chat_json(self, system, user, schema, retries=2):
        self.calls.append((schema.__name__, system, user))
        if schema is Classification:
            return self._classification
        if schema is RoutingAndResponse:
            return self._routing
        raise AssertionError(f"Unexpected schema requested: {schema}")


def _make_pipeline(classification: Classification, routing: RoutingAndResponse) -> TriagePipeline:
    fake_llm = FakeLLMClient(classification, routing)
    kb = KBIndex(DEFAULT_KB_ROOT)
    return TriagePipeline(llm_client=fake_llm, kb_index=kb)


def test_high_confidence_bug_with_kb_match_does_not_need_review():
    classification = Classification(
        product_area="Connectors",
        issue_category="Bug",
        urgency=Urgency.P2,
        reasoning="Timeout error affecting many users in production.",
        impact_summary="47 users unable to sync data.",
        affected_user_count_estimate=47,
        search_keywords=["ERR_CONNECTION_TIMEOUT", "connectors", "timeout"],
        confidence=0.9,
    )
    routing = RoutingAndResponse(
        recommended_team="Platform Engineering",
        routing_reasoning="Connector-level production bug.",
        draft_response="Hi, thanks for flagging this -- we're on it...",
        escalation_flag=False,
    )
    pipeline = _make_pipeline(classification, routing)

    result = pipeline.run(
        {
            "ticket_id": "TKT-TEST-1",
            "subject": "Unable to connect DataBridge Pro to Connectors",
            "body": "ERR_CONNECTION_TIMEOUT after 30s, affecting 47 users in production.",
            "plan_tier": "Enterprise",
        }
    )

    assert result.ticket_id == "TKT-TEST-1"
    assert result.classification.urgency == Urgency.P2
    assert len(result.kb_matches) > 0
    assert result.kb_matches[0].doc_path == "products/databridge-pro.md"
    assert result.needs_human_review is False
    assert result.model_used == "fake-model"
    assert result.pipeline_latency_ms >= 0


def test_low_confidence_flags_for_human_review():
    classification = Classification(
        product_area="Unknown",
        issue_category="Other",
        urgency=Urgency.P3,
        reasoning="Ambiguous ticket, hard to classify.",
        impact_summary="Unclear.",
        search_keywords=["vague"],
        confidence=0.2,
    )
    routing = RoutingAndResponse(
        recommended_team="General Support",
        routing_reasoning="Default queue for unclear tickets.",
        draft_response="Thanks for reaching out, could you clarify...",
        escalation_flag=False,
    )
    pipeline = _make_pipeline(classification, routing)

    result = pipeline.run({"subject": "???", "body": "not sure what's going on"})
    assert result.needs_human_review is True


def test_bug_without_kb_match_flags_for_human_review():
    classification = Classification(
        product_area="Some Other Product",
        issue_category="Bug",
        urgency=Urgency.P3,
        reasoning="Novel bug not covered in KB.",
        impact_summary="One user affected.",
        search_keywords=["completely_unrelated_zzz_term"],
        confidence=0.85,
    )
    routing = RoutingAndResponse(
        recommended_team="Product Engineering",
        routing_reasoning="No existing KB coverage, needs investigation.",
        draft_response="Thanks for the report, we're looking into it...",
        escalation_flag=False,
    )
    pipeline = _make_pipeline(classification, routing)

    result = pipeline.run({"subject": "Weird novel bug", "body": "something totally new broke"})
    assert result.needs_human_review is True


def test_feature_request_without_kb_match_does_not_need_review():
    # Low-risk categories shouldn't be auto-flagged just for lacking a KB hit.
    classification = Classification(
        product_area="Connectors",
        issue_category="Feature Request",
        urgency=Urgency.P4,
        reasoning="Cosmetic feature request, no functional impact.",
        impact_summary="Nice-to-have for a few users.",
        search_keywords=["dark mode", "dashboard"],
        confidence=0.9,
    )
    routing = RoutingAndResponse(
        recommended_team="Product Management",
        routing_reasoning="Feature requests are triaged by product, not support.",
        draft_response="Thanks for the suggestion, we've logged it...",
        escalation_flag=False,
    )
    pipeline = _make_pipeline(classification, routing)

    result = pipeline.run({"subject": "Dark mode please", "body": "would love a dark theme"})
    assert result.needs_human_review is False
