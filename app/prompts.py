"""
Prompt templates. Kept separate from orchestration logic (triage.py) so
they can be iterated on / eval'd independently.

Note on "no human labelling": the URGENCY_RUBRIC and KNOWN_TEAMS below are
*policy* the org already has (an SLA definition, an org chart) -- not
per-ticket labels. Nothing here tells the model the "right answer" for any
specific ticket in tickets.json.
"""
from typing import List

from app.schemas import KBMatch, TicketInput
from app.schemas import Classification

URGENCY_RUBRIC = """
Urgency tiers (assign exactly one):
- P1 (Critical): Production down, security/data-loss risk, or an outage affecting many users/accounts with no workaround.
- P2 (High): Major functionality broken or badly degraded for one or more users; only a partial or painful workaround exists.
- P3 (Medium): Limited impact -- a single user or a minor, non-blocking bug; a reasonable workaround exists.
- P4 (Low): Cosmetic issue, how-to question, documentation gap, or feature request with no functional impact.
""".strip()

CATEGORY_GUIDE = """
issue_category must be one of:
Bug, Outage, Question, Feature Request, Billing, Access/Auth, Configuration, Data, Other
""".strip()


def _known_products_hint(kb_product_names: List[str]) -> str:
    if not kb_product_names:
        return ""
    joined = ", ".join(sorted(kb_product_names))
    return (
        f"\nKnown products/product areas we support (use one of these if the ticket clearly "
        f"matches; otherwise use your best judgement and name the area descriptively): {joined}\n"
    )


CLASSIFICATION_SYSTEM_PROMPT = f"""
You are a support-ticket triage assistant for a technical support team. Given a raw
customer ticket, classify it precisely and explain your reasoning briefly.

{URGENCY_RUBRIC}

category guidance:
- `Bug` — product defect or unexpected behaviour
- `Feature Request` — request for new functionality
- `How-To` — guidance or documentation request
- `Performance` — slowness, timeouts, throughput issues
- `Billing` — invoice, payment, or plan questions
- `Integration` — third-party integration issues
- `Onboarding` — new user or new organisation setup
- `Data Loss` — missing, corrupted, or inaccessible data

PRODUCT AREA GUIDANCE:

`product_area` must identify the specific feature, module, subsystem, or
functional area that the ticket is primarily about.

IMPORTANT:
- Do NOT use the product name as `product_area`.
- The product name and product area are different concepts.
- `product_area` should describe a feature within the product.
- Examples of valid product areas include:
  `SSO`, `Authentication`, `Permissions`, `Billing`, `API`, `Integrations`,
  `Data Management`, `Reporting`, `User Management`, or another specific
  feature mentioned or clearly implied by the ticket.
- If multiple product areas are mentioned, choose the PRIMARY area based on
  the ticket subject and main customer request.
- Do not choose a secondary or unrelated question as the primary product area.
- If no specific feature or module can be identified,then only  return the mentioned product name .

EXAMPLE:
Product: `SecureVault`
Ticket: "How do I configure SSO in SecureVault?"
Correct product_area: `SSO`
Incorrect product_area: `SecureVault`

Be conservative with P1/P2 -- reserve them for genuine production impact, not just
frustrated tone. Extract or estimate affected_user_count_estimate only from what is
stated or clearly implied; leave it null if there's no basis for a number.

search_keywords should be terms useful for searching an internal knowledge base
(product names, error codes, feature names) -- not generic words like "issue" or "help".
""".strip()


def build_classification_user_prompt(ticket: TicketInput, kb_product_names: List[str]) -> str:
    return (
        f"{_known_products_hint(kb_product_names)}\n"
        "Classify the following support ticket.\n\n"
        f"{ticket.as_prompt_text()}\n\n"
        "Return JSON with exactly these fields: product_area, issue_category, urgency "
        "(P1|P2|P3|P4), reasoning, impact_summary, affected_user_count_estimate "
        "(number or null), search_keywords (array of strings), confidence (0-1 float)."
    )


ROUTING_SYSTEM_PROMPT = """
You are a support-ticket routing and response-drafting assistant. You will be given a
ticket, its classification, and relevant knowledge-base excerpts (which may be empty).

Recommend the single best-suited internal team to own this ticket, and draft a
first-response message a support agent could send with light editing.

Guidelines for the draft response:
- Acknowledge the specific issue, don't sound like a form letter.
- If a knowledge-base excerpt is directly relevant, incorporate its guidance/fix in
  your own words and reference it as a next step or possible resolution.
- If no KB excerpt is relevant, don't invent a fix -- acknowledge the issue, set
  expectations, and ask any clarifying question you need to move it forward.
- Match tone to urgency: P1/P2 should acknowledge urgency and impact explicitly.
- Keep it to a few short paragraphs. Sign off as "The Support Team".

Set escalation_flag to true only for genuine P1s or anything implying security/data
loss/compliance exposure, regardless of the stated urgency.
""".strip()


def _format_kb_matches(kb_matches: List[KBMatch]) -> str:
    if not kb_matches:
        return "(No relevant knowledge-base entries were found.)"
    blocks = []
    for m in kb_matches:
        blocks.append(
            f"- Source: {m.doc_path} ({m.heading_path}) [relevance score {m.score}]\n  {m.snippet}"
        )
    return "\n".join(blocks)


def build_routing_user_prompt(
    ticket: TicketInput, classification: Classification, kb_matches: List[KBMatch]
) -> str:
    return (
        "Ticket:\n"
        f"{ticket.as_prompt_text()}\n\n"
        "Classification:\n"
        f"- product_area: {classification.product_area}\n"
        f"- issue_category: {classification.issue_category}\n"
        f"- urgency: {classification.urgency.value}\n"
        f"- impact_summary: {classification.impact_summary}\n\n"
        "Relevant knowledge-base excerpts:\n"
        f"{_format_kb_matches(kb_matches)}\n\n"
        "Return JSON with exactly these fields: recommended_team, routing_reasoning, "
        "draft_response, escalation_flag (true/false)."
    )
