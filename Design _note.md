# Task 4 · Design Note

## 1. Failure Modes

**->Misclassified urgency (silent P1 downgraded to P3):** 

The single highest-risk failure: a genuine outage gets under-prioritised because the
model reads a calmly-worded ticket as low-impact.

 *Detection:* 
 
 Log
`classification.confidence` and `urgency` on every ticket and alert when a
ticket contains outage-adjacent language (e.g. "down", "all users",
"production") but is scored P3/P4 — a cheap regex tripwire alongside the
LLM's own judgement. 

*Mitigation:* 

The deterministic `needs_human_review`
flag already catches low-confidence cases; extend its rule set with a
keyword-based override so certain phrases force a human look regardless
of the model's stated confidence.

**-> Malformed or fabricated structured output:** 

The model returns invalid
JSON, or valid JSON that doesn't match reality — e.g. inventing a
`draft_response` fix that isn't in the KB. 

*Detection:* 

schema validation
already catches the first case (with automatic retry in
`LLMClient.chat_json`); the second is harder to detect automatically.


*Mitigation:* 

the routing prompt is explicitly instructed not to invent a
fix when no KB excerpt is relevant, and `needs_human_review` flags
Bug/Outage tickets with no strong KB match — but this is a mitigation, not
a guarantee. A production rollout should sample and manually audit a
percentage of auto-sent drafts, especially on unfamiliar product areas.

**-> Provider outage or rate-limiting:** 

OpenRouter (or a local LM Studio
instance) becomes unreachable or slow under load. 

*Detection:*


`LLMClient` raises `LLMError` on API failure, which `run_batch.py` logs
per-ticket without aborting the batch, and which the FastAPI layer
surfaces as a 502.

 *Mitigation:*
 
  add a circuit breaker with fallback to a
secondary model/provider, and a dead-letter queue so failed tickets are
retried rather than silently dropped from the triage queue.

## 2. Latency vs Quality

**The concrete trade-off made:** the pipeline uses two sequential LLM
calls (classify, then route+draft) instead of one merged call, because
classification's `search_keywords` output is needed *before* KB retrieval
can run, and drafting needs the retrieved excerpts as grounding. This
roughly doubles per-ticket latency and cost compared to a single prompt,
but keeps each concern independently testable/eval-able and lets the
draft response cite real KB content instead of guessing.

**If latency were the hard constraint,** I would merge both steps into
one prompt: ask the model to classify *and* propose search terms *and*
draft a placeholder response in one call, run KB retrieval, then do a
second, much cheaper call only to revise the draft with the retrieved
excerpts — or skip revision entirely and only inject KB context when a
match scores above threshold. This trims one full model round-trip per
ticket for the common case.

## 3. Data Sensitivity

Ticket bodies and account metadata can contain PII (names, emails,
internal hostnames, sometimes credentials pasted into error logs). Two
protections are built in: (1) the KB itself contains no customer data —
only product documentation — so retrieval never leaks between accounts;
(2) the LLM client is provider-swappable specifically so PII-bearing
tickets can be routed to an **LM Studio** instance running fully inside
the corporate network, keeping ticket content off any external API
entirely, while lower-sensitivity tickets or non-production testing use a
hosted provider. What this design does *not* yet do: field-level PII
redaction before the prompt is built. The natural next step is a
redaction pass (regex/NER for emails, IPs, tokens) in `TicketInput.as_prompt_text()`
before any external call, with an audit log of what was stripped.

## 4. Scaling (10× ticket volume)

The KB index and BM25 search are in-memory and O(chunks) per query —
trivial at 10×, since KB size doesn't grow with ticket volume. **What
breaks first is LLM throughput and cost**: two sequential calls per
ticket means 10× tickets is 20× more API calls, quickly hitting
per-minute rate limits (hosted) or saturating a single local GPU (LM
Studio). Second bottleneck: `run_batch.py` processes tickets serially —
at 10× volume this becomes a multi-hour job. Fixes, in order of effort:
add async/concurrent request batching with a semaphore-limited worker
pool; move the FastAPI service behind a queue (so ticket ingestion is
decoupled from LLM processing throughput); and revisit the two-call
design (§2) to cut call volume per ticket before scaling infrastructure
around it.