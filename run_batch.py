"""
Run the triage pipeline over every ticket in data/tickets.json and write
results to triage_results.json. Handles per-ticket failures without
aborting the whole batch.

Usage:
    python run_batch.py
    python run_batch.py --tickets path/to/tickets.json --out results.json --limit 10
"""
import argparse
import json
import sys

from app.config import DEFAULT_TICKETS_PATH
from app.triage import TriagePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", default=DEFAULT_TICKETS_PATH)
    parser.add_argument("--out", default="triage_results.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.tickets, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    if args.limit:
        tickets = tickets[: args.limit]

    pipeline = TriagePipeline()
    results = []
    failures = []

    for i, raw_ticket in enumerate(tickets, start=1):
        ticket_id = raw_ticket.get("ticket_id", f"ticket-{i}")
        print(f"[{i}/{len(tickets)}] Triaging {ticket_id} ... ", end="", flush=True)
        try:
            result = pipeline.run(raw_ticket)
            results.append(result.model_dump(mode="json"))
            summary = result.to_summary_dict()
            print(
                f"{summary['urgency']} | {summary['product_area']} | "
                f"{summary['issue_category']} -> {summary['recommended_team']}"
                f"{'  [NEEDS HUMAN REVIEW]' if summary['needs_human_review'] else ''}"
            )
        except Exception as e:  # noqa: BLE001 - keep the batch going on single-ticket failures
            print(f"FAILED: {e}")
            failures.append({"ticket_id": ticket_id, "error": str(e)})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "failures": failures}, f, indent=2)

    print(f"\nDone. {len(results)} succeeded, {len(failures)} failed. Written to {args.out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
