import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.kb_index import KBIndex  # noqa: E402
from app.config import DEFAULT_KB_ROOT  # noqa: E402


def test_kb_index_builds_chunks():
    kb = KBIndex(DEFAULT_KB_ROOT)
    assert len(kb.chunks) > 0
    # every chunk should know which file it came from
    assert all(c.doc_path.endswith(".md") for c in kb.chunks)


def test_kb_search_finds_error_code():
    kb = KBIndex(DEFAULT_KB_ROOT)
    results = kb.search("ERR_CONNECTION_TIMEOUT connectors timeout")
    assert len(results) > 0
    assert results[0]["doc_path"] == "products/databridge-pro.md"
    assert "ERR_CONNECTION_TIMEOUT" in results[0]["snippet"]


def test_kb_search_finds_billing_doc():
    kb = KBIndex(DEFAULT_KB_ROOT)
    results = kb.search("seat billing prorated invoice")
    assert len(results) > 0
    assert results[0]["doc_path"] == "billing/enterprise-plan.md"


def test_kb_search_empty_query_returns_nothing():
    kb = KBIndex(DEFAULT_KB_ROOT)
    assert kb.search("") == []


def test_known_product_names():
    kb = KBIndex(DEFAULT_KB_ROOT)
    names = kb.known_product_names()
    assert "Databridge Pro" in names
