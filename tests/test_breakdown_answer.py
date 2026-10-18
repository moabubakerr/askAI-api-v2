"""
What happens after the resolver: choosing between same-named series by the
period asked about, and the template fallback stating a missing breakdown.
"""
from datetime import date
from types import SimpleNamespace

from app.compute.verifier import render_template_fallback
from app.core import graph_v2
from app.resolvers.indicator_resolver import IndicatorMatch


def _match(detail_id, parent, **extra):
    return IndicatorMatch(detail_id, f"ind-{detail_id}", "Total Expenditure", True,
                          f"pub-{detail_id}", True, "bn QAR", None, None, "MoF", ".0",
                          None, None, 0.8, None, parent_name_en=parent, **extra)


YEARLY = _match("te-eb", "Expenditure Breakdown", dropped_qualifier="non-oil",
                related=["Non-Hydrocarbon Revenue"])
QUARTERLY = _match("te-sbp", "State Budget Performance")
Q4_2024 = SimpleNamespace(raw="2024-Q4", start_date=date(2024, 10, 1),
                          end_date=date(2024, 12, 31))


def _install(monkeypatch, granularities, rows):
    monkeypatch.setattr(graph_v2, "same_named_siblings", lambda m: [QUARTERLY])
    monkeypatch.setattr(graph_v2.retriever, "get_available_granularities",
                        lambda pub, detail: granularities[detail])
    monkeypatch.setattr(graph_v2.retriever, "get_series",
                        lambda detail, pub, gran, start_date=None, end_date=None:
                        rows.get((detail, gran), []))


def test_the_series_with_the_asked_quarter_is_chosen(monkeypatch):
    _install(monkeypatch,
             {"te-eb": {"yearly"}, "te-sbp": {"quarterly", "yearly"}},
             {("te-sbp", "quarterly"): [{"actual": 47.8}]})
    chosen = graph_v2._prefer_sibling_with_data(YEARLY, {}, Q4_2024, "spending in 2024-Q4")
    assert chosen.indicator_detail_id == "te-sbp"
    # What the resolver found out about the question moves with it.
    assert chosen.dropped_qualifier == "non-oil"
    assert chosen.related == ["Non-Hydrocarbon Revenue"]


def test_the_resolved_series_is_kept_when_it_can_answer(monkeypatch):
    _install(monkeypatch,
             {"te-eb": {"yearly"}, "te-sbp": {"quarterly", "yearly"}},
             {("te-eb", "yearly"): [{"actual": 200.9}]})
    year_2024 = SimpleNamespace(raw="2024", start_date=date(2024, 1, 1),
                                end_date=date(2024, 12, 31))
    chosen = graph_v2._prefer_sibling_with_data(YEARLY, {}, year_2024, "spending in 2024")
    assert chosen.indicator_detail_id == "te-eb"


def test_fallback_says_the_breakdown_is_missing_before_the_figure():
    text = render_template_fallback({"ok": True, "facts": {
        "indicator": "Total Expenditure", "actual": 47.8, "period": "2024-Q4",
        "unit": "bn QAR",
        "breakdown_not_published": {"asked_for": "non-oil Total Expenditure",
                                    "reported_instead": "Total Expenditure",
                                    "related_published": []},
    }})
    assert text.startswith("SCEAI doesn't publish \"non-oil Total Expenditure\"")
    assert "47.8" in text
