"""
Shared fixtures. Unit tests here need no database, embedding server or model:
the catalogue is a small hand-built one with the real names that matter, and
scoring is replaced where a test needs to control it.
"""
import functools

import pytest

from app.resolvers import indicator_resolver as resolver


def _row(detail_id, name, parent=None, published=True, points=10, name_ar=None):
    return {
        # Series of one indicator share its id, as indicator_details rows do.
        "indicator_detail_id": detail_id, "indicator_id": f"ind-{parent or name}",
        "name_en": name, "name_ar": name_ar,
        "parent_name_en": parent or name, "parent_name_ar": None,
        "is_published": published, "published_detail_id": f"pub-{detail_id}" if published else None,
        "is_active": True, "unit_en": "bn QAR", "unit_ar": None, "polarity_en": None,
        "data_source_en": "MoF", "format": "bn0.0",
        "definition_en": "-", "definition_ar": "-", "data_point_count": points,
    }


CATALOG = [
    _row("gdp", "Real GDP"),
    _row("nh-gdp", "Non-Hydrocarbon Real GDP"),
    _row("exp", "Expenditure", "Revenue vs Expenditure"),
    _row("cur-exp", "Current Expenditure", "Expenditure Breakdown"),
    _row("te-sbp", "Total Expenditure", "State Budget Performance"),
    _row("te-eb", "Total Expenditure", "Expenditure Breakdown"),
    _row("sal", "Salaries", "State Budget Performance"),
    _row("nh-rev", "Non-Hydrocarbon Revenue", "State Budget Performance"),
    _row("h-rev", "Hydrocarbon Revenue", "State Budget Performance"),
    _row("t-rev", "Total Revenue", "State Budget Performance"),
    _row("infl", "Inflation"),
    _row("nh-exp", "Non-Hydrocarbon Exports (share of total exports)"),
    # Named by the split alone, as the catalogue does under some indicators.
    _row("nh-exp-bare", "Non-Hydrocarbon", "Total Exports (Goods and Services)"),
    _row("t-exp", "Total Exports (Goods and Services)"),
    _row("visitors", "Number of International Visitors"),
    _row("q-jobs", "Number of Qatari Jobs"),
    _row("nq-jobs", "Number of Non-Qatari Jobs"),
    _row("debt", "Public Debt as a Percentage of GDP (%)"),
    _row("trade", "Trade Balance (Goods & Services)"),
    _row("teachers", "Share of Qatari Teachers"),
    _row("energy", "Energy Intensity"),
]


@pytest.fixture
def catalog(monkeypatch):
    """Installs CATALOG as the resolver's catalogue, with fresh derived caches."""
    # Cached like the real one, so clear_catalog_cache() can still clear it.
    fake = functools.lru_cache(maxsize=1)(lambda: tuple(CATALOG))
    resolver.clear_catalog_cache()
    monkeypatch.setattr(resolver, "_fetch_catalog_cached", fake)
    yield CATALOG
    resolver.clear_catalog_cache()


@pytest.fixture
def scores(monkeypatch, catalog):
    """Controls scoring: a test sets what each catalogue NAME scores, separately
    for phrase forms that still carry a negation ("non-oil …") and forms that
    do not (the broader re-search). Embeddings are never computed."""
    table = {"negated": {}, "plain": {}}

    def fake_score(rows, forms, _embeddings):
        negated = any(resolver._NEGATED.search(f.lower()) for f in forms)
        chosen = table["negated" if negated else "plain"]
        scored = [(r, chosen.get(r["name_en"], 0.1)) for r in rows]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    monkeypatch.setattr(resolver, "_score_catalog", fake_score)
    monkeypatch.setattr(resolver, "get_embedding", lambda text: [0.0])
    # The model picker is an LLM call; it declines, as it does on a bad field.
    monkeypatch.setattr(resolver, "pick_indicator", lambda *a, **k: None)
    return table
