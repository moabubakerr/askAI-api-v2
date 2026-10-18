"""
How the resolver decides between answering, answering with the broader series,
offering the closest names, and refusing. Scores are set by the test (see the
`scores` fixture); what is under test is the decision made from them.
"""
from app.resolvers import indicator_resolver as resolver
from app.resolvers.indicator_resolver import (
    _contradiction_kind, _embedding_text, phrase_forms, resolve_indicator,
    same_named_siblings, vocabulary_form,
)


# --- vocabulary --------------------------------------------------------------

def test_everyday_words_are_rewritten_into_catalogue_terms():
    assert vocabulary_form("non-oil spending") == "non-hydrocarbon expenditure"
    assert vocabulary_form("oil exports") == "hydrocarbon exports"
    assert vocabulary_form("government spend") == "government expenditure"


def test_phrase_forms_include_the_rewritten_form_once(catalog):
    forms = phrase_forms("What was non-oil spending in 2024-Q4?")
    assert "non-hydrocarbon expenditure" in forms
    assert len(forms) == len(set(forms))


# --- what kind of mismatch ---------------------------------------------------

def test_dropping_only_the_negation_is_a_dropped_qualifier(catalog):
    assert _contradiction_kind("non-oil spending", "Total Expenditure") == "qualifier_dropped"
    assert _contradiction_kind("non-hydrocarbon GDP", "Real GDP") == "qualifier_dropped"


def test_a_negated_series_about_another_subject_is_a_different_measure(catalog):
    # Both say "non-", but revenue is not spending.
    assert _contradiction_kind("non-oil spending", "Non-Hydrocarbon Revenue") == "different_measure"


def test_matching_negation_and_subject_is_a_fair_match(catalog):
    assert _contradiction_kind("non-hydrocarbon GDP", "Non-Hydrocarbon Real GDP") is None
    # The alias table maps this phrasing to this name; the subject test defers to it.
    assert _contradiction_kind("How much of Qatar's exports are non-oil",
                               "Non-Hydrocarbon Exports (share of total exports)") is None


def test_the_opposite_side_of_a_pair_is_still_refused(catalog):
    assert _contradiction_kind("Nominal GDP", "Real GDP") == "opposite"


def test_arabic_subject_is_not_compared_with_an_english_name(catalog):
    # No comparable words, so the subject test abstains; only the negation counts.
    assert _contradiction_kind("الإنفاق غير النفطي", "Total Expenditure") == "qualifier_dropped"


# --- duplicate names ---------------------------------------------------------

def test_series_sharing_a_name_are_embedded_with_their_parent(catalog):
    by_id = {r["indicator_detail_id"]: r for r in catalog}
    assert "State Budget Performance" in _embedding_text(by_id["te-sbp"])
    assert "Expenditure Breakdown" in _embedding_text(by_id["te-eb"])
    assert resolver._row_key(by_id["te-sbp"]) != resolver._row_key(by_id["te-eb"])


def test_same_named_siblings_are_found(catalog, scores):
    scores["plain"] = {"Total Expenditure": 0.9}
    match = resolve_indicator("total expenditure").match
    siblings = same_named_siblings(match)
    assert len(siblings) == 1
    assert siblings[0].name_en == "Total Expenditure"
    assert siblings[0].parent_name_en != match.parent_name_en


def test_two_series_with_one_name_are_not_offered_as_a_choice(catalog, scores):
    scores["plain"] = {"Total Expenditure": 0.80}
    result = resolve_indicator("government spending")
    assert result.status == "resolved"
    assert result.match.name_en == "Total Expenditure"


def test_a_tie_goes_to_the_candidate_that_does_not_narrow_the_question(catalog, scores):
    # The real scores for "government spending" (scripts/diagnose_resolver.py on
    # the VM). The picker chose Current Expenditure; nothing asked for "current".
    scores["plain"] = {"Expenditure": 0.8115, "Current Expenditure": 0.762,
                       "Total Expenditure": 0.760, "Government Revenues": 0.678}
    result = resolve_indicator("government spending")
    assert result.status == "resolved"
    assert result.match.name_en == "Expenditure"


def test_a_qualifier_the_user_named_is_not_counted_as_narrowing(catalog, scores):
    scores["plain"] = {"Current Expenditure": 0.80, "Expenditure": 0.78}
    result = resolve_indicator("current expenditure")
    assert result.match.name_en == "Current Expenditure"


# --- the non-oil spending question -------------------------------------------

def test_non_oil_spending_answers_with_total_and_names_the_gap(catalog, scores):
    # What the embeddings plausibly do: the non-hydrocarbon REVENUE series and
    # total expenditure both look close, neither is what was asked.
    scores["negated"] = {"Non-Hydrocarbon Revenue": 0.70, "Total Expenditure": 0.66,
                         "Non-Hydrocarbon Real GDP": 0.55}
    scores["plain"] = {"Total Expenditure": 0.82, "Salaries": 0.60}

    result = resolve_indicator("non-oil spending")

    assert result.status == "resolved"
    assert result.match.name_en == "Total Expenditure"
    assert result.match.dropped_qualifier == "non-oil"
    assert result.match.related == ["Non-Hydrocarbon Revenue"]


def test_a_series_named_only_by_its_split_is_listed_with_its_parent(catalog, scores):
    # The real main-pass scores from the VM.
    scores["negated"] = {"Non-Hydrocarbon Revenue": 0.796, "Non-Hydrocarbon": 0.745,
                         "Total Expenditure": 0.642}
    scores["plain"] = {"Expenditure": 0.968, "Total Expenditure": 0.830}
    result = resolve_indicator("non-oil spending")
    assert result.match.name_en == "Expenditure"
    assert result.match.related == ["Non-Hydrocarbon Revenue",
                                    "Total Exports (Goods and Services) — Non-Hydrocarbon"]


def test_contradicting_candidates_do_not_make_an_ambiguity(catalog, scores):
    scores["negated"] = {"Non-Hydrocarbon Revenue": 0.70, "Total Expenditure": 0.68}
    scores["plain"] = {}
    result = resolve_indicator("non-oil spending")
    # Nothing broader clears the bar either, so it is refused — not offered as
    # a choice between two wrong answers.
    assert result.status == "not_found"
    assert not result.candidates


def test_definition_questions_never_fall_back_to_the_broader_series(catalog, scores):
    scores["negated"] = {"Total Expenditure": 0.70}
    scores["plain"] = {"Total Expenditure": 0.85}
    result = resolve_indicator("non-oil spending", require_data=False)
    assert result.status == "not_found"


def test_a_real_non_oil_series_is_still_preferred(catalog, scores):
    scores["negated"] = {"Non-Hydrocarbon Real GDP": 0.78, "Real GDP": 0.76}
    result = resolve_indicator("non-hydrocarbon GDP")
    assert result.status == "resolved"
    assert result.match.name_en == "Non-Hydrocarbon Real GDP"
    assert result.match.dropped_qualifier is None


# --- not found ---------------------------------------------------------------

def test_not_found_offers_the_closest_published_names(catalog, scores):
    scores["plain"] = {"Number of International Visitors": 0.60}
    result = resolve_indicator("visitor arrivals")
    assert result.status == "not_found"
    assert [c.name_en for c in result.candidates] == ["Number of International Visitors"]
    assert "closest published" in result.message


def test_nonsense_still_gets_a_plain_refusal(catalog, scores):
    # 0.60 on "Number of …" alone is the penguin false positive from calibration.
    scores["plain"] = {"Number of Qatari Jobs": 0.60}
    result = resolve_indicator("number of penguins in Qatar")
    assert result.status == "not_found"
    assert not result.candidates
