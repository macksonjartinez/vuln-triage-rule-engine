"""Contract tests.

These are deliberately written as *claims about the domain* rather than as
coverage of the code. Each one encodes a decision that cost a measurement to
make, so if someone changes the behaviour the test says why it mattered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import definitions as d
from src.agreement import agreement, inverse_control, score_rule, totals
from src.rules import (ACTIVE, NOT_SCANNED, SUPERSEDED, classify, frame_from_names,
                       is_superseded)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name,shape",
    [
        ("Contoso Browser (Chromium) < 151.0.4129.59 Multiple Vulnerabilities", d.COMPARATOR),
        ("KB5101650: Windows 11 Version 24H2 Security Update (July 2026)", d.KB_TITLED),
        ("Oracle Java JRE Unsupported Version Detection", d.EOL),
        ("Security Updates for Contoso Office Products (July 2026)", d.MONTHLY),
        ("TLS Version 1.0 Protocol Detection", d.PROTOCOL),
        ("Insecure Windows Service Permissions", d.CONFIG),
        ("Compromised System Check", d.OTHER),
        ("", d.OTHER),
    ],
)
def test_shape_classification(name, shape):
    assert d.parse(name).shape == shape


def test_product_and_major_come_from_the_threshold():
    parsed = d.parse("Contoso Browser (Chromium) < 150.0.4078.83 Multiple Vulnerabilities")
    assert parsed.product == "contoso browser (chromium)"
    assert parsed.threshold == "150.0.4078.83"
    assert parsed.major == 150


def test_a_definition_without_a_version_has_no_branch():
    assert d.parse("TLS Version 1.0 Protocol Detection").has_version_branch is False


def test_only_a_leading_kb_is_the_subject_of_the_finding():
    """A KB in the body is a reference. Matching on it inflates any KB rule."""
    body = "Windows PrintNightmare Exposure, superseded by KB5101650, still present"
    assert d.parse(body).kb is None
    assert d.any_kb(body) == "KB5101650"
    assert d.parse("KB5101650: Windows 11 Security Update (July 2026)").kb == "KB5101650"


def test_a_comparator_inside_the_product_name_is_a_known_wart():
    """`Apache Log4j SEoL (<= 1.x)` parses to the product `apache log4j seol (`.

    Documented rather than fixed: the product string is only ever used as a
    grouping key, so a scruffy-but-stable key costs nothing, and every attempt to
    "clean" it so far has broken a product that legitimately contains a
    parenthesis, such as `Contoso Browser (Chromium)`. If it ever needs fixing,
    fix it here with this test as the specification.
    """
    parsed = d.parse("Apache Log4j SEoL (<= 1.x)")
    assert parsed.product == "apache log4j seol ("
    assert parsed.major == 1


def test_parsing_never_raises_on_junk():
    for junk in (None, "", "   ", "<<<", "< 1", "()"):
        assert d.parse(junk).shape in {
            d.COMPARATOR, d.KB_TITLED, d.EOL, d.MONTHLY, d.PROTOCOL, d.CONFIG, d.OTHER
        }


# --------------------------------------------------------------------------- #
# The supersedence rule
# --------------------------------------------------------------------------- #

def _cut(names, days_since_scan=0):
    frame = pd.DataFrame({
        "finding_id": [f"F{i}" for i in range(len(names))],
        "definition_name": names,
        "asset": [f"a{i}" for i in range(len(names))],
        "days_since_scan": days_since_scan,
    })
    return frame_from_names(frame)


def test_a_stale_branch_is_superseded_and_the_current_one_is_not():
    frame = _cut([
        "Contoso Browser (Chromium) < 151.0.4129.59 Multiple Vulnerabilities",
        "Contoso Browser (Chromium) < 150.0.4078.83 Multiple Vulnerabilities",
    ])
    assert is_superseded(frame).tolist() == [False, True]


def test_a_patch_within_the_same_branch_does_not_supersede():
    """The decision that took a measurement to get right.

    `151.0.7922.71` and `151.0.7922.108` are the same branch. Comparing full
    versions marks the first one superseded, which the analyst does not; on real
    data that single choice moved rule precision from 1.00 to 0.79.
    """
    frame = _cut([
        "Contoso Browser (Chromium) < 151.0.7922.108 Multiple Vulnerabilities",
        "Contoso Browser (Chromium) < 151.0.7922.71 Multiple Vulnerabilities",
        "Contoso Browser (Chromium) < 150.0.7871.128 Multiple Vulnerabilities",
    ])
    assert is_superseded(frame).tolist() == [False, False, True]


def test_supersedence_does_not_leak_across_products():
    frame = _cut([
        "Contoso Browser (Chromium) < 151.0.4129.59 Multiple Vulnerabilities",
        "Northwind Reader < 11.0.23 Multiple Vulnerabilities",
    ])
    assert is_superseded(frame).tolist() == [False, False]


def test_a_product_with_one_branch_is_undecidable_not_superseded():
    """The structural ceiling, stated as a test so nobody 'fixes' it later."""
    frame = _cut(["Fabrikam Node.js 22.x < 22.23.0 Multiple Vulnerabilities"])
    assert is_superseded(frame).tolist() == [False]


# --------------------------------------------------------------------------- #
# Two-level precedence
# --------------------------------------------------------------------------- #

def test_superseded_state_wins_over_a_stale_scan():
    """The fix worth 225 findings: a replaced plugin is superseded regardless of
    when the asset was last scanned. Resolving it the other way round buried them
    in the scan bucket, where they were counted as healthy."""
    frame = _cut([
        "Contoso Browser (Chromium) < 151.0.4129.59 Multiple Vulnerabilities",
        "Contoso Browser (Chromium) < 150.0.4078.83 Multiple Vulnerabilities",
    ], days_since_scan=90)
    result = classify(frame)
    assert result["state"].tolist() == [NOT_SCANNED, SUPERSEDED]


def test_a_finding_in_a_state_never_gets_a_workflow():
    frame = _cut(["TLS Version 1.0 Protocol Detection"], days_since_scan=90)
    result = classify(frame)
    assert result.loc[0, "state"] == NOT_SCANNED
    assert pd.isna(result.loc[0, "workflow"])


def test_the_cascade_assigns_exactly_one_workflow():
    frame = _cut(["SSL Certificate Cannot Be Trusted"])
    result = classify(frame)
    assert result.loc[0, "state"] == ACTIVE
    assert result.loc[0, "category"] == result.loc[0, "workflow"]


def test_every_row_ends_up_with_a_category():
    from src.synthetic import generate
    result = classify(frame_from_names(generate()))
    assert result["category"].notna().all()


# --------------------------------------------------------------------------- #
# The measurement harness
# --------------------------------------------------------------------------- #

def _scored(engine, analyst):
    return pd.DataFrame({"category": engine, "analyst_category": analyst})


def test_misrouted_and_over_detected_are_different_failures():
    """Same headline error, opposite fixes: one is precedence, one is threshold."""
    frame = _scored(
        engine=["SUPERSEDED", "SUPERSEDED", "SUPERSEDED"],
        analyst=["SUPERSEDED", "PATCH", None],
    )
    row = agreement(frame).set_index("category").loc["SUPERSEDED"]
    assert (row["agreed"], row["misrouted"], row["over_detected"]) == (1, 1, 1)


def test_the_net_variance_hides_errors_of_opposite_sign():
    """Two categories, +2 and -2, net zero, four rows misclassified."""
    frame = _scored(
        engine=["A", "A", "A", "A", "B", "B"],
        analyst=["A", "A", "B", "B", "B", "B"],
    )
    report = agreement(frame)
    assert totals(report)["net_variance"] == 0
    assert totals(report)["absolute_error"] == 4


def test_similarity_cannot_be_gamed_by_cancelling_errors():
    frame = _scored(engine=["A", "A", "B"], analyst=["A", "B", "B"])
    assert totals(agreement(frame))["similarity"] == pytest.approx(0.5)


def test_score_rule_reports_precision_over_the_whole_population():
    """Scoring only against labelled target rows measures recall and calls it
    precision — the mistake that made a 16%-precision rule look like a 50% win."""
    frame = pd.DataFrame({"analyst_category": ["SUPERSEDED", "PATCH", "PATCH", None]})
    mask = pd.Series([True, True, False, True])
    score = score_rule(mask, frame, "SUPERSEDED")
    assert score == {"fired": 3, "hits": 1, "other_category": 1, "unlabelled": 1,
                     "precision": pytest.approx(1 / 3, abs=1e-4), "recall": 1.0}


def test_a_rule_matching_its_own_inverse_carries_no_information():
    """The control that killed three plausible rules on the real data."""
    analyst = ["SUPERSEDED"] + ["PATCH"] * 9 + ["SUPERSEDED"] + ["PATCH"] * 9
    frame = pd.DataFrame({"analyst_category": analyst})
    domain = pd.Series([True] * 20)
    mask = pd.Series([True] * 10 + [False] * 10)
    control = inverse_control(mask, domain, frame, "SUPERSEDED")
    assert control["rule"]["precision"] == control["inverse"]["precision"]
    assert control["rule"]["precision"] == pytest.approx(control["base_rate"])
