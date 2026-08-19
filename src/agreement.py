"""The measurement harness — the part that keeps the rule engine honest.

A rule engine that reproduces a manual process has an unusual property: there is
a ground truth, but it is another human's spreadsheet, and it arrives days after
the data does. That makes the *measurement* at least as much engineering as the
rules, and it is where this project got most of its wrong answers caught.

Three ideas here, all of them learned the expensive way.

**Disagreement has two causes that need opposite fixes.**
When the engine flags a row the analyst did not flag in that category, either
the analyst put it in a *different* category — a precedence problem — or they
did not consider it a finding at all — a threshold problem. Reporting a single
"accuracy" number blends them and points at neither.

**A net variance hides its own magnitude.**
The first version of this report compared totals: calculated vs reported, per
category. It read -394 across the export and looked like a rounding issue. Row
by row it was 1,642 misclassified findings — sub-detection and over-detection in
different categories cancelling each other out. Always sum magnitudes.

**Measure a rule against its own inverse, not against zero.**
A rule that fires on 62 rows and gets 6 right sounds weak but plausible. Then you
measure the *exact opposite* rule and it also gets 9%, because 9% is simply the
base rate — the axis carries no information at all. Two candidate rules died this
way after looking reasonable on a recall-only chart.
"""

from __future__ import annotations

import pandas as pd

AGREED = "agreed"
MISROUTED = "misrouted"
OVER_DETECTED = "over_detected"
MISSED = "missed"


def agreement(frame: pd.DataFrame, *, predicted: str = "category",
              actual: str = "analyst_category") -> pd.DataFrame:
    """Per-category agreement between the engine and the analyst.

    `actual` may be null, meaning the analyst did not classify that row at all.
    That distinction is what separates a misrouted row from an over-detected one.

    Returns one row per category with:
        engine_count   rows the engine put in this category
        analyst_count  rows the analyst put in this category
        agreed         both agree
        misrouted      engine says this category, analyst says a different one
        over_detected  engine says this category, analyst classified nothing
        missed         analyst says this category, engine does not
        precision      agreed / engine_count
        recall         agreed / analyst_count
        absolute_error misrouted + over_detected + missed
    """
    engine = frame[predicted]
    analyst = frame[actual]
    claimed = analyst.notna()

    rows = []
    for category in sorted(set(engine.dropna()) | set(analyst.dropna())):
        picked = engine.eq(category)
        wanted = analyst.eq(category)
        agreed = int((picked & wanted).sum())
        engine_count = int(picked.sum())
        analyst_count = int(wanted.sum())
        rows.append(
            {
                "category": category,
                "engine_count": engine_count,
                "analyst_count": analyst_count,
                AGREED: agreed,
                MISROUTED: int((picked & ~wanted & claimed).sum()),
                OVER_DETECTED: int((picked & ~claimed).sum()),
                MISSED: analyst_count - agreed,
            }
        )

    out = pd.DataFrame(rows)
    out["precision"] = (out[AGREED] / out["engine_count"]).where(out["engine_count"] > 0)
    out["recall"] = (out[AGREED] / out["analyst_count"]).where(out["analyst_count"] > 0)
    out["net_variance"] = out["engine_count"] - out["analyst_count"]
    out["absolute_error"] = out[MISROUTED] + out[OVER_DETECTED] + out[MISSED]
    return out.sort_values("absolute_error", ascending=False).reset_index(drop=True)


def totals(report: pd.DataFrame) -> dict:
    """Headline numbers, including the similarity the net variance hides."""
    engine = int(report["engine_count"].sum())
    analyst = int(report["analyst_count"].sum())
    agreed = int(report[AGREED].sum())
    union = engine + analyst - agreed
    return {
        "engine_count": engine,
        "analyst_count": analyst,
        "agreed": agreed,
        "misrouted": int(report[MISROUTED].sum()),
        "over_detected": int(report[OVER_DETECTED].sum()),
        "missed": int(report[MISSED].sum()),
        "precision": round(agreed / engine, 4) if engine else None,
        "recall": round(agreed / analyst, 4) if analyst else None,
        # Jaccard. Unlike the net variance it cannot be gamed by two errors of
        # opposite sign, which is exactly why it is the number worth tracking.
        "similarity": round(agreed / union, 4) if union else None,
        "net_variance": engine - analyst,
        "absolute_error": int(report["absolute_error"].sum()),
    }


def score_rule(mask: pd.Series, frame: pd.DataFrame, target: str, *,
               actual: str = "analyst_category") -> dict:
    """Score one candidate rule over the *whole* eligible population.

    The four counts are not optional. Scoring a rule only against the rows the
    analyst already labelled measures recall and calls it accuracy: a rule can
    reach 44% of the target category while firing on a third of the export.
    """
    analyst = frame[actual]
    fired = int(mask.sum())
    hits = int((mask & analyst.eq(target)).sum())
    other = int((mask & analyst.notna() & ~analyst.eq(target)).sum())
    unlabelled = int((mask & analyst.isna()).sum())
    target_total = int(analyst.eq(target).sum())
    return {
        "fired": fired,
        "hits": hits,
        "other_category": other,
        "unlabelled": unlabelled,
        "precision": round(hits / fired, 4) if fired else 0.0,
        "recall": round(hits / target_total, 4) if target_total else 0.0,
    }


def base_rate(frame: pd.DataFrame, target: str, *,
              actual: str = "analyst_category") -> float:
    """How often the target occurs. Any rule must beat this to mean anything."""
    return round(frame[actual].eq(target).mean(), 4)


def inverse_control(mask: pd.Series, domain: pd.Series, frame: pd.DataFrame,
                    target: str, *, actual: str = "analyst_category") -> dict:
    """Score the rule's exact complement inside the same domain.

    If a rule and its inverse score the same, the axis carries no information and
    the rule is reading the base rate back to you. This is the cheapest way to
    kill a plausible-looking rule, and it should be run before writing any code
    that depends on one.
    """
    return {
        "rule": score_rule(mask & domain, frame, target, actual=actual),
        "inverse": score_rule(~mask & domain, frame, target, actual=actual),
        "base_rate": base_rate(frame, target, actual=actual),
    }
