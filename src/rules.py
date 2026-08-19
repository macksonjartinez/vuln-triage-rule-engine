"""The rule engine: turn a scan export into one operational category per finding.

Two design decisions carry most of the weight here, and both were forced by
measurement rather than taste.

**1. States and workflows are two levels, not one.**
A finding has a *state* (superseded, not scanned, not seen) and, if it is
genuinely active, a *workflow* (uninstall the app, patch it, it is in a control
group). Flattening both into one list of categories makes them compete, and the
loser is whichever rule happens to run later. Measured on real data, the scan
state was swallowing 225 findings that belonged to the superseded state — 26% of
all the misclassification in the export — purely because the scan check ran
first. Splitting the levels fixed it without touching any individual rule.

**2. Precedence is data, not control flow.**
The cascade is a list of `(code, predicate)` pairs. That makes the order
reviewable, testable and printable, and it makes "why did this row get that
category" answerable by index instead of by reading nested `if`s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .definitions import COMPARATOR, parse

# States: mutually exclusive facts about whether a finding is actionable at all.
SUPERSEDED = "SUPERSEDED"
NOT_SCANNED = "NOT_SCANNED"
ACTIVE = "ACTIVE"

# Workflows: how an active finding gets remediated.
UNINSTALL = "UNINSTALL"
CONTROL_GROUP = "CONTROL_GROUP"
APP_UPDATE = "APP_UPDATE"
ENCRYPTION = "ENCRYPTION"
CERTIFICATE = "CERTIFICATE"
PATCH = "PATCH"

SCAN_STALE_DAYS = 10


@dataclass(frozen=True)
class Rule:
    code: str
    describe: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def newest_major_per_product(frame: pd.DataFrame) -> pd.Series:
    """Newest live branch per product, computed *within the cut being processed*.

    This is the crux of the whole engine. Supersedence is not a property you can
    look up: the same plugin is current one week and superseded the next. What
    makes it decidable is that the scanner keeps reporting findings under the old
    plugin after publishing a new one, so both branches are visible side by side
    in the same export. The newest branch present *is* the current one.

    The consequence is a hard limit worth stating out loud: if a product has only
    one branch in the export, no rule can tell whether it is superseded, because
    there is nothing to compare it against.
    """
    return frame.groupby("product", dropna=True)["major"].transform("max")


def is_superseded(frame: pd.DataFrame) -> pd.Series:
    """A newer branch of the same product exists in this cut."""
    newest = newest_major_per_product(frame)
    return (
        frame["shape"].eq(COMPARATOR)
        & frame["product"].notna()
        & frame["major"].notna()
        & frame["major"].lt(newest)
    ).fillna(False)


def is_not_scanned(frame: pd.DataFrame) -> pd.Series:
    """The asset has no recent authenticated scan, so its state is unknown."""
    return frame["days_since_scan"].gt(SCAN_STALE_DAYS).fillna(False)


def _name_matches(frame: pd.DataFrame, pattern: str) -> pd.Series:
    return frame["definition_name"].str.contains(pattern, case=False, regex=True,
                                                 na=False)


# Order matters and is the reviewable part of the engine. Read top to bottom:
# the first predicate that matches wins.
WORKFLOW_CASCADE: list[Rule] = [
    Rule(UNINSTALL, "product is not approved for this fleet",
         lambda f: _name_matches(f, r"unsupported application|\bzoom\b")),
    Rule(CONTROL_GROUP, "developer tooling, patched in a pilot ring first",
         lambda f: _name_matches(f, r"node\.js|jetbrains|pgadmin|git for windows")),
    Rule(ENCRYPTION, "deprecated transport security",
         lambda f: _name_matches(f, r"\btls\b|cipher")),
    Rule(CERTIFICATE, "certificate hygiene",
         lambda f: _name_matches(f, r"certificate")),
    Rule(APP_UPDATE, "vendor application update owned by the app team",
         lambda f: _name_matches(f, r"adobe|oracle java|apache")),
    Rule(PATCH, "ordinary patch cycle",
         lambda f: pd.Series(True, index=f.index)),
]


def classify(frame: pd.DataFrame, *, supersedence_first: bool = True) -> pd.DataFrame:
    """Assign one state and, for active findings, one workflow.

    Returns a copy with `state`, `workflow` and `category` added. `category` is
    the flat label used for reporting and is what gets compared against the
    analyst's own labels in `agreement.py`.

    `supersedence_first` exists so the precedence decision can be *measured*
    rather than argued about. Set it to False to get the original behaviour, in
    which a stale scan hid the supersedence underneath it. Keeping the losing
    branch runnable is cheap and it is what turns a design opinion into a number.
    """
    out = frame.copy()

    superseded = is_superseded(out)
    not_scanned = is_not_scanned(out)
    out["state"] = ACTIVE
    if supersedence_first:
        # A finding reported under a plugin the vendor has already replaced is
        # superseded whether or not the machine was scanned recently.
        out.loc[not_scanned & ~superseded, "state"] = NOT_SCANNED
        out.loc[superseded, "state"] = SUPERSEDED
    else:
        out.loc[superseded & ~not_scanned, "state"] = SUPERSEDED
        out.loc[not_scanned, "state"] = NOT_SCANNED

    out["workflow"] = pd.NA
    active = out["state"].eq(ACTIVE)
    remaining = active.copy()
    for rule in WORKFLOW_CASCADE:
        if not remaining.any():
            break
        hits = remaining & rule.predicate(out)
        out.loc[hits, "workflow"] = rule.code
        remaining &= ~hits

    out["category"] = out["workflow"].where(active, out["state"])
    return out


def explain(frame: pd.DataFrame, index) -> str:
    """Why did this row get its category? Answerable without reading the code."""
    row = classify(frame).loc[index]
    if row["state"] != ACTIVE:
        newest = newest_major_per_product(frame).loc[index]
        detail = (f" (product '{row['product']}' is on branch {newest} in this cut, "
                  f"this finding is branch {row['major']})"
                  if row["state"] == SUPERSEDED else
                  f" (last authenticated scan {row['days_since_scan']} days ago, "
                  f"threshold {SCAN_STALE_DAYS})")
        return f"{row['category']}: state resolved before any workflow{detail}"
    position = [r.code for r in WORKFLOW_CASCADE].index(row["workflow"])
    rule = WORKFLOW_CASCADE[position]
    return (f"{row['category']}: rule {position + 1} of {len(WORKFLOW_CASCADE)} "
            f"in the cascade — {rule.describe}")


def frame_from_names(records: pd.DataFrame) -> pd.DataFrame:
    """Attach parsed definition facts to a raw export."""
    parsed = records["definition_name"].map(parse)
    return records.assign(
        shape=[p.shape for p in parsed],
        product=[p.product for p in parsed],
        major=pd.array([p.major for p in parsed], dtype="Int64"),
        kb=[p.kb for p in parsed],
        period=pd.array([p.period for p in parsed], dtype="Int64"),
    )
