"""Parsing of vulnerability *definition names* into structured facts.

A vulnerability scanner reports findings under a plugin whose entire semantics
live in a free-text title:

    Microsoft Edge (Chromium) < 151.0.4129.59 Multiple Vulnerabilities
    KB5101650: Windows 11 Version 24H2 Security Update (July 2026)
    Oracle Java JRE Unsupported Version Detection
    TLS Version 1.0 Protocol Detection

Every downstream rule depends on reading those titles correctly, so the parsing
lives in one place, is pure, and is the most heavily tested module in the repo.

The `shape` classification is not cosmetic: it turned out to be the single most
useful lens on the problem. Each shape has a different *generative process*, and
whether supersedence is detectable at all depends on which shape you are looking
at. See `notebooks/01_profiling_the_definition_space.ipynb`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# `<product> < <version>` — the comparator shape. The product is everything
# before the comparator; the threshold is the first version after it.
THRESHOLD_PATTERN = re.compile(r"<\s*=?\s*(\d+(?:\.[0-9A-Za-z]+)*)", re.I)
PRODUCT_PATTERN = re.compile(r"^(.*?)\s*<\s*=?\s*\d", re.I)

# `KB5101650: ...` — a titled Windows update. Only a *leading* KB counts as the
# subject of the finding; a KB mentioned mid-sentence is a reference, not the
# subject, and treating those alike inflates any KB-based rule.
LEADING_KB_PATTERN = re.compile(r"^\s*(KB\d+)\s*:", re.I)
ANY_KB_PATTERN = re.compile(r"\b(KB\d+)\b", re.I)

# `(July 2026)` — the reporting period of a monthly rollup.
PERIOD_PATTERN = re.compile(r"\(([A-Za-z]+)\s+(\d{4})\)\s*$")

EOL_PATTERN = re.compile(r"seol|unsupported|end[ -]of[ -]life", re.I)
PROTOCOL_PATTERN = re.compile(r"\btls\b|\bssl\b|cipher|certificate", re.I)
CONFIG_PATTERN = re.compile(
    r"policy|macro|registry|activex|protected view|permissions", re.I
)

MONTHS = {
    name: index
    for index, name in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}

COMPARATOR = "comparator"
KB_TITLED = "kb_titled"
EOL = "eol"
MONTHLY = "monthly"
PROTOCOL = "protocol"
CONFIG = "config"
OTHER = "other"


@dataclass(frozen=True)
class Definition:
    """Everything a rule is allowed to know about a plugin title."""

    name: str
    shape: str
    product: str | None = None
    threshold: str | None = None
    major: int | None = None
    kb: str | None = None
    period: int | None = None  # YYYYMM, comparable as an int

    @property
    def has_version_branch(self) -> bool:
        """True when this definition names a version branch of a product."""
        return self.product is not None and self.major is not None


def _product(name: str) -> str | None:
    match = PRODUCT_PATTERN.match(name)
    if not match:
        return None
    product = match.group(1).strip().lower()
    return product or None


def _threshold(name: str) -> str | None:
    match = THRESHOLD_PATTERN.search(name)
    return match.group(1) if match else None


def _major(threshold: str | None) -> int | None:
    """The major is the leading integer of the threshold.

    Comparing majors rather than full versions is a deliberate choice, and the
    one place where this parser encodes a business decision. `151.0.7922.71` and
    `151.0.7922.108` are two patches of the *same* branch; `150.x` and `151.x`
    are different branches. Only a branch change supersedes a plugin. Measured
    on real data, comparing full versions dropped rule precision from 1.00 to
    0.79 by flagging same-branch patches.
    """
    if not threshold:
        return None
    match = re.match(r"^(\d+)", threshold)
    return int(match.group(1)) if match else None


def _period(name: str) -> int | None:
    match = PERIOD_PATTERN.search(name)
    if not match:
        return None
    month = MONTHS.get(match.group(1).lower())
    return int(match.group(2)) * 100 + month if month else None


def _shape(name: str, *, threshold: str | None, leading_kb: str | None,
           period: int | None) -> str:
    if threshold is not None:
        return COMPARATOR
    if leading_kb is not None:
        return KB_TITLED
    if EOL_PATTERN.search(name):
        return EOL
    if period is not None:
        return MONTHLY
    if PROTOCOL_PATTERN.search(name):
        return PROTOCOL
    if CONFIG_PATTERN.search(name):
        return CONFIG
    return OTHER


def parse(name: str | None) -> Definition:
    """Parse a definition name. Never raises: unparseable input is `other`."""
    name = (name or "").strip()
    threshold = _threshold(name)
    leading = LEADING_KB_PATTERN.match(name)
    leading_kb = leading.group(1).upper() if leading else None
    period = _period(name)
    return Definition(
        name=name,
        shape=_shape(name, threshold=threshold, leading_kb=leading_kb, period=period),
        product=_product(name),
        threshold=threshold,
        major=_major(threshold),
        kb=leading_kb,
        period=period,
    )


def any_kb(name: str | None) -> str | None:
    """Any KB mentioned anywhere in the title, subject or not.

    Kept separate from `Definition.kb` on purpose: a rule that matches on *any*
    mention fires on references and is measurably worse than one that matches on
    the subject. Having both makes the difference testable instead of arguable.
    """
    match = ANY_KB_PATTERN.search(name or "")
    return match.group(1).upper() if match else None
