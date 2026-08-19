"""A synthetic scan export that reproduces the *generative process*, not the data.

No client data ships in this repository. What ships is a generator whose
structure mirrors the real one closely enough that the analytical conclusions
reproduce — which is the only property that matters for reasoning about rules.

Four structural facts are encoded on purpose, because they are what makes
supersedence detectable in one place and undetectable in another:

1. **Browsers leave leftovers.** The vendor ships a new branch every few weeks
   and the scanner keeps reporting findings under the old plugin until every
   machine catches up, so two or three branches are visible in the same export.
   The stale branches are exactly what the analyst marks superseded.
2. **Cumulative updates do not.** If a machine is missing July's rollup it is
   also missing June's, but the scanner reports only the newest missing one. One
   finding per machine per line means there is no "old" row to mark — so no rule
   over that axis can work, however reasonable it sounds.
3. **Some products only ever have one branch present**, which puts them outside
   what any comparison rule can see.
4. **A residual is human judgement.** A fraction of findings are marked
   superseded for reasons that are not in the export at all. Encoded as uniform
   noise, so any rule measured against it lands on the base rate — which is what
   happened with three candidate rules on the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fictional vendors; any resemblance to a real fleet is the point, not the data.
BROWSER = "Contoso Browser (Chromium)"
READER = "Northwind Reader"
RUNTIME = "Fabrikam Node.js 22.x"

#: Product branches present in the cut, newest first. Only products with more
#: than one branch are decidable by a comparison rule.
#: `151.0.4129.12` is the case that separates a good rule from a plausible one:
#: an older *patch* of the branch that is still current. The vendor revised the
#: plugin, it did not replace it, so the analyst does not call it superseded —
#: but a rule comparing full version strings does.
BROWSER_BRANCHES = [("151.0.4129.59", 1800), ("151.0.4129.12", 90),
                    ("150.0.4078.83", 120), ("149.0.4022.67", 30)]
READER_BRANCHES = [("26.001.21662", 40), ("11.0.23", 8)]
RUNTIME_BRANCHES = [("22.23.0", 60)]

KB_LINE = "Windows 11 Version 24H2 Security Update"
KB_ROLLUPS = [("KB5101650", "July 2026", 700), ("KB5094126", "June 2026", 45),
              ("KB5089549", "May 2026", 5)]

MONTHLY = [("Security Updates for Contoso Office Products", "July 2026", 120),
           ("Security Updates for Contoso Office Products", "June 2026", 12)]

FLAT_DEFINITIONS = [
    ("TLS Version 1.0 Protocol Detection", 260),
    ("SSL Certificate Cannot Be Trusted", 180),
    ("Oracle Java JRE Unsupported Version Detection", 70),
    ("Apache Log4j SEoL (<= 1.x)", 40),
    ("Insecure Windows Service Permissions", 35),
    ("Contoso Media Player Unsupported Application Detection", 25),
]

#: Share of findings the analyst marks superseded for reasons not in the export.
HUMAN_SUPERSEDED_RATE = 0.09
#: Share of assets that have drifted past the scan threshold.
STALE_ASSET_RATE = 0.18
#: Share of findings the analyst never got to, which is what makes an engine hit
#: "over-detected" rather than "misrouted".
UNREVIEWED_RATE = 0.14
#: Share of stale-scan assets where the analyst overrode the scan state and used
#: a workflow anyway. Real analysts are not consistent, and a harness that
#: assumes they are will report a precision it has not earned.
SCAN_OVERRIDE_RATE = 0.12


def _rows(rng, definitions, n_assets):
    records = []
    for name, count in definitions:
        for _ in range(count):
            records.append({"definition_name": name,
                            "asset": f"wks{rng.integers(0, n_assets):05d}"})
    return records


def generate(seed: int = 20260819, n_assets: int = 900) -> pd.DataFrame:
    """Build one week's export with the analyst's own labels attached."""
    rng = np.random.default_rng(seed)

    definitions: list[tuple[str, int]] = []
    for product, branches in ((BROWSER, BROWSER_BRANCHES), (READER, READER_BRANCHES),
                              (RUNTIME, RUNTIME_BRANCHES)):
        definitions += [(f"{product} < {version} Multiple Vulnerabilities", count)
                        for version, count in branches]
    definitions += [(f"{kb}: {KB_LINE} ({period})", count)
                    for kb, period, count in KB_ROLLUPS]
    definitions += [(f"{family} ({period})", count) for family, period, count in MONTHLY]
    definitions += FLAT_DEFINITIONS

    frame = pd.DataFrame(_rows(rng, definitions, n_assets))
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame.insert(0, "finding_id", [f"F{i:06d}" for i in range(len(frame))])

    # Scan freshness is an asset property, so it has to be consistent per asset.
    # Most of a managed fleet is scanned within the week and a minority drifts,
    # so the distribution is deliberately not uniform: a uniform one would put
    # half the export behind the scan gate and hide every workflow rule behind it.
    assets = sorted(frame["asset"].unique())
    fresh = rng.integers(0, 9, size=len(assets))
    drifted = rng.integers(11, 45, size=len(assets))
    days = np.where(rng.random(len(assets)) < STALE_ASSET_RATE, drifted, fresh)
    frame["days_since_scan"] = frame["asset"].map(dict(zip(assets, days.tolist())))

    return _label(frame, rng)


def _label(frame: pd.DataFrame, rng) -> pd.DataFrame:
    """Attach the analyst's category — the ground truth the engine is scored on."""
    from .definitions import parse
    from .rules import (ACTIVE, APP_UPDATE, CERTIFICATE, CONTROL_GROUP, ENCRYPTION,
                        NOT_SCANNED, PATCH, SCAN_STALE_DAYS, SUPERSEDED, UNINSTALL)

    parsed = frame["definition_name"].map(parse)
    product = pd.Series([p.product for p in parsed], index=frame.index)
    major = pd.Series([p.major for p in parsed], index=frame.index, dtype="Float64")
    newest = major.groupby(product).transform("max")

    labels = pd.Series(pd.NA, index=frame.index, dtype="object")

    # 1. A stale branch of a product that has a newer one is superseded, always.
    stale_branch = product.notna() & major.notna() & major.lt(newest)
    labels[stale_branch] = SUPERSEDED

    # 2. Human judgement: superseded for reasons the export does not carry.
    #    Uniform on purpose — this is what makes every non-version axis measure
    #    at the base rate, exactly as observed on the real export.
    residual = (~stale_branch) & (rng.random(len(frame)) < HUMAN_SUPERSEDED_RATE)
    labels[residual] = SUPERSEDED

    # 3. Stale scan, with the analyst sometimes overriding it.
    stale_scan = (frame["days_since_scan"] > SCAN_STALE_DAYS) & labels.isna()
    override = stale_scan & (rng.random(len(frame)) < SCAN_OVERRIDE_RATE)
    labels[stale_scan & ~override] = NOT_SCANNED

    # 4. Everything else gets a workflow from the definition name.
    name = frame["definition_name"].str.lower()
    remaining = labels.isna()
    for pattern, code in (
        (r"unsupported application|zoom", UNINSTALL),
        (r"node\.js|jetbrains|pgadmin", CONTROL_GROUP),
        (r"\btls\b|cipher", ENCRYPTION),
        (r"certificate", CERTIFICATE),
        (r"adobe|oracle java|apache|northwind reader", APP_UPDATE),
    ):
        hit = remaining & name.str.contains(pattern, regex=True, na=False)
        labels[hit] = code
        remaining &= ~hit
    labels[remaining] = PATCH

    # 5. Not every finding gets reviewed.
    unreviewed = rng.random(len(frame)) < UNREVIEWED_RATE
    labels[unreviewed] = pd.NA

    frame["analyst_category"] = labels
    assert frame["finding_id"].is_unique, "finding_id must be a key"
    return frame
