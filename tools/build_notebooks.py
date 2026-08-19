"""Build the notebooks from source, executing every cell so outputs are real.

Notebooks are checked in *with* their outputs, because the audience for this repo
reads them on GitHub rather than running them. Committing hand-written outputs
would be a lie waiting to happen, so they are generated: this script executes
each code cell in a shared namespace and captures the actual stdout.

    python tools/build_notebooks.py

Deliberately dependency-free — it writes the .ipynb JSON directly rather than
pulling in jupyter just to serialise a list of dicts.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(ROOT))

SETUP = """import sys
sys.path.insert(0, "..")
import pandas as pd
pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 20)
"""


def md(text: str) -> tuple[str, str]:
    return ("markdown", text.strip("\n"))


def code(text: str) -> tuple[str, str]:
    return ("code", text.strip("\n"))


# --------------------------------------------------------------------------- #
# 01 — profiling
# --------------------------------------------------------------------------- #

PROFILING = [
    md("""
# 1 · Profiling the definition space

A vulnerability scanner reports findings under *plugins*, and a plugin's entire
meaning lives in its title. Before writing a single rule, the question worth
answering is: **what kinds of titles are there, and does each kind behave the
same way?**

That turned out to matter more than any individual rule. Supersedence is
decidable for one kind of title and provably undecidable for another, and no
amount of rule tuning moves that.

> The data here is synthetic (`src/synthetic.py`) and generated to reproduce the
> *structure* of a real export — not its content. No client data is in this repo.
"""),
    code(SETUP + """
from src.synthetic import generate
from src.rules import frame_from_names

raw = generate()
findings = frame_from_names(raw)

print(f"findings          {len(findings):,}")
print(f"assets            {findings['asset'].nunique():,}")
print(f"distinct plugins  {findings['definition_name'].nunique():,}")
print(f"base rate of SUPERSEDED  "
      f"{findings['analyst_category'].eq('SUPERSEDED').mean():.3f}")
"""),
    md("""
## The taxonomy

Every title falls into one of a few syntactic shapes. Grouping by shape and
looking at how often the analyst calls each one *superseded* is the cheapest
possible orientation — one `groupby` that decides where to spend the next week.
"""),
    code("""
profile = (findings.groupby("shape")
    .agg(findings=("finding_id", "size"),
         plugins=("definition_name", "nunique"),
         superseded=("analyst_category", lambda s: s.eq("SUPERSEDED").sum()))
    .assign(rate=lambda d: (d["superseded"] / d["findings"]).round(3))
    .sort_values("findings", ascending=False))
print(profile.to_string())
"""),
    md("""
## The precondition nobody states

A comparison rule can only see supersedence when **two branches of the same
product are present in the same export**. That happens because the scanner keeps
reporting findings under the old plugin until every machine catches up.

If a product has only one branch in the cut, there is nothing to compare against
— and no rule over that axis can ever fire, however well written.

This is a *structural* ceiling, not a modelling one, so it is worth measuring
explicitly rather than discovering later as unexplained recall.
"""),
    code("""
branches = (findings[findings["shape"].eq("comparator")]
    .groupby("product")
    .agg(findings=("finding_id", "size"),
         branches=("major", "nunique"),
         which=("major", lambda s: sorted(s.dropna().unique().tolist())),
         superseded=("analyst_category", lambda s: s.eq("SUPERSEDED").sum()))
    .sort_values("findings", ascending=False))
print(branches.to_string())

decidable = branches["branches"] > 1
print(f"\\nproducts a comparison rule can decide: {int(decidable.sum())} of {len(branches)}")
print(f"findings inside them:                  {int(branches.loc[decidable, 'findings'].sum()):,}")
print(f"findings structurally out of reach:    {int(branches.loc[~decidable, 'findings'].sum()):,}")
"""),
    md("""
## What this buys

Two facts, both of which shaped everything downstream:

1. Only the **comparator** shape carries a usable signal. The others sit at or
   near the base rate, which is the profiling equivalent of a flat line.
2. Even inside that shape, the reachable population is bounded by how many
   products happen to have two live branches — a property of the fleet, not of
   the code.

Next: turn that into a rule, and resist the obvious version of it.
"""),
]

# --------------------------------------------------------------------------- #
# 02 — the rule
# --------------------------------------------------------------------------- #

RULE = [
    md("""
# 2 · Building the supersedence rule

The obvious rule is *"a newer version exists, so this one is superseded"*. The
obvious rule is wrong in a way that only shows up if you measure it over the
whole population instead of over the rows you are trying to catch.

This notebook builds the rule the way it actually got built: propose, measure
precision **and** recall, discard, repeat.
"""),
    code(SETUP + """
from src.synthetic import generate
from src.rules import frame_from_names, classify, is_superseded, explain
from src.agreement import score_rule, base_rate

findings = frame_from_names(generate())
print("base rate:", base_rate(findings, "SUPERSEDED"))
"""),
    md("""
## Attempt 1 — any older version at all

Compare the full dotted version and flag anything below the highest one seen for
that product. It reads correctly in English, which is exactly the problem.
"""),
    code("""
def version_key(series):
    # Zero-pad each segment so 150.0.4078.83 sorts below 150.0.4078.105
    return (series.fillna("")
            .str.split(".")
            .apply(lambda parts: ".".join(p.zfill(6) for p in parts if p)))

versions = version_key(findings["definition_name"].str.extract(
    r"<\\s*=?\\s*(\\d+(?:\\.[0-9A-Za-z]+)*)", expand=False))
newest_version = versions.groupby(findings["product"]).transform("max")
naive = findings["product"].notna() & versions.ne("") & versions.lt(newest_version)

print("full-version comparison:", score_rule(naive, findings, "SUPERSEDED"))
"""),
    md("""
## Attempt 2 — only a newer *branch* counts

`151.0.7922.71` and `151.0.7922.108` are two patches of the same branch. The
vendor did not replace the plugin, it revised it. Only a change of major version
means the old plugin has been superseded.

The difference between the two rules is a single `.str.split(".")` — and a large
difference in precision.
"""),
    code("""
print("major comparison:      ", score_rule(is_superseded(findings), findings, "SUPERSEDED"))
print()
print("rows the naive rule adds:", int((naive & ~is_superseded(findings)).sum()))
extra = findings[naive & ~is_superseded(findings)]
print(extra["analyst_category"].value_counts(dropna=False).to_string())
"""),
    md("""
Same recall, better precision. The extra rows the naive rule picks up are
same-branch patches the analyst does not consider superseded at all.

## The precedence problem

A finding on a machine that has not been scanned recently has *two* things true
about it at once: the plugin was replaced, **and** the scan is stale. Which one
wins is not a detail — it decides where the row is reported and who acts on it.

The engine models these as two levels: a **state** (is this actionable at all?)
resolves first, and only genuinely active findings get a **workflow**. The
`supersedence_first` switch keeps the losing design runnable so the decision can
be measured instead of argued.
"""),
    code("""
from src.agreement import agreement, totals

for supersedence_first in (False, True):
    result = classify(findings, supersedence_first=supersedence_first)
    report = agreement(result)
    row = report.set_index("category").loc["SUPERSEDED"]
    print(f"supersedence_first={supersedence_first!s:5}  "
          f"SUPERSEDED agreed={int(row['agreed']):4}  "
          f"missed={int(row['missed']):4}  "
          f"overall agreed={totals(report)['agreed']:5}  "
          f"similarity={totals(report)['similarity']}")
"""),
    md("""
Resolving the scan state first silently swallows findings that belong to the
supersedence state — and, because they land in a bucket that looks healthy, the
loss is invisible in any per-category total.

## Explaining a single row

A rule engine that cannot answer *"why this category?"* does not survive contact
with the analyst whose work it is reproducing. Precedence is stored as data, so
the answer is an index lookup rather than an archaeology exercise.
"""),
    code("""
result = classify(findings)
for category in ("SUPERSEDED", "NOT_SCANNED", "CERTIFICATE", "PATCH"):
    index = result.index[result["category"].eq(category)][0]
    print(f"{result.loc[index, 'definition_name'][:58]:60} -> {explain(findings, index)}")
"""),
]

# --------------------------------------------------------------------------- #
# 03 — measurement
# --------------------------------------------------------------------------- #

MEASUREMENT = [
    md("""
# 3 · Measuring agreement honestly

The engine reproduces a process a human already performs, so there is a ground
truth — their spreadsheet. Comparing against it is where most of the wrong
answers in this project got caught, and it needed three ideas that are not in the
usual precision/recall toolkit.
"""),
    code(SETUP + """
from src.synthetic import generate
from src.rules import frame_from_names, classify
from src.agreement import agreement, totals, score_rule, base_rate, inverse_control

findings = frame_from_names(generate())
result = classify(findings)
report = agreement(result)
print(report.to_string(index=False))
"""),
    md("""
## Idea 1 — disagreement has two causes, and they need opposite fixes

When the engine flags a row the analyst did not flag *in that category*, either:

- **misrouted** — the analyst filed it under a different category. The engine was
  right that it is a finding and wrong about which. That is a *precedence* bug.
- **over-detected** — the analyst did not classify it at all. That is a
  *threshold* bug.

One number blends them and points at neither. Splitting them tells you which part
of the engine to open.
"""),
    code("""
print(totals(report))
"""),
    md("""
## Idea 2 — a net variance hides its own magnitude

The first version of this report compared totals per category. Errors of opposite
sign cancel, and a badly broken engine reports a small variance.
"""),
    code("""
worst = report.assign(
    net=lambda d: d["net_variance"].abs(),
    real=lambda d: d["absolute_error"])[["category", "net", "real"]]
print(worst.to_string(index=False))
print(f"\\nsum of net variances (what a totals report shows): "
      f"{report['net_variance'].sum():+,}")
print(f"sum of magnitudes (what is actually wrong):      "
      f"{report['absolute_error'].sum():,}")
"""),
    md("""
The gap between those two numbers is the entire argument for row-level
reconciliation. A totals report is not a weaker version of this — it points the
wrong way.

## Idea 3 — score a rule against its own inverse, not against zero

This is the cheapest way to kill a plausible rule, and it retired three of them.

A rule that fires on a few dozen rows and gets ~10% right sounds weak but real.
Then you score the **exact opposite** rule on the same domain. If it also gets
10%, the axis carries no information — the rule was reading the base rate back.
"""),
    code("""
# Candidate: within a family of monthly Windows rollups, the older month is
# superseded by the newer one. Reasonable, and the same shape of argument that
# works for browsers.
line = findings["definition_name"].str.extract(r":\\s*(.*?)\\s*\\(", expand=False)
newest_period = findings["period"].groupby(line).transform("max")
domain = findings["shape"].eq("kb_titled")
older_kb = (domain & findings["period"].lt(newest_period)).fillna(False)

control = inverse_control(older_kb, domain, findings, "SUPERSEDED")
for label in ("rule", "inverse"):
    print(f"{label:8} {control[label]}")
print(f"base rate {control['base_rate']}")
"""),
    md("""
The rule and its inverse land on the same number, and that number is the base
rate. The axis is noise.

Contrast with the version axis, where the rule separates and its inverse does
not:
"""),
    code("""
from src.rules import is_superseded

comparator = findings["shape"].eq("comparator")
control = inverse_control(is_superseded(findings), comparator, findings, "SUPERSEDED")
for label in ("rule", "inverse"):
    print(f"{label:8} {control[label]}")
print(f"base rate {control['base_rate']}")
"""),
    md("""
## The ceiling

What is left after the version axis is exhausted is not a rule waiting to be
found. In the generator it is literally uniform noise, because that is what the
real residual measured like: findings the analyst marks superseded for reasons
that are not in the export at all.

Knowing that is worth as much as a rule. It is the difference between *"recall is
0.35 and we don't know why"* and *"recall is 0.35, here is the boundary, and the
rest needs a conversation rather than more code"*.
"""),
    code("""
residual = result[result["analyst_category"].eq("SUPERSEDED")
                  & ~result["category"].eq("SUPERSEDED")]
print(f"findings the analyst calls superseded and the engine does not: {len(residual):,}")
print()
print(residual.groupby("shape")["finding_id"].size().sort_values(ascending=False).to_string())
"""),
]

NOTEBOOKS_SPEC = {
    "01_profiling_the_definition_space.ipynb": PROFILING,
    "02_building_the_supersedence_rule.ipynb": RULE,
    "03_measuring_agreement_honestly.ipynb": MEASUREMENT,
}


def run(cells):
    namespace: dict = {}
    out = []
    for kind, source in cells:
        if kind == "markdown":
            out.append({"cell_type": "markdown", "metadata": {},
                        "source": source.splitlines(keepends=True)})
            continue
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102
        except Exception:  # surface failures instead of shipping a broken notebook
            raise RuntimeError(f"cell failed:\n{source}\n\n{traceback.format_exc()}")
        text = buffer.getvalue()
        outputs = ([{"output_type": "stream", "name": "stdout",
                     "text": text.splitlines(keepends=True)}] if text else [])
        out.append({"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": outputs, "source": source.splitlines(keepends=True)})
    return out


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS_SPEC.items():
        document = {
            "cells": run(cells),
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": sys.version.split()[0]},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (NOTEBOOKS / name).write_text(
            json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote notebooks/{name}  ({len(document['cells'])} cells)")


if __name__ == "__main__":
    main()
