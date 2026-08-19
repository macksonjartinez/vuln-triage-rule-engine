# vuln-triage-rule-engine

A rule engine that reproduces a manual vulnerability-triage process, and the
measurement harness that keeps it honest.

Every week a security analyst takes a scan export of tens of thousands of
findings and sorts each one into an operational bucket: this is superseded by a
newer plugin, this machine has not been scanned, this app has to be uninstalled,
this one is an ordinary patch. It is a day of work, it is done in a spreadsheet,
and it is the kind of judgement that looks trivial to automate right up until you
measure yourself against it.

This repo is the distilled version of that problem: the parsing, the rule engine,
and — the part that mattered most — the harness that decides whether a proposed
rule is real or is reading the base rate back to you.

> **The data is synthetic.** `src/synthetic.py` generates an export whose
> *structure* reproduces the real one: the shapes of plugin titles, which
> products have two live version branches, and a residual of human judgement that
> nothing in the data explains. No client data, names or identifiers are in this
> repository.

## Start here

| | |
|---|---|
| **[1 · Profiling the definition space](notebooks/01_profiling_the_definition_space.ipynb)** | What kinds of findings exist, and why supersedence is decidable for one kind and provably not for another |
| **[2 · Building the supersedence rule](notebooks/02_building_the_supersedence_rule.ipynb)** | The obvious rule, why it fails, and what fixes it |
| **[3 · Measuring agreement honestly](notebooks/03_measuring_agreement_honestly.ipynb)** | Misrouted vs over-detected, why net variance lies, and the control that kills plausible rules |

Notebooks are checked in with real outputs — `tools/build_notebooks.py` executes
every cell and writes the results, so nothing in them is hand-typed.

## The three ideas worth the read

**Supersedence is a property of the cut, not of the plugin.** The same plugin is
current one week and superseded the next. What makes it decidable is that the
scanner keeps reporting findings under the old plugin after publishing a new one,
so both branches sit in the same export and the newest one present *is* the
current one. The corollary is a hard ceiling: a product with only one branch in
the export is undecidable, and no rule can fix that.

**Compare branches, not versions.** `151.0.4129.12` and `151.0.4129.59` are two
patches of one branch — the vendor revised the plugin, it did not replace it.
Comparing full version strings flags the first one and the analyst does not:

| Rule | Fires | Correct | Precision |
|---|---:|---:|---:|
| any older full version | 248 | 136 | **0.55** |
| older *major* only | 158 | 130 | **0.82** |

Same recall, and the difference is one `.split(".")`.

**States and workflows are two levels.** A finding has a state — superseded, not
scanned — and, only if it is genuinely active, a workflow. Flattening both into
one list makes them compete, and the loser is whichever rule runs later.
`classify(..., supersedence_first=False)` keeps the losing design runnable so the
decision is a number instead of an opinion:

```
supersedence_first=False   SUPERSEDED agreed=104   similarity=0.672
supersedence_first=True    SUPERSEDED agreed=130   similarity=0.683
```

## The harness

Reproducing a human process means the ground truth is somebody's spreadsheet, and
scoring against it needs three things the usual precision/recall toolkit does not
give you.

**Disagreement has two causes that need opposite fixes.** If the engine flags a
row the analyst filed elsewhere, that is a *precedence* bug. If the analyst never
classified it at all, that is a *threshold* bug. `agreement()` reports them as
separate columns because a single accuracy number points at neither.

**A net variance hides its own magnitude.** Per-category totals let errors of
opposite sign cancel. In the notebook the totals report reads +490 while 1,190
rows are actually misclassified. Always sum magnitudes, and track a similarity
that cannot be gamed by cancellation.

**Score a rule against its own inverse.** The cheapest way to kill a plausible
rule. A candidate that fires on 50 rows and gets 6% right sounds weak but real —
until the *exact opposite* rule scores 9% on the same domain, which is the base
rate. The axis carries no information and the rule was never doing anything:

```
rule      fired=50    hits=3    precision=0.060
inverse   fired=700   hits=65   precision=0.093
base rate                        0.1135
```

Compare with an axis that does carry signal, where the inverse collapses:

```
rule      fired=158   hits=130  precision=0.823
inverse   fired=1940  hits=146  precision=0.075
```

## Layout

```
src/definitions.py   parsing plugin titles into product / branch / KB / period
src/rules.py         the two-level engine; precedence is data, not control flow
src/agreement.py     the measurement harness
src/synthetic.py     the generator, and what it deliberately encodes
tests/test_rules.py  26 tests written as claims about the domain
tools/               notebook builder
```

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q
python tools/build_notebooks.py     # re-executes the notebooks in place
```

Python 3.11+, pandas, numpy. No Spark needed.

## On the production version

The original runs on PySpark over Microsoft Fabric against ~150k findings a week.
The logic ports directly — the one piece of real work is the window:

```python
# pandas
frame.groupby("product")["major"].transform("max")

# pyspark
F.max("major").over(Window.partitionBy("product"))
```

Everything else in `rules.py` is boolean masks, which are the same expression in
either engine. That symmetry is deliberate: the rules were developed and tested
locally on a laptop and deployed to the cluster unchanged, which is the only way
the measure-discard-repeat loop above is fast enough to be worth doing.
