"""
Human validation of the paraphrase judge.

sample:  draws the annotation set (stratified by judge verdict, dataset and kind of
         edit; one item per source text) and writes a blind annotation sheet (no
         verdicts) plus the sampling record.
analyse: reads the returned sheets (one CSV per annotator) and reports
         - inter-annotator agreement: raw agreement, pairwise Cohen's kappa,
           Fleiss' kappa (all annotators complete) and Krippendorff's alpha (nominal,
           tolerates missing labels);
         - agreement of the judge with the human majority: accuracy and Cohen's
           kappa, the precision of judge-accepted edits (share a human majority
           also accepts) and the share of judge-rejected edits humans accept, with
           Wilson 95% intervals, per stratum and weighted to the full candidate pool.

Usage (from the repository root):
    uv run python -m analysis.human_validation sample --out DIR
    uv run python -m analysis.human_validation analyse --sheets DIR [--out FILE.md]
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt
from typing import Dict, List

from stability_framework.stress_tests import DEFAULT_JUDGMENTS_PATH

DESIGN = {  # (judge verdict, dataset, kind) -> items
    ('Y', 'sst2', 'synonym'): 45, ('Y', 'sst2', 'reorder'): 5,
    ('Y', 'ag_news', 'synonym'): 48, ('Y', 'ag_news', 'reorder'): 2,
    ('N', 'sst2', 'synonym'): 20, ('N', 'sst2', 'reorder'): 5,
    ('N', 'ag_news', 'synonym'): 20, ('N', 'ag_news', 'reorder'): 5,
}
SEED = 2026
FIELDS = ['item_id', 'dataset', 'kind', 'change', 'original', 'changed', 'label', 'reason', 'comment']


def _judgments() -> List[Dict]:
    with open(DEFAULT_JUDGMENTS_PATH, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def _dataset(text: str) -> str:
    return 'ag_news' if any(ch.isupper() for ch in text) else 'sst2'


def _stratum(e: Dict):
    return (e['verdict'], _dataset(e['text']), e.get('type') or 'synonym')


def _display(e: Dict):
    """Original and changed text with the edited span marked ⟦ ⟧, and a short description."""
    words = e['text'].split()
    if (e.get('type') or 'synonym') == 'synonym':
        i = e['position']
        core = e['original']
        before = " ".join(w.replace(core, f"⟦{core}⟧", 1) if k == i else w for k, w in enumerate(words))
        after = " ".join(w.replace(core, f"⟦{e['substitute']}⟧", 1) if k == i else w for k, w in enumerate(words))
        return before, after, f"{core} → {e['substitute']}"
    return e['text'], e['variant'], 'clauses swapped'


def sample(out: str):
    rng = random.Random(SEED)
    pools = defaultdict(list)
    for e in _judgments():
        pools[_stratum(e)].append(e)
    used, chosen = set(), []
    for stratum, n in DESIGN.items():
        pool = sorted(pools[stratum], key=lambda e: e['id'])
        rng.shuffle(pool)
        picked = []
        for e in pool:
            if e['text'] not in used:
                used.add(e['text'])
                picked.append(e)
            if len(picked) == n:
                break
        if len(picked) < n:
            raise RuntimeError(f"stratum {stratum} has only {len(picked)} items with distinct texts")
        chosen += picked
    rng.shuffle(chosen)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'annotation_sheet.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in chosen:
            before, after, change = _display(e)
            w.writerow({'item_id': e['id'], 'dataset': 'SST-2 (movie review)' if _dataset(e['text']) == 'sst2'
                        else 'AG News (news headline + lead)',
                        'kind': 'word substitution' if (e.get('type') or 'synonym') == 'synonym' else 'clause swap',
                        'change': change, 'original': before, 'changed': after,
                        'label': '', 'reason': '', 'comment': ''})
    with open(os.path.join(out, 'sampling_record.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'design': {'|'.join(k): v for k, v in DESIGN.items()},
                   'pool_sizes': {'|'.join(k): len(v) for k, v in pools.items()},
                   'item_ids': [e['id'] for e in chosen]}, f, indent=2)
    print(f"wrote {len(chosen)} items to {out}")


# --- agreement statistics ---

def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(table: List[List[str]]) -> float:
    """table: items x raters (complete)."""
    cats = sorted({x for row in table for x in row})
    n, m = len(table), len(table[0])
    counts = [[row.count(c) for c in cats] for row in table]
    p_i = [(sum(c * c for c in row) - m) / (m * (m - 1)) for row in counts]
    p_bar = sum(p_i) / n
    p_j = [sum(row[j] for row in counts) / (n * m) for j in range(len(cats))]
    pe = sum(p * p for p in p_j)
    return (p_bar - pe) / (1 - pe) if pe < 1 else 1.0


def krippendorff_alpha_nominal(table: List[List[str]]) -> float:
    """table: items x raters, '' for missing."""
    units = [[x for x in row if x] for row in table]
    units = [u for u in units if len(u) >= 2]
    coincidence = Counter()
    for u in units:
        m = len(u)
        for i, x in enumerate(u):
            for j, y in enumerate(u):
                if i != j:
                    coincidence[(x, y)] += 1 / (m - 1)
    n_c = Counter()
    for (x, _), v in coincidence.items():
        n_c[x] += v
    n = sum(n_c.values())
    d_o = sum(v for (x, y), v in coincidence.items() if x != y) / n
    d_e = sum(n_c[x] * n_c[y] for x in n_c for y in n_c if x != y) / (n * (n - 1))
    return 1 - d_o / d_e if d_e > 0 else 1.0


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyse(sheets: str, out: str = None):
    judgments = {e['id']: e for e in _judgments()}
    annotators = {}
    for path in sorted(glob.glob(os.path.join(sheets, '*.csv'))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding='utf-8-sig') as f:
            annotators[name] = {r['item_id']: r['label'].strip().upper()[:1] for r in csv.DictReader(f)
                                if r['label'].strip().upper()[:1] in ('Y', 'N')}
    names = sorted(annotators)
    items = sorted({i for a in annotators.values() for i in a} & set(judgments))
    lines = ["# Human validation of the paraphrase judge", "",
             f"Annotators: {', '.join(names)} ({len(names)}); items with at least one label: {len(items)}", ""]

    lines += ["## Inter-annotator agreement", "", "| Pair | Items | Raw agreement | Cohen's κ |", "|---|---|---|---|"]
    for a, b in combinations(names, 2):
        shared = [i for i in items if i in annotators[a] and i in annotators[b]]
        if shared:
            x, y = [annotators[a][i] for i in shared], [annotators[b][i] for i in shared]
            lines.append(f"| {a} – {b} | {len(shared)} | {sum(p == q for p, q in zip(x, y)) / len(shared):.1%} | "
                         f"{cohen_kappa(x, y):.3f} |")
    table = [[annotators[n].get(i, '') for n in names] for i in items]
    complete = [row for row in table if all(row)]
    lines += [""]
    if len(names) >= 3 and complete:
        lines.append(f"Fleiss' κ (items labelled by all {len(names)}): {fleiss_kappa(complete):.3f} (n = {len(complete)})")
    if len(names) >= 2:
        lines.append(f"Krippendorff's α (nominal): {krippendorff_alpha_nominal(table):.3f}")

    majority, ties = {}, 0
    for i, row in zip(items, table):
        votes = Counter(x for x in row if x)
        if len(votes) == 2 and votes['Y'] == votes['N']:
            ties += 1
            continue
        majority[i] = votes.most_common(1)[0][0]
    lines += ["", f"Human majority label for {len(majority)} items ({ties} ties excluded; resolve ties by adjudication "
              "and rerun if needed).", "", "## Judge vs human majority", ""]
    pairs = [(judgments[i]['verdict'], majority[i]) for i in majority]
    if pairs:
        judge, human = zip(*pairs)
        lines.append(f"Accuracy {sum(j == h for j, h in pairs) / len(pairs):.1%}; Cohen's κ {cohen_kappa(list(judge), list(human)):.3f} "
                     "(on the stratified sample; see the weighted estimate below)")
        lines += ["", "| Stratum (judge, dataset, kind) | Items | Humans accept | Share | Wilson 95% CI |", "|---|---|---|---|---|"]
        by = defaultdict(list)
        for i in majority:
            by[_stratum(judgments[i])].append(majority[i] == 'Y')
        for s in sorted(by):
            k, n = sum(by[s]), len(by[s])
            lo, hi = wilson(k, n)
            lines.append(f"| {' / '.join(s)} | {n} | {k} | {k / n:.1%} | [{lo:.1%}, {hi:.1%}] |")
        for verdict, label in (('Y', 'Precision of judge-accepted edits (humans also accept)'),
                               ('N', 'Judge-rejected edits that humans accept')):
            k = sum(majority[i] == 'Y' for i in majority if judgments[i]['verdict'] == verdict)
            n = sum(judgments[i]['verdict'] == verdict for i in majority)
            lo, hi = wilson(k, n)
            lines.append(f"\n{label}: {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else f"\n{label}: n/a")
        # Accuracy weighted to the full candidate pool (strata are sampled at different rates)
        pool = Counter(_stratum(e) for e in judgments.values())
        acc_by = {s: sum(_stratum(judgments[i]) == s and judgments[i]['verdict'] == majority[i] for i in majority) /
                  max(1, sum(_stratum(judgments[i]) == s for i in majority)) for s in by}
        weighted = sum(pool[s] * acc_by[s] for s in by) / sum(pool[s] for s in by)
        lines.append(f"\nJudge accuracy weighted to the candidate pool: {weighted:.1%}")
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--out', required=True)
    a = sub.add_parser('analyse')
    a.add_argument('--sheets', required=True)
    a.add_argument('--out')
    args = parser.parse_args(argv)
    if args.cmd == 'sample':
        sample(args.out)
    else:
        analyse(args.sheets, args.out)


if __name__ == '__main__':
    main()
