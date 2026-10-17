"""
Prepare and collect LLM-judge verdicts for paraphrase candidates (generator v4).

export: generates every locally validated candidate for the texts of the given
        experiments (their correctly classified originals), stores them in
        results/paraphrase_candidates/candidates.jsonl, and writes the candidates
        without a verdict into batch files of --batch-size items
        (results/paraphrase_candidates/batches/batch_NNN.jsonl). Each batch line has
        the judge's input: id, text with the word marked [[like this]] (or, for a
        clause swap, the reordered text), word and substitute.
merge:  reads judge outputs (one {"id", "v", "c"} object per line) from --outputs
        and appends the new verdicts to stability_framework/resources/paraphrase_judgments.jsonl.
status: counts candidates, verdicts and pending items.

Usage (from the repository root):
    uv run python experiment_runners/paraphrase_judging.py export --experiments experiment_1 experiment_3 --batch-size 300
    uv run python experiment_runners/paraphrase_judging.py merge --outputs DIR --judge claude-haiku-5-5 --label full-1
    uv run python experiment_runners/paraphrase_judging.py status
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stability_framework.stress_tests import (DEFAULT_JUDGMENTS_PATH, JUDGE_PROMPT_VERSION,  # noqa: E402
                                              SemanticParaphrasingTest, load_judgments)

OUT = Path(__file__).resolve().parent.parent / 'results' / 'paraphrase_candidates'
CANDIDATES = OUT / 'candidates.jsonl'


def _texts(experiments):
    from analysis.report import latest_instance_dir
    from analysis.stability import load_per_instance
    texts = {}
    for e in experiments:
        for (_, test), rows in load_per_instance(latest_instance_dir(e)).items():
            for r in rows:
                texts.setdefault(r['original_text'], None)
    return list(texts)


def _judge_input(c):
    if c['type'] == 'synonym':
        words = c['source'].split()
        marked = " ".join(f"[[{w}]]" if i == c['position'] else w for i, w in enumerate(words))
        return {'id': c['id'], 'text': marked, 'word': c['original'], 'substitute': c['substitute']}
    return {'id': c['id'], 'text': c['source'], 'reordered': c['text']}


def export(experiments, batch_size):
    OUT.mkdir(parents=True, exist_ok=True)
    known = {}
    if CANDIDATES.exists():
        for line in CANDIDATES.read_text().splitlines():
            c = json.loads(line)
            known.setdefault(c['source'], []).append(c)
    t = SemanticParaphrasingTest(judgments_path=None)
    texts = _texts(experiments)
    with open(CANDIDATES, 'a', encoding='utf-8') as f:
        for n, text in enumerate(texts):
            if text in known:
                continue
            cands = [dict(c, source=text) for c in t.candidates(text)]
            known[text] = cands
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
            if n % 200 == 0:
                print(f"{n}/{len(texts)} texts", flush=True)
    verdicts = load_judgments(DEFAULT_JUDGMENTS_PATH)
    pending = [c for text in texts for c in known.get(text, []) if c['id'] not in verdicts]
    seen, unique = set(), []
    for c in pending:
        if c['id'] not in seen:
            seen.add(c['id'])
            unique.append(c)
    batches = OUT / 'batches'
    batches.mkdir(exist_ok=True)
    for old in glob.glob(str(batches / 'batch_*.jsonl')):
        os.remove(old)
    for b in range(0, len(unique), batch_size):
        with open(batches / f'batch_{b // batch_size + 1:03d}.jsonl', 'w', encoding='utf-8') as f:
            for c in unique[b:b + batch_size]:
                f.write(json.dumps(_judge_input(c), ensure_ascii=False) + '\n')
    print(f"texts {len(texts)}, candidates {sum(len(known.get(t, [])) for t in texts)}, "
          f"pending {len(unique)} in {-(-len(unique) // batch_size)} batch(es) of up to {batch_size}")


def merge(outputs, judge, label):
    cands = {}
    for line in CANDIDATES.read_text().splitlines():
        c = json.loads(line)
        cands[c['id']] = c
    existing = load_judgments(DEFAULT_JUDGMENTS_PATH)
    added = 0
    with open(DEFAULT_JUDGMENTS_PATH, 'a', encoding='utf-8') as f:
        for path in sorted(glob.glob(os.path.join(outputs, '*.jsonl'))):
            for line in open(path, encoding='utf-8'):
                if not line.strip():
                    continue
                o = json.loads(line)
                if o['id'] in existing or o['id'] not in cands or o.get('v') not in ('Y', 'N'):
                    continue
                c = cands[o['id']]
                entry = {'id': o['id'], 'type': c['type'], 'text': c['source'], 'position': c.get('position'),
                         'original': c.get('original'), 'substitute': c.get('substitute'),
                         'variant': c['text'] if c['type'] == 'reorder' else None,
                         'verdict': o['v'], 'code': o.get('c'), 'judge': judge,
                         'prompt_version': JUDGE_PROMPT_VERSION, 'batch': f"{label}:{os.path.basename(path)}"}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                existing[o['id']] = o['v']
                added += 1
    print(f"added {added} verdicts; total {len(existing)}")


def status():
    verdicts = load_judgments(DEFAULT_JUDGMENTS_PATH)
    cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines()] if CANDIDATES.exists() else []
    ids = {c['id'] for c in cands}
    judged = [i for i in ids if i in verdicts]
    print(f"candidates {len(ids)}, judged {len(judged)} (Y {sum(verdicts[i] == 'Y' for i in judged)}), "
          f"pending {len(ids) - len(judged)}; verdicts on file {len(verdicts)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--experiments', nargs='+', default=['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4'])
    e.add_argument('--batch-size', type=int, default=300)
    m = sub.add_parser('merge')
    m.add_argument('--outputs', required=True)
    m.add_argument('--judge', required=True)
    m.add_argument('--label', required=True)
    sub.add_parser('status')
    args = parser.parse_args(argv)
    if args.cmd == 'export':
        export(args.experiments, args.batch_size)
    elif args.cmd == 'merge':
        merge(args.outputs, args.judge, args.label)
    else:
        status()


if __name__ == '__main__':
    main()
