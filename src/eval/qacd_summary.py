"""Aggregate POPE matrix answer files into a mean+/-std table over seeds.

Reads answer files named  <method>_<dataset>_<split>_seed<N>.jsonl  (as written
by qacd_matrix.bash), computes Accuracy / F1 / yes-proportion per file against
the matching gt file, then averages over seeds and prints a table grouped by
(dataset, split, method) -- the paper-style results layout.

Usage:
    python eval/qacd_summary.py output/matrix/*.jsonl
"""
import glob
import json
import os
import sys
from collections import defaultdict


def _metrics(gt_path, ans_path):
    gt = {json.loads(l)['question_id']: json.loads(l)['label'].lower().strip()
          for l in open(gt_path) if l.strip()}
    tp = tn = fp = fn = yes = n = 0
    for l in open(ans_path):
        if not l.strip():
            continue
        d = json.loads(l)
        q = d['question_id']
        if q not in gt:
            continue
        pred = d['text'].lower().strip()
        g = gt[q]
        n += 1
        if g == 'yes':
            if 'yes' in pred:
                tp += 1; yes += 1
            else:
                fn += 1
        elif g == 'no':
            if 'no' in pred:
                tn += 1
            else:
                fp += 1; yes += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {'acc': 100 * acc, 'f1': 100 * f1, 'yes': 100 * yes / n if n else 0.0, 'n': n}


def _mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def main():
    paths = []
    for pat in sys.argv[1:]:
        paths.extend(glob.glob(pat))
    if not paths:
        print('No answer files given.')
        return

    # group metric values by (dataset, split, method) over seeds
    groups = defaultdict(lambda: defaultdict(list))  # key -> metric -> [values]
    for p in sorted(paths):
        stem = os.path.basename(p)[:-len('.jsonl')] if p.endswith('.jsonl') else os.path.basename(p)
        parts = stem.split('_')
        if len(parts) < 4 or not parts[-1].startswith('seed'):
            continue
        seed = parts[-1]; split = parts[-2]; dataset = parts[-3]
        method = '_'.join(parts[:-3])           # robust to method names w/ '_'
        gt_path = f'./data/POPE/{dataset}/{dataset}_pope_{split}.jsonl'
        if not os.path.exists(gt_path):
            print(f'[warn] gt not found for {p}: {gt_path}')
            continue
        m = _metrics(gt_path, p)
        key = (dataset, split, method)
        for k in ('acc', 'f1', 'yes'):
            groups[key][k].append(m[k])
        groups[key]['_seeds'].append(seed)

    # print table
    hdr = f'{"dataset":<8} {"split":<12} {"method":<11} {"seeds":>5}  {"Acc":>14} {"F1":>14} {"Yes%":>13}'
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    last_ds_split = None
    for (dataset, split, method) in sorted(groups):
        g = groups[(dataset, split, method)]
        ns = len(g['_seeds'])
        am, asd = _mean_std(g['acc'])
        fm, fsd = _mean_std(g['f1'])
        ym, ysd = _mean_std(g['yes'])
        if last_ds_split and last_ds_split != (dataset, split):
            print()
        last_ds_split = (dataset, split)
        print(f'{dataset:<8} {split:<12} {method:<11} {ns:>5}  '
              f'{am:>6.2f}+-{asd:<5.2f} {fm:>6.2f}+-{fsd:<5.2f} {ym:>6.2f}+-{ysd:<4.2f}')
    print('=' * len(hdr))
    print('(yes% << 50 indicates over-suppression of present objects)')


if __name__ == '__main__':
    main()
