"""Aggregate QACD fallback rates and recipe distributions across run files.

Reads one or more JSONL files (POPE answer files written by pope.py, or
recipes.jsonl from a debug run) and reports:
    - parse-fallback rate   (planner output unparseable)
    - region-fallback rate  (attention grounding -> center region)
    - operation distribution
    - intensity distribution
    - mean attended-region coverage

Usage:
    python eval/qacd_stats.py output/qacd/pope/**/*.jsonl
    python eval/qacd_stats.py output/qacd_debug/recipes.jsonl
"""
import argparse
import glob
import json
from collections import Counter


def _iter_records(paths):
    for pat in paths:
        for path in glob.glob(pat, recursive=True):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # answer files nest the metric under 'qacd'; recipes are flat
                    meta = rec.get('qacd', rec)
                    if meta and 'op' in meta:
                        yield meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='JSONL file(s) or globs')
    args = ap.parse_args()

    n = parse_fb = region_fb = 0
    ops, intensities = Counter(), Counter()
    covs = []

    for m in _iter_records(args.paths):
        n += 1
        parse_fb += int(m.get('parse_fallback', not m.get('parsed_ok', True)))
        region_fb += int(m.get('region_fallback', False))
        ops[m.get('op')] += 1
        intensities[m.get('intensity')] += 1
        if m.get('mask_coverage') is not None:
            covs.append(m['mask_coverage'])
    covs.sort()
    coverage_sum = sum(covs)
    coverage_n = len(covs)

    if not n:
        print('No QACD records found.')
        return

    print('=' * 60)
    print(f'QACD stats over {n} questions')
    print('-' * 60)
    print(f'parse fallback : {parse_fb:>6}/{n} = {parse_fb / n:6.1%}')
    print(f'region fallback: {region_fb:>6}/{n} = {region_fb / n:6.1%}')
    if coverage_n:
        cmin, cmed, cmax = covs[0], covs[len(covs) // 2], covs[-1]
        print(f'mean coverage  : {coverage_sum / coverage_n:6.1%}'
              f'  (over {coverage_n} masked questions)')
        print(f'coverage range : {cmin:.1%} .. median {cmed:.1%} .. {cmax:.1%}'
              '  (wide spread = region adapts to object size)')
    print('-' * 60)
    print('operation distribution:')
    for op, c in ops.most_common():
        print(f'  {str(op):<12} {c:>6}  {c / n:6.1%}')
    print('intensity distribution:')
    for lvl, c in sorted(intensities.items(), key=lambda x: (x[0] is None, x[0])):
        print(f'  {str(lvl):<12} {c:>6}  {c / n:6.1%}')
    print('=' * 60)


if __name__ == '__main__':
    main()
