#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

"""
Summarize mujoco_closed_loop.py result files. Episodes are paired across controllers (same seed,
same disturbance sequence), so costs are compared per episode against a baseline controller on the
episodes where both survive, and survival with an exact McNemar test on the discordant pairs.

    summarize_closed_loop.py "output/closed_loop_go2_trot_w*.txt" [--pool 10 12.5 15] [--baseline STOCK] [--out summary.md]
"""

import argparse
import glob
import math
import re

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k/n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1 - p)/n + z*z/(4*n*n))
    return ((c - h)/d, (c + h)/d)


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1))/2**n
    return min(1.0, 2*p)


def sign_test_p(x):
    """Two-sided sign test that the median of x is zero."""
    x = np.asarray(x)
    x = x[x != 0]
    return mcnemar_p(int(np.sum(x > 0)), int(np.sum(x < 0)))


def load(path):
    lines = open(path).read().splitlines()
    head = lines[0]
    fields = lines[1].split()
    rows = []
    for line in lines[2:]:
        v = line.split()
        r = dict(zip(fields, v))
        for k in fields[1:]:
            r[k] = float(r[k])
        rows.append(r)
    m = re.search(r'wrench \+-([0-9.]+)', head)
    return (float(m.group(1)) if m else float('nan')), head, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--baseline', default='STOCK')
    parser.add_argument('--out', default=None)
    parser.add_argument('--pool', type=float, nargs='*', default=[],
                        help='levels A over which to pool the paired survival and cost comparisons')
    args = parser.parse_args()
    files = sorted({f for p in args.files for f in glob.glob(p)}, key=lambda f: load(f)[0])

    out = []
    # overview: survival and paired median cost ratio per controller and level
    levels = [load(f) for f in files]
    order = list(dict.fromkeys(r['controller'] for _, _, rows in levels for r in rows))
    out.append('### Overview: episodes survived, and median cost ratio to the baseline on episodes both survive\n')
    out.append('| controller | ' + ' | '.join(f'A = {A:g}' for A, _, _ in levels) + ' |')
    out.append('|---|' + '---|'*len(levels))
    for c in order:
        cells = []
        for A, _, rows in levels:
            by = {int(r['episode']): r for r in rows if r['controller'] == c}
            base = {int(r['episode']): r for r in rows if r['controller'] == args.baseline}
            if not by:
                cells.append('')
                continue
            k = int(sum(r['success'] for r in by.values()))
            both = [e for e in by if e in base and by[e]['success'] and base[e]['success']]
            ratio = f', {np.median([by[e]["cost"]/base[e]["cost"] for e in both]):.3f}' if c != args.baseline and both else ''
            cells.append(f'{k}/{len(by)}{ratio}')
        out.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    out.append('')
    pooled = [lv for lv in levels if lv[0] in args.pool]
    if pooled:
        out.append(f'### Pooled over A = {", ".join(f"{A:g}" for A, _, _ in pooled)}\n')
        out.append('| controller | survived | only this / only baseline | McNemar p | cost ratio, both survive: median | better on | sign-test p |')
        out.append('|---|---|---|---|---|---|---|')
        for c in order:
            k = n = only_c = only_b = 0
            ratios = []
            for A, _, rows in pooled:
                by = {int(r['episode']): r for r in rows if r['controller'] == c}
                base = {int(r['episode']): r for r in rows if r['controller'] == args.baseline}
                for e, r in by.items():
                    n += 1
                    k += int(r['success'])
                    if e in base:
                        only_c += int(r['success'] and not base[e]['success'])
                        only_b += int(base[e]['success'] and not r['success'])
                        if r['success'] and base[e]['success'] and c != args.baseline:
                            ratios.append(r['cost']/base[e]['cost'])
            if c == args.baseline:
                out.append(f'| {c} | {k}/{n} | - | - | - | - | - |')
            else:
                ratios = np.array(ratios)
                med = f'{np.median(ratios):.3f} ({len(ratios)} ep.)' if len(ratios) else '-'
                better = f'{int(np.sum(ratios < 1))}/{len(ratios)}' if len(ratios) else '-'
                sp = f'{sign_test_p(np.log(ratios)):.3g}' if len(ratios) else '-'
                out.append(f'| {c} | {k}/{n} | {only_c} / {only_b} | {mcnemar_p(only_c, only_b):.3g} | {med} | {better} | {sp} |')
        out.append('')
    for f in files:
        A, head, rows = load(f)
        controllers = list(dict.fromkeys(r['controller'] for r in rows))
        by = {c: {int(r['episode']): r for r in rows if r['controller'] == c} for c in controllers}
        base = by[args.baseline]
        out.append(f'### A = {A:g} N / N m\n')
        out.append(f'`{head.lstrip("# ")}`\n')
        out.append('| controller | success | 95% CI | mean survival s | cost, both survive: median ratio to baseline | '
                   'better on | sign-test p | survival vs baseline: only this / only baseline, McNemar p | '
                   'iterations mean | capped calls |')
        out.append('|---|---|---|---|---|---|---|---|---|---|')
        for c in controllers:
            eps = sorted(by[c])
            n = len(eps)
            k = int(sum(by[c][e]['success'] for e in eps))
            lo, hi = wilson(k, n)
            surv = np.mean([by[c][e]['t_end'] for e in eps])
            both = [e for e in eps if e in base and by[c][e]['success'] and base[e]['success']]
            if c != args.baseline and both:
                ratio = np.array([by[c][e]['cost']/base[e]['cost'] for e in both])
                med = f'{np.median(ratio):.3f} ({len(both)} ep.)'
                better = f'{int(np.sum(ratio < 1))}/{len(both)}'
                pval = f'{sign_test_p(np.log(ratio)):.3g}'
            else:
                med, better, pval = '-', '-', '-'
            if c != args.baseline:
                only_c = sum(1 for e in eps if e in base and by[c][e]['success'] and not base[e]['success'])
                only_b = sum(1 for e in eps if e in base and base[e]['success'] and not by[c][e]['success'])
                surv_cmp = f'{only_c} / {only_b}, p = {mcnemar_p(only_c, only_b):.3g}'
            else:
                surv_cmp = '-'
            calls = sum(by[c][e]['n_calls'] for e in eps)
            iters = sum(by[c][e]['iters_mean']*by[c][e]['n_calls'] for e in eps)/max(calls, 1)
            capped = int(sum(by[c][e]['status2'] for e in eps))
            out.append(f'| {c} | {k}/{n} | {lo:.2f} to {hi:.2f} | {surv:.2f} | {med} | {better} | {pval} | {surv_cmp} | '
                       f'{iters:.2f} | {capped}/{int(calls)} |')
        out.append('')
    text = '\n'.join(out)
    print(text)
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(text + '\n')


if __name__ == '__main__':
    main()
