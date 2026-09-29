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
Figure of README.md from the closed-loop result files: survival rate and median tracking-cost ratio
to STOCK (on episodes both survive) against the push level A.

    python plot_results.py "output/closed_loop_go2_trot_w*.txt" --output output/closed_loop.png
"""

import argparse
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_closed_loop import load

SHOWN = [('STOCK', 'stock (SQP_GN, 1 iteration)', 'k', '-'),
         ('SQP_GN:3', 'SQP_GN, 3 iterations', 'tab:blue', '-'),
         ('SQP_GN:300', 'SQP_GN, converged', 'tab:blue', ':'),
         ('FILTERDDP_GN:15', 'FILTERDDP_GN, 15 iterations', 'tab:orange', '-'),
         ('FILTERDDP_GN:5:vg+ws', 'FILTERDDP_GN warm started, 5', 'tab:green', '--'),
         ('FILTERDDP_GN:15:vg+ws', 'FILTERDDP_GN warm started, 15', 'tab:green', '-')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--baseline', default='STOCK')
    parser.add_argument('--min_pairs', type=int, default=5, help='fewest episodes both survive to plot a cost ratio')
    parser.add_argument('--output', default='output/closed_loop.png')
    args = parser.parse_args()
    levels = sorted((load(f) for f in sorted({f for p in args.files for f in glob.glob(p)})), key=lambda lv: lv[0])

    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(10, 3.8))
    for spec, label, color, style in SHOWN:
        A_s, surv, A_c, ratio = [], [], [], []
        for A, _, rows in levels:
            by = {int(r['episode']): r for r in rows if r['controller'] == spec}
            base = {int(r['episode']): r for r in rows if r['controller'] == args.baseline}
            if not by:
                continue
            A_s.append(A)
            surv.append(np.mean([r['success'] for r in by.values()]))
            both = [e for e in by if e in base and by[e]['success'] and base[e]['success']]
            if len(both) >= args.min_pairs:
                A_c.append(A)
                ratio.append(np.median([by[e]['cost']/base[e]['cost'] for e in both]))
        ax_s.plot(A_s, surv, style, color=color, marker='o', ms=4, label=label)
        ax_c.plot(A_c, ratio, style, color=color, marker='o', ms=4, label=label)
    ax_s.set_xlabel('push level A (N, N m)')
    ax_s.set_ylabel('episodes survived')
    ax_s.set_ylim(-0.03, 1.03)
    ax_c.set_xlabel('push level A (N, N m)')
    ax_c.set_ylabel('tracking cost / stock, median')
    ax_c.axhline(1.0, color='0.7', lw=0.8)
    for ax in (ax_s, ax_c):
        ax.grid(alpha=0.3)
    ax_s.legend(fontsize=8, loc='lower left')
    n = len(levels[0][2])//len({r['controller'] for r in levels[0][2]})
    fig.suptitle(f'Go2 trot, 0.5 m/s, random base wrench redrawn every 2 s, {n} paired episodes per level', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
