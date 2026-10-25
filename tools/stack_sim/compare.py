#!/usr/bin/env python3
"""Runs stack_sim on several seeds per configuration and prints the mean of every metric.

  compare.py path.csv --seeds 8 --common "speed=1.5" -- "ctrl=pf" "ctrl=ekf" "ctrl=ekf gyro_bias=0.01"

Each configuration after `--` is one row; --common is added to all of them.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(HERE, '..', '..', 'build', 'stack_sim', 'stack_sim')
COLUMNS = ['elat_rms', 'elat_max', 'elat_st', 'elat_st_mean', 'elat_cu', 'steer_std_st', 'ekf_err', 'ekf_err_max',
           'ekf_hdg', 'jump_max', 'rejected', 'resets', 'aborts', 'beta_max', 'elat_lap1', 'elat_last', 'laps', 'time']


def run(binary, path, args):
    out = subprocess.run([binary, path] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'{args}: {out.stderr.strip()}')
    return {k: float(v) for k, v in (kv.split('=') for kv in out.stdout.split())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--common', default='')
    ap.add_argument('--bin', default=DEFAULT_BIN)
    ap.add_argument('configs', nargs='+')
    a = ap.parse_args()
    jobs = [(c, s) for c in a.configs for s in range(1, a.seeds + 1)]
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(lambda j: run(a.bin, a.path, a.common.split() + j[0].split() + [f'seed={j[1]}']), jobs))
    width = max(len(c) for c in a.configs)
    print(f'{"config":<{width}}  ' + ' '.join(f'{c:>12}' for c in COLUMNS))
    for c in a.configs:
        rows = [r for (cc, _), r in zip(jobs, res) if cc == c]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in COLUMNS}
        # the worst seed for the max metrics: one bad run matters
        for k in ('elat_max', 'ekf_err_max', 'jump_max', 'beta_max'):
            mean[k] = max(r[k] for r in rows)
        print(f'{c:<{width}}  ' + ' '.join(f'{mean[k]:>12.4f}' for k in COLUMNS))


if __name__ == '__main__':
    sys.exit(main())
