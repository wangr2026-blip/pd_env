#!/usr/bin/env python3
"""Aggregate completed baseline comparisons, excluding smoke tests by default."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.benchmark import configure
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/frozen_benchmark.yaml')
    p.add_argument('--include-smoke', action='store_true')
    args = p.parse_args()
    root = configure(args.config)['benchmark_root']
    tables = [pd.read_csv(f) for f in (root / 'runs').glob('*/comparison.csv')]
    if not tables:
        raise SystemExit('No completed experiments to compare')
    frame = pd.concat(tables, ignore_index=True)
    if not args.include_smoke:
        frame = frame[frame['mode'].eq('baseline')]
    # Different cohort snapshots cannot be interpreted as a controlled model comparison.
    import json
    frame['manifest_digest'] = frame.run.map(lambda r: json.loads((Path(r) / 'provenance.json').read_text())['manifest_digest'])
    frame.to_csv(root / 'comparison.csv', index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
