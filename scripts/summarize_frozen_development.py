#!/usr/bin/env python3
"""Validation-only ranking across completed shared-manifest frozen jobs."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import yaml
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from mpower_common import save_json,write_table


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());root=Path(a.root).resolve();tables=[];digests=set()
    for model in cfg['models']:
        for context in cfg['protocol']['contexts']:
            directory=root/'runs'/model/context;done=json.loads((directory/'DONE.json').read_text())
            if done['test_labels_read']:raise ValueError('Test labels were opened')
            digests.add(done['manifest_sha256']);tables.append(pd.read_csv(directory/'validation_leaderboard.csv'))
    if len(digests)!=1:raise ValueError('Models did not consume the same manifest')
    table=pd.concat(tables,ignore_index=True);write_table(table,root/'validation_leaderboard.csv')
    eligible=table.arm.eq('technical_primary') & table.peak_gpu_gib.le(cfg['protocol']['max_peak_gpu_gib']) & table.seconds_per_recording.le(cfg['protocol']['max_seconds_per_recording'])
    primary=table[eligible].sort_values(
        ['auroc','log_loss','model','context','representation','head'],ascending=[False,True,True,True,True,True],kind='stable')
    selected=primary.drop_duplicates('model').head(cfg['protocol']['shortlist'])
    write_table(selected,root/'provisional_validation_shortlist.csv')
    save_json(root/'SUMMARY.json',{'models':len(cfg['models']),'conditions':len(cfg['protocol']['contexts']),
        'pipelines':len(table),'manifest_sha256':next(iter(digests)),'test_labels_read':False,
        'shortlist':selected[['model','context','representation','head']].to_dict('records'),
        'status':'development-only shortlist; near-duplicate/task review required before fine-tuning/final evaluation'})
    print((root/'SUMMARY.json').read_text(),flush=True)
if __name__=='__main__':main()
