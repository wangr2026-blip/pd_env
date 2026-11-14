#!/usr/bin/env python3
"""Snapshot code and submit common preprocessing plus 18 dependent GPU tasks."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import yaml
REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output');p.add_argument('--max-concurrent',type=int,default=3);p.add_argument('--after',help='Successful validation job dependency');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    for name in ['MPOWER_DATA_ROOT','MPOWER_OUTPUT_ROOT','MPOWER_MODEL_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_BENCHMARK_ROOT']:
        if not os.environ.get(name):raise ValueError(f'Set {name}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=Path(a.output or str(Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'protocol_experiments'/stamp)).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Experiment must be outside Git')
    cfg=yaml.safe_load((REPO/'configs/frozen_model_testing.yaml').read_text());tasks=[{'task':i,'model':m,'context':c} for i,(m,c) in enumerate((m,c) for m in cfg['models'] for c in cfg['protocol']['contexts'])]
    if a.plan_only:print(json.dumps({'root':str(root),'tasks':tasks},indent=2));return
    if a.max_concurrent<1:raise ValueError('Positive concurrency required')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir()
    for folder in ['scripts','src','configs']:
        shutil.copytree(REPO/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    logs=root/'logs';logs.mkdir();env={**os.environ,'MPOWER_PROTOCOL_ROOT':str(root),'MPOWER_CODE_ROOT':str(code)}
    def submit(args):
        r=subprocess.run(['sbatch','--parsable',*args],env=env,text=True,capture_output=True,check=True)
        return r.stdout.strip().split(';')[0]
    receipt={'root':str(root),'tasks':tasks,'code_snapshot':str(code)}
    def save(): (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        dependency=['--dependency=afterok:'+a.after,'--kill-on-invalid-dep=yes'] if a.after else []
        prep=submit([*dependency,'--output='+str(logs/'prepare_%j.out'),'--error='+str(logs/'prepare_%j.err'),str(code/'scripts/frozen_protocol_prepare.slurm')]);receipt['preparation_job']=prep;save()
        tests=submit(['--dependency=afterok:'+prep,'--kill-on-invalid-dep=yes',f'--array=0-{len(tasks)-1}%{a.max_concurrent}',
            '--output='+str(logs/'frozen_%A_%a.out'),'--error='+str(logs/'frozen_%A_%a.err'),str(code/'scripts/frozen_protocol_test.slurm')]);receipt['gpu_array_job']=tests;save()
        aggregate=submit(['--dependency=afterok:'+tests,'--kill-on-invalid-dep=yes','--partition=defq','--cpus-per-task=1','--mem=4G','--time=01:00:00',
            '--output='+str(logs/'summary_%j.out'),'--error='+str(logs/'summary_%j.err'),
            '--wrap',f'cd {shlex.quote(str(code))} && {shlex.quote(env["MPOWER_BENCHMARK_PYTHON"])} scripts/summarize_frozen_development.py --config configs/frozen_model_testing.yaml --root {shlex.quote(str(root))}'])
        receipt['summary_job']=aggregate;save()
    except Exception:
        save();raise
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
