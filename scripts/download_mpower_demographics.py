#!/usr/bin/env python3
"""Explicit opt-in metadata-only download; does not run in preprocessing jobs.

Requires prior research-data authorization. Inspect actual exported schema before
configuring linkage. Never print participant rows or credential values.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import synapseclient

TABLE_ID='syn5511429'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorized',action='store_true',help='Confirm explicit authorization for this metadata-only retrieval')
    args=p.parse_args()
    if not args.authorized: p.error('Explicit authorization is required')
    os.umask(0o077)
    root=Path(os.environ['MPOWER_DATA_ROOT']).expanduser().resolve()
    repo=Path(__file__).resolve().parents[1]
    if root==repo or repo in root.parents: raise ValueError('Raw data root must be outside Git')
    dest=root/'demographics'; dest.mkdir(parents=True,exist_ok=True)
    target=dest/'Demographics.csv'
    if target.exists(): raise FileExistsError('Existing export preserved; inspect before choosing a new snapshot')
    token=os.environ.get('SYNAPSE_AUTH_TOKEN')
    if not token: raise ValueError('Missing SYNAPSE_AUTH_TOKEN; use Slurm wrapper')
    syn=synapseclient.Synapse(cache_root_dir=str(root/'synapse_cache'),configPath=str(root/'synapse_no_config.ini'))
    syn.login(authToken=token,silent=True)
    response=syn.tableQuery(f'SELECT * FROM {TABLE_ID}')
    frame=response.asDataFrame()
    # No file columns are downloaded; this is a table export only.
    fd,temp=tempfile.mkstemp(prefix='demographics_',suffix='.csv',dir=dest)
    os.close(fd); tmp=Path(temp)
    frame.to_csv(tmp,index=False)
    tmp.replace(target)
    (dest/'export_provenance.json').write_text(json.dumps({'synapse_table':TABLE_ID,'rows':len(frame),
        'columns':list(frame.columns),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'query':f'SELECT * FROM {TABLE_ID}','audio_downloaded':False},indent=2))
    print(f'Demographics metadata exported: {len(frame)} rows, {len(frame.columns)} columns; no audio downloaded',flush=True)


if __name__=='__main__': main()
