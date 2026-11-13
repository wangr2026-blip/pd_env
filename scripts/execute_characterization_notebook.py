#!/usr/bin/env python3
"""Execute the source notebook; all outputs stay in restricted output storage."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
repo=Path(__file__).resolve().parents[1]
out=Path(os.environ['MPOWER_OUTPUT_ROOT']).resolve()
if out==repo or repo in out.parents: raise ValueError('Output must be outside Git')
nb=nbformat.read(repo/'notebooks/01_mPower_Dataset_Characterization.ipynb',as_version=4)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(repo)}}).execute()
nbformat.write(nb,out/'eda/01_mPower_Dataset_Characterization.executed.ipynb')
print('Executed notebook saved outside repository',flush=True)
