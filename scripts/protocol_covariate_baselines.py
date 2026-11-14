#!/usr/bin/env python3
"""Development-only age/device controls at initial assessment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from mpower_common import save_json,write_table
from src.training.development import score


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--snapshot',required=True);a=p.parse_args();root=Path(a.snapshot)
    frame=pd.read_parquet(root/'development_manifest.parquet');frame=frame[frame.initial_recording].copy()
    if not frame.split.isin(['train','val']).all():raise ValueError('Test data forbidden')
    train=frame.split.eq('train');val=frame.split.eq('val');frame['age']=pd.to_numeric(frame.age,errors='coerce')
    for column in ['phoneInfo','appVersion']:frame[column]=frame[column].fillna('unknown').astype(str)
    results={}
    for name,numeric,categorical in [('age_only',['age'],[]),('device_only',[],['phoneInfo','appVersion']),('age_device',['age'],['phoneInfo','appVersion'])]:
        transforms=[]
        if numeric:transforms.append(('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric))
        if categorical:transforms.append(('categorical',OneHotEncoder(handle_unknown='ignore'),categorical))
        model=make_pipeline(ColumnTransformer(transforms),LogisticRegression(C=1,max_iter=1000,random_state=20261112))
        model.fit(frame.loc[train],frame.loc[train,'label']);prob=model.predict_proba(frame.loc[val])[:,1]
        results[name]=score(frame.loc[val],prob)
    save_json(root/'covariate_baselines.json',{'test_labels_read':False,'models':results,'medication_used_as_predictor':False})
    # Common support is descriptive, not an outcome-dependent eligibility filter.
    frame['age_band']=pd.cut(frame.age,[0,30,50,65,80,120]).astype(str)
    write_table(frame.groupby(['split','diagnosis','age_band','sex'],dropna=False).size().reset_index(name='participants'),root/'demographic_common_support.csv')
if __name__=='__main__':main()
