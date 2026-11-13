#!/usr/bin/env python3
"""Pilot-only repeat extraction and plausibility audit; intended for Slurm."""
import numpy as np
import pandas as pd
from mpower_common import configure, decode, parser, save_json, write_table
from extract_acoustic_features import extract


def main():
    args=parser(__doc__).parse_args(); cfg=configure(args.config); out=cfg['output_root']
    f=pd.read_parquet(out/'processed/pilot_acoustic_features.parquet')
    r=pd.read_parquet(out/'processed/pilot_selection.parquet')
    merged=r.merge(f,on='recording_id',validate='one_to_one')
    checks=[]; errors=[]
    successful=merged[merged.feature_status.eq('ok')].sort_values('recording_id').head(5)
    for row in successful.to_dict('records'):
        try:
            y,sr=decode(row['audio_path'],cfg); new,_,_=extract(y,sr,cfg)
            shared=[c for c in new if c in row and isinstance(row[c],(int,float,np.number))]
            a=np.array([row[c] for c in shared],dtype=float); b=np.array([new[c] for c in shared],dtype=float)
            agree=np.isclose(a,b,atol=1e-8,rtol=1e-7,equal_nan=True)
            checks.append({'recording_id':row['recording_id'],'compared_features':len(shared),'disagreements':int((~agree).sum()),'reproducible':bool(agree.all())})
        except Exception as exc: errors.append({'recording_id':row['recording_id'],'error':str(exc)})
    write_table(pd.DataFrame(checks),out/'eda/diagnostics/pilot_repeat_processing.csv')
    anomalies=[]
    for col in f:
        if not col.startswith(('praat_full__','praat_stable__')): continue
        v=pd.to_numeric(f[col],errors='coerce')
        bad=(v<0) if 'hnr_db' not in col else pd.Series(False,index=v.index)
        if 'fraction' in col: bad |= v>1
        if 'f0_' in col and col.endswith('_hz') and not any(x in col for x in ('sd','range')):
            bad |= v.notna() & ~v.between(cfg['pitch']['floor'],cfg['pitch']['ceiling'])
        anomalies.append({'feature':col,'finite':int(np.isfinite(v).sum()),'missing_or_nonfinite':int((~np.isfinite(v)).sum()),
                          'technical_bound_violations':int(bad.sum()),'denominator':len(f),
                          'interpretation':'HNR may legitimately be negative; bounds are technical diagnostics, not clinical cutoffs' if 'hnr' in col else 'No disease-specific clinical cutoff applied'})
    write_table(pd.DataFrame(anomalies),out/'eda/tables/pilot_feature_plausibility.csv')
    save_json(out/'eda/diagnostics/pilot_validation.json',{'selected':len(f),'repeated':len(checks),'repeat_errors':errors,
        'all_repeated_reproducible':bool(checks) and all(c['reproducible'] for c in checks),
        'status':'review_required','full_scale_gate':'Review statuses, missingness, interval validity and pitch/trim sensitivity before full Slurm submission',
        'pitch_trim_sensitivity':'Retained per recording in pilot_acoustic_features.parquet',
        'no_claim':'Synthetic or periodic-signal validity does not establish performance for severely dysphonic human voices'})
    print(f'Pilot repeat validation: {len(checks)} recordings checked',flush=True)


if __name__=='__main__': main()
