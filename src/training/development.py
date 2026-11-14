"""Participant-balanced development-only fitting; no final-test metrics."""
import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import StratifiedGroupKFold
from src.data.frozen_protocol import participant_weights,development_frame


def initial_predictions(frame,p):
    d=frame[[c for c in ['participant_id','recording_id','label','initial_recording','duplicate_component'] if c in frame]].copy();d['probability']=p
    d=d[d.initial_recording]
    if d.participant_id.duplicated().any():raise ValueError('Multiple initial assessments')
    if d.empty:raise ValueError('No initial assessments')
    return d


def score(frame,p):
    d=initial_predictions(frame,p);y=d.label.to_numpy();prob=d.probability.to_numpy()
    return {'auroc':float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,
            'log_loss':float(log_loss(y,prob,labels=[0,1])),'participants':len(d)}


def fit_development(x,frame,options,seed):
    if not frame.split.isin(['train','val']).all():raise ValueError('Development fitting refuses sealed test rows')
    frame=development_frame(frame);train=frame.split.eq('train').to_numpy();val=~train
    tr=frame.loc[train].reset_index(drop=True);va=frame.loc[val].reset_index(drop=True)
    xt=x[train];xv=x[val];y=tr.label.to_numpy();w=participant_weights(tr.participant_id)
    groups=tr.get('duplicate_component',tr.participant_id).to_numpy()
    folds=StratifiedGroupKFold(n_splits=options['cv_folds'],shuffle=True,random_state=seed)
    search=[]
    for c in options['C_grid']:
        scores=[]
        for a,b in folds.split(xt,y,groups):
            weights=participant_weights(tr.iloc[a].participant_id)
            scaler=StandardScaler().fit(xt[a],sample_weight=weights)
            head=LogisticRegression(C=c,max_iter=1000,random_state=seed)
            head.fit(scaler.transform(xt[a]),y[a],sample_weight=weights)
            metrics=score(tr.iloc[b],head.predict_proba(scaler.transform(xt[b]))[:,1])
            scores.append(metrics['auroc'])
        search.append({'C':c,'cv_auroc':float(np.mean(scores))})
    chosen=sorted(search,key=lambda d:(-d['cv_auroc'],d['C']))[0]['C']
    scaler=StandardScaler().fit(xt,sample_weight=w);z=scaler.transform(xt);zv=scaler.transform(xv)
    head=LogisticRegression(C=chosen,max_iter=1000,random_state=seed).fit(z,y,sample_weight=w)
    predictions=head.predict_proba(zv)[:,1]
    results={'logistic':{'state':{'scaler':scaler,'head':head},'probabilities':predictions,
                          'metrics':score(va,predictions),'search':search,'chosen_C':chosen}}
    # Weighted PyTorch MLP avoids relying on sklearn's version-specific sample_weight API.
    import torch
    from torch import nn
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=nn.Sequential(nn.Linear(z.shape[1],options['hidden_dim']),nn.ReLU(),nn.Linear(options['hidden_dim'],1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    tx=torch.tensor(z,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32)
    visits=[np.flatnonzero(tr.participant_id.to_numpy()==pid) for pid in sorted(tr.participant_id.unique())]
    tv=torch.tensor(zv,dtype=torch.float32);best_key=None;logs=[]
    for epoch in range(options['epochs']):
        model.train();order=rng.permutation(len(visits))
        for start in range(0,len(order),options['batch_size']):
            people=order[start:start+options['batch_size']]
            ix=np.array([rng.choice(visits[i]) for i in people]);optimizer.zero_grad()
            losses=nn.functional.binary_cross_entropy_with_logits(model(tx[ix]).squeeze(-1),ty[ix],reduction='none')
            losses.mean().backward();optimizer.step()
        model.eval()
        with torch.no_grad():p=torch.sigmoid(model(tv).squeeze(-1)).numpy()
        metrics=score(va,p);logs.append({'epoch':epoch+1,**metrics});key=(metrics['auroc'],-metrics['log_loss'])
        if best_key is None or key>best_key:best_key=key;best=copy.deepcopy(model.state_dict());best_p=p.copy()
    results['mlp']={'state':{'scaler':scaler,'state_dict':best,'input_dim':z.shape[1],'hidden_dim':options['hidden_dim']},
                    'probabilities':best_p,'metrics':score(va,best_p),'search':logs}
    for result in results.values():
        d=initial_predictions(va,result['probabilities']);controls=d.loc[d.label.eq(0),'probability']
        threshold=float(np.nextafter(np.quantile(controls,options['specificity_target'],method='higher'),np.inf))
        result['threshold']=threshold;pred=d.probability.ge(threshold)
        result['metrics'].update(threshold=threshold,specificity=float((~pred[d.label.eq(0)]).mean()),sensitivity=float(pred[d.label.eq(1)].mean()))
        result['initial_validation_predictions']=d
    return results
