from pathlib import Path
import sys;sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"training"))
"""Fair external baselines on the reconstructed dataset.
B0 mean, B1 Morgan/Ridge, B2 Morgan/RandomForest. Fold-local preprocessing,
shared outer grouped folds, and no use of downloaded model weights.
"""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from runtime import ROOT,seed_everything
ART=ROOT/'training/artifacts';OUT=ROOT/'external_baselines/results';OUT.mkdir(parents=True,exist_ok=True)
TARGETS=['Tg','FFV','Tc','Density','Rg']
BASELINES=['mean','morgan_ridge','morgan_rf']

def fingerprints(smiles):
 out=np.zeros((len(smiles),2048),dtype=np.uint8)
 for i,s in enumerate(smiles):
  m=Chem.MolFromSmiles(s)
  if m is None:raise ValueError('invalid canonical SMILES')
  bv=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
  arr=np.zeros((2048,),dtype=np.uint8);from rdkit.DataStructs import ConvertToNumpyArray
  ConvertToNumpyArray(bv,arr);out[i]=arr
 return out

def score(baseline,target,fold,model,ids,X,y):
 pred=model.predict(X[ids]) if baseline!='mean' else np.full(len(ids),model)
 actual=y[ids];err=pred-actual
 return pd.DataFrame({'row_id':ids,'fold':fold,'target':target,'baseline':baseline,
  'measured':actual,'predicted':pred,'absolute_error':abs(err),'squared_error':err**2})

def main():
 seed_everything(42);df=pd.read_csv(ART/'analysis_dataset.csv');folds=pd.read_csv(ART/'outer_folds.csv');inner=pd.read_csv(ART/'inner_splits.csv')
 xpath=ART/'morgan_2048_uint8.npy'
 if xpath.exists():X=np.load(xpath)
 else:X=fingerprints(df.SMILES.tolist());np.save(xpath,X)
 print('fingerprints',X.shape,flush=True)
 all_metrics=[];all_oof=[]
 for target in TARGETS:
  y=df[target].to_numpy(dtype=float);available=np.flatnonzero(np.isfinite(y))
  for baseline in BASELINES:
   fold_metrics=[];oof=[]
   for fold in range(1,6):
    test=folds.loc[(folds.target==target)&(folds.fold==fold),'row_id'].to_numpy(dtype=int)
    train=np.setdiff1d(available,test)
    if baseline=='mean':
     model=float(y[train].mean());selected={}
    else:
     # Selection is performed only on inner training/early-stop rows.
     fit=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='fit'),'row_id'].to_numpy(dtype=int)
     stop=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='early_stop'),'row_id'].to_numpy(dtype=int)
     if baseline=='morgan_ridge':
      candidates=[.01,.1,1.,10.,100.]
      trial=[]
      for alpha in candidates:
       m=Ridge(alpha=alpha);m.fit(X[fit],y[fit]);trial.append((mean_absolute_error(y[stop],m.predict(X[stop])),alpha))
      selected={'alpha':min(trial)[1]};model=Ridge(alpha=selected['alpha']);model.fit(X[train],y[train])
     elif baseline=='morgan_rf':
      candidates=[50,200]
      trial=[]
      for leaf in candidates:
       m=RandomForestRegressor(n_estimators=200,max_features='sqrt',min_samples_leaf=leaf,n_jobs=4,random_state=4200+fold)
       m.fit(X[fit],y[fit]);trial.append((mean_absolute_error(y[stop],m.predict(X[stop])),leaf))
      selected={'min_samples_leaf':min(trial)[1]};model=RandomForestRegressor(n_estimators=300,max_features='sqrt',min_samples_leaf=selected['min_samples_leaf'],n_jobs=4,random_state=42000+fold)
      model.fit(X[train],y[train])
     else:raise ValueError(baseline)
    pred=np.full(len(test),model) if baseline=='mean' else model.predict(X[test]);actual=y[test];err=pred-actual
    part=pd.DataFrame({'row_id':test,'fold':fold,'target':target,'baseline':baseline,'measured':actual,'predicted':pred,'absolute_error':abs(err),'squared_error':err**2});oof.append(part);all_oof.append(part)
    fold_metrics.append({'baseline':baseline,'target':target,'fold':fold,'n_test':len(test),'n_train':len(train),'mae':float(abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2))),'r2':float(r2_score(actual,pred)),'selected':selected})
    print(baseline,target,'fold',fold,fold_metrics[-1],flush=True)
   f=pd.concat(oof);vals=np.array([m['mae'] for m in fold_metrics])
   all_metrics.append({'baseline':baseline,'target':target,'n':len(f),'oof_mae':float(f.absolute_error.mean()),'fold_mae_mean':float(vals.mean()),'fold_mae_sd':float(vals.std(ddof=1)),'oof_rmse':float(np.sqrt(f.squared_error.mean())),'oof_r2':float(r2_score(f.measured,f.predicted))})
   pd.concat(oof).to_csv(OUT/f'OOF_{baseline}_{target}.csv',index=False)
 pd.DataFrame(all_metrics).to_csv(OUT/'classical_performance.csv',index=False);pd.concat(all_oof,ignore_index=True).to_csv(OUT/'classical_oof_all.csv',index=False)
 metadata={'protocol':'same reconstructed dataset, outer_folds.csv and inner_splits.csv as main experiment','fingerprint':'RDKit Morgan radius=2, nBits=2048; calculated from canonical SMILES','mean':'outer training mean only','ridge':'inner MAE alpha selection, refit outer training','random_forest':'inner MAE min_samples_leaf selection; 300 trees, sqrt features, refit outer training','no_downloaded_weights':True}
 (OUT/'classical_protocol.json').write_text(json.dumps(metadata,indent=2))
 print(pd.DataFrame(all_metrics).to_string(index=False),flush=True)
if __name__=='__main__':main()
