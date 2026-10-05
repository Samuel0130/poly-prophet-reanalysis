"""Actual seven-variant, five-property, five-fold training with held-out OOF scores.
Outer folds never choose epochs; inner validation selects epochs then a fresh
model is refit on all outer-training rows for that fixed number of epochs.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,json,os,time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from runtime import ROOT,ORIGINAL,AblationModel,VARIANTS,seed_everything

ART=ROOT/'training/artifacts';RESULT=ROOT/'training/results'
TARGETS=['Tg','Tc','Density','Rg','FFV']

def atomic_save(value,path):
 tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');torch.save(value,tmp);os.replace(tmp,path)

def save_json(value,path):
 tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def predict(model,ids,tokens,descriptors,embeddings,batch_size):
 output=[];model.eval()
 with torch.no_grad():
  for lo in range(0,len(ids),batch_size):
   i=ids[lo:lo+batch_size]
   output.append(model(tokens[i,0],descriptors[i],embeddings[i,0]).cpu().numpy())
 return np.concatenate(output)


def train_phase(variant,target,fold,phase,fit_ids,val_ids,epochs,args,data,seed,directory):
 df,tokens,desc,embeddings=data
 seed_everything(seed)
 config={'learning_rate':.0005 if target in ['Tc','FFV'] else .001,
         'weight_decay':.001 if target in ['Tc','FFV'] else .0001,
         'patience':20 if target in ['Tc','FFV'] else 15}
 ymean=float(df.iloc[fit_ids][target].mean());ystd=float(df.iloc[fit_ids][target].std(ddof=0))
 # Fit learned preprocessing on phase training rows only.
 scaler=StandardScaler().fit(desc[fit_ids]);scaled_desc=scaler.transform(desc).astype(np.float32)
 d=torch.as_tensor(scaled_desc,device='cuda');y=torch.as_tensor(((df[target].to_numpy()-ymean)/ystd).astype(np.float32),device='cuda')
 model=AblationModel(variant,len(ORIGINAL['create_exact_training_vocab']())).cuda()
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=8)
 latest=directory/(phase+'_latest.pt');bestpath=directory/(phase+'_best.pt')
 start=0;best=np.inf;best_epoch=0;no_improve=0;history=[]
 if latest.exists():
  state=torch.load(latest,map_location='cuda',weights_only=False)
  model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
  start=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];no_improve=state['no_improve'];history=state['history']
  print('RESUME',variant,target,fold,phase,'epoch',start,flush=True)
 for epoch in range(start,epochs):
  seed_everything(seed+epoch*17)
  rng=np.random.default_rng(seed+epoch*17)
  order=rng.permutation(fit_ids);model.train();loss_sum=0;began=time.time()
  for lo in range(0,len(order),args.batch_size):
   ids=order[lo:lo+args.batch_size];choices=np.zeros(len(ids),dtype=np.int64)
   if VARIANTS[variant][4]:
    augmented=rng.random(len(ids))<.3;choices[augmented]=rng.integers(1,4,size=int(augmented.sum()))
   optimizer.zero_grad(set_to_none=True)
   pred=model(tokens[ids,choices],d[ids],embeddings[ids,choices]);loss=torch.nn.functional.mse_loss(pred,y[ids])
   if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss {variant} {target} {fold}')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
   loss_sum+=float(loss.detach())*len(ids)
  train_loss=loss_sum/len(order)
  if phase=='select':
   predictions=predict(model,val_ids,tokens,d,embeddings,args.batch_size)
   raw=predictions*ystd+ymean;actual=df.iloc[val_ids][target].to_numpy()
   mae=float(np.mean(abs(raw-actual)));scheduler.step(mae)
   if mae<best:
    best=mae;best_epoch=epoch+1;no_improve=0
    atomic_save(model.state_dict(),bestpath)
   else:no_improve+=1
  else:
   mae=None;scheduler.step(train_loss);best_epoch=epochs
  history.append({'epoch':epoch+1,'train_mse_standardized':train_loss,'inner_validation_mae':mae,
   'lr':optimizer.param_groups[0]['lr'],'seconds':time.time()-began})
  atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
   'epoch':epoch+1,'best':best,'best_epoch':best_epoch,'no_improve':no_improve,'history':history},latest)
  pd.DataFrame(history).to_csv(directory/(phase+'_history.csv'),index=False)
  print(variant,target,'fold',fold,phase,'epoch',epoch+1,'/',epochs,'loss',round(train_loss,5),
        'inner_mae',None if mae is None else round(mae,6),'sec',round(time.time()-began,1),flush=True)
  if phase=='select' and no_improve>=config['patience']:break
 if phase=='select':
  save_json({'epochs_selected':best_epoch,'inner_best_mae':best,'epochs_run':len(history)},directory/'selected_epochs.json')
 else:
  metadata={'model':model.state_dict(),'variant':variant,'target':target,'fold':fold,
   'epochs':epochs,'seed':seed,'target_mean':ymean,'target_sd':ystd,
   'descriptor_mean':scaler.mean_.tolist(),'descriptor_scale':scaler.scale_.tolist(),
   'vocab':ORIGINAL['create_exact_training_vocab'](),'config':config}
  atomic_save(metadata,directory/'model.pt')
 for p in [latest,bestpath]:
  if p.exists():p.unlink()
 return model,best_epoch,ymean,ystd,d


def aggregate():
 lock=(RESULT/'.aggregate.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 files=sorted(RESULT.glob('*/*/fold_*/metrics.json'))
 rows=[json.loads(f.read_text()) for f in files]
 if rows:pd.DataFrame(rows).to_csv(RESULT/'completed_fold_metrics.csv',index=False)
 progress={'completed_models':len(rows),'expected_models':175,'last_updated':time.strftime('%Y-%m-%d %H:%M:%S'),
           'status':'complete' if len(rows)==175 else 'running'}
 save_json(progress,RESULT/'progress.json')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=80)
 parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--variants',nargs='+',default=list(VARIANTS))
 parser.add_argument('--targets',nargs='+',default=TARGETS);parser.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
 args=parser.parse_args()
 RESULT.mkdir(exist_ok=True)
 assert torch.cuda.is_available(),'RTX GPU required'
 progress=json.loads((ART/'feature_progress.json').read_text());assert progress['completed_rows']==progress['total_rows']
 seed_everything(42)
 df=pd.read_csv(ART/'analysis_dataset.csv');desc=np.load(ART/'descriptors.npy')
 tokens=torch.as_tensor(np.load(ART/'tokens.npy'),device='cuda')
 emb=torch.as_tensor(np.load(ART/'molformer.npy').reshape(len(df),4,64),device='cuda')
 folds=pd.read_csv(ART/'outer_folds.csv');inner=pd.read_csv(ART/'inner_splits.csv')
 configuration={'epochs_max':args.epochs,'batch_size':args.batch_size,'seed':42,'variants':list(VARIANTS),
  'targets':['Tg','FFV','Tc','Density','Rg'],'outer_folds':5,'inner_holdout':.2,
  'architecture':'user original SimpleMolFormerCNNLSTM; removed branches physically excluded from fusion',
  'model_input':'original inference 64D MolFormer; 17D original backbone descriptor; 256D char embedding',
  'changes_from_original':'batch=64; fold-local z-score target/descriptor scaling; shared non-stereo groups; inner early stopping plus full outer-train refit; deterministic cached MolFormer with 3 randomized SMILES',
  'outer_folds_sha256':hashlib.sha256((ART/'outer_folds.csv').read_bytes()).hexdigest(),
  'dataset_sha256':hashlib.sha256((ART/'analysis_dataset.csv').read_bytes()).hexdigest()}
 configpath=RESULT/'experiment_config.json'
 if configpath.exists() and json.loads(configpath.read_text())!=configuration:raise ValueError('Existing experiment configuration differs')
 save_json(configuration,configpath)
 data=(df,tokens,desc,emb)
 for target in args.targets:
  target_rows=np.flatnonzero(df[target].notna().to_numpy())
  for fold in args.folds:
   test_ids=folds.loc[(folds.target==target)&(folds.fold==fold),'row_id'].to_numpy()
   outer_train=np.setdiff1d(target_rows,test_ids)
   subs=inner[(inner.target==target)&(inner.fold==fold)]
   fit_ids=subs.loc[subs.role=='fit','row_id'].to_numpy();stop_ids=subs.loc[subs.role=='early_stop','row_id'].to_numpy()
   assert not set(df.iloc[outer_train].cv_group_id)&set(df.iloc[test_ids].cv_group_id)
   for variant in args.variants:
    directory=RESULT/variant/target/f'fold_{fold}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metrics.json').exists():continue
    print('START',variant,target,'fold',fold,'fit',len(fit_ids),'stop',len(stop_ids),'outer_test',len(test_ids),flush=True)
    chosen=directory/'selected_epochs.json'
    if chosen.exists():epochs=json.loads(chosen.read_text())['epochs_selected']
    else:
     model,epochs,_,_,_=train_phase(variant,target,fold,'select',fit_ids,stop_ids,args.epochs,args,data,42+fold,directory)
     del model;torch.cuda.empty_cache()
    model,_,mean,sd,d=train_phase(variant,target,fold,'refit',outer_train,None,epochs,args,data,10042+fold,directory)
    prediction=predict(model,test_ids,tokens,d,emb,args.batch_size)*sd+mean
    actual=df.iloc[test_ids][target].to_numpy();errors=prediction-actual
    outputs=pd.DataFrame({'row_id':test_ids,'polymer_group_id':df.iloc[test_ids].polymer_group_id.to_numpy(),
      'cv_group_id':df.iloc[test_ids].cv_group_id.to_numpy(),'fold':fold,'target':target,'variant':variant,
      'measured':actual,'predicted':prediction})
    outputs.to_csv(directory/'oof_predictions.csv',index=False)
    metrics={'variant':variant,'target':target,'fold':fold,'n_test':len(test_ids),'n_train':len(outer_train),
     'selected_epochs':epochs,'mae':float(np.mean(abs(errors))),'rmse':float(np.sqrt(np.mean(errors**2))),
     'r2':float(1-np.sum(errors**2)/np.sum((actual-actual.mean())**2))}
    save_json(metrics,directory/'metrics.json');aggregate();print('DONE',metrics,flush=True)
    del model,d;torch.cuda.empty_cache()
 aggregate()
 print('Training invocation finished',flush=True)
if __name__=='__main__':main()
