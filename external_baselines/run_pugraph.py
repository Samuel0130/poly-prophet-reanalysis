"""Author Polymer-Unit Graph network transferred to the current five targets.

Failed PURS decompositions use one whole repeat-unit node, explicitly recorded.
This is an adapted author-code baseline, not an exact paper reproduction.
"""
import sys, json, time, hashlib, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external_baselines/vendor'))
from graph_compat import install
install()
import torch, torchgraphs as tg
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.metrics import r2_score
from purs.graph.pu_gn_exp.polymer_unit.networks import SolubilityGN

ART=ROOT/'training/artifacts';OUT=ROOT/'external_baselines/results/pugraph_cpu'

def load_graphs(df):
    graphs=torch.load(ROOT/'external_baselines/cache/pugraph.pt',weights_only=False)
    fallback=[]
    for i,s in enumerate(df.SMILES):
        if i in graphs:continue
        m=Chem.MolFromSmiles(s)
        fp=torch.tensor([float(c) for c in MACCSkeys.GenMACCSKeys(m).ToBitString()]).reshape(1,167)
        graphs[i]=tg.Graph(num_nodes=1,num_edges=0,node_features=fp,
                           edge_features=torch.empty((0,8)),senders=torch.empty(0,dtype=torch.long),
                           receivers=torch.empty(0,dtype=torch.long))
        fallback.append(i)
    # Legacy PURS returns integer tensors for some all-zero/unit fingerprints.
    # The author network has floating weights, so normalize every cached graph
    # to the same feature dtypes before batching.
    for i,g in list(graphs.items()):
        graphs[i]=g.evolve(node_features=g.node_features.to(torch.float32),
                           edge_features=g.edge_features.to(torch.float32),
                           senders=g.senders.to(torch.long),receivers=g.receivers.to(torch.long))
    (OUT/'fallback_rows.json').write_text(json.dumps(fallback))
    return graphs,fallback

def prediction(model,ids,graphs):
    model.eval();parts=[]
    with torch.no_grad():
        for lo in range(0,len(ids),50):
            g=tg.GraphBatch.from_graphs([graphs[int(i)] for i in ids[lo:lo+50]]).to('cpu')
            parts.append(model(g).global_features.reshape(-1).cpu().numpy())
    return np.concatenate(parts)

def train(ids,val,y,graphs,epochs,seed,folder,label):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    mu=float(y[ids].mean());sd=max(float(y[ids].std()),1e-12)
    scaled=(y-mu)/sd
    # Exact downloaded network/config, including upstream dropout layer choices.
    model=SolubilityGN(num_layers=3,hidden_bias=True,hidden_node=32,dropout=50,aggregation='sum').to('cpu')
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=np.inf;best_epoch=0;bad=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0;order=rng.permutation(ids)
        for lo in range(0,len(order),50):
            ix=order[lo:lo+50]
            g=tg.GraphBatch.from_graphs([graphs[int(i)] for i in ix]).to('cpu')
            truth=torch.tensor(scaled[ix],dtype=torch.float32,device='cpu')
            opt.zero_grad();pred=model(g).global_features.reshape(-1)
            mse=torch.nn.functional.mse_loss(pred,truth)
            loss=mse+.0001*sum(p.abs().sum() for p in model.parameters())
            if not torch.isfinite(loss):raise ValueError('Nonfinite graph training loss')
            loss.backward();opt.step();loss_sum+=float(mse.detach())*len(ix)
        score=float(np.mean(abs(prediction(model,val,graphs)*sd+mu-y[val]))) if val is not None else None
        history.append(dict(epoch=epoch,train_mse=loss_sum/len(ids),inner_mae=score))
        if score is not None:
            if score<best:best=score;best_epoch=epoch;bad=0
            else:bad+=1
            if bad>=15:break
    pd.DataFrame(history).to_csv(folder/f'{label}_history.csv',index=False)
    return model,mu,sd,best_epoch if val is not None else epochs

def aggregate():
    rows=[]
    for target in ['Tg','FFV','Tc','Density','Rg']:
        files=sorted(OUT.glob(f'{target}/fold_*/oof.csv'))
        if len(files)!=5:continue
        d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        d.to_csv(OUT.parent/f'OOF_pugraph_{target}.csv',index=False)
        maes=d.groupby('fold').absolute_error.mean()
        rows.append(dict(baseline='pugraph_adapted',target=target,n=len(d),oof_mae=d.absolute_error.mean(),
                         fold_mae_mean=maes.mean(),fold_mae_sd=maes.std(ddof=1),
                         oof_rmse=np.sqrt(d.squared_error.mean()),oof_r2=r2_score(d.measured,d.predicted)))
    pd.DataFrame(rows).to_csv(OUT.parent/'pugraph_performance.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    df=pd.read_csv(ART/'analysis_dataset.csv');folds=pd.read_csv(ART/'outer_folds.csv');inner=pd.read_csv(ART/'inner_splits.csv')
    graphs,fallback=load_graphs(df)
    protocol=dict(source_repo='https://github.com/yecaichao/PURS-PUFp-PUGraph',
                  source_commit='f0815d512387ea3a5fc9b6c4b07a1ecc26a0e21f',doi='10.1021/acs.jctc.3c01385',
                  status='author-code adaptation, not exact numerical paper reproduction',
                  model='original SolubilityGN: 3 layers hidden32 sum aggregation, config dropout enabled, Adam .001 L1 .0001 batch50',
                  adaptations=['PURS per molecule to avoid test-dependent unit index vocabulary',
                               'two DataFrame chained writes replaced with loc for pandas3',
                               'torch_scatter sum replaced with native torch index_add',
                               'failed unit decomposition: one whole-repeat-unit MACCS167 node, zero edges',
                               'fold-local target scaling, shared grouped folds, training-only early stop patience15, full outer refit'],
                  fallback_count=len(fallback),n=len(df),epochs_max=80,device='cpu',threads=1,
                  dataset_sha256=hashlib.sha256((ART/'analysis_dataset.csv').read_bytes()).hexdigest(),
                  outer_folds_sha256=hashlib.sha256((ART/'outer_folds.csv').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('START',protocol,flush=True)
    for target in ['Tg','Tc','Density','Rg','FFV']:
        y=df[target].to_numpy(float);available=np.flatnonzero(np.isfinite(y))
        for fold in range(1,6):
            folder=OUT/target/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'metrics.json').exists():continue
            start=time.time();test=folds.loc[(folds.target==target)&(folds.fold==fold),'row_id'].to_numpy(int)
            train_ids=np.setdiff1d(available,test)
            fit=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='fit'),'row_id'].to_numpy(int)
            stop=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='early_stop'),'row_id'].to_numpy(int)
            assert set(fit)|set(stop)==set(train_ids)
            assert not set(df.cv_group_id.iloc[train_ids])&set(df.cv_group_id.iloc[test])
            m,_,_,selected=train(fit,stop,y,graphs,80,42+fold,folder,'select');del m
            m,mu,sd,_=train(train_ids,None,y,graphs,selected,1042+fold,folder,'refit')
            pred=prediction(m,test,graphs)*sd+mu;err=pred-y[test]
            d=pd.DataFrame(dict(row_id=test,fold=fold,target=target,baseline='pugraph_adapted',measured=y[test],predicted=pred,
                                absolute_error=abs(err),squared_error=err**2,cv_group_id=df.cv_group_id.iloc[test].values,
                                whole_unit_fallback=np.isin(test,fallback)))
            d.to_csv(folder/'oof.csv',index=False)
            metrics=dict(target=target,fold=fold,mae=float(abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),
                         r2=float(r2_score(y[test],pred)),n_train=len(train_ids),n_test=len(test),selected_epochs=selected,
                         fallback_train=int(np.isin(train_ids,fallback).sum()),fallback_test=int(np.isin(test,fallback).sum()),
                         runtime_seconds=time.time()-start)
            (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));aggregate()
            print('DONE',metrics,flush=True);del m
    aggregate()

if __name__=='__main__':main()
