"""Use the downloaded Chen et al. Keras model factory on shared nested folds.

Published best architecture: length 120, embedding 15, two BiLSTMs (60
units/direction), time-distributed Dense(30), flatten, scalar regression.
Only the factory is extracted; upstream top-level data loading is not executed.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_BACKEND', 'torch')
import argparse, ast, hashlib, json, time, fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import keras
import torch
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Reshape
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'training/artifacts'
OUT = ROOT / 'external_baselines/results/chen_rnn'
SOURCE = ROOT / 'external_baselines/chen_rnn_tg/model/ChemicalLanguageProcessingModel.py'
LENGTH = 120

def factory(vocab_size):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'getRNNmodel')
    ns = dict(Sequential=Sequential, Embedding=Embedding, Bidirectional=Bidirectional,
              LSTM=LSTM, TimeDistributed=TimeDistributed, Dense=Dense, Reshape=Reshape, input_len=LENGTH)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['getRNNmodel'](60, 15, vocab_size)

def encode(smiles, train):
    # Training-only vocabulary; reserve PAD=0 and UNK=1 for transfer to new data.
    vocab = {c: i + 2 for i, c in enumerate(sorted(set(''.join(smiles[i] for i in train))))}
    x = np.zeros((len(smiles), LENGTH), dtype=np.int32)
    for i, s in enumerate(smiles):
        seq = [vocab.get(c, 1) for c in s[:LENGTH]]
        x[i, :len(seq)] = seq
    return x, vocab

def fit_stage(x, y, ids, val, epochs, seed, folder, label):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    mu, sd = float(y[ids].mean()), float(y[ids].std())
    sd = max(sd, 1e-12)
    model = factory(int(x.max()) + 1)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=.001, epsilon=1e-7))
    kw = {}
    if val is not None:
        kw = dict(validation_data=(x[val],((y[val]-mu)/sd).astype('float32')),
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    h = model.fit(x[ids],((y[ids]-mu)/sd).astype('float32'),batch_size=128,epochs=epochs,verbose=0,**kw).history
    pd.DataFrame(h).assign(epoch=lambda d: np.arange(1,len(d)+1)).to_csv(folder/f'{label}_history.csv', index=False)
    return model, mu, sd, h

def aggregate():
    lock=(OUT/'.aggregate.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[]
    for target in ['Tg','FFV','Tc','Density','Rg']:
        files=sorted(OUT.glob(f'{target}/fold_*/oof.csv'))
        if len(files)!=5: continue
        d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        d.to_csv(OUT.parent/f'OOF_chen_rnn_{target}.csv',index=False)
        maes=d.groupby('fold').absolute_error.mean()
        rows.append(dict(baseline='chen_rnn',target=target,n=len(d),oof_mae=d.absolute_error.mean(),
                         fold_mae_mean=maes.mean(),fold_mae_sd=maes.std(ddof=1),
                         oof_rmse=np.sqrt(d.squared_error.mean()),oof_r2=r2_score(d.measured,d.predicted)))
    dest=OUT.parent/'chen_performance.csv'
    tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',nargs='+',default=['Tg','FFV','Tc','Density','Rg']);p.add_argument('--epochs',type=int,default=500)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    df=pd.read_csv(ART/'analysis_dataset.csv');folds=pd.read_csv(ART/'outer_folds.csv');inner=pd.read_csv(ART/'inner_splits.csv')
    smiles=df.SMILES.tolist()
    protocol=dict(source_repo='https://github.com/figotj/RNN-Tg',source_commit='6a4f61554d8a9ce42dd99468dace793f5e02afca',
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),factory='original getRNNmodel via AST',
                  architecture='length120 embedding15 BiLSTM60 x2 TimeDistributedDense30 flatten Dense1',
                  adaptations=['training-only char vocabulary PAD=0 UNK=1, post-truncate/post-pad length120',
                               'fold-local target standardization','shared grouped outer5 and inner training-only split',
                               'published optimal architecture fixed, inner MSE early stopping patience10; refit outer train selected epochs',
                               'FFV Tc Density Rg are regression transfers of original Tg method'],
                  epochs_max=a.epochs,batch_size=128,optimizer='Adam lr=.001 epsilon=1e-7',
                  keras=keras.__version__,backend='torch',torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  dataset_sha256=hashlib.sha256((ART/'analysis_dataset.csv').read_bytes()).hexdigest(),
                  outer_folds_sha256=hashlib.sha256((ART/'outer_folds.csv').read_bytes()).hexdigest(),
                  truncated_count=sum(len(s)>120 for s in smiles))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('START',protocol,flush=True)
    for target in a.targets:
        y=df[target].to_numpy(float);available=np.flatnonzero(np.isfinite(y))
        for fold in range(1,6):
            folder=OUT/target/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'metrics.json').exists(): continue
            start=time.time()
            test=folds.loc[(folds.target==target)&(folds.fold==fold),'row_id'].to_numpy(int)
            train=np.setdiff1d(available,test)
            fit=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='fit'),'row_id'].to_numpy(int)
            stop=inner.loc[(inner.target==target)&(inner.fold==fold)&(inner.role=='early_stop'),'row_id'].to_numpy(int)
            assert not set(train)&set(test) and set(fit)|set(stop)==set(train)
            assert not set(df.cv_group_id.iloc[train])&set(df.cv_group_id.iloc[test])
            x,vocab=encode(smiles,fit)
            _,_,_,h=fit_stage(x,y,fit,stop,a.epochs,42+fold,folder,'select')
            selected=int(np.argmin(h['val_loss']))+1
            x,vocab=encode(smiles,train)
            model,mu,sd,_=fit_stage(x,y,train,None,selected,1042+fold,folder,'refit')
            with torch.no_grad():
                pred=np.concatenate([keras.ops.convert_to_numpy(model(x[test[i:i+128]],training=False)).reshape(-1) for i in range(0,len(test),128)])*sd+mu
            err=pred-y[test]
            d=pd.DataFrame(dict(row_id=test,fold=fold,target=target,baseline='chen_rnn',measured=y[test],predicted=pred,
                                absolute_error=abs(err),squared_error=err**2,cv_group_id=df.cv_group_id.iloc[test].values))
            d.to_csv(folder/'oof.csv',index=False)
            m=dict(target=target,fold=fold,selected_epochs=selected,n_train=len(train),n_test=len(test),
                   mae=float(abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),r2=float(r2_score(y[test],pred)),
                   runtime_seconds=time.time()-start,vocab=vocab)
            (folder/'metrics.json').write_text(json.dumps(m,indent=2));aggregate()
            print('DONE',target,fold,'MAE',m['mae'],'epochs',selected,'seconds',round(m['runtime_seconds']),flush=True)
    aggregate()

if __name__=='__main__':main()
