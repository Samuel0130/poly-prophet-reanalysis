"""Prepare immutable data, folds and frozen original-inference MolFormer features."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from runtime import ROOT,ORIGINAL,seed_everything

OUT=ROOT/'training/artifacts';OUT.mkdir(exist_ok=True)

def main():
 seed_everything(42)
 df=pd.read_csv(ROOT/'table1_rebuilt/merged_labeled_polymers.csv')
 # Group specified stereoisomers together as a conservative protection; does not
 # claim to resolve all repeat-unit shifts or different repeat-unit multiples.
 group_smiles=[Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in df.SMILES]
 df['cv_group_id']=[hashlib.sha256(s.encode()).hexdigest() for s in group_smiles]
 df.to_csv(OUT/'analysis_dataset.csv',index=False)
 folds=[];inner=[]
 for target in ['Tg','FFV','Tc','Density','Rg']:
  ids=np.flatnonzero(df[target].notna().to_numpy());sub=df.iloc[ids]
  bins=pd.qcut(sub[target],q=10,labels=False,duplicates='drop').to_numpy()
  splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
  for fold,(tr,te) in enumerate(splitter.split(ids,bins,sub.cv_group_id),1):
   trids=ids[tr];teids=ids[te]
   assert not set(df.iloc[trids].cv_group_id)&set(df.iloc[teids].cv_group_id)
   for i in teids:folds.append({'target':target,'fold':fold,'row_id':int(i),'polymer_group_id':df.iloc[i].polymer_group_id,'cv_group_id':df.iloc[i].cv_group_id})
   gs=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=4200+fold)
   fit,valid=next(gs.split(trids,groups=df.iloc[trids].cv_group_id))
   for role,indices in [('fit',trids[fit]),('early_stop',trids[valid])]:
    for i in indices:inner.append({'target':target,'fold':fold,'role':role,'row_id':int(i)})
  print('folds',target,len(ids),'groups',sub.cv_group_id.nunique(),flush=True)
 pd.DataFrame(folds).to_csv(OUT/'outer_folds.csv',index=False)
 pd.DataFrame(inner).to_csv(OUT/'inner_splits.csv',index=False)
 backbone=[]
 for s in df.SMILES:
  d=ORIGINAL['get_backbone_sidechain_features'](s)
  backbone.append([d[k] for k in ORIGINAL['BACKBONE_SIDECHAIN_FEATURE_NAMES']])
 desc=np.array(backbone,dtype=np.float32)
 assert np.isfinite(desc).all();np.save(OUT/'descriptors.npy',desc)
 # Generate three reproducible equivalent random SMILES per polymer, reused
 # across folds and variants, and cache corresponding frozen embeddings.
 if (OUT/'augmentation_bank.json').exists():
  bank=json.loads((OUT/'augmentation_bank.json').read_text())
 else:
  bank=[];rejected=[]
  for row_id,s in enumerate(df.SMILES):
   mol=Chem.MolFromSmiles(s);variants=[s]
   for _ in range(3):
    randomized=s
    for attempt in range(10):
     candidate=Chem.MolToSmiles(Chem.RenumberAtoms(mol,np.random.permutation(mol.GetNumAtoms()).tolist()),canonical=False)
     parsed=Chem.MolFromSmiles(candidate)
     if parsed is not None and Chem.MolToSmiles(parsed,canonical=True)==s:
      randomized=candidate;break
     rejected.append({'row_id':row_id,'original':s,'candidate':candidate,'reason':'canonical stereochemical identity mismatch'})
    variants.append(randomized)
   bank.append(variants)
  (OUT/'augmentation_bank.json').write_text(json.dumps(bank))
  pd.DataFrame(rejected,columns=['row_id','original','candidate','reason']).to_csv(OUT/'rejected_augmentations.csv',index=False)
 vocab=ORIGINAL['create_exact_training_vocab']()
 maxlen=max(len(s) for row in bank for s in row)+20
 tokens=np.zeros((len(bank),4,maxlen),dtype=np.int64)
 for i,row in enumerate(bank):
  for j,s in enumerate(row):tokens[i,j,:len(s)]=[vocab.get(c,vocab['!']) for c in s]
 np.save(OUT/'tokens.npy',tokens)
 (OUT/'tokenization.json').write_text(json.dumps({'vocab':vocab,'max_length':maxlen,'augmentation_bank_size':3}))
 print('prepared tokens',tokens.shape,'frozen MolFormer features next',flush=True)
 import torch
 from transformers import AutoModel,AutoTokenizer
 path=ROOT/'kaggle_inputs/molfold/molformer-xl-local'
 tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
 model=AutoModel.from_pretrained(path,trust_remote_code=True,local_files_only=True).cuda().eval()
 # Local model defaults to re-sampling its attention random-feature map even in
 # eval. Fix that map for repeatable cached features; do not use label information.
 for module in model.modules():
  if hasattr(module,'deterministic'):module.deterministic=True
 model.config.deterministic_eval=True
 import transformers
 metadata={'feature_definition':'original inference: (CLS + mean(hidden[1:]))/2, first 64 channels',
  'frozen_pretrained_model':True,'max_molformer_tokens':256,'padding':'max_length',
  'deterministic_eval':True,'augmentation':'3 RDKit atom permutations per structure; p=0.3 only in training',
  'torch':torch.__version__,'transformers':transformers.__version__,
  'feature_fallback_allowed':False,'cv_grouping':'canonical non-isomeric SMILES; does not resolve all repeat-unit equivalences'}
 (OUT/'feature_metadata.json').write_text(json.dumps(metadata,indent=2))
 cache_path=OUT/'molformer.npy';progress_path=OUT/'feature_progress.json'
 shape=(len(bank)*4,64)
 if cache_path.exists():
  cache=np.lib.format.open_memmap(cache_path,mode='r+');start=json.loads(progress_path.read_text())['completed_rows']
 else:
  cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=shape);start=0
 flat=[s for row in bank for s in row]
 began=time.time()
 for lo in range(start,len(flat),64):
  hi=min(lo+64,len(flat))
  encoded=tokenizer(flat[lo:hi],max_length=256,padding='max_length',truncation=True,return_tensors='pt')
  with torch.no_grad():
   y=model(**{k:v.cuda() for k,v in encoded.items()}).last_hidden_state
   emb=((y[:,0,:]+y[:,1:,:].mean(dim=1))/2)[:,:64].float().cpu().numpy()
  if not np.isfinite(emb).all():raise ValueError('Nonfinite real MolFormer features; no fallback allowed')
  cache[lo:hi]=emb;cache.flush()
  progress_path.write_text(json.dumps({'completed_rows':hi,'total_rows':len(flat)}))
  if lo%640==0 or hi==len(flat):print('MolFormer',hi,'/',len(flat),'elapsed',round(time.time()-began,1),'s',flush=True)
 del cache
 print('Preparation complete',flush=True)
if __name__=='__main__':main()
