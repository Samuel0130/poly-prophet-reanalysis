#!/usr/bin/env python3
"""Download and verify all fork inputs plus the original inference model output.
Run .venv/bin/python download_kaggle_inputs.py [--verify-only].
Uses the official Kaggle SDK. Dataset version numbers are not source IDs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kaggle_inputs'
SOURCES = [
 (12846515, 'wliilamsam/m61-1315-0-006022-0-027956-0-029068-1-491555', 1, 70654201),
 (12946886, 'wliilamsam/molfold', 1, 375055277),
 (12947898, 'wliilamsam/mordred-clean', 1, 42711490),
 (12237259, 'negoto/mordred-1-2-0-py3-none-any', 2, 2201943),
 (12189904, 'senkin13/rdkit-2025-3-3-cp311', 1, 34881186),
 (12971223, 'bfladderbean/cnnlstm-bamod4c-26793f-output', 1, 201640185),
]
COMPETITION = 'neurips-open-polymer-prediction-2025'
COMP_FILES = {'sample_submission.csv':87, 'test.csv':263, 'train.csv':691254,
 'train_supplement/dataset1.csv':37682, 'train_supplement/dataset2.csv':463888,
 'train_supplement/dataset3.csv':3621, 'train_supplement/dataset4.csv':67188}

def inventory(directory):
 files=[]
 for p in sorted(directory.rglob('*')):
  if not p.is_file() or p.suffix == '.zip': continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
  files.append({'name':p.relative_to(directory).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 return files

def retry(fn,label):
 for attempt in range(1,4):
  try: return fn()
  except Exception as e:
   print(f'{label}: attempt {attempt}/3 failed ({type(e).__name__})',flush=True)
   if attempt==3: raise
   time.sleep(2)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
 OUT.mkdir(exist_ok=True);api=None
 if not args.verify_only:
  import requests
  send=requests.Session.send
  def bounded_send(self,request,**kwargs):
   kwargs['timeout']=(15,90);return send(self,request,**kwargs)
  requests.Session.send=bounded_send
  from kaggle.api.kaggle_api_extended import KaggleApi
  api=KaggleApi();retry(api.authenticate,'authenticate')
 manifest=[];cd=OUT/COMPETITION;cd.mkdir(exist_ok=True)
 for name,size in COMP_FILES.items():
  f=cd/name
  if (not f.exists() or f.stat().st_size!=size) and api:
   # SDK saves the URL basename; give it the original relative parent.
   f.parent.mkdir(parents=True,exist_ok=True)
   retry(lambda:api.competition_download_file(COMPETITION,name,path=str(f.parent),quiet=True,force=True),name)
   import zipfile
   for archive in f.parent.glob('*.zip'):
    with zipfile.ZipFile(archive) as z: z.extractall(f.parent)
 files=inventory(cd);found={f['name']:f['bytes'] for f in files};ok=found==COMP_FILES
 manifest.append({'sourceType':'competition','sourceId':74608,'ref':COMPETITION,
  'scope':['fork','original'],'files':files,'expectedBytes':sum(COMP_FILES.values()),
  'status':'verified' if ok else 'incomplete','verification':'exact official names/sizes and local SHA256'})
 for sid,ref,version,expected in SOURCES:
  dest=OUT/ref.split('/')[-1];dest.mkdir(exist_ok=True);files=inventory(dest);error=None
  if sum(f['bytes'] for f in files)!=expected and api:
   try:
    retry(lambda:api.dataset_download_files(f'{ref}/{version}',path=str(dest),unzip=True,quiet=True,force=True),ref)
    files=inventory(dest)
   except Exception as e: error=type(e).__name__
  ok=sum(f['bytes'] for f in files)==expected
  entry={'sourceType':'datasetVersion','sourceId':sid,'ref':ref,'versionNumber':version,
   'scope':['original'] if sid==12971223 else ['fork','original'],
   'expectedBytes':expected,'actualBytes':sum(f['bytes'] for f in files),
   'status':'verified' if ok else 'incomplete','files':files,
   'verification':'official uncompressed total bytes and local SHA256'}
  if error: entry['errorType']=error
  manifest.append(entry);print(ref,entry['status'],entry['actualBytes'],'/',expected,flush=True)
 result={'createdUTC':datetime.now(timezone.utc).isoformat(),
  'originalNotebook':'cnnlstm-bamod4c-molfold-aug-infer.ipynb',
  'forkNotebook':'fork-of-neurips-open-polymer-prediction.ipynb','sources':manifest}
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 rows=['# Kaggle input 下载核验','','| Input | 范围 | 状态 | 文件数 | 解压字节数 |','|---|---|---|---:|---:|']
 for m in manifest:
  rows.append(f"| {m['ref']} | {', '.join(m['scope'])} | {m['status']} | {len(m['files'])} | {sum(f['bytes'] for f in m['files']):,} |")
 rows+=['','manifest.json 保存版本、文件名、大小及本地 SHA256。SHA256 用于后续复验，Kaggle 没有提供远端 SHA256。',
  '竞赛核验逐文件官方大小；各 dataset 核验官方解压总字节数。模型、描述符和安装包不计为训练聚合物样本。']
 (OUT/'下载核验.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
 return 0 if all(m['status']=='verified' for m in manifest) else 1
if __name__=='__main__': raise SystemExit(main())
