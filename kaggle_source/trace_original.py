"""Read-only provenance lookup using Kaggle SDK; no credential output."""
from pathlib import Path
import json,time,requests
send=requests.Session.send
requests.Session.send=lambda self,req,**kw:send(self,req,**dict(kw,timeout=(12,45)))
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest,ApiGetKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelsListViewType
root=Path(__file__).resolve().parent
client=KernelsApiClient(KaggleHttpClient(api_token=(Path.home()/'.kaggle/access_token').read_text().strip()))
r=ApiListKernelsRequest();r.user='bfladderbean';r.page_size=100;r.group=KernelsListViewType.PROFILE
for attempt in range(3):
 try:
  data=json.loads(str(client.list_kernels(r)));break
 except Exception as e:
  print('list failed',type(e).__name__,flush=True)
  if attempt==2: raise
(root/'bfladderbean-kernels.json').write_text(json.dumps(data,indent=2))
print('kernel list',str(data)[:9000],flush=True)
for k in data.get('kernels',[]):
 ref=k.get('ref','');title=k.get('title','')
 if 'cnnlstm' not in (ref+' '+title).lower(): continue
 req=ApiGetKernelRequest();req.user_name=ref.split('/')[0];req.kernel_slug=ref.split('/')[1]
 try:
  result=client.get_kernel(req);obj=json.loads(str(result))
  dest=root/'original_training'/req.kernel_slug;dest.mkdir(parents=True,exist_ok=True)
  (dest/'pull-response.json').write_text(json.dumps(obj,ensure_ascii=False))
  print('pulled',ref,'keys',list(obj),flush=True)
  # API uses blob.source in its pull response.
  blob=obj.get('blob',{})
  if isinstance(blob,dict):
   code=blob.get('source') or blob.get('text')
   if code: (dest/(req.kernel_slug+'.ipynb')).write_text(code,encoding='utf-8')
 except Exception as e: print('pull failed',ref,type(e).__name__,flush=True)
