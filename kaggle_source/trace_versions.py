from pathlib import Path
import json,requests,concurrent.futures
send=requests.Session.send
requests.Session.send=lambda self,req,**kw:send(self,req,**dict(kw,timeout=(12,45)))
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
root=Path(__file__).resolve().parent
token=(Path.home()/'.kaggle/access_token').read_text().strip()
def fetch(task):
 slug,version=task;r=ApiGetKernelRequest();r.user_name='bfladderbean';r.kernel_slug=slug
 if version:r.version_label=version
 try:
  c=KernelsApiClient(KaggleHttpClient(api_token=token));obj=json.loads(str(c.get_kernel(r)))
  p=root/'original_training'/(slug+('-v'+version if version else ''));p.mkdir(parents=True,exist_ok=True)
  (p/'pull-response.json').write_text(json.dumps(obj))
  (p/(slug+'.ipynb')).write_text(obj['blob']['source'])
  return {'slug':slug,'version':version,'status':'downloaded','metadata':obj['metadata']}
 except Exception as e:return {'slug':slug,'version':version,'status':'unavailable','errorType':type(e).__name__}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(fetch,[('neurips-open-polymer-prediction','1'),('cnnlstm-bamod4c',''),('cnnlstm-bamod4c-26793f','')]))
(root/'original-provenance-lookup.json').write_text(json.dumps(results,indent=2))
for r in results:print(r,flush=True)
