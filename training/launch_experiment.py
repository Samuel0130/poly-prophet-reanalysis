"""Persistent supervisor: two disjoint training workers; auto report all results."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'training/results';OUT.mkdir(exist_ok=True)

def write_status(status):
 p=OUT/'supervisor_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,p)

def run():
 lock=(ROOT/'training/experiment.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Experiment already running')
 (OUT/'supervisor_pid.txt').write_text(str(os.getpid()))
 status={'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),'state':'training',
  'worker_groups':{'small_targets':['Tg','Tc','Density','Rg'],'FFV':['FFV']},
  'reason_for_parallel_restart':'same protocol/seeds; resume checkpoints; move largest target onto a concurrent GPU worker'}
 workers={};logs=[]
 for name,targets in status['worker_groups'].items():
  path=ROOT/'training'/('train.log' if name=='small_targets' else 'train_FFV.log')
  log=path.open('a',buffering=1);logs.append(log)
  command=[sys.executable,'-u',str(ROOT/'training/run_ablation.py'),'--targets',*targets]
  workers[name]=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 status['worker_pids']={name:p.pid for name,p in workers.items()};write_status(status)
 (OUT/'training_pid.txt').write_text(str(workers['small_targets'].pid))
 while any(p.poll() is None for p in workers.values()):
  time.sleep(30)
  if list(OUT.glob('*/*/fold_5/metrics.json')):
   with (ROOT/'training/report.log').open('a') as report:
    subprocess.run([sys.executable,str(ROOT/'training/report.py')],cwd=ROOT,stdout=report,stderr=subprocess.STDOUT)
  status['worker_exit_codes']={name:p.poll() for name,p in workers.items()};write_status(status)
 codes={name:p.wait() for name,p in workers.items()}
 with (ROOT/'training/report.log').open('a') as report:
  reported=subprocess.run([sys.executable,str(ROOT/'training/report.py')],cwd=ROOT,stdout=report,stderr=subprocess.STDOUT)
 progress=json.loads((OUT/'progress.json').read_text())
 ok=all(c==0 for c in codes.values()) and progress['completed_models']==175 and reported.returncode==0
 status.update(state='complete' if ok else 'failed',worker_exit_codes=codes,report_exit_code=reported.returncode,
  completed_models=progress['completed_models'],finished=time.strftime('%Y-%m-%d %H:%M:%S'))
 write_status(status)
 return 0 if ok else 1
if __name__=='__main__':raise SystemExit(run())
