"""Run an explicit, finite list of already-pushed experiment commits in order.

Stops on any CLI/API error; never retries mutations or automatically promotes
an experiment to the challenge. Each run's correctness is reported separately.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from fherma.api import Api
from fherma.config import load

ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'finished','failed','cancelled','error','aborted','expired'}
parser=argparse.ArgumentParser()
parser.add_argument('commits',nargs='+')
parser.add_argument('--wait-run')
parser.add_argument('--seeds',type=int,default=3)
args=parser.parse_args()
if not 1<=args.seeds<=20: parser.error('seeds must be 1..20')
commits=[]
for ref in args.commits:
    if not re.fullmatch('[a-f0-9]{7,40}',ref): parser.error('Use explicit commit hashes')
    commits.append(subprocess.check_output(['git','rev-parse',ref+'^{commit}'],cwd=ROOT,text=True).strip())
if args.wait_run:
    if not re.fullmatch('[a-f0-9]{24}',args.wait_run): parser.error('Invalid run ID')
    api=Api(load())
    try:
        while True:
            result=api._request('GET','/runs/'+args.wait_run)
            status=result['run']['status']
            print(json.dumps({'waiting_for':args.wait_run,'status':status}),flush=True)
            if status in TERMINAL: break
            time.sleep(20)
    finally: api.close()

def cli(tool,*arguments):
    subprocess.run([sys.executable,str(ROOT/'tools'/tool),*arguments],cwd=ROOT,check=True)

for commit in commits:
    print(json.dumps({'experiment_commit':commit,'seeds':args.seeds}),flush=True)
    cli('fherma_submit.py','attach','--repository','https://github.com/ilyausmanov2015/fherma-polymul-baseline','--commit',commit)
    cli('fherma_submit.py','benchmark','--seeds',str(args.seeds))
    cli('watch_runs.py','--commit',commit,'--wait')
    reports=[]
    for path in (ROOT/'local').glob('run-*.json'):
        report=json.loads(path.read_text())
        if report.get('submission',{}).get('commit')==commit: reports.append(report['run'])
    latest=max(reports,key=lambda r:r.get('started_at') or '')
    print(json.dumps({'completed_commit':commit,**{k:latest.get(k) for k in ['id','status','cases','passed','failure']},'points':[{k:p.get(k) for k in ['point','passed','seeds','seconds']} for p in latest.get('points',[])]}),flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools'/'collect_results.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
