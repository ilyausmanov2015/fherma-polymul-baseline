"""Collect reproducible summaries from downloaded official run responses."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for path in (ROOT/'local').glob('run-*.json'):
    payload=json.loads(path.read_text())
    run=payload.get('run',payload)
    if run.get('status') not in ['finished','failed','error','aborted','expired']:
        continue
    row={k:run.get(k) for k in ['id','status','cases','passed','failure','init_s','started_at','finished_at']}
    row['commit']=payload.get('submission',{}).get('commit')
    row['image']=payload.get('submission',{}).get('image')
    row['runner']=run.get('runner',{}).get('name')
    row['points']=[{k:p.get(k) for k in ['point','passed','seeds','seconds']} for p in run.get('points',[])]
    row['case_errors']=sorted({c['note'] for p in run.get('points',[]) for c in p.get('cases',[]) if c.get('note') and not c.get('passed')})
    row['profiles_us']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
        for line in str(run.get('build_log','')).splitlines() if 'PROFILE_US' in line]
    row['host_us']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
        for line in str(run.get('build_log','')).splitlines() if 'HOST_US' in line]
    row['host_graph_us']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
        for line in str(run.get('build_log','')).splitlines() if 'HOST_GRAPH_US' in line]
    row['host_alloc_us']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
        for line in str(run.get('build_log','')).splitlines() if 'HOST_ALLOC_US' in line]
    row['host_staged_us']=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
        for line in str(run.get('build_log','')).splitlines() if 'HOST_STAGED_US' in line]
    for marker,key in [('MAPPED_GPU_US','mapped_gpu_us'),('ASYNC_ALLOC_US','async_alloc_us')]:
        profiles=[{k:float(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',line)}
            for line in str(run.get('build_log','')).splitlines() if marker in line]
        if profiles: row[key]=profiles
    rows.append(row)
rows.sort(key=lambda r:r.get('started_at') or '')
(ROOT/'results'/'experiments.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
    print(r['commit'][:7],r['id'],r['status'],f"{r['passed']}/{r['cases']}",
          [(p.get('seconds') or {}).get('median') for p in r['points']],r['profiles_us'])
