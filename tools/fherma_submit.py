"""FHERMA operations for this implementation; credentials stay in CLI config.

Benchmark/entry use the same authenticated server actions as the website.
Action IDs are read afresh from its published client, not hard-coded.
Mutations are explicit subcommands and are never automatically retried.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess

import httpx
from fherma.api import Api
from fherma.config import load

ROOT=Path(__file__).resolve().parents[1]
KERNEL='polynomial-multiplication'
SLUG='ntt-cupqc-baseline'
OWNER='ilya-usmanov'
CHALLENGE='polynomial-multiplication-2025'
IMPL_ID='6aaabc211be2e96f7342bb92'
IMAGE='6a9833940171486f4faae443'
RUNNER='6aa6690fd4413d278e91947e'
BASE='https://www.fherma.io'
PAGE=f'/kernels/{KERNEL}/{OWNER}/{SLUG}'
API_PATH=f'/kernels/{KERNEL}/implementations/{SLUG}'

def server_action(page, name, args):
    profile=load()
    if not profile.token: raise RuntimeError('Authenticate with fherma auth login first')
    with httpx.Client(base_url=BASE,cookies={'fherma_session':profile.token},timeout=45) as client:
        r=client.get(page); r.raise_for_status()
        scripts=re.findall(r'<script[^>]*src="([^"]+)"',r.text)
        pattern=re.compile(r'createServerReference\)\("([a-f0-9]+)".{0,160}?"'+re.escape(name)+r'"\)')
        def find(src):
            if not src.startswith('/_next/static/'):
                return None
            res=client.get(src); res.raise_for_status()
            m=pattern.search(res.text)
            return m[1] if m else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            ids={x for x in pool.map(find,scripts) if x}
        if len(ids)!=1: raise RuntimeError(f'Cannot identify website action {name}: {len(ids)} matches')
        r=client.post(page,headers={'Next-Action':ids.pop(),'Origin':BASE,
                       'Content-Type':'text/plain;charset=UTF-8','Accept':'text/x-component'},
                       content=json.dumps(args))
        r.raise_for_status()
        local=ROOT/'local';local.mkdir(exist_ok=True)
        (local/f'{name}-response.txt').write_text(r.text)
        for line in r.text.splitlines():
            _,sep,payload=line.partition(':')
            if not sep: continue
            try: result=json.loads(payload)
            except ValueError: continue
            if isinstance(result,dict) and isinstance(result.get('ok'),bool):
                if not result['ok']: raise RuntimeError(result.get('message',str(result)))
                return result
        raise RuntimeError('Unrecognized action response. Check local response AND platform state before retrying.')

def main():
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('status')
    attach=commands.add_parser('attach');attach.add_argument('--repository',required=True)
    attach.add_argument('--commit',help='Defaults to current HEAD')
    bench=commands.add_parser('benchmark');bench.add_argument('--seeds',type=int,default=2)
    commands.add_parser('enter')
    run=commands.add_parser('run');run.add_argument('id')
    args=parser.parse_args()
    if args.command=='benchmark':
        if not 1<=args.seeds<=20: parser.error('Use 1..20 seeds per baseline run')
        print(json.dumps(server_action(PAGE,'runBenchmark',[KERNEL,SLUG,{
            'runners':[RUNNER],'image':IMAGE,'language':'cpp',
            'points':[{'point':{'N':32768,'W':868}}],'seeds':{'count':args.seeds}}]),indent=2))
        return
    if args.command=='enter':
        page=f'/kernels/{KERNEL}/challenges/{CHALLENGE}'
        print(json.dumps(server_action(page,'enterChallenge',[KERNEL,CHALLENGE,IMPL_ID]),indent=2));return
    api=Api(load())
    try:
        if args.command=='attach':
            commit=args.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if not re.fullmatch('[a-f0-9]{40}',commit): parser.error('Need an exact 40-character commit SHA')
            if not args.repository.startswith('https://github.com/ilyausmanov2015/'):
                parser.error('Expected a repository in the authorized GitHub account')
            payload={'artifacts':[{'kind':'repository','location':args.repository,'digest':commit,'path':''}],
                     'language':'cpp','image':IMAGE,'lifecycle_status':'development'}
            result=api._request('PATCH',API_PATH,json=payload)
            print(json.dumps({k:result.get(k) for k in ['id','slug','language','image','artifacts']},indent=2))
        elif args.command=='run':
            if not re.fullmatch('[a-f0-9]{24}',args.id): parser.error('Invalid run ID')
            result=api._request('GET','/runs/'+args.id)
            out=ROOT/'local'/('run-'+args.id+'.json');out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2))
        else:
            result=api._request('GET',API_PATH)
            print(json.dumps({k:result.get(k) for k in ['id','slug','language','image','artifacts',
                 'measured','verification_status','lifecycle_status','visibility']},indent=2))
            print(BASE+PAGE)
    finally: api.close()

if __name__=='__main__': main()
