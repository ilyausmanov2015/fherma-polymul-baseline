"""Watch our runs without dumping credentials or full case payloads."""
import argparse
import json
from pathlib import Path
import time

from fherma.api import Api
from fherma.config import load
from fherma_submit import history

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit')
    parser.add_argument('--run')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    last = None
    api = Api(load())
    try:
        while True:
            if args.run:
                ids = [args.run]
            else:
                entries = history()
                if args.commit:
                    entries = [e for e in entries if e['submission']['commit'].startswith(args.commit)]
                ids = [r['id'] for e in entries for r in e.get('runs', [])]
                ids = ids[:1]
                if not ids:
                    line = json.dumps({'waiting_jobs': [{k:j.get(k) for k in ['id','status','created_at']}
                        for e in entries for j in e.get('jobs',[])]})
                    if line != last:
                        print(line, flush=True)
                        last = line
            done = False
            for rid in ids:
                result = api._request('GET', '/runs/' + rid)
                (ROOT / 'local' / ('run-' + rid + '.json')).write_text(json.dumps(result, indent=2))
                run = result.get('run', result)
                summary = {k: run.get(k) for k in ['id','status','progress','cases','passed','failure']}
                summary['points'] = [{k:p.get(k) for k in ['point','passed','seeds','seconds']} for p in run.get('points',[])]
                summary['log_tail'] = str(run.get('build_log', ''))[-3500:]
                line = json.dumps(summary, ensure_ascii=False)
                if line != last:
                    print(line, flush=True)
                    last = line
                done = run.get('status') in ['finished','failed','cancelled','error']
            if not args.wait or done:
                return
            time.sleep(20)
    finally:
        api.close()

if __name__ == '__main__':
    main()
