"""Independent exact oracle: integer multiplication via Kronecker substitution.

The oracle uses neither NTT nor cuPQC. Each base-B digit holds a whole ordinary
convolution coefficient, with B > N*(q-1)**2. We then fold X**N = -1.
Local validation is not the official FHERMA verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import gmpy2

ROOT = Path(__file__).resolve().parents[1]

def modulus(n, w):
    q = (1 << w) - 2*n + 1
    while not gmpy2.is_prime(q, 40):
        q -= 2*n
    return q

def encode(a, width):
    return b''.join(int(x).to_bytes(width, 'little') for x in a)

def oracle(a, b, q):
    n = len(a)
    width = ((n*(q-1)**2).bit_length()+7)//8
    x = gmpy2.mpz.from_bytes(encode(a, width), 'little')
    y = gmpy2.mpz.from_bytes(encode(b, width), 'little')
    raw = (x*y).to_bytes(2*n*width, 'little')
    return [(int.from_bytes(raw[k*width:(k+1)*width], 'little') -
             int.from_bytes(raw[(k+n)*width:(k+n+1)*width], 'little')) % q
            for k in range(n)]

def schoolbook(a, b, q):
    n=len(a); c=[0]*n
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            c[(i+j)%n] += x*y*(1 if i+j<n else -1)
    return [x%q for x in c]

def cases(n, q):
    rng = random.Random(20260916+n)
    a = [rng.randrange(q) for _ in range(n)]
    b = [rng.randrange(q) for _ in range(n)]
    yield 'dense-random-1', a, b
    yield 'dense-random-2', [rng.randrange(q) for _ in range(n)], [rng.randrange(q) for _ in range(n)]
    yield 'zero', a, [0]*n
    yield 'identity', a, [1]+[0]*(n-1)
    yield 'wrap-sign', [0]*(n-1)+[q-1], [0,q-1]+[0]*(n-2)
    yield 'all-q-minus-one', [q-1]*n, [q-1]*n
    yield 'alternating', [q-1 if i%2 else 0 for i in range(n)], a
    edge=[0,1,q-1,q-2,1<<864,(1<<864)-1,1<<32,(1<<32)-1]
    yield 'limb-boundaries', [edge[i%len(edge)] for i in range(n)], b
    yield 'random-after-edges', b, a  # Reuse the same initialized state.

def case_directory(n,delta=None):
    return ROOT/'local'/(f'n{n}-w868'+(f'-c{delta}' if delta is not None else ''))

def make(n,delta=None):
    w=868; limbs=28; q=modulus(n,w) if delta is None else (1<<w)-delta
    work=case_directory(n,delta)
    (work/'point').mkdir(parents=True, exist_ok=True)
    (work/'point'/'q.bin').write_bytes(q.to_bytes(limbs*4,'little'))
    labels=[]
    for idx,(name,a,b) in enumerate(cases(n,q)):
        start=time.monotonic(); d=work/'cases'/f'{idx:06d}'; d.mkdir(parents=True,exist_ok=True)
        (d/'a.bin').write_bytes(encode(a,112)); (d/'b.bin').write_bytes(encode(b,112))
        c=oracle(a,b,q)
        if n<=32: assert c==schoolbook(a,b,q), 'independent oracle disagreement'
        expected=work/'expected'/f'{idx:06d}'; expected.mkdir(parents=True,exist_ok=True)
        (expected/'c.bin').write_bytes(encode(c,112)); labels.append(name)
        print(f'make N={n} {name}: {time.monotonic()-start:.3f}s',flush=True)
    (work/'manifest.json').write_text(json.dumps({'point':{'N':n,'W':w,'L':limbs},'cases':len(labels)}))
    (work/'labels.json').write_text(json.dumps(labels))
    return work

def verify(work,source):
    labels=json.loads((work/'labels.json').read_text())
    results=json.loads((work/'out'/'results.json').read_text())
    verdicts=[]
    for idx,label in enumerate(labels):
        output=work/'out'/f'{idx:06d}'/'c.bin'; expected=work/'expected'/f'{idx:06d}'/'c.bin'
        result=results['cases'][idx]
        passed=result['status']=='ok' and output.exists() and output.read_bytes()==expected.read_bytes()
        verdicts.append({'case':label,'passed':passed, 'seconds':result.get('seconds')})
        print(label, 'PASS' if passed else 'FAIL', result,flush=True)
    report={'kind':'local-independent-oracle','point':json.loads((work/'manifest.json').read_text())['point'],
            'q_delta':(1<<868)-int.from_bytes((work/'point'/'q.bin').read_bytes(),'little'),
            'passed':all(v['passed'] for v in verdicts),'verdicts':verdicts,
            'source':str(source.relative_to(ROOT)),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'median_seconds':statistics.median(v['seconds'] for v in verdicts if v['seconds'] is not None),
            'init_seconds':results['init_s']}
    (work/'validation.json').write_text(json.dumps(report,indent=2))
    assert report['passed'], 'exact verification failed'
    print(json.dumps({k:v for k,v in report.items() if k!='verdicts'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=32)
    p.add_argument('--binary',type=Path); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--existing',action='store_true',help='reuse existing generated cases')
    p.add_argument('--source',default='solve.cu',help='source used to build the checked binary')
    p.add_argument('--delta',type=int,help='test q=2^868-delta, including non-prime boundary moduli')
    args=p.parse_args()
    if args.delta is not None and not 0<args.delta<(1<<28): p.error('delta must be in (0,2^28)')
    work=case_directory(args.n,args.delta)
    if not args.verify_only:
        if not args.existing: work=make(args.n,args.delta)
        if args.binary: subprocess.run([str(args.binary.resolve()),str(work)],check=True,timeout=600)
    if args.verify_only or args.binary: verify(work,(ROOT/args.source).resolve())
