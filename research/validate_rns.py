"""Check the proposed CRT bounds, integer quotient estimate and limb folding.
This is a mathematical prototype, not a GPU implementation or speed benchmark.
"""
import json
from pathlib import Path
import random
import gmpy2

rng=random.Random(731191)
N=32768
C=11993087
Q=(1<<868)-C
primes=[]
candidate=(1<<31)-65535
while len(primes)<57:
    if gmpy2.is_prime(candidate): primes.append(candidate)
    candidate-=65536
P=1
for p in primes: P*=p
bound=N*(Q-1)**2
assert P>2*bound and (57*P).bit_length()<=1792
assert min(primes)>(1<<32)//3
bases=[P//p for p in primes]
inverses=[pow(m,-1,p) for m,p in zip(bases,primes)]
reciprocals=[(1<<64)//p for p in primes]
MASK=(1<<32)-1

def fold_q(value,c):
    words=[(value>>(32*i))&MASK for i in range(56)]+[0]
    folded=[];carry=0
    for k in range(29):
        high=(words[k+27]>>4)|((words[k+28]<<28)&MASK)
        low=words[k] if k<27 else (words[k]&15 if k==27 else 0)
        v=high*c+low+carry
        folded.append(v&MASK);carry=v>>32
    assert carry==0
    high0=(folded[27]>>4)|((folded[28]<<28)&MASK)
    high1=folded[28]>>4
    folded[27]&=15
    carry=0
    for k in range(28):
        addition=high0*c if k==0 else high1*c if k==1 else 0
        v=folded[k]+addition+carry
        folded[k]=v&MASK;carry=v>>32
    assert carry==0
    out=sum(v<<(32*k) for k,v in enumerate(folded[:28]))
    q=(1<<868)-c
    assert out<2*q
    return out-q if out>=q else out

crt_cases=0
edges=[0,1,-1,Q-1,Q,Q+1,-Q,-Q-1,bound,-bound,P//2,-(P//2)]
for value in edges+[rng.randrange(-bound,bound+1) for _ in range(10000)]:
    t=[(value%p)*inv%p for p,inv in zip(primes,inverses)]
    S=sum(ti*m for ti,m in zip(t,bases))
    alpha0=sum(ti*recip for ti,recip in zip(t,reciprocals))>>64
    assert S//P-alpha0 in (0,1)
    canonical=S-alpha0*P
    if canonical>=P: canonical-=P
    assert canonical==value%P
    reduced=fold_q(canonical,C)
    if canonical>P//2: reduced=(reduced-P%Q)%Q
    assert reduced==value%Q
    crt_cases+=1
shoup_cases=0
for p in primes:
    for _ in range(2000):
        a=rng.randrange(p);w=rng.randrange(p)
        reciprocal=(w<<32)//p
        approximate=(a*reciprocal)>>32
        r=(a*w-approximate*p)&MASK
        assert r<2*p
        if r>=p: r-=p
        assert r==a*w%p
        shoup_cases+=1
fold_cases=0
for c in [1,C,(1<<28)-1]:
    q=(1<<868)-c
    for v in [0,1,q-1,q,q*q-1,P-1]+[rng.randrange(P) for _ in range(2000)]:
        assert fold_q(v,c)==v%q
        fold_cases+=1
report={'kind':'rns-mathematical-prototype','prime_count':len(primes),'prime_min':min(primes),
        'prime_max':max(primes),'product_bits':P.bit_length(),'required_bound_bits':(2*bound).bit_length(),
        'sum_bits_bound':(57*P).bit_length(),'crt_cases':crt_cases,'shoup_cases':shoup_cases,
        'fold_cases':fold_cases,'passed':True}
Path(__file__).resolve().parents[1].joinpath('results/rns-arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
