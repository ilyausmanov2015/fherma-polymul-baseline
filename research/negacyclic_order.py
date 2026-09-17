"""Check CT/GS odd-root ordering against direct evaluation and convolution."""
import random

def reverse(x, bits):
    return int(f"{x:0{bits}b}"[::-1], 2)

def transform(a, psi, p, inverse=False):
    a=a.copy();n=len(a);bits=n.bit_length()-1
    stages=range(bits) if inverse else range(bits-1,-1,-1)
    for level in stages:
        half=1<<level;m=n//(2*half)
        for group in range(m):
            w=pow(psi,reverse(m+group,bits),p)
            if inverse: w=pow(w,-1,p)
            for j in range(half):
                i=2*half*group+j;u,v=a[i],a[i+half]
                if inverse: a[i],a[i+half]=(u+v)%p,(u-v)*w%p
                else: a[i],a[i+half]=(u+w*v)%p,(u-w*v)%p
    if inverse: a=[x*pow(n,-1,p)%p for x in a]
    return a

rng=random.Random(48113);p=65537;checked=0
for n in [2,4,8,16,32,128,1024]:
    psi=pow(3,(p-1)//(2*n),p);assert pow(psi,n,p)==p-1
    for trial in range(3):
        a=[rng.randrange(p) for _ in range(n)];b=[rng.randrange(p) for _ in range(n)]
        A=transform(a,psi,p);B=transform(b,psi,p)
        assert transform(A,psi,p,True)==a
        if n<=32:
            for k in range(n):
                root=pow(psi,2*reverse(k,n.bit_length()-1)+1,p)
                assert A[k]==sum(x*pow(root,j,p) for j,x in enumerate(a))%p
        product=transform([x*y%p for x,y in zip(A,B)],psi,p,True)
        exact=[0]*n
        for i,x in enumerate(a):
            for j,y in enumerate(b): exact[(i+j)%n]+=x*y*(1 if i+j<n else -1)
        assert product==[x%p for x in exact];checked+=1
print(f"{checked} direct negacyclic products, inverse identities and small direct DFT checks passed")
