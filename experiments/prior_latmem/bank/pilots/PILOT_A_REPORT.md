> Committed copy of the run-generated report (out/run_2026-07-28/, --limit 100
> --synth-tests, seed 42, macOS arm64 devbox, 2026-07-29). Raw artifacts stay in
> the gitignored out dir. Orchestrator addendum at the end.

# Pilot A report

This report summarizes the staged-corpus Pilot A run.

## Registered questions

1. Band yield: 11 in-band pairs among 60 floor-eligible front pairs (0.183); 4 of 66 problems.
2. Dominated yield: 67 clean pairs among 171 evaluated front-vs-dominated pairs; 19 problems yielded one.
3. Cost: 120.224s total, 1.822s/problem. Projected seconds for 1,000 in-band pairs: 10929.5s.
4. Cleanup: 35 candidates had Z-silence hits, 0 sources failed parsing, and 81 candidates tripped a report-only style indicator.

## Synthesized workloads

66 of 100 problems produced measurement workloads. Generator failures: 0 (0.000); consensus failures: 15 (0.185); dissenting solutions dropped: 2; solutions too slow at scale: 8.

## Best in-band pairs (up to 10)

| Problem | Speed-side excerpt (≤15 lines) | Memory-side excerpt (≤15 lines) | Time ratio | Peak ratio |
|---|---|---|---:|---:|
| `1208_D. Restore Permutation` | <code>import sys<br>input = sys.stdin.readline<br><br>nn = 18<br>bit=[0]*(2**nn+1)<br> <br>def addbit(i, x):<br>    while i &lt;= 2**nn:<br>        bit[i] += x<br>        i += i &amp; (-i)<br> <br>def getsum(i):<br>    ret = 0<br>    while i != 0:<br>        ret += bit[i]</code> | <code>import sys<br>input=sys.stdin.readline<br>n=int(input())<br>s=list(map(int,input().split()))<br>BIT=[0]*(n+1)<br>def update(i,w):<br>    while i&lt;=n:<br>        BIT[i]+=w<br>        i+=(i&amp;-i)<br>def get_sum(i):<br>    res=0<br>    while i&gt;0:<br>        res+=BIT[i]<br>        i-=(i&amp;-i)<br>    return res</code> | 2.510× | 0.527× |
| `1136_D. Nastya Is Buying Lunch` | <code>from sys import stdin,stdout<br>from itertools import combinations<br>from collections import defaultdict<br>import math<br>import heapq<br><br>def listIn():<br>    return list((map(int,stdin.readline().strip().split())))<br><br>def stringListIn():<br>    return([x for x in stdin.readline().split()])<br>    <br>def intIn():<br>    return (int(stdin.readline()))<br></code> | <code>n,m=map(int,input().split())<br>l=[int(x) for x in input().split()]<br>pairs=[[] for i in range(n+1)]<br>ans=0<br>N=[0 for i in range(n+1)]<br>for i in range(m):<br>    a,b=map(int,input().split())<br>    pairs[b].append(a)<br>for i in range(n-1,-1,-1):<br>    if N[l[i]]==n-1-i-ans and i!=n-1:<br>        ans+=1<br>    else:<br>        for I in pairs[l[i]]:<br>            N[I]+=1<br>print(ans)</code> | 2.112× | 0.547× |
| `1136_D. Nastya Is Buying Lunch` | <code>import math<br>from collections import deque, defaultdict<br>from sys import stdin, stdout<br>input = stdin.readline<br># print = stdout.write<br>listin = lambda : list(map(int, input().split()))<br>mapin = lambda : map(int, input().split())<br>n, m = mapin()<br>a = listin()<br>s = set([])<br>for _ in range(m):<br>    s.add(tuple(mapin()))<br>z = [a.pop()]<br>count = 0<br>while a:</code> | <code>n,m=map(int,input().split())<br>l=[int(x) for x in input().split()]<br>pairs=[[] for i in range(n+1)]<br>ans=0<br>N=[0 for i in range(n+1)]<br>for i in range(m):<br>    a,b=map(int,input().split())<br>    pairs[b].append(a)<br>for i in range(n-1,-1,-1):<br>    if N[l[i]]==n-1-i-ans and i!=n-1:<br>        ans+=1<br>    else:<br>        for I in pairs[l[i]]:<br>            N[I]+=1<br>print(ans)</code> | 2.493× | 0.541× |
| `1208_D. Restore Permutation` | <code>def sumsegtree(l,seg,st,en,x):<br>    if st==en:<br>        seg[x]=l[st]<br>    else:<br>        mid=(st+en)&gt;&gt;1<br>        sumsegtree(l,seg,st,mid,2*x)<br>        sumsegtree(l,seg,mid+1,en,2*x+1)<br>        seg[x]=seg[2*x]+seg[2*x+1]<br> <br>def query(seg,st,en,val,x):<br>    if st==en:<br>        return seg[x]<br>    mid=(st+en)&gt;&gt;1<br>    if seg[2*x]&gt;=val:<br>        return query(seg,st,mid,val,2*x)</code> | <code>import sys<br>input=sys.stdin.readline<br>n=int(input())<br>s=list(map(int,input().split()))<br>BIT=[0]*(n+1)<br>def update(i,w):<br>    while i&lt;=n:<br>        BIT[i]+=w<br>        i+=(i&amp;-i)<br>def get_sum(i):<br>    res=0<br>    while i&gt;0:<br>        res+=BIT[i]<br>        i-=(i&amp;-i)<br>    return res</code> | 2.279× | 0.592× |
| `1208_D. Restore Permutation` | <code>from operator import add<br><br>class Stree:<br>    def __init__(self, f, n, default, init_data):<br>        self.ln = 2**(n-1).bit_length()<br>        self.data = [default] * (self.ln * 2)<br>        self.f = f<br>        for i, d in init_data.items():<br>            self.data[self.ln + i] = d<br>        for j in range(self.ln - 1, 0, -1):<br>            self.data[j] = f(self.data[j*2], self.data[j*2+1])<br><br>    def update(self, i, a):<br>        p = self.ln + i<br>        self.data[p] = a</code> | <code>import sys<br>input=sys.stdin.readline<br>n=int(input())<br>s=list(map(int,input().split()))<br>BIT=[0]*(n+1)<br>def update(i,w):<br>    while i&lt;=n:<br>        BIT[i]+=w<br>        i+=(i&amp;-i)<br>def get_sum(i):<br>    res=0<br>    while i&gt;0:<br>        res+=BIT[i]<br>        i-=(i&amp;-i)<br>    return res</code> | 3.019× | 0.511× |
| `1012_B. Chemical table` | <code>import sys<br>n,m,q=map(int,input().split())<br>p=[-1]*(n+m)<br>r=[0]*(n+m)<br>def par(i):<br> if p[i]==-1: return i<br> p[i]=par(p[i])<br> return p[i]<br>def merge(a,b):<br> a,b=par(a),par(b)<br> if a==b: return 0<br> if r[a]&lt;r[b]:p[a]=b<br> elif r[b]&lt;r[a]:p[b]=a<br> else:p[a]=b;r[b]+=1<br> return 1</code> | <code>n,m,q = map(int,input().split())<br>f = [-1]*(n+m+1)<br>def find(x):<br>    if f[x]==-1:<br>        return x<br>    else:<br>        f[x] = find(f[x])<br>        return f[x]<br>ans = n+m-1<br>for i in range(q):<br>    r,c = map(int,input().split())<br>    c+=n<br>    R = find(r)<br>    C = find(c)<br>    if R!=C :</code> | 1.877× | 0.299× |
| `1208_D. Restore Permutation` | <code># 1208D<br>class segTree():<br>    def __init__(self, n):<br>        self.t = [0] * (n &lt;&lt; 2)<br><br>    def update(self, node, l, r, index, value):<br>        if l == r:<br>            self.t[node] = value<br>            return<br>        mid = (l + r) &gt;&gt; 1<br>        if index &lt;= mid:<br>            self.update(node*2, l, mid, index, value)<br>        else:<br>            self.update(node*2 + 1, mid + 1, r, index, value)<br>        self.t[node] = self.t[node*2] + self.t[node*2 + 1]</code> | <code>import sys<br>input=sys.stdin.readline<br>n=int(input())<br>s=list(map(int,input().split()))<br>BIT=[0]*(n+1)<br>def update(i,w):<br>    while i&lt;=n:<br>        BIT[i]+=w<br>        i+=(i&amp;-i)<br>def get_sum(i):<br>    res=0<br>    while i&gt;0:<br>        res+=BIT[i]<br>        i-=(i&amp;-i)<br>    return res</code> | 1.757× | 0.620× |
| `1012_B. Chemical table` | <code>from sys import stdin<br><br>class DSU:<br>    def __init__(self, n) -&gt; None:<br>        self.parent = [i for i in range(n)]<br>        self.rank =[0]*n<br><br>    def find_set(self, v):<br>        w = v<br>        parent = self.parent<br>        while parent[v] != v:<br>            v = parent[v]<br>        while parent[w] != w:<br>            t = parent[w]<br>            parent[w] = v</code> | <code>n,m,q = map(int,input().split())<br>f = [-1]*(n+m+1)<br>def find(x):<br>    if f[x]==-1:<br>        return x<br>    else:<br>        f[x] = find(f[x])<br>        return f[x]<br>ans = n+m-1<br>for i in range(q):<br>    r,c = map(int,input().split())<br>    c+=n<br>    R = find(r)<br>    C = find(c)<br>    if R!=C :</code> | 1.396× | 0.495× |
| `1195_D2. Submarine in the Rybinsk Sea (hard edition)` | <code>import os<br>import sys<br>from io import BytesIO, IOBase<br># region fastio<br><br>BUFSIZE = 8192<br><br>class FastIO(IOBase):<br>    newlines = 0<br><br>    def __init__(self, file):<br>        self._fd = file.fileno()<br>        self.buffer = BytesIO()<br>        self.writable = &quot;x&quot; in file.mode or &quot;r&quot; not in file.mode<br>        self.write = self.buffer.write if self.writable else None</code> | <code>n=int(input())<br>l=list(input().strip().split())<br>sum=0<br>dic={}<br>for i in range(n):<br>    if len(l[i]) in dic:<br>        dic[len(l[i])]+=1<br>    else:<br>        dic[len(l[i])]=1<br><br>def calc(a,k):<br>    a0=a[-k:]<br>    a1=a[:-k]<br>    A1=0<br>    if a1!=&#x27;&#x27;:</code> | 1.487× | 0.668× |
| `1012_B. Chemical table` | <code>import sys<br>n,m,q=map(int,input().split())<br>p=[-1]*(n+m)<br>r=[0]*(n+m)<br>def par(i):<br> if p[i]==-1: return i<br> p[i]=par(p[i])<br> return p[i]<br>def merge(a,b):<br> a,b=par(a),par(b)<br> if a==b: return 0<br> if r[a]&lt;r[b]:p[a]=b<br> elif r[b]&lt;r[a]:p[b]=a<br> else:p[a]=b;r[b]+=1<br> return 1</code> | <code>from sys import stdin<br><br>class DSU:<br>    def __init__(self, n) -&gt; None:<br>        self.parent = [i for i in range(n)]<br>        self.rank =[0]*n<br><br>    def find_set(self, v):<br>        w = v<br>        parent = self.parent<br>        while parent[v] != v:<br>            v = parent[v]<br>        while parent[w] != w:<br>            t = parent[w]<br>            parent[w] = v</code> | 1.344× | 0.604× |

## Z-silence hits

`{"efficien": 1, "fast": 28, "memory": 6}`

## Drop reasons

`{"correctness_crash": 5, "extraction_ast_duplicates": 30, "extraction_sampled_out": 1852, "synth_consensus_dropped": 10, "wrong_answer": 14}`

## Limitations

- Process RSS is not the bank gate's tracemalloc metric.
- Each solution is measured on one largest available input.
- Interpreter startup, allocator state, and platform scheduling add noise.

## Orchestrator addendum (2026-07-29)

- Review trail: 4 independent review passes (Opus), final verdict GO; the
  full pipeline was fixture-verified before any real-data number was read.
- Headline: 11 in-band pairs across 4/66 measured problems (~6%); 67 clean
  dominated pairs across 19/66 (29%). Consensus failures 15/81 (18.5%),
  cleanly separated from slowness (too_slow_at_scale: 8 solutions, 7 crash /
  1 timeout, 4 problems).
- Recoverable headroom before treating 6% as final: 45/81 problems hit the
  n=1e6 scale cap under the 30ms floor (cap raise = config experiment on the
  same resumable out dir); 270/497 solutions sat under the 512KB peak floor
  at RSS granularity.
- Pool math: at pilot yield, ~1,000 in-band instances would consume most of
  the ~6,000-problem code_contests pool plus ~75 generator-authoring runs —
  marginal for training bulk, comfortable for the ~120-instance eval split
  (~1,100 problems, ~10 authoring runs).
- In-band pairs are face-valid real tradeoffs (fixed 2^18 BIT vs n-sized
  BIT; segment tree vs BIT; DSU with/without rank), each side written by a
  different human and accepted by the judge.
