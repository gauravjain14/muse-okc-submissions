"""Attribute a judge-score change to changed vs unchanged shapes.
python3 attr.py A1[,A2..] B1[,B2..] layer:m,layer:m,...   (A = before build's judge txt files, B = after build's)
Per shape: speedup = mean over runs of sglang/yours.  Score(S) = geomean over loads of sum(sglang_A)/sum(sglang_A/sp_S).
Reports the score with A's speedups everywhere, with B's speedups on the listed (changed) shapes only, on the
unlisted shapes only, and everywhere."""
import sys, math, statistics as st
sys.path.insert(0, __import__('os').path.dirname(__file__))
from jt import parse  # jt.py: judge .txt parser (same directory)
A=[parse(f) for f in sys.argv[1].split(',')]; B=[parse(f) for f in sys.argv[2].split(',')]
ch={(s.split(':')[0],int(s.split(':')[1])) for s in sys.argv[3].split(',')}
keys=[(c[0],c[1]) for c in A[0][1]]
def sp(P,j): return st.mean(p[1][j][2]/p[1][j][3] for p in P)
sgl=[st.mean(p[1][j][2] for p in A) for j in range(30)]
def score(pick):
    loads={}
    for j,k in enumerate(keys):
        s=sp(B,j) if pick(k) else sp(A,j)
        loads.setdefault(k[1],[0,0]); loads[k[1]][0]+=sgl[j]; loads[k[1]][1]+=sgl[j]/s
    return math.exp(sum(math.log(a/b) for a,b in loads.values())/6)
a=score(lambda k:False); c=score(lambda k:k in ch); u=score(lambda k:k not in ch); b=score(lambda k:True)
print('A-only %.4f | B on changed only %.4f (%+.2f%%) | B on unchanged only %.4f (%+.2f%%) | B everywhere %.4f (%+.2f%%)'%(a,c,100*(c/a-1),u,100*(u/a-1),b,100*(b/a-1)))
for j,k in enumerate(keys):
    if k in ch: print('  %-9s m%-3d sp %.3f -> %.3f (%+.1f%%)'%(k[0],k[1],sp(A,j),sp(B,j),100*(sp(B,j)/sp(A,j)-1)))
