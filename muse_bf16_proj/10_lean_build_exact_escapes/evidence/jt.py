import re,sys
def parse(f):
    L=open(f).read().split('\n'); cases=[]; kern={}
    i=0
    for l in L:
        m=re.match(r'layer=(\S+) load=(\S+ \d+) m=(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)',l.strip())
        if m: cases.append((m.group(1),int(m.group(3)),float(m.group(4)),float(m.group(5)),float(m.group(7))))
    cur=None
    for idx,l in enumerate(L):
        m=re.match(r'profile (\d+): layer=(\S+) load=\S+ \d+ m=(\d+)',l)
        if m: cur=(m.group(2),int(m.group(3))); kern[cur]=[]; continue
        if cur and l and not l.startswith('kernel') and not l.startswith('total') and 'µs' in l:
            kern[cur].append(l.split()[0])
    sp=float(re.search(r'speedup\s+([\d.]+)',open(f).read()).group(1))
    return sp,cases,kern
if __name__=='__main__':
  files=sys.argv[1:]
  P=[parse(f) for f in files]
  print('speedups',[round(p[0],4) for p in P])
  for j,(lay,m,sg,y,sol) in enumerate(P[0][1]):
      row=[f'{lay:9s} m{m:<3d}']
      for sp,c,k in P:
          row.append(f"{c[j][2]:6.2f}/{c[j][3]:6.2f} {','.join(k.get((lay,m),[]))[:28]:28s}")
      print(' | '.join(row))
