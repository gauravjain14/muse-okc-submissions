"""Combine the four WGS full-unit readiness releases into one CTA release."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d10_pool_router_union.py'
s=base.read_text()
old='''  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }'''
new='''  if (pd.kind == 1) {
    // All four warpgroups own the same complete unit and reach this branch.
    // The CTA barrier orders all epilogue stores before one cumulative release.
    // Phase B still acquires exactly eight completed warpgroup contributions.
    CTA_SYNC();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(4) : "memory");
    pd.kind = 0;
    return;
  }'''
assert s.count(old)==1
s=s.replace(old,new)
ast.parse(s)
out=ROOT/'variants/d11_expert_release.py';out.write_text(s)
(ROOT/'runs/d11_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='Full phase-A units synchronize their four warpgroups once and publish one release adding four readiness contributions; split-unit handling unchanged.'),indent=2))
print(out,len(s.encode()))
