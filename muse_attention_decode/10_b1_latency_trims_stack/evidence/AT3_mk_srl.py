"""SRL: the speculation-failure path re-reads seq_lens (asm volatile load) instead of reusing the early `seq` register,
so ptxas cannot hoist min(seq, seq_g) (uniform) above the burst and stall the prologue on the seq_lens DRAM load.
python mk_srl.py IN OUT"""
import sys
s = open(sys.argv[1]).read()
i = s.index('_XG5_SRC = r"""')
head, tail = s[:i], s[i:]
assert tail.count("    n = min(seq, p.seq_g);") == 2
tail = tail.replace("    n = min(seq, p.seq_g);", "    n = min(xg_ldg_s32(p.seq_lens, true), p.seq_g);")
assert head.count("    n = min(seq, p.seq_g);") == 2
head = head.replace("    n = min(seq, p.seq_g);", "    n = min(xa_ldg_s32(p.seq_lens, true), p.seq_g);")
open(sys.argv[2], "w").write(head + tail)
print("ok")
