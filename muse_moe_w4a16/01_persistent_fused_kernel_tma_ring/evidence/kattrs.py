"""Registers and local-memory (spill) bytes of each compiled kernel config: kattrs.py <module> [fast]."""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
M = importlib.import_module(sys.argv[1])
mod = M._module(len(sys.argv) > 2)
d = mod.driver
for cfg in M._all_cfgs():
    f = mod.func(M._name(cfg))
    regs = d.cuFuncGetAttribute(d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, f)[1]
    local = d.cuFuncGetAttribute(d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)[1]
    print(f"{M._name(cfg):24s} regs {regs:4d} local {local:6d} B")
