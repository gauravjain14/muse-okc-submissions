"""Probe: build a 4-D UINT64 tiled tensor map with cuda.bindings and print its bytes."""
import torch
from cuda.bindings import driver as d
import inspect
x = torch.zeros(32 * 16 * 512 * 2048, dtype=torch.uint8, device="cuda")
err, tm = d.cuTensorMapEncodeTiled(
    d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64, d.cuuint32_t(4), x.data_ptr(),
    [d.cuuint64_t(v) for v in (256, 512, 16, 32)], [d.cuuint64_t(v) for v in (2048, 512 * 2048, 16 * 512 * 2048)],
    [d.cuuint32_t(v) for v in (256, 4, 4, 1)], [d.cuuint32_t(1)] * 4,
    d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
print("err", err)
print(type(tm), [a for a in dir(tm) if not a.startswith("_")])
op = tm.opaque
print(type(op), len(op), [int(v) for v in op][:4])
import struct
b = b"".join(struct.pack("<Q", int(v)) for v in op)
print(len(b))
