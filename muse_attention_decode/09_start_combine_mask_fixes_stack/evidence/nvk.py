"""Tiny NVRTC + driver launcher for microbenchmarks (CUDA 13 bindings)."""
import ctypes
import struct

import torch
from cuda.bindings import driver as D
from cuda.bindings import nvrtc as N


class Mod:
    def __init__(self, src, opts=()):
        torch.zeros(1, device="cuda")
        err, prog = N.nvrtcCreateProgram(src.encode(), b"m.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17", b"-default-device"] + [x.encode() for x in opts]
        (err,) = N.nvrtcCompileProgram(prog, len(o), o)
        _, n = N.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        N.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace")
        if int(err):
            raise RuntimeError(self.log)
        _, n = N.nvrtcGetCUBINSize(prog)
        cub = b" " * n
        N.nvrtcGetCUBIN(prog, cub)
        self.cubin = cub
        err, self.m = D.cuModuleLoadData(cub)
        assert int(err) == 0, err
        self.f = {}

    def fn(self, name, smem=0):
        if name not in self.f:
            err, f = D.cuModuleGetFunction(self.m, name.encode())
            assert int(err) == 0, (name, err)
            self.f[name] = [f, 48 * 1024]
        if smem > self.f[name][1]:
            D.cuFuncSetAttribute(self.f[name][0], D.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            self.f[name][1] = smem
        return self.f[name][0]

    def launch(self, name, grid, block, args, smem=0, cluster=1):
        f = self.fn(name, smem)
        parts = []
        for a in args:
            if isinstance(a, torch.Tensor):
                parts.append(struct.pack("Q", a.data_ptr()))
            elif isinstance(a, float):
                parts.append(struct.pack("f", a))
            elif isinstance(a, tuple):  # ("q", value)
                parts.append(struct.pack(a[0], a[1]))
            else:
                parts.append(struct.pack("i", a))
        raw = ctypes.create_string_buffer(sum(len(p) for p in parts) + 8 * len(parts) + 64)
        base = (ctypes.addressof(raw) + 15) // 16 * 16
        ptrs, off = [], 0
        for p in parts:
            off = (off + len(p) - 1) // len(p) * len(p)
            ctypes.memmove(base + off, p, len(p))
            ptrs.append(base + off)
            off += len(p)
        argv = (ctypes.c_void_p * len(ptrs))(*ptrs)
        self._keep = (raw, argv)
        g = grid if isinstance(grid, tuple) else (grid, 1, 1)
        b = block if isinstance(block, tuple) else (block, 1, 1)
        stream = D.CUstream(torch.cuda.current_stream().cuda_stream)
        if cluster == 1:
            (err,) = D.cuLaunchKernel(f, *g, *b, smem, stream, ctypes.addressof(argv), 0)
        else:
            cfg = D.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = g
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = b
            cfg.sharedMemBytes = smem
            cfg.hStream = stream
            at = D.CUlaunchAttribute()
            at.id = D.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            (err,) = D.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, err
