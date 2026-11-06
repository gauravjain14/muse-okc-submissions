"""Exhaustive cuBLASLt config search (BF16 in/out, FP32 compute; FP32 split-K reductions only) under judge-like timing.

    python3 claude/lt_tune.py --shapes 128,256,32 --json /tmp/lt.json
Built on Codex's dev/cublaslt_candidate.py (descriptor/plan/run code) plus an enumerator over algo ids x tiles x stages
x cluster shapes x split-K x swizzle x custom option, each validated by cublasLtMatmulAlgoCheck.
"""
import argparse, json, math, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch
from torch.utils.cpp_extension import load_inline
import reference as R

_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

constexpr size_t MAX_WORKSPACE = 256ull * 1024 * 1024;

static void lt_check(cublasStatus_t status, const char* operation)
{
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation, " failed with cuBLAS status ", static_cast<int>(status));
}

static const cublasLtMatmulAlgoConfigAttributes_t config_attributes[9] = {
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
};

static int64_t lt_get_attribute(const cublasLtMatmulAlgo_t& algo, int index)
{
    size_t written = 0;
    if (index >= 7) {
        uint16_t value = 0;
        lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo, config_attributes[index],
                 &value, sizeof(value), &written), "get 16-bit algorithm attribute");
        return value;
    }
    uint32_t value = 0;
    lt_check(cublasLtMatmulAlgoConfigGetAttribute(&algo, config_attributes[index],
             &value, sizeof(value), &written), "get 32-bit algorithm attribute");
    return value;
}

static void lt_set_attributes(cublasLtMatmulAlgo_t& algo, const std::vector<int64_t>& settings)
{
    TORCH_CHECK(settings.size() == 9, "nine algorithm settings required");
    for (int i = 1; i < 9; ++i) {
        if (i >= 7) {
            TORCH_CHECK(settings[i] >= 0 && settings[i] <= UINT16_MAX, "invalid 16-bit algorithm setting");
            const uint16_t value = static_cast<uint16_t>(settings[i]);
            lt_check(cublasLtMatmulAlgoConfigSetAttribute(&algo, config_attributes[i],
                     &value, sizeof(value)), "set 16-bit algorithm attribute");
        } else {
            const uint32_t value = static_cast<uint32_t>(settings[i]);
            lt_check(cublasLtMatmulAlgoConfigSetAttribute(&algo, config_attributes[i],
                     &value, sizeof(value)), "set 32-bit algorithm attribute");
        }
    }
}

struct LtDescriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t output = nullptr;

    LtDescriptors(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn)
    {
        lt_check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create matmul descriptor");
        const cublasOperation_t trans_a = nn ? CUBLAS_OP_N : CUBLAS_OP_T;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        lt_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                 &trans_a, sizeof(trans_a)), "set transpose A");
        lt_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                 &trans_b, sizeof(trans_b)), "set transpose B");
        // TN: stored A[k,n] -> op(A)[n,k]. NN: stored A[n,k].
        lt_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, nn ? n : k, nn ? k : n, lda),
                 "create weight layout");
        lt_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k), "create activation layout");
        lt_check(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, m, n), "create output layout");
    }

    ~LtDescriptors()
    {
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
    LtDescriptors(const LtDescriptors&) = delete;
    LtDescriptors& operator=(const LtDescriptors&) = delete;
};

struct LtPlan : LtDescriptors {
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;

    LtPlan(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn,
           const std::vector<int64_t>& settings, int64_t version)
        : LtDescriptors(m, n, k, lda, nn)
    {
        TORCH_CHECK(version == static_cast<int64_t>(cublasLtGetVersion()),
                    "static algorithm was selected with another cuBLASLt version");
        TORCH_CHECK(settings.size() == 9, "nine algorithm settings required");
        // Exclude reductions that store intermediate sums in the BF16 output type.
        TORCH_CHECK(settings[3] == CUBLASLT_REDUCTION_SCHEME_NONE ||
                    settings[3] == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                    "FP32 split-K reductions required");
        const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
        lt_check(cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
                 CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                 static_cast<int>(settings[0]), &algorithm), "initialize static algorithm");
        lt_set_attributes(algorithm, settings);
        cublasLtMatmulHeuristicResult_t result{};
        lt_check(cublasLtMatmulAlgoCheck(handle, operation, a, b, output, output,
                 &algorithm, &result), "check static algorithm");
        lt_check(result.state, "static algorithm state");
        workspace_bytes = result.workspaceSize;
        TORCH_CHECK(workspace_bytes <= MAX_WORKSPACE, "algorithm exceeds workspace limit");
    }
};

static std::shared_ptr<LtPlan> lt_plan(int64_t m, int64_t n, int64_t k, int64_t lda,
                                     bool nn, const std::vector<int64_t>& settings,
                                     int64_t version, int64_t device)
{
    static std::mutex mutex;
    static std::map<std::vector<int64_t>, std::shared_ptr<LtPlan>> cache;
    std::vector<int64_t> key{m, n, k, lda, nn, version, device};
    key.insert(key.end(), settings.begin(), settings.end());
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto plan = std::make_shared<LtPlan>(m, n, k, lda, nn, settings, version);
    cache.emplace(std::move(key), plan);
    return plan;
}

std::vector<std::vector<int64_t>> lt_heuristics(
    int64_t m, int64_t n, int64_t k, bool nn, int64_t count, int64_t device)
{
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && count > 0, "positive shape/count required");
    const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    LtDescriptors desc(m, n, k, nn ? n : k, nn);
    cublasLtMatmulPreference_t preference = nullptr;
    lt_check(cublasLtMatmulPreferenceCreate(&preference), "create heuristic preference");
    const size_t workspace = MAX_WORKSPACE;
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
             &workspace, sizeof(workspace)), "set workspace preference");
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    lt_check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
             &reduction, sizeof(reduction)), "set FP32 reduction preference");
    const uint32_t alignment = 256;
    for (const auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
        lt_check(cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment)),
                 "set alignment preference");
    }
    std::vector<cublasLtMatmulHeuristicResult_t> results(count);
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc.operation, desc.a, desc.b,
                        desc.output, desc.output, preference, static_cast<int>(count), results.data(), &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    lt_check(status, "query heuristics");
    std::vector<std::vector<int64_t>> output;
    for (int i = 0; i < returned; ++i) {
        if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > MAX_WORKSPACE) continue;
        std::vector<int64_t> row;
        for (int attr = 0; attr < 9; ++attr) row.push_back(lt_get_attribute(results[i].algo, attr));
        if (row[3] != CUBLASLT_REDUCTION_SCHEME_NONE && row[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) continue;
        row.push_back(static_cast<int64_t>(results[i].workspaceSize));
        row.push_back(static_cast<int64_t>(cublasLtGetVersion()));
        output.push_back(std::move(row));
    }
    return output;
}

int64_t lt_workspace(int64_t m, int64_t n, int64_t k, int64_t lda, bool nn,
                     std::vector<int64_t> settings, int64_t version, int64_t device)
{
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    return static_cast<int64_t>(lt_plan(m, n, k, lda, nn, settings, version, device)->workspace_bytes);
}

at::Tensor lt_run(at::Tensor x, at::Tensor weight, int64_t n, bool nn,
                  std::vector<int64_t> settings, int64_t version, at::Tensor workspace)
{
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && workspace.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == weight.device() && x.device() == workspace.device(), "matching devices required");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "rank-two matrices required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "BF16 activation and weight tensors required");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && workspace.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(workspace.scalar_type() == at::kByte, "byte workspace required");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 256 == 0 &&
                reinterpret_cast<uintptr_t>(weight.data_ptr()) % 256 == 0,
                "this heuristic set requires 256-byte aligned tensor bases");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(nn ? (weight.size(0) == k && weight.size(1) >= n)
                   : (weight.size(1) == k && weight.size(0) >= n), "incompatible weight shape");
    const int64_t lda = nn ? weight.size(1) : k;
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({m, n}, x.options());
    if (m == 0 || n == 0) return y;
    auto plan = lt_plan(m, n, k, lda, nn, settings, version, x.get_device());
    TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= plan->workspace_bytes, "workspace too small");
    const float alpha = 1.0f, beta = 0.0f;
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    lt_check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), plan->operation,
             &alpha, weight.data_ptr(), plan->a, x.data_ptr(), plan->b,
             &beta, y.data_ptr(), plan->output, y.data_ptr(), plan->output,
             &plan->algorithm, plan->workspace_bytes ? workspace.data_ptr() : nullptr,
             plan->workspace_bytes, stream.stream()), "execute static cuBLASLt algorithm");
    return y;
}

std::vector<std::vector<int64_t>> lt_enumerate(int64_t m, int64_t n, int64_t k, bool nn, int64_t device, int64_t max_split)
{
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    LtDescriptors desc(m, n, k, nn ? n : k, nn);
    int ids[512]; int nids = 0;
    lt_check(cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
             CUDA_R_16BF, 512, ids, &nids), "get algo ids");
    std::vector<std::vector<int64_t>> out;
    for (int ia = 0; ia < nids; ++ia) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                   CUDA_R_16BF, ids[ia], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        size_t sz = 0;
        std::vector<uint32_t> tiles(1, 0), stages(1, 0);
        std::vector<uint16_t> clusters(1, 0);
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) {
            tiles.resize(sz / sizeof(uint32_t)); cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
        }
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) {
            stages.resize(sz / sizeof(uint32_t)); cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz);
        }
        clusters.clear();
        for (uint16_t c = 0; c <= 20; ++c) clusters.push_back(c);   // cluster shape enum values; AlgoCheck filters
        int32_t splitk_support = 0; uint32_t red_mask = 0, swz = 0; int32_t custom_max = 0;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_support, sizeof(splitk_support), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &red_mask, sizeof(red_mask), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz, sizeof(swz), &sz);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_max, sizeof(custom_max), &sz);
        std::vector<int> splits{1};
        if (splitk_support) for (int s : {2, 3, 4, 5, 6, 7, 8, 10, 12, 16}) if (s <= max_split) splits.push_back(s);
        for (uint32_t tile : tiles) for (uint32_t st : stages) for (uint16_t cl : clusters) for (int sk : splits)
        for (uint32_t sw = 0; sw <= (swz ? 1u : 0u); ++sw) for (int co = 0; co <= std::min(custom_max, 1); ++co) {
            const uint32_t red = sk == 1 ? CUBLASLT_REDUCTION_SCHEME_NONE : CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            if (sk > 1 && !(red_mask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) continue;
            std::vector<int64_t> settings{ids[ia], tile, sk, red, sw, co, st, 0, cl};
            cublasLtMatmulAlgo_t a2 = algo;
            bool ok = true;
            for (int i = 1; i < 9 && ok; ++i) {
                if (i >= 7) { const uint16_t v = (uint16_t)settings[i]; ok = cublasLtMatmulAlgoConfigSetAttribute(&a2, config_attributes[i], &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS; }
                else { const uint32_t v = (uint32_t)settings[i]; ok = cublasLtMatmulAlgoConfigSetAttribute(&a2, config_attributes[i], &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS; }
            }
            if (!ok) continue;
            cublasLtMatmulHeuristicResult_t res{};
            if (cublasLtMatmulAlgoCheck(handle, desc.operation, desc.a, desc.b, desc.output, desc.output, &a2, &res) != CUBLAS_STATUS_SUCCESS) continue;
            if (res.state != CUBLAS_STATUS_SUCCESS || res.workspaceSize > MAX_WORKSPACE) continue;
            settings.push_back((int64_t)res.workspaceSize);
            settings.push_back((int64_t)cublasLtGetVersion());
            out.push_back(settings);
        }
    }
    return out;
}
"""


def ext():
    return load_inline(name="claude_lt_tune_v2", cpp_sources=_CPP, functions=["lt_enumerate", "lt_workspace", "lt_run"],
                       extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcublasLt", "-lcublas"], with_cuda=True, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="128,256")
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler")
    ap.add_argument("--json", default="/tmp/lt.json")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--max-split", type=int, default=16)
    args = ap.parse_args()
    E = ext()
    w = R.generate_weights(R.WEIGHTS_SEED)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"); flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    ge = torch.cuda.CUDAGraph(); tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(ge): tmp.add_(1)
    def tg(g):
        flush.sum(); ev[0].record(); g.replay(); ev[1].record(); torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000
    for _ in range(10): tg(ge)
    fixed = statistics.median([tg(ge) for _ in range(30)])
    results = {}
    for m in [int(v) for v in args.shapes.split(",")]:
        for layer in args.layers.split(","):
            n = R.out_features(layer, m); W = w[layer][:n].contiguous(); WT = w[layer].t().contiguous()
            k = W.shape[1]
            x = R.generate_input(m, layer, 3)["x"]
            ref = torch.mm(x, W.t())
            cands = []
            for layout in ("tn", "nn"):
                nn = layout == "nn"
                t0 = time.time()
                rows = E.lt_enumerate(m, n, k, nn, 0, args.max_split)
                cands += [(layout, tuple(r)) for r in rows]
                print(f"m={m} {layer} {layout}: {len(rows)} valid configs ({time.time() - t0:.1f}s)", flush=True)
            # quick eager timing (flush first so the CPU runs ahead)
            scored = []
            for layout, r in cands:
                nn = layout == "nn"; wt = WT if nn else W; lda = wt.shape[1] if nn else k
                st = list(r[:9]); ver = r[10]
                try:
                    y = E.lt_run(x, wt, n, nn, st, ver, ws)
                    if (y.float() - ref.float()).abs().max().item() > 0.05 * ref.float().abs().max().item():
                        continue
                    ts = []
                    for _ in range(3):
                        flush.sum(); ev[0].record(); E.lt_run(x, wt, n, nn, st, ver, ws); ev[1].record(); torch.cuda.synchronize()
                        ts.append(ev[0].elapsed_time(ev[1]) * 1000)
                    scored.append((min(ts), layout, r))
                except Exception as e:
                    pass
            scored.sort(key=lambda z: z[0])
            # precise graph timing of the top candidates + torch.mm reference
            final = []
            for t_quick, layout, r in scored[:args.top]:
                nn = layout == "nn"; wt = WT if nn else W
                st = list(r[:9]); ver = r[10]
                wsz = E.lt_workspace(m, n, k, wt.shape[1] if nn else k, nn, st, ver, 0)
                wsl = torch.empty(max(wsz, 1), dtype=torch.uint8, device="cuda")
                for _ in range(2): E.lt_run(x, wt, n, nn, st, ver, wsl)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): E.lt_run(x, wt, n, nn, st, ver, wsl)
                t = statistics.median([tg(g) - fixed for _ in range(25)])
                final.append((t, layout, r))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): torch.mm(x, W.t())
            tref = statistics.median([tg(g) - fixed for _ in range(25)])
            final.sort(key=lambda z: z[0])
            print(f"== m={m} {layer}: torch.mm {tref:.2f} us; best lt " + "; ".join(f"{t:.2f} {lay} {list(r[:9])}" for t, lay, r in final[:4]), flush=True)
            results[f"{m}/{layer}"] = {"torch": tref, "best": [(t, lay, list(r)) for t, lay, r in final[:6]]}
    json.dump(results, open(args.json, "w"))


if __name__ == "__main__":
    main()
