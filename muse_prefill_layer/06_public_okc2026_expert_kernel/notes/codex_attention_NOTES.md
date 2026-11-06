# Attention optimization, 2026-10-29

## Resume here

Prepared independent builders and candidates, CPU syntax checked; no GPU work launched (root owns one shared Modal instance). Incumbent attention uses `PrefillAttention`, three consumer warpgroups, 24 tokens / block, FP8 QK/PV WGMMA, F16 score accumulators, approximate exponent coding, stream-K-only full attention module, and separate combine grid. Short sliding units split in two when occupancy allows. `qkv_post2` fuses normalization, RoPE, cache store; `kv_pack` overlaps projection on a side stream.

Ready probe: `python research/attention/probe.py --variants 'incumbent.py;research/attention/maxpair.py;research/attention/inline_comb.py;research/attention/fullunits.py;research/attention/grid128.py' --cases 0,1,2,3,5 --reps 12 --out /tmp/attention_probe.json`.
The probe compares attention outputs to incumbent, not official correctness; accepted gains still need full-layer evaluator.

Candidates from `build.py BASE OUT`:
- `--maxpair`: exact half-precision row max reduction stays packed through warp shuffle; halves shuffle count and saves half/float conversions. All existing scores and exponent work retained.
- `--inline-combine`: inline the standalone combine kernel's function instead of forcing noinline; no arithmetic change. Previous inline-INCOMB experiments are distinct: those added combine to forward kernel and slowed its hot loop.
- `--full-units`: use whole query units for full attention when unit count fills grid; skips partial publication and separate combine naturally, attends every key. Test B4 and 16K for balance tradeoff.
- `--grid 128/96/64`: stream-K grid alignment / parallelism probe. 753-token shape has 64 units at CW3; 128 avoids uneven 132-way divisions. These are candidate hypotheses, not measured improvements.

Archived evidence:
- `logs/attn_incomb.log`: fusing combine into forward slowed full attention 224.9->237.4us, 843->921.6us; avoid redoing.
- `logs/attn_cw2_f16s2.log`: CW2 and F16S2 both regress full shapes; CW3/F16S1 established.
- `logs/attn_earlykv.log`: early KV prefetch gave no consistent gain.
- `logs/attn_nomaxh.log`: omitting max produced NaN, not usable.
- `logs/bench_full26.log`: small shapes 1.2–1.5% slower than full25a.
- `logs/bench_full27.log`: tie to full25b.
- `logs/bench_full28.log`: first-turn cases +1.5%, +1.95%, +0.24%, +0.63% relative to full25a from dense fused changes.
- `logs/bench_full29.log`: tie small shapes, small uncertain 16K gains.

Saved roofline (`logs/roofline_full24.txt`) puts large full-attention compute near roofline, sliding attention has roughly 100us headroom at 16K but layer dominated by MoE/dense. Attention alone unlikely to supply all required 9% score gain. Most valuable short shape attention wins are combined with MoE.

Additional frozen probes (unmeasured):
- `post4.py` / `post2.py`: qkv_post2 4 or 2 warps/CTA; 753-token original has only95 blocks, whereas189/377 better cover132SMs. Exact per-token arithmetic retained.
- `cw1dual.py`: one consumer WG,4 KV stages,256threads/CTA and launch_bounds256,2. Local NVRTC12.4 confirms128 initial regs,0spills, allowing2CTAs/SM with ~79KB shared perCTA. `cw1ns4.py` without changed launch bounds compiled160 initialregs so only1CTA; not a true dual occupancy probe.
- `fastdiv.py`: use CUDA FP32 __fdividef for sigmoid reciprocal to avoid exact-division correction sequence. Representation staysFP32; bound error~2ulp; whole-layer correctness still needed.
- `factornorm.py`: factor attention output scaling into RMSNorm epsilon; epsilon/sign retained, eliminates32 scaling multiplies/thread. `epilogue.py` stacks fastdiv+factornorm.
- `static_slide.py`: existing unit-only attention forward module is selected only for window2048,unit2; make those fields constants inside this compiled kernel. Other shapes/window values use existing generic/full path. QKV and packing kernels are unchanged. CPU NVRTC12.4 compiled128regs0spills.

CPU compiler diagnostics: `compile_cpu.py` uses local NVRTC12.4 without GPU. Incumbent and maxpair/post4/epilogue compile128regs0spills forward. Inline combine raises standalone combine register count80->103, zero spills; benchmark whether call removal pays. Judge compiler might differ; these diagnostics do not prove speed.

## Round1 GPU result (root's one H200)

`research/attention_round1b.log`, copied parsed results to `round1_results.json`. First attempt failed before kernels because no CUDA context; `probe.py` now calls set_device + tiny CUDA allocation before raw-driver module load.

Attention cold graph medians (us):

| case | incumbent | epilogue | gain |
| --- | ---: | ---: | ---: |
| 0,753 sliding |35.28|33.52|1.76|
| 1,753 full |244.69|243.18|1.50|
| 2,3012 sliding |62.82|60.00|2.82|
| 3,3012 full |899.02|897.02|2.00|
| 5,16384 sliding |243.90|228.30|15.60|

Epilogue output relative L2 to control <=2.13e-5, finite all cases. Paired gain positive12/12 repeats for cases2,5,11/12 case1,10/12 case0,8/12 case3. Still needs full-layer official evaluator, and16K full shapes not yet measured.

`static_slide` paired median+1.52us case0 (10/12 wins), neutral elsewhere because other shapes execute unchanged source but placement noise. `maxpair` slight case0 gain, ~1–2us regressions elsewhere; do not stack globally. `post4` mostly tie. `cw1dual` reject: +28.5us full case1,+120us case3,+35.8us case5.

Frozen round1: `deliver_r1/epilogue.py`, `deliver_r1/epi_static.py` and builder. `epi_static` untested stack combines winning epilogue with static sliding-unit specialization. `epi_post4` untested small-post configuration stack. Next falsification: isolate fastdiv vs factornorm; measure stack and remaining grid/full-units/inline-combine, then full-layer verification.

Additional round2 followups:
- `epi_page1.py`: specialized `kv_pack_page1` sets page=1 at kernel entry, selected only for actual page1; retains generic kernel for other pages. Removes integer division/modulo in page-table/cache addressing. CPU12.4 compilation succeeds,36regs0spills for both pack variants.
- `epi_scalediv.py`: epilogue plus FP32 __fdividef for attention output/QKV scale divisions. CPU compile succeeds; qkv_post2 initial register count rises48->64, possible occupancy tradeoff.
- Probe supports `--nvtx` labels around each timed graph replay for subsequent nsys per-kernel confirmation; no candidate/evaluator changes.

## Exact lazy EXPQ max guard (rejected after GPU)

Root requested a current-F16S lazy scaling mechanism. Archived `ATT_LAZY` applies only to FP32 score path and is compiled out with F16S=1; `logs/a8_lazy.log` showed that older path slower than eager. Current F16S computes max on every tile, but only rescales when max exceeds m+8.

`build_lazy.py BASE OUT` adds an exact guarded fast path; `epi_lazy_exact.py` built from epilogue.py.
- Preserve all original half scores while computing trial codes with encoding bias+8.
- For offsets ma,mb within[-100,100] (each an eighth-step), constants1080-8m and1088-8m are exact half integers.
- Valid trial halves0x6408..0x647f contain biased codes8..127. OR across every score, masked by0x9b809b80, detects any overflow. Warp-wide guard covers every active row/lane; no scores or keys omitted.
- If no overflow, original score maximum is strictly below m+8, so no max update is needed. Pack and subtract0x08080808; per-byte values>=8 preclude borrow.
- Otherwise use untouched original scores and unchanged original max/rescale/encode path. Extreme offsets also use original path. No extra quantization, approximated max, changed offset sequence, or lost tie information.
- CPU exhaustive `prove_lazy_encoding.py`:101,647,490 half-score/offset pairs,53,627,295 accepted fast paths, zero code differences and zero missed maxima. Includes all finite half values and infinities, offsets -100..100 in1/8 increments.
- CPU NVRTC12.4:128 initial registers, no spills. Runtime/compiler placement may still erase gain; GPU falsification pending.

Command: `python research/attention/probe.py --variants 'research/attention/epilogue.py;research/attention/epi_lazy_exact.py' --cases 0,1,2,3,4,5,6,7 --reps 16 --out /tmp/attention_lazy.json`.

Root GPU result `research/attention_lazy.log`: outputs bit-identical on all eight cases, but latency increases 7–20%; case1 244->289us and case3 899->1082us. The extra code path costs more than max work saved. Reject; do not integrate.

## B4 request attention + FFN pipeline (pending GPU)

Root's FFN-only token pipeline passed correctness but cost 2% sequential or 3–4% concurrent at16K. Full attention may offer sufficient independent work to recover that cost; root requested B4 request splitting first.

`build_request_pipeline.py BASE OUT --streams 1/2/4 [--att-sms 132] [--ffn-sms 132] [--full-only]` builds complete standalone candidates. Initial frozen `request_s1.py`, `request_s2.py`, `request_s4.py` derive from `research/integrated/bundle1_bk4.py`.

- Applies only batch4,q_len753; other shapes retain original layer. Full/sliding masks both remain unchanged.
- QKV projection runs once over3012 tokens. Then each request's existing attention and full FFN execute on its assigned stream.
- Each request gets its original sequence length/page-table row/cache slots. Batch=1 and tokens=753 are local metadata; token positions and cached/full-window key range are unchanged.
- Each stream owns attention accumulators/counters/output/KV packing buffers and all mutable native-expert state. Compiled modules, RoPE table, weights are shared. DenseLt workspace and custom Sk2Plan flags/workspace are private. Global shape-only SK2 plan reuse is deliberately bypassed.
- Router IDs and final boundary outputs write directly into final request slices. No concatenation pass.
- Cached packing presently occurs inside each request attention call; original cached-pack overlap with QKV is not retained in this first bound.
- CPU AST/py_compile passed; no GPU testing launched by attention worker. Root schedules cases2,3 correctness/timing. Single stream variant quantifies splitting overhead; concurrent variants quantify recoverable overlap.

Root result: all B4 pipeline checks pass, but concurrent candidates lose about11% full and17% sliding; sequential loses about15%/20%. Reject B4 splitting at753 rows; MoE/dense batch-efficiency loss dominates.

## B1 full16K attention + FFN chunks (pending GPU)

`build_chunk_pipeline.py BASE OUT --chunk 8192 --streams 1/2 [--att-sms 132] [--ffn-sms 132] [--staged]` composes the private stream-state builder and replaces B4 request splitting with B1 full-attention chunks. Initial `chunk8192_s1.py` and `chunk8192_s2.py` derive from bundle1_bk4.

- Original full QKV GEMM plus original qkv_post2 run first, writing every new KV cache row and the complete quantized Q array. Each attention chunk reuses a view of that Q8 data and the corresponding QKV gate rows, so Q/K normalization, RoPE and quantization are unchanged.
- `seq_lens` represents cached-prefix length. Each chunk computes original seq_lens + start on GPU inside the measured call. Thus local query i has absolute position original_prefix + start + i, and the original attention mask sees precisely its original causal keys. No chunk truncates the key range.
- Each chunk packs KV and its query-tile prefix table with the existing kv_pack kernel; redundant packing is a measured cost, not an approximation. Forward/combine kernels are unchanged.
- Child FFN uses private DenseLt/experts and UbPlan8192, matching root's prior FFN chunk experiment. Final arrays are written directly by the router and UB boundary.
- Changed dispatch applies only B1,T16384,window0; all other shapes retain the base path.
- Staged variants serialize attention chunks on stream0 and FFN chunks on stream1, with an event after each attention chunk. Each chunk retains its own attention output buffer. This explicitly lets attention1 overlap FFN0, whereas whole-chain streams may launch both attention chunks first.
- Frozen staged candidates: `chunk8192_staged.py`132/132SM, `chunk8192_staged_a64.py`64/132SM, `chunk8192_staged_64.py`64/64SM. All pass CPU AST/py_compile; root schedules GPU evaluation.

Root results: every fresh correctness check passes, but all chunk pipelines lose. `research/attention_chunk.log`: sequential2x8192 about1.8% slower; whole-chain parallel2x8192 about0.9% slower. `research/attention_staged.log`: staged132/132 about1.6% slower; attention64/FFN132 about24% slower;64/64 about48% slower. Reject structural chunking: attention and FFN compete for the same SM capacity, and lost batch efficiency exceeds recovered overlap.

## Public attention comparison and exact early vote (pending GPU)

Parent imported Apache-2.0 `vjeux/okc-2026` public sources under `research/public_okc_2026/`. Public prefill attention variant43 uses FP32 QK scores, FP32 exp2 with round-robin SFU scheduling, two alternating P-register sets and4 KV stages; ordinary query tiles by default, not stream-K. Our F16S/EXPQ path avoids much of that SFU work, so source inspection alone does not establish a gain.

`public_attention.py` is a development-only `probe.py` adapter: original epilogue QKV-post followed by public variant8(sliding)/43(full). It imports `research/moe/public_moe_adapter.py` and calls its `_build(True)` so the already-built full extension can reuse compilation cache. It does not invoke public prepare, tuning or synthetic cases. Original cache writes and Q8 retained. Ready command:

`python research/attention/probe.py --variants 'research/attention/epilogue.py;research/attention/public_attention.py;research/attention/epi_votemax.py' --cases 0,1,2,3,4,5,6,7 --reps 12 --out /tmp/public_attention.json`

`build_votemax.py BASE OUT` ports the public exact vote ordering into our existing half-score path. Every lane first reduces all its scores exactly as before; move existing warp rescale vote before four row-max shuffles. If no lane exceeds its row's existing threshold, no row maximum can trigger rescale, so skip the shuffles. Otherwise perform the unchanged shuffle/update sequence. No trial encodings, no extra score arrays, no precision change. Initial `epi_votemax.py` derives from epilogue; CPU NVRTC128regs0spills, expected bit-identical output. Root GPU falsification pending.

Root GPU result: public attention is far slower; full753 368vs243us, full3012 1425vs896us, full16K 6848vs3862us. Reject public attention algorithm. Early votemax is bit-identical on all8 cases and improves full attention roughly1.7%,1.4%,2%,2.5%; sliding753 is~1us noisily better, sliding3012/16K tie.

Root requested whole-layer ablations on bundle2: frozen `bundle2_vote_full.py` and `bundle2_vote_all.py`. Builder now supports `--full-only`; this leaves the original max path compiled into every pre-existing module and builds one extra stream-K vote module, selected only for window0. Sliding unit1 sometimes uses the stream-K module, so merely putting the vote in that existing module would not isolate full attention. Exotic shapes outside existing stream-K slot support retain original behavior. Whole-layer GPU compare pending.

Root requested public-MoE stack: frozen `bundle2_public_vote_full.py`, SHA256`9f8e2ce351e4cd67c11afb74e9177dafd2565c2cc8f0567f7da16aaf05cda546`, built from `research/moe/bundle2_public.py` using `build_votemax.py --full-only`. CPU AST comparison confirms only native `ATT_CUDA_SRC` assignment and `PrefillAttention` class changed; all public `_pm_*` functions and three public C++/CUDA source strings are byte-identical. Public extension build hash/cache remains unchanged. New attention module compiles in native prepare before the existing public-MoE prepare wrapper. Timed dispatch adds only a host config branch selecting the full-attention module; no synchronization or extra activation work. GPU whole-stack validation remains root-owned.

## Polynomial exponential reassessment: closed without GPU work

Parent requested a fresh degree3/4 FP32 range-reduced exp2 investigation. Current ATT_F16S=1 / ATT_EXPQ=1 bypasses all per-score exponentials: packed half2 FMA/max directly creates E4M3 probability codes, followed by byte permutes. Scalar ex2 remains for two rescale factors per thread when the running maximum grows, plus segment combination. Former per-score work was64 exponentials per thread per128-key tile; the remaining rescale upper bound is2, and actual rescaling is conditional. Existing ex2_poly/ex2_poly8 helpers compile dead under these flags.

Archive `logs/ncu_poly.log` already compares the older scalar path: baseline378.66us; degree3 mixed-poly475.04us; lighter mixture430.98us. XU activity fell72.97% to43.34%/56.06%, but extra instructions increased runtime14–25%. Older half2 polynomial experiments did improve369us to353us (`logs/a8_polyh.log`), then EXPQ superseded them at about252us (`logs/a8_expq_sweep.log`). These are historical measurements, not a paired current benchmark. Parent agreed this is not a useful current bottleneck; no new polynomial candidate or GPU work was created.

## Q16 per consumer warpgroup / CW2 feasibility review

Read-only source analysis after PG tensor candidate freeze. Current source copied for inspection to `current_attention_review.cu` from `bundle3_native_epi.py`; no working candidate changed. Current Q8/CW3 maps each consumer WG's64 WGMMA rows to8 tokens×8 GQA heads. `tok_a=c*8+2*warp`, `tok_b=tok_a+1`, head=`lane/4`. Three consumer WGs therefore cover24 tokens/CTA. The current QK opcode is m64n128k32 FP8→F16; PV is m64n72k32 FP8→F32, including eight repeated ones columns for denominators.

There is no tested Q16-per-WG variant in the searched current/dev/archive sources. `cand_att16.py`/`logs/bench_att16.log` are an older BF16-probability attention variant and still use8 queries/WG; their name is not evidence against Q16. `logs/attn_cw2_f16s2.log` changes consumer count and score buffering, always retaining Q8/WG. Its Q8/CW2/F16S1 full timings are280.2 vs251.9us at753,1076.7 vs952.6us at3012,2772.7 vs2358.8us at16384/ctx32768. F16S2 is slower still. These are pertinent occupancy/issue warnings but do not test the proposed reuse.

Hopper WGMMA's M dimension here remains64. Q16 needs two independent m64 query blocks per WG, with `tok_a=c*16+qgroup*8+2*warp`; it does not increase FLOPs per instruction. With CW2,32 tokens/CTA reduces producer stages by about25%, while total tensor instructions and scalar softmax work remain essentially unchanged. Grouping both query blocks in each QK/PV commit can halve commit/wait/control overhead per query, provided the compiler keeps healthy async groups. Each physical WG must release a ring slot only after both query blocks' PV reads finish.

Exact full-attention counts (both KV heads, before stream-K segmentation, which partitions the same tiles):

| Context / new tokens | KV stages Q8/CW3 → Q16/CW2 | KV copy bytes Q8/CW3 → Q16/CW2 | Tensor instruction ratio | Ring arrival ratio |
|---|---:|---:|---:|---:|
|80000 /753|40228 →30168|659095552 →494272512|0.999901|0.499950|
|0 /16384|88234 →66048|1445625856 →1082130432|0.998073|0.499037|
|32768 /16384|437930 →328192|7175045120 →5377097728|0.999222|0.499611|
|65536 /16384|787626 →590336|12904464384 →9672065024|0.999351|0.499675|

Register/shared/workspace bounds: CW2 already requests240 consumer registers, not232. Producer24×128 + consumer240×256 =64512 registers, identical to CW3 consumer160. Two simultaneous query blocks require essential arrays Q16 + O72 + FP8-P32 + FP16-S64 =184 32-bit registers/thread, leaving56 for max/rescale, masks, softmax temporaries, descriptors and loop state. Reusing a single score tile lowers essential arrays to152, at the cost of less query-block overlap. This is a liveness estimate, not compiler proof. Shared memory remains152768 bytes at NS8, because queries stay in registers and both groups reuse the same K/V ring. Stream-K partial slots need four logical query groups instead of three, growing105.1875 to140.25MiB at MAX_B8/grid132. Q8 global layout, output tensors and per-token accumulators need no expansion.

A restricted full-only prototype would separate physical consumer count2 from logical query-group count4. Producer/barrier counts use2; query prefix/QB uses32; combine/workspace indexing and combine grid use4. Each WG owns logical groups2*c and2*c+1. Full-only stream-K compilation can avoid rewriting sliding unit handoffs initially, but the wrapper must retain the existing sliding module and original QB24 metadata for that module. Global key coverage and each group's causal mask must remain exact.

A20% full-attention gain is possible only through substantial synchronization/ILP improvement, not the25% traffic reduction alone. Archived F16S/EXPQ NCU (`logs/ncu_att_src.log`) records DRAM2.31%, L2 throughput25.81%, issue slots45.02%, SM compute56.12%, and significant integer/minmax/branch activity. Both tensor arithmetic and per-score conversion/max work remain unchanged, while active consumer warps fall12→8. The new layout is not ruled out by old tests, but it needs a CPU codegen pilot and a paired attention microbenchmark before claiming such headroom.

Parent authorized a prototype. `build_query16.py` now generates `query16.py` (SHA256 `847e4ab57fa6c441a2bc539315e3b34cb0068951b05ce3caf96111e5b8e9fda4`) and standalone `query16.cu` (SHA256 `3014f134defa0e1c0cc12ad1ea078ae8117cd0600ea8001c9e42621eda004597`). It retains the original Q8 attention source byte-for-byte and original class AST, then adds a full-only Q16 instance. Four logical Q8 groups use existing QB32/prefix/combine/partial-slot code, while the new kernel launches384 threads with two physical consumer WGs, two empty-barrier arrivals, and240 consumer registers. Each consumer independently owns query groups2*c and2*c+1. Existing max/rescale/FP8-code and output epilogue bodies are duplicated through token renaming, preserving their arithmetic.

The primary pipeline issues both QKs in one group, waits, computes softmax0 and launches PV0, computes softmax1 while PV0 runs, then launches PV1 and the next stage's two QKs. Wait_group1 retires both PV groups before the old ring slot is returned; wait_group0 completes the next QKs before the loop back edge. Both query blocks' outputs and FP8 probability fragments stay independent. Final PVs drain with wait_group0 before the last slot is released. `query16_grouped.py` is an unqueued alternative using the original intertile QK-current/PV-previous overlap with both query blocks grouped together.

The wrapper selects Q8 for sliding, batches above8, and geometries whose minimum zero-cache causal work is smaller than the CTA grid. This last guard prevents the shortened QB32 prefix from introducing empty-CTA gaps into the unchanged contiguous-CTA partial-slot combine rule. All scored full753/16384 shapes still select Q16. The guard uses only query length and launch grid, not input-derived values or reference results.

`query16_audit.py` passes source identity, exact token/head ownership, mocked wrapper dispatch, ring-lifetime ordering, and43296 stream-K query-block partitions across boundary lengths/contexts/grids. It proves partial-slot writers and readers agree without collisions and the full causal tile ranges are covered exactly. Local CUDA12.1 compiles at168 static registers with zero stack/spills but emits C7516; parent CUDA13 NVCC gate compiles at168 registers with no C75 warning. The parent is running the exact NVRTC13 gate before any GPU probe. `query16_nvrtc_gate.py` derives defines from the actual constructor and uses the exact `_AttCuda` options, writes CUBIN/resources/SASS, and performs no CUDA context creation or GPU launch.

Parent exact-codegen command: `python research/attention/query16_nvrtc_gate.py --candidate research/attention/query16.py --out research/attention/query16_nvrtc`.

Parent attention micro command after source/codegen GO: `python research/attention/probe.py --variants 'research/attention/bundle3_native_epi.py;research/attention/query16.py' --cases 0,1,2,3,4,5,6,7 --reps 16 --out research/query16_attention.json`.

Exact target NVRTC13 gate passes:168 static registers, zero stack/local traffic,48 static QGMMA/12 ARRIVE/8 DEPBAR instructions, no C75 warnings. Target SASS `query16_target_nvrtc.sass` uses dynamic consumer allocation240 and registers throughR237, confirming the liveness estimate nearly fills the available budget. The main loop contains PV0×4, softmax1 scalar work, PV1×4, next-QK×4, DEPBAR1, ring ARRIVE, DEPBAR0, then back edge. Score arraysR128/R96 and output arraysR60/R24 are disjoint. Harness gave initial exact-source GO for a controlled GPU probe, with ring counts/release lifetimes and logical row/slot ownership checked.

If primary wins only modestly, the prepared grouped variant changes only scheduling: primary commits3n groups and waits2n times for an n-tile segment; grouped commits2n and still waits2n. Grouped runs combined PV(j−1) under both softmaxes(j), instead of primary PV0(j) under softmax1(j). It requires a separate target gate before GPU. No additional occupancy change or precision change is involved. An unused generic `p.unit` branch currently causes compiler duplication of the Q16 loop; the wrapper always passes unit0, but primary remains frozen until its first timing result.

Parent GPU falsified both Q16/CW2 schedules. Primary `research/query16_attention.log` full cases1/3/4/6/7:241.232→296.800,884.560→1097.456,539.824→657.632,2163.568→2658.720,3781.232→4602.736us. Grouped `research/query16_grouped_attention.log`:240.1→293.8,882.8→1050.8,540.2→625.5,2160.8→2469.6,3786.9→4300.1us. Outputs finite; full relative differences0.16–1.21% to control reflect changed stream-K partitioning and were not promoted to a hard correctness claim. Both schedules lose14–24%, so closed. Harness final independent source audit passed all32768 scalar owners,30720 valid batch/length bounds, and exact source/dispatch identity; codegen correctness was not the performance issue.

## Q8/CW4 resource gate: rejected before GPU

No measured128-key Q8/CW4 run was found in searched logs. An old unmeasured `cw4ns4.py` exists; `dev/attn10.py` separately used64-key tiles/shared Q with four consumer WGs. Current640-thread launch permits at most96 static registers:20 warps×round_up(R×32,256)<=65536. Thus the CTA starts with61440 registers. Producer24 plus four consumer WGs120 would demand64512 and cannot all obtain registers from the CTA-local pool. The existing112 consumer cap uses60416 and is legal. NVIDIA's shipped `cuda_occupancy.h` implements the allocation rule; [PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) specifies the per-CTA pool and blocking acquisition.

`q8_cw4_resource.cu` is source-only current full-vote Q8 with CW4. Parent exact NVRTC13 confirms C7512,24QGMMA/24ARRIVE/24DEPBAR,24STL/24LDL,96static registers and96-byte stack. No GPU candidate was launched. Static full specialization below still serializes/spills at CW4; no useful resource margin yet.

## Full-only constants and exact packed causal masks

`build_full_static.py` / `full_static.py` add a separate CUDA source used only by the already-full-only `mod_sk_vote`. `attn_fwd` and `attn_comb` set `p.window=0,p.unit=0`; existing host dispatch and `_params` guarantee both. Original ATT_CUDA_SRC/sliding/public AST remains exact. `full_static_audit.py` proves the isolated changes and dispatch preconditions. SHA256 b99bfde62cbd7f048edc207ef64f513b6f6bc5e43550f3521dc4cf6f8026feb4. Target13 compiles healthy12QGMMA/4ARRIVE/4DEPBAR,zero spills,maxR155: code halves but register peak does not improve.

First full-static probe bit-identical, mixed timings: cases1/3/4/6/7 control→static239.84→238.96,883.86→885.97,536.62→534.30,2150.26→2162.26,3928.8→3970.24us. Pending repeat alongside mask; not integrated. User explicitly asks to preserve and stack repeatable tiny gains, with no minimum gain cutoff.

`build_full_mask.py` replaces only the full-module causal mask inside `softmax_h`. In a masked tail tile, the integer query-minus-key-base differences are small. Convert `(pa-k0-2*tq,pa-k0-2*tq-1)` exactly to packed FP16 and compare against `(8*n8,8*n8)` using `set.ge.u32.f16x2`. NVIDIA PTX defines this as two independently packed0xffff/0 masks. The original `hs&mask | -inf&~mask` expression remains unchanged. No score/probability precision or key coverage changes.

Frozen `full_mask.py` SHA2560c9a4ceee0205e321c3c4b8c4fc5718b7db1ede25d02b141abaf5ea71121a948 combines constants+mask; runtime source is byte-identical to gated `full_mask.cw3.cu`. `full_mask_only.py` SHA2568e251003a148bd2dc63ea68e84a29f985b9cb352f94738e13157c069971cde5f isolates the mask without entry constants. `full_mask_audit.py` passes exact AST/source isolation,572448 residue/tail/row states for QB24/32/48,28288 packed comparisons,30277 concrete blocks near shape boundaries and large contexts. Enumerated query-minus-key-base range[-47,173]; all further lane/pair offsets remain exactly representable in FP16. Absolute context cancels, so residue enumeration covers arbitrary supported cached prefix lengths.

Local12.1 peakR155→R140 and SASS instructions2808→2568; target13 peakR155→R143 with healthy12/4/4,zero spills. CW4 still fails the resource gate. Parent schedules controlled probe; harness independent audit pending. Command variants: `bundle3_native_epi.py;full_static.py;full_mask_only.py;full_mask.py` under this directory, cases1,3,4,6,7.

Parent proposed Q16/CW3 with one reused S32/P16 and two Q/O sets, maintaining12 consumer warps and48 queries/CTA. Essential arrays total136 registers before metadata; direct original overhead projects beyond160. Correct overlap requires QK(next query group) before PV(previous), then wait1 before new scores/softmax and wait0 before reusing shared P registers; release a KV ring slot only after query-group1 PV retires. No runnable sequential candidate built. Packed-mask pressure reduction is assessed independently first; do not repeat the rejected Q16/CW2 schedules.

Harness independently approved both frozen packed-mask candidates: exact signed difference precedes half conversion, low/high packing matches the previous bit masks, removed lower key bounds are automatically true for full attention, and module dispatch guarantees window/unit0. Root GPU probe `research/full_mask_attention.log` confirms bit-identical outputs on all scored full cases. Small753/B4 shapes tie/slightly regress; long cached full shapes lose about0.3–2%. Cacheless16K consistently improves this round:538.64→533.78us for static+mask (~0.9%), versus536.94us mask-only. Keep that small win pending nearby-shape validation; do not enable globally.

`build_mask_dispatch.py` creates `empty_mask.py` (SHA7560792eec0e955262026bebc7b5688b310f95c7252142fefabca1a5da48fc9f) and `empty_mask_only.py` (SHA9ddeec4ca0ffa5048e3016073a36bab8b9ce755105cba00a2e567b8460e9ffa6). They preserve the original full-vote module and add one compiled mask module. Select it only for full stream-K inputs whose page-table capacity `max_pages*page<=L`. Valid inputs require `seq_len+L<=capacity`, so this proves an empty prefix from host-visible shape metadata without reading device values or synchronizing. Overallocated tables conservatively use the original module. No fixed scored length/batch is used. `mask_dispatch_audit.py` passes17280 mocked selections, source/AST isolation and capacity implication.

The probe now supports `--case-specs research/attention/full_mask_nearby_cases.json --cases 0,1,2,3,4`: full L8192/16384 ctx0, L16384 ctx1024/4096, L8192 ctx1024. Compare original, mask-only, static+mask to resolve the constants interaction and whether a broader low-context capacity bound is warranted. Parent owns GPU runs and integration.

Parent nearby probe passed bit-identical outputs in all five cases. Static+mask improves ctx0/L8192 181.34→177.25us and ctx0/L16384 539.89→534.50us; low contexts1024/4096 also improve0.26–1.84%. Empty-prefix gains repeat across lengths. Preserve the broader low-context option in these notes, but the current dispatch stays conservatively empty-only; no exact scored-length predicate and no full cached regressions are introduced.

On parent request, integrated the empty-only static+mask module onto canonical `research/integrated/bundle5_ten_pg.py` (base SHAbf9b279a4c04c39f82bf13e7d6c859e513f43f0672d6d8cc1a8c0ef7010332d6). Frozen sibling `research/attention/bundle5_ten_pg_empty_mask.py` SHA8bcab32e0439f333af9cb9efba6575f40da3bfb880e6ea324a3ec3589d2f6331. `bundle5_mask_audit.py`/`.json` PASS:95 original literal values unchanged, non-attention AST exact, attention AST exact after removing the four module assignments and one dispatch branch, and added CUDA byte-identical to probed full_mask.py. Root performs full-stack checks; no canonical/BEST pointer changed by this worker.

## Cached EXPQ encoding bias: exact hot-loop work removal

After composing mask+unit-combine, root requested a small unmasked-path scalar redundancy. Source and healthy SASS show every key tile recomputes `1080-8*clamp(ma/mb,-1000,1000)` and converts each result to packed half, even when running maxima remain unchanged. This is four FP32 min/max, two FFMA and two F32→half2 packs. No matching previous cache/hoist experiment was found.

`build_expq_cache.py BASE OUT` preserves every original CUDA literal and adds two derived sources for the full-vote and empty-mask modules. Two persistent half2 bias registers initialize from the exact original expression once per segment and refresh immediately after `ma=ma2;mb=mb2` inside the existing `softmax_h` rescale branch. The common path uses those cached bits. Compile guards require F16S1/EXPQ1/noBIAS; all scored full paths satisfy these. Sliding/unit kernels, host dispatch, attention keys/math/precision, barriers, workspaces and all non-attention code stay unchanged.

Frozen `bundle6_expq_cache.py` SHA20cc196cb26fa21ca89ee58acabd30209d33cecb20921849d85ee8ca4cf1b490 derives from `research/integrated/bundle6_mask_unit.py` SHA155dd2ac4e122bc0cd878aa2c8a1514d3f6308160e83216202d30de191b1cbd7. Exact gate inputs are `.full.cu` and `.empty.cu`. `expq_cache_audit.py/json` PASS:96 original literals unchanged; entire AST identical after removing two literals/restoring two module-source names; source transforms invert exactly;190488 recurrence steps over all finite half-score-derived eighth maxima, initial all-masked state, one-row updates, infinities and clamp boundaries. The stronger equivalence proof is induction on identical function F: initialize cache=F(m), update it immediately after every active m assignment, otherwise both stay unchanged.

Parent target13 gates PASS: full24QGMMA/8ARRIVE/8DEPBAR,maxR156; empty12/4/4,maxR145; both zero local stores/loads and no C75 warning. Local healthy empty-kernel SASS confirms the no-rescale branch now jumps over all four clamps/two FFMA/two half packs. Original no-rescale branches landed at the first clamp. No extra barrier or branch was added. Parent will probe against bundle6 full cases1,3,4,6,7; expected bit-identical, timing pending. Builder remains composable onto the forthcoming bundle7 stack.

Parent full-attention probe: all five outputs bit-identical, consistent positive timings. Cases1/3/4/6/7:239.95→235.86,885.44→865.79,535.31→522.66,2210.42→2154.98,3941.62→3809.44us (1.7–3.35%). Root is composing this full-only change onto the new stack. Harness independently approved active-path maximum mutation, initialization/all-masked rows, one-row updates, per-segment reset, and independence from pending PV operands.

Parent requested bounded expansion to sliding. `build_expq_cache_all.py BASE OUT` modifies only original ATT_CUDA_SRC in an already-full-cached base; cached full/empty CUDA literals and Python code remain unchanged. It enables the exact same transform only under compile-time `ATT_F16S==1 && ATT_EXPQ==1 && !ATT_BIAS`. Inactive FP32/F16S2/EXPQ2/BIAS configurations retain their exact original text after preprocessing, preserving exploratory fallback configurations.

Frozen `bundle6_expq_cache_all.py` SHAd6d609a0bdcd3f317e77e353547a6d48c2fa86d8e665bc6407dd35382b48d99a. `expq_cache_all_audit.py/json` PASS proves only one literal changes, full/empty sources exact, active transform equal to proven full transform, only softmax_h writes active maxima, split-unit handoff/combination never modify maxima, and initialization occurs inside each segment. Runtime unit/window selection does not alter the F16S1 helper path. Target source inputs `.unit.cu`, `.sk.cu`, `.generic.cu` cover the actual module define sets. Parent owns target gates and cases0/2/5 probe. Local12.1 unit/sk compile with zero spills and the known non-authoritative C7516 warning.

Parent sliding probe passes bit-identical all three: case0 32.608→33.184us (loss), case2 60.592→59.776us (+0.816us), case5 228.544→226.000us (+2.544us). At grid132/QB24, small753 has64 query units and chooses unit2/grid128; B4×753 has256 units and chooses unit1; L16384 has1366 units and chooses unit1. Unit1 uses the `mod_sk` compiled module but executes whole query units through `for_segments`' p.unit branch, not actual stream-K partitions. Split-unit2 initialization/scheduling overhead limits the small case.

Scoped the gain to unit1. `build_expq_cache_unsplit_shared.py` changes only self.mod_sk's constructor source argument to the existing ATT_EXPQ_CACHE_CUDA_SRC; original unit2/generic and all full cached sources stay untouched. No extra CUDA literal or host branch. Frozen `bundle9_unsplit_shared.py` SHA6242548366e259f24e64d9b4611a837109aef5a9fd11afe5ded1b71a1c1b7da9 derives from root's `bundle8_compact.py` SHAf2d0fbb38214576316a2460adfb60f40d6ab5627c2e1c8464aac18f4c65f8c51,941598 bytes. `expq_unsplit_shared_audit.py/json` proves exact single-name byte/AST change and actual nvcc-E output identity with the previously tested conditional sliding source. This avoids the submission1MB cap; root removed an unreferenced old empty-mask literal to compact bundle8. Earlier materialized research candidate `bundle8_expq_unsplit.py` remains available but is not the compact integration artifact.

## Exact packed early rescale vote

Parent proposed avoiding four half→FP32 conversions, two scalar maxima and two threshold additions on tiles where no rescale is needed. Existing pa2/pb2 packed half maxima can be compared directly against cached `(ma+8,ma+8)` / `(mb+8,mb+8)` half2 thresholds with `set.gt.u32.f16x2`; OR the four comparison bits and retain the existing warp-any. Only after a positive vote perform the unchanged FP32 conversions, row shuffles and maximum update.

`prove_packed_threshold.py/json` proves the recurrence before implementation. All65536 half encodings examined: every non-NaN updated FP32 threshold `rintf(8*(ta-8))/8+8` is exactly representable in half (20479 distinct finite thresholds). Initial -1e30 converts to -inf, with identical comparisons against every half encoding. All63491 reachable unchanged-row states are fixed points under the existing re-rounding; infinities included. Ordered NaN updates cannot occur.7,929,856 comparisons supplement the identity `fmaxf(x,y)>t == (x>t || y>t)` across NaNs/infinities/signed zeros and all half inputs. The ordering identity establishes all finite pairs, so no query/key approximation is introduced.

`build_packed_vote.py` modifies only the two existing full cached CUDA literals. Two packed thresholds initialize and refresh alongside cached EXPQ bias. Declarations/updates are guarded by ATT_VOTEMAX, so non-vote sliding preprocessing is unchanged. `bundle9_packed_vote.py` SHA33288571d05b43c1ca3c712f2e82a97b8f58ca1295c297e2b88aa8e09b0626ff,942734 bytes, derives from bundle9_unsplit_shared. `packed_vote_audit.py/json` PASS exact reversible source changes, untouched remaining AST, numerical proof linkage, and actual nvcc-E identity of the non-vote sliding module. Local empty source compiles healthy12/4/4,zero spills,maxR144 versus142 bias-cached baseline. Parent target gates/probe and independent harness audit pending.

If packed full wins, a distinct later follow-up could enable ATT_VOTEMAX in the unit1 sliding module, activating the packed vote there while retaining original unit2. Earlier scalar early-vote sliding tests were ties, but still paid FP32 conversions; no new variant has been built for that follow-up yet.

Parent unit1 integration comparison against compact8 passed all checks over40 repetitions: medium1794.368→1793.655us (+0.713us), large8804.016→8800.324us (+3.69us). Alongside isolated+0.816/+2.544us, the small gain is retained. Unit1 also passed18 hard cases.

Packed vote parent probe passed bit-identical all five full cases but has mixed performance:236.08→236.58us small,865.70→867.78us B4,523.39→527.07us empty,2115.79→2111.98us ctx32K,3829.01→3825.81us ctx64K. Keep the original packed candidate frozen; do not integrate without repeatable evidence. Long-context≈3us gains remain a possible small win requiring repeat, not an accepted result. Root's compact8 official offboard measurement is199.955ms; no worker submission occurred.

Parent fetched exact target SASS into `expq_cache_{full,empty}_target.sass` and `packed_vote_{full,empty}_target.sass`. The common row reduction is identical18 instructions (12VHMNMX+6HMNMX2). Counting that reduction through VOTE: scalar bias-cached baseline29 instructions, packed full23, packed empty24. Packed comparison needs one/two IMAD copies, two HSET2 masks, one LOP3 integer-to-predicate operation, then VOTE. Positive rescale now pays the deferred original four conversions/two FP32 maxima after the vote, plus four threshold-refresh operations; the no-rescale savings therefore do not establish shorter runtime by themselves.

Read primary [PTX half setp](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp) and [vote.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync). `predicate_vote_probe.cu` is a CPU-only code-generation diagnostic, no GPU launch helper. Direct predicate OR of ordered half2 comparisons plus vote.sync.any compiles locally to two HSETP2, two PLOP3, one VOTE; the logically equivalent unordered≤/vote.all form has the same count. This is five instructions, tying the target full packed path and saving only one instruction versus empty while adding another predicate-combine dependency. No full predicate candidate was built because the probe does not demonstrate a clear instruction/dependency improvement. Original packed candidate remains unchanged.

## Exact cached scalar thresholds, preserving original vote schedule

Parent requested an independent exact small saving after packed-vote mixed results. `build_scalar_threshold.py` caches FP32 `ma+8` and `mb+8` alongside the accepted EXPQ bias cache, under ATT_VOTEMAX only. The exact same FP32 additions run once at segment initialization and immediately after every active maximum update; comparisons before and after warp shuffles use the current cached values. All original half→FP32 conversions, scalar maxima, warp vote and subsequent update ordering remain unchanged. This isolates two redundant FADDs without the packed vote's deferred conversions or mask-to-predicate operations.

Frozen `bundle10_scalar_threshold.py` SHA5d910f4a8b1074cd4ab38f0dccdc522059207770468e4a331b13b16aeb3c33ba,942195 bytes, derives from root's retained `research/integrated/bundle10_clear.py` SHAc1163a90eeeea7843ba7e2bd38bf715f363389c495eea9b8f2b3403499141351. Only the two cached full CUDA literals change. `scalar_threshold_audit.py/json` PASS source reversibility, identical FP32 expressions, cache-refresh induction, unchanged remaining AST, and actual nvcc-E identity for non-vote unit1 sliding. Original unit2 is untouched. Local healthy empty SASS12/4/4,maxR144,zero spills confirms exactly two FADDs disappear before VOTE and no common-path instructions are added.

Parent target gates/probe and independent harness audit pending. Standalone `.full.cu` and `.empty.cu` use exact runtime defines; `.sk.cu` is the untouched non-vote proof. Root's separate kv_pack prefix-scan work changes only ATT_CUDA_SRC and composes independently with this variant's two cached literals.

Parent target13 scalar-threshold gate is healthy with unchanged resources. All five full cases are bit-identical and faster:234.592→233.040,866.256→859.712,524.176→519.392,2134.128→2120.176,3818.352→3787.744us. Harness independently approved the exact pre-update comparisons, initialization, sole maximum writer and unchanged vote/conversion schedule. Root is composing with boundary coefficient retention into bundle11.

## Independent root CTAB warp-scan audit

Audited root-owned `research/packing/build_warp_scan.py` and frozen `bundle10_warp_scan.py` SHA34fa6d30f9490c16ecd9d501a089f79e5921693b80d16d84b85e44744fef394e. The only changed literal is ATT_CUDA_SRC; every byte outside the integer CTAB scan and all other AST are unchanged. All three kv_pack launch sites use256 threads. The8-warp inclusive scan uses3 CTA barriers per256-entry chunk instead of18: publish warp totals, publish warp-prefix results, close every reader before next-chunk overwrite. Warp0's first shuffle consumes all shared totals before any lane overwrites them. Invalid tail lanes stay active with zero; each valid qi writes unique CTAB[qi+1], and CTAB[0] remains unchanged.

`warp_scan_independent_audit.py/json` PASS:3105 arbitrary scans and26352 real query geometries, including32/256 boundaries and multi-chunk carry. Largest enumerated prefix2407633 fits int32. Parent GPU all16 CTAB/KP/VP byte comparisons pass, but timing ties/slightly regresses≈0.1us, so this candidate is not promoted. Parent closed further scan variants without critical-path evidence.

## Unsplit sliding scalar threshold cache

`build_scalar_threshold_unsplit.py` extends the accepted scalar threshold cache to non-vote unsplit sliding in shared ATT_EXPQ_CACHE_CUDA_SRC. It removes only the ATT_VOTEMAX guards around the two threshold registers/refresh and substitutes the two non-vote comparisons after the unchanged shuffles. Two exact FP32 additions are now computed at initialization/max updates instead of every tile. No packed comparison, conversion delay, or precision change. Full vote uses identical preprocessed code; empty-mask and original unit2 sources are untouched.

Frozen `bundle10_scalar_threshold_unsplit.py` SHAe4fe7f0235eab40ce960e9770672a2f2906416b1b933d8ac01af12836ffac890,942161bytes. `scalar_threshold_unsplit_audit.py/json` PASS exact single-literal reversible edit, unchanged remaining AST, actual nvcc-E full-vote identity, and inherited init/sole-maximum-write induction. Exact target gate input `.sk.cu`; parent probe cases0/2/5 against `bundle10_scalar_threshold.py`. Root owns target gate/GPU and integration.

Harness independently approved unsplit scalar-threshold extension after rerunning its exact-source/preprocessor audit. Parent target gate and GPU timing remain pending.

## Exact first-tile zero rescale removal

Target SASS shows that the first `softmax_h` computes two scale exponentials and multiplies untouched zero O accumulators by those factors. Compiler CSE reduces36 zero multiplies to2, then broadcasts the resulting row zeros, but retains2 EX2,2 zero-FMUL and2 FSEL plus exponent argument subtracts. Before any PV, O is exactly+0. Every possible half row maximum, including NaN/±Inf, either updates initial m=-1e30 to a finite half-derived offset/+Inf (factor0) or retains initial m (factor1). Ordered comparisons reject NaN/-Inf. Thus this rescale preserves every+0 bit.

`build_first_zero.py` makes only the first F16S1 `softmax_h` call pass compile-time first_zero=true; this skips its unused scale exponentials and substitutes `resc=false` for the following zero rescale. Original maximum update, cached bias, thresholds and probability encoding remain unchanged. Every recurrent helper call defaultsfalse and retains existing scale calculations/rescale timing. The explicit flag is necessary because ex2_mufu uses volatile inline assembly, so merely deleting the multiply does not remove its dead EX2s. The documented [PTX ex2 semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2) establish ex2(±0)=1, ex2(-Inf)=+0 and FTZ underflow behavior.

Frozen `bundle10_first_zero.py` SHAbf076db8e3e687943fc62979ddf3c8d1cbed59161c7652f6a004b4e7855c177d,942475bytes, derives from accepted full scalar-threshold candidate. `first_zero_audit.py/json` PASS all65536 half maxima plus reversible two-literal edit, untouched remaining AST/unit2, per-segment zero init and all subsequent writes/reads. Local12.1 empty codegen stays healthy12QGMMA/4ARRIVE, zero spills/maxR144; startup loses2 EX2/2 FMUL/2 FSEL. Exact target inputs `.full.cu`,`.empty.cu`,`.sk.cu`; parent owns gate/GPU. Independent harness review requested.

First-zero second independent audit `first_zero_active_audit.py/json` derives exact defines from all three actual PrefillAttention constructors, checks source-byte equality with supplied gates, and preprocesses each module without importing the builder. Reversing only the intended edits recovers the baseline active kernel. All active initial maxima are -1e30; ATT_BIAS1 (nonzero initialization), F16S2 and EXPQ2 fail actual preprocessing. A +Inf first max produces scale+0; NaN/-Inf do not update and produce scale1 on a warp rescale. Original unit2 is untouched and the obsolete FP32 pipeline is statically if(false). Harness also independently approved zero initialization, unused stale-scale fields and recurrent writes.

Parent first-zero16-pair attention trial: paired gains full cases1+.528us(10/16),3+2.240us(15/16),4+1.112us(13/16),6+1.752us(10/16),7−.176us(8/16); sliding2−.04us,5+.208us. Keep pending reversed40-pair full repeat, not yet accepted.

Unsplit scalar repeat is accepted: first pairs medium+.296us(12/20),large+1.480us(15/20); reversed order medium+.512us(15/20),large+1.864us(18/20), all bit-identical. Parent official offboard bundle11 passed199454.169947us, improving compact8 by≈.501ms, while the no-submission-under194ms constraint remains active.

## pnr2y coefficient retention bound: closed before GPU

Parent requested a persistent-row reuse inspection. Current pnr2y uses four normalization warps and one top8/router warp, one CTA per token. The same16KiB FP32 coefficient vector is shared across all rows, so reuse would primarily reduce L1/load-issue traffic rather than save16KiB of HBM per token. A CPU-only straightforward register-retention loop `pnr2y_keep_sketch.cu` loads32 coefficients per normalization thread once, then strides rows by gridDim. It includes a required whole-CTA barrier after every row's PG scatter to protect both srow and sh_slot (and red) before the next iteration. Existing before-scatter barrier remains.

Local codegen rises56→106 registers with8320B shared/zero spills. Rounded register allocation reduces7CTAs/35warps perSM to3CTAs/15warps, a poor exchange for cache-resident coefficients. Parent closed full register retention before GPU. Shared coefficient staging could preserve occupancy but exchanges L1 loads for shared loads and extra synchronization; no strong bound motivated a candidate. No runnable Python candidate was produced.

## Exact router selection sentinel

Parent proposed replacing topk8's repeated taken-bit tests with destructive -Inf marking in biased score st, while preserving sigmoid scores sc for output weights. Source invariant: local bv starts -Inf and increases only under ordered st>bv, so selected st=-Inf can never win again. Every unselected value and comparison/tie order remains unchanged. NaNs/-Inf never win. If all available candidates are exhausted, bi remains1<<30 and no lane matches the winning branch, so both old and new code retain their existing fallback behavior without a sentinel write. +Inf can win once and is then excluded exactly as before; signed-zero/tie ordering is preserved in both FAST variants.

Before building a runnable candidate, compared CPU codegen with actual non-fast-math options on the accepted fastsig source. pnr5 instructions1688→1512, regs54→48; pnr5g2048→1840, regs56 unchanged; pnr2y2112→1896, regs56→48. All havezero spills. Sources/SASS/logs are `router_sentinel_{base,sketch}.*`. This removes repeated bit tests while retaining all expert candidates and precision.

Frozen `bundle11_router_sentinel.py` SHA3b6a8de8831daa7a7473b50da7ec363e51c61726b4d5a1ca9ade24c156c38f6e,942354bytes, derives from root `research/router/bundle11_fast_sigmoid.py` SHAcc7efe9b982932f72d8701ba58d492641b47722bee321a1da90dbe49c4021afe. Builder `build_router_sentinel.py` changes only topk8 within ROUTE_SRC. `.runtime.cu` is exactly byte-identical to CPU-gated sketch, `.route.cu` supplies isolated source. `router_sentinel_audit.py/json` PASS131074 local patterns and2048 full rows for each FAST0/1, exact IDs/normalized weight bits over random float bit patterns, ties, signed zeros, NaN/±Inf and fewer than8 viable scores. Entire remaining AST/source stays unchanged. Parent owns target gate/component GPU; independent harness review requested.

Component command: `python research/harness/router_sigmoid_probe.py --control research/router/bundle11_fast_sigmoid.py --candidate research/attention/bundle11_router_sentinel.py --cases 0,2,5 --inputs 2 --reps 24 --json /tmp/router_sentinel.json`.

The exact non-fast-math fastsig PNR retention recheck (`pnr2y_keep_exact.*`) uses108 registers versus56 baseline; both preliminary106 and exact108 round to112, so3CTA/15warps remains the bound. No GPU run requested.

Harness independent router sentinel GO: reviewed strict ordered comparison/-Inf invariant, unchanged sc/ties/reduction, exhausted fallback, and bucket publication lifetimes; reran full CPU proof read-only. Parent notified. Candidate remains frozen while root schedules target/GPU validation.

Parent router sentinel GPU component checks all pass. Paired large gain+5.552us(16/16), medium+.072us(9/16) tie, small−.136us(6/16) regression. Parent requests generic PG-template-only gating so the large win is retained and original non-PG small routing stays intact.

`build_router_sentinel_pg.py` selects the complete measured sentinel topk8 body with `if constexpr(PG)` and retains the complete original fastsig body in else. A first statement-level constexpr sketch restored pnr5 but changed PG codegen and lost some instruction savings; it remains archived as `bundle11_router_sentinel_pg.py` and is NOT the integration artifact. Whole-body selection preserves compiler SSA/scheduling for both branches.

Frozen integration candidate `bundle11_router_sentinel_pg_whole.py` SHA61836829e13d660e6f34c0f6ddb5b86bf17faeddc3c62d6b0296f368e48841cf,944771bytes. Builder is composable onto stack12 after fastsig. `router_sentinel_pg_audit.py/json` PASS exact two body identities, unchanged remaining AST/source, no new host/shape branch. LocalCUDA12.1 SASS including encoded instruction/control words is byte-identical for pnr5/pnr/route to fastsig baseline and for pnr2y/pnr5g to the measured all-sentinel candidate. Runtime source `.runtime.cu` exactly matches the compiled `router_sentinel_pg_whole.cu`; independent harness review requested. Original frozen all-sentinel and fastsig candidates remain untouched.

Parent target13 PG-only sentinel compilation passed: pnr5 retains54registers matching fastsig, pnr2y48registers matching sentinel. Root composed bundle12 and runs full8/hard validation.

## Peer audit: public N64 four-MMA grouping

Root requested read-only audit of `research/moe/bundle11_public_n64g4.py` SHA08c422255f98ec8a608d1168d8235a547cfbed57d416cdfd4aa358c874c9c939. `n64g4_peer_audit.py/json` independently proves only consumer_item's N64 branch/A-fragment allocation changes; other widths and remaining Python AST are exact. N64 retains two32-float accumulator chains, even/odd K64 assignments, j0..3 issue order and original final acc+=acc2. Two16-register A quartets alternate by kk parity; after commit/wait1 the earlier bank retires before its next overwrite. At the next stage's kk0 wait1, all previous-slot asynchronous readers are complete before all256 consumers release it. The finalwait0 drains the last slot before release and scalar reduction.

CPU enumeration passes40 item/start-slot/parity states and2048 fragment/weight/scale address states; N64 stage footprint25600B fits43008B. Both consumer warpgroups and unchanged8warps×32 empty-barrier arrival count agree. Target13 `research/public_n64g4_gate.log` is healthy with noC75: FINE0 68HGMMA/34ARRIVE/40DEPBAR, FINE1 88/54/64,168regs. FINE1's32B stack/additional local operations may offset fewer commits; no timing conclusion inferred. Source GO sent root/MoE; GPU remains root-owned.

## Rank1 restart:64-key compute decomposition

Latest retained base is26, SHA328cfc3a…881e5. New architectural work and its exact audit/resource/work bounds are in `halfkey_NOTES.md`, `halfkey_audit.json` and `halfkey_bound.json`. Frozen `bundle26_halfkey.py` preserves every original26 AST node as fallback, adds a cached-full path with four Q8 consumer WGs and two64-key compute subtiles per unchanged128-key physical KV stage. Target13 gate is healthy at maxR107 withinC112, zero spills,8QGMMA/4ARRIVE/4DEPBAR. Root now owns reference-checked full-layer validation. No speedup or correctness promotion claimed from compilation/source audit alone.
