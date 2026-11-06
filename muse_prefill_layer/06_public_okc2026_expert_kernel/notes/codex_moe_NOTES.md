# MoE research checkpoint

## Resume here

Coordinator owns the only Modal GPU and all official checks/submissions. Do not launch GPU jobs here.
Pending candidate batches, all based on incumbent.py:

1. stsm.py / deq6.py / deq6_stsm.py
2. wg_small4.py / wg_small8.py / wg_small2.py

All candidates have passed Python syntax checks. deq6 packing passed 10,000 random CPU exact-code comparisons and a bijection check over all 32 input bits. GPU compile, numerical correctness and timing are pending.

## Current path

incumbent.py CUDA_SRC equals cand_dense2_moe_v29.py byte-for-byte (172982 bytes, SHA256 prefix 73108fff0a8b).
`kernel_fn` computes fresh routing, buckets local expert token IDs and optionally pre-gathers activations. `_experts` calls `_native_expert_kernel`. This runs a persistent 384-thread CTA per SM (128 producer threads, 256 consumer threads), fresh per-call gate/up and down GEMMs, and a separate `_native_finish` Triton norm/weighted combine. NVFP4 codes/scales remain compressed; BF16 dequantization occurs in the consumer loop. T=128 token tiles, runtime trimmed width; WGMMA starts at four groups (25 routes), smaller tiles use mma.sync with shared-memory ldmatrix.

## Existing evidence

Official incumbent MoE profile: 288.4us (49.3%) at M753 sliding; 271.9us (33.5%) at M753 full; 839us (43.7%) at M3012 sliding.

Most later candidates are measured regressions, already explored in logs:
- v27 loop arrangement: ~7% slower M753.
- v28 variable width: ~1% improvement over v24b; v29 already integrates it.
- v30: tie; v31 1/2 warp variants slower.
- v41 reorder and v42 merge: M753 slower 1–3%.
- v45 stmatrix down epilogue: ~0.3–0.6% MoE win, correct relative errors 0.00287.
- v46 stmatrix gate/up relabel: incorrect, relative error 1.0. Do not stack.
- nt8 overall: ~10–25% slower.
- lower consumer register caps: slower.
- prefetch and A3 pipelining: not reliable wins.

Traces (`moe_real_timing2.log`) show large-token compute approaching the 4096-cycle per 256x128x256 stage tensor ceiling. At M16384 stage compute 4241 clocks; large gains require structural changes rather than another simple pipeline knob.

Tiny routes are the better near-term target: `moe_tl_v42_v29.log` shows n<=32 consuming about 35% of CTA time at M753, although those experts carry around 8% of routes. `moe_small_only.log` shows ~100us on 112 routes across 11 experts. Unlike large tiles these are issue/data movement bound.

## Candidates

`build.py BASE OUT --stsm --deq6` composes independent exact changes.
- stsm: frozen equivalent of prior v45 change, replacing scalar BF16 output shared stores with stmatrix.
- deq6: lossless bit permutation + alternate decode trades one integer ALU op for one IMAD on the FMA pipe. Prior timing-only small_probe.py predicted 1499->1437 cycles at n8, but used a non-invertible code mapping. This version fixes the mapping: output3 mask must be q&0x81808180 | rot8(q)&0x00400040 (old timing prototype selected overlapping bits). The `_POS` map uses each packed input bit exactly once, preserving all 8 4-bit codes.

`build_small_wgmma.py BASE OUT [--group8] [--threshold 1|2]` enables n8/n16 WGMMA correctly.
- Current wg_issue4 omits NTL1/2; runtime WGMMA dispatch omits cases1/2. Simply lowering MOE_WGMIN is insufficient on incumbent.
- wg_small4: fixes dispatch and uses existing four-MMA-group pipeline.
- wg_small8: eight-MMA groups (two A matrices), double buffers and continuous stages for widths8/16; reduces fence/commit/wait rate.
- wg_small2: keeps <=8 routes on mma.sync, only 9–16 routes use small WGMMA.

## Commands to coordinator

```
python dev/moe_real_ab.py --variants 'base|incumbent.py|;stsm|research/moe/stsm.py|;deq6|research/moe/deq6.py|;both|research/moe/deq6_stsm.py|' --ms 753,3012,16384 --seeds 2 --reps 12 --check
python dev/moe_real_ab.py --variants 'base|incumbent.py|;wg4|research/moe/wg_small4.py|;wg8|research/moe/wg_small8.py|;wg2|research/moe/wg_small2.py|' --ms 753,3012 --seeds 2 --reps 12 --check
```

No observed new performance claims yet. Keep changes only if correct on full-layer checks and consistently faster in the coordinator's frozen evaluator.

## Observed first-round results (2026-10-29)

`research/moe_round1.log`: all candidates match reference relative errors (about0.00287). stsm gives 0–0.4% MoE improvements at3012/16384, neutral753. deq6+stsm gains0.5–1.0% at753 but loses0.6–0.9% on larger inputs. Do not indiscriminately stack deq6.

`research/moe_concurrent.log`: corrected retained-state probe runs without memory faults. All321.8us, conc16=699.4,24=491.6,32=395.4,44=339.6; threshold48 concurrency loses even at44SM. Separate full-SM calls total384.3us. Each sum has1.06e-3 relative error vs unsplit; all repeated graphs survived. Dynamic partition candidate built but not worth running on this bound.

Ready next: tiny-only old-packing decode (`small_decode.py`), pair-scale conversion (`pair_scales.py`, `pair_scales_diet.py`), tiny WGMMA (`wg_small8.py`, `wg_small2.py`). All latter CUDA code compiles to PTX with local CPU-onlyNVRTC12.1; target13.x builds still needed.

Final concurrency bound probe amortizes one preparation over thresholds8/16/32:
`python research/moe/concurrent_thresholds.py incumbent.py --ms 753 --ths 8,16,32 --ks 24,40,56 --reps 12 --nopdl`. CPU source compiled. Holds all state owners; unusedPG buffers stay shared forM<=2048 only. Every capture/replay has90second watchdog; incorrect branch aborts before timing.

## Small-expert theoretical bound

Packed weight+scale bytes per expert: w13=16MiB+2MiB; w2=8MiB+1MiB; total27MiB. Eleven tiny experts stream297MiB, an ideal64.9us at4.8TB/s. Historical isolated tiny-expert span~100us, leaving at most~35us vs this bandwidth lower bound. Saving35us in both753 layer classes reduces their load by~5.7%; because the score has three equal geometric-mean loads this is only~1.9% total score. Tiny-path improvements alone cannot plausibly supply the full9% target; they need to stack with larger phases or structural overlap that changes the critical path.

## Round 2 and structural bounds (2026-10-29)

`research/moe_round2.log`: all errors identical 2.87e-3. Tiny WGMMA8/2 lose 8-18% across all M, reject. small_decode/pair/pairdiet are near tie at M753/3012, pairdiet 0.4-1.0% better in paired large M measurements (within noise). Built `pairdiet_stsm.py` to test combination; compose builders `build_pair_scales.py BASE TMP --decode-diet`, `build.py TMP OUT --stsm`. No full-layer acceptance yet.

CPU ptxas12.1: wg_small8 and wg_small2 emit C7512 (`wgmma.mma_async instructions are serialized due to insufficient register resources`). Native/pairdiet/pairdiet_stsm have no such warning. All 168 launch registers, native and pair 248 spill stores / 452 loads, wg 204/376. This plausibly explains globally slower wide WGMMA even though source changes only tiny dispatch. Target-image compiler diagnostics still needed.

Large math bound: M16384 useful compute~3.006 TF, tiled compute~3.105 TF (3.3% padding). Historical 4241-clk stages vs4096 ideal means 3.543ms compute vs3.421ms cycle floor at1.679GHz. 989TF/s nominal floor assumes higher clock; native already ~908TF/s cycle limit at measuredclock. Remaining measured overhead~0.25ms epilogue +0.16ms waits.

Transient fresh BF16 expansion is compatible with stated rules if rerun each timed call. Threshold>256 routes selects9 experts /99.2% of large routes; weight expansion reads243MiB compressed and writes864MiB BF16 =>242us ideal@4.8TB/s. All20 active wouldcost537us ideal. No expansion outside timedcalls allowed.

`bf16_bound.py` is a feasibility probe, NOT candidate: uses host token lists outside timing to measure optimistic fresh-weight-expansion +cuBLAS bound. Timed total includes activation gather, fresh dequant,FP32 GEMMs,SiLU, BF16intermediate, down BF16 rounding,norm/weightedcombine. CPUsyntaxchecked; GPU pending:
`python research/moe/bf16_bound.py incumbent.py --ms 16384 --threshold 256 --reps 12`.
NVIDIA CuTe Hopper groupedGEMM reference available read-only under kernel-design-agents KernelWiki PR-3091; supports runtime group problem sizes and pergroup TMA descriptors. Local quack install not found; target image availability can be checked if bound justifiesimplementation.

Update: `wg_small4.py` CPU ptxas has NO C7512 and native-identical register/spill counts; queued as the clean test of tiny WGMMA. SASS WARPGROUP.ARRIVE/DEPBAR sites: native112/126, pairdiet112/126, wg_small4128/144, wg_small8512/512. Extra group8 live registers force global WGMMA serialization; this does not reject group4 tiny WGMMA.

## Round 3 / BF16 rejection / gate-stmatrix repair

Round3 `research/moe_round3.log`: proper width8/16 WGMMA group4 still loses~1.8–2% small/M3012,~0.3–0.4%large. pairdiet_stsm replicated against duplicate controls:~0.3%753,0.3–0.5%3012,0.4%large, all errors identical. Wide pair `pairdiet_wide.py` CPU clean but GPU `research/moe_wide.log` ~1.1%large regression vs pairdiet_stsm control (which includes down-stmatrix), likely reject; no need rewrite from this.

`bf16_bound.py` initial FP8 masked-load integer-other compile error fixed by omitting mask on exact full8192-element blocks. CPU-offline Triton compiler (`offline_triton.py`, noGPUdriver) verifies unpack/activate/post/combine. GPU second run correct relerr4.99e-4 but negative: native_hot4411.2us,dequant535us,dense4929.7us,total5428.4us for9experts/29620routes. Dense alone loses; reject fresh-expanded BF16 rewrite.

Register experiments are negative: PREG24/CREG240 raises native spill stores/loads248/452→440/748; group8 C7512 remains. Trimming pin_acc to only live width columns and loading q per kk also retain C7512. Do not queue these CPU-diagnostic variants.

**NEW concrete old-v46 bug:** historical `dev/make_stsm1.py` inserted its alternate w13 packing block between w13 and w2 repacks, accidentally removing w2 from original expert loop. Only expert31 got w2 initialized; relerr1.0 does not reject gate-stmatrix math. `build_gate_stsm.py BASE OUT` invokes old builder then repairs all32 down expert packing and supplies matching alternate tensor maps. It checks AST loop placement and all128 relabeled intermediate mapping entries. `gate_stsm.py` built on pairdiet_stsm, CPU NVRTC+ptxas clean, noC7512/native-identical spills. GPU priority candidate.

**NEW real queue changes:** bucket_tables already descending-sorts current counts; MOE_INTERLEAVE only touches legacy bitmask routing, not incumbent bucket_tables, so historical interleave tests likely no-ops. `build_interleave.py BASE OUT --mode alternate|tinyfirst --max-m4096 --threshold32` changes the used GPU bucket table rank permutation, preserving all experts/routes. alternate largest,smallest,2ndlargest,...; tinyfirst rotates tiny experts ahead of hot, preserving descending order inside each. Default only M<=4096; big cases unchanged. `alternate.py`/`tinyfirst.py` based on pairdiet_stsm. Rank bijections checked all32liveexpert populations. Historical traces show tiny down work tails ~15us beyond final hot CTA, so bounded potential10–15us753.

`chunk_bound.py` is optional MoE token-chunk/SM-budget overhead feasibility probe for cross-phase pipelines. Root already found fullFFN sequential2x8192 costs2%; initial unbounded concurrent paths lose3–4%, now investigating actual SM caps. Parent owns GPU.

## Public implementation architecture and component adapter

Upstream Apache2 repository `https://github.com/vjeux/okc-2026`, root-cloned to `research/public_okc_2026`, commit `f35a41ee1c13e316dabbf772830d47b23b453393`.

`muse_prefill_layer.py` differences worth testing:
- CTA128 weight rows ×256tokens vs ours256×128. Both128 accumulator regs/fulltile, but public halves weightdecode work peroutputFLOP.
- Small width32 uses4 independent accumulator chains,64/128 use2; mitigates WGMMA recurrence latency. Unlike our tinyWG experiment, minimumtokenwidth32 incurspadding but strongerpipelining.
- Dedicated epilogue staging buffer and producer publisherwarp issues gate bulkS2G, downscalarcopies while consumers start nextitem. Our publisher only signals completion; consumers do the actual copies before proceeding.
- Both phases share one persistentqueue; allgate before alldown. Public tileplan sorts byWIDTH ratherthan expert-total count; widths32/64/128/160/192/256, tailrounding fine fromM2048.
- Public comments confirm excessive livefinewidth specializations causeC7512 globalWGMMAserialization; defaultfine_mask6 selects160/192 only.

`public_moe_adapter.py` freezes original source plus `public_adapter_extension.py`. It prepares MoEonly (whole CUDAextension compile, no dense/attention tuning). Exact normalized-input semantics: gatherRMSinverse replaced by1 and gather normweightszeros, so BF16x remains unchanged. FreshTritonGPU bucket computesroutecounts/positions eachcall; publicgather/plan/fusedexpert/combine alltimed. Optionalprenorm input uses sameF.rms_norm path asbenchmark beforeidentitygather. Provenance retained. Root owns microbenchmark scheduling.

Gate-stmatrix GPU nowcorrect on6inputs, modestpositive .2–.4%M753,.3–.4%M3012,.1%16384 overpairdiet_stsm. Actualalternate andtinyfirstqueues regress1–2%small, largeunchanged;rejectorderingchanges.


## Public MoE gain and full-layer integration

Root GPU `research/moe_public_bound.log` verified all6 inputs at reference-relative errors~2.87e-3. M753 public roughlytie seed0 and3.7%slower seed1; keep native small. M3012 paired speedups2.9/4.6%; M16384 paired5.4/5.7%. Public shape aspect and publisher epilogue overlap produce a genuine structural gain.

`build_public_integration.py research/integrated/bundle2.py research/moe/bundle2_public.py` constructs self-contained namespaced34-symbol component. Full Apache2 license and upstreamcommit retained. Public route selected only2048<=M<=MAX_M, native small/chunked fallback unchanged. Native fused router bucket counts/positions reused; native0..255 moduloIDs guarded against>=32 in publicgather+combine. Native PG gatherdisabled on public branch so fusedrouterwritesnormalized expert_input; public identitygather merely rearranges it. Reset32nativecountsfused into ordinary ordered combine CTA0, making everycall/graphreplay fresh without extra reset launch. UnusedPDLcombinebranch removed forolderTritoncompatibility. Resetcombine offlinecompiled forSM90, PythonASTvalid. ParentGPU full-layervalidation pending.


## Integrated public success / remaining bounds

Parent full-layer checks pass all8 shapes. `bundle2_public` improves aggregate1.013469 versus bundle1; large loads~2.62%, B4~1.36%, small effectively unchanged. Parent public+attention-vote officialcheck passes at201438us; submitted455b4f71a2a940b9 (ranking pending when noted).

`--moe-only` on `build_public_integration.py` removes unrelated public attention/QK/boundary extension definitions and wrappers. `bundle2_public_trim.py` is677KB vs787KB original; CPU nvcc compiled. All public MoE code is retained. Optional compile-time reduction, no claimed timing gain; not yet GPU accepted.

`research/public_moe_trace.log`: gate256 median39.36us, gate19229.74, gate12822.528, gate6416.128, gate3211.168. A128-output-row×N256×K4096 item is268435456 FLOP /4096 BF16 FLOP perSMcycle=65536 cycles; at historical1.679GHz theoretical39.03us. Thus full256 gate is within~1%of TCfloor. Width128 has15%overhead,64~65%,32~129%; remainder tiles have optimization scope, but dominantlarge compute does not. Down25641.152us is~5.4%above samefloor (clock stability needs confirmation). Totalconsumer span3662.8us, main3521.6us,tail118.3us,gaps18.7us; excellentloadbalance.

Fine remainder audit: mask6 means160/192;96/224 WGMMA helpers exist but their fused consumerdispatch cases do NOT exist. Changingonlymask is invalid! `build_public_fine.py` updatesbothmaskanddispatch. All4extras reduce sample16384 padding30400→30240rows vs29865real, only0.53%math gain. `public_fine5.py`96+192 and `public_fine9.py`96+224 both CPU/GPUcorrect. `research/public_fine.log`: fine5+1.2%M3012seed0 butotherwise~noise; fine9 roughlytie/regressesseed1M3012, large~0–0.3%. Do not integrate yet.

Native tiny chains: `build_tiny_chains.py` creates tight[chains][2][NTL][4] temporaries; firstversion andaliasvariant cause12.1C7515. Attentionpeer first-stage scale_d0/outputonly peel removesC7515 but leavesC7512; target13GPUboth2/4chains~6%slower, reject. Note unchangedpublicMoE also emitsC7515 on local12.1 even withuse_fast_math, so localcompilerwarning alone is not definitive target13 behavior.

Next frozen probes: `public_chain64.py` raises four-accumulatorchain widthlimit32→64, keeping128live accumulatorregisters equalto supportedwidth128twochainpath. `build_public_stages.py BASE OUT` gives2×83968byte ring slots instead of4×43008 and doublesKextent perstage. Footprints32:69632,64:51200,96:67584,128:83968,160:50176,192:58368,224:66560,256:74752 bytes, totalheader+ring+epilogue214016<232448. ProducerandconsumerKPSmatching/staticasserts added. `public_stage2.py` frozen forparent GPU, CPUcompile pending atnote time.


## Stage and fine-width negatives; small public extension

`research/public_stage.log`: public stage2 passes all6 inputs but loses~2–4%M753,~1.6–2.2%M3012,~0.5–0.6%M16384. Four-chain width64 is mostly noise,<=0.1%larger cases. Do not integrate. Native peeled tiny-chain variants also lose~6%on target13; reject.

`build_public_tiny.py BASE OUT` implements public N8/N16 tiles only forFINE=0 (small specialization). New WGMMA RS specializations, ten-class GPU plan (descending widths256,...,32,16,8), matching fused dispatch, scalar tiny gate epilogue writing the same128B-swizzled staging layout. CPU address coverage proves one unique store per8/16×64 BF16element. Public gather/pad fill and producer already use runtime tilewidth; down epilogue already supports8/16. Four accumulator chains mean16registers forN8 and32forN16. Wider FINE=1 kernels keep existing widths and math. `public_tiny.py` frozen forGPU microbench; CPU nvcc passes (same12.1 C7515 aspublicbaseline).

`public_mma32.py` enables upstream's implemented MOE_MMA32 path: N32 uses mma.sync+ldmatrix and skips token groups beyondceil(nreal/8), preserving32-row storage. This is an alternative public hot plus native-style tiny math bound, avoiding extra WGMMA specializations. Parent queues afterfirsttiny result.


## WG1 / two CTAs per SM prototype

Root requested distinct same-SM hot/cold overlap mechanism. `build_public_wg1.py` makes64weight-row blocks,128consumerthreads+128producer/publisherthreads,2ring slots,110KiB dynamic shared,256-thread launch with2CTA launchbounds. First N256 design fails CPU C7600:128accumulatorregisters+operands cannot fit128initialregistertarget. Revised maximumtokenwidth128 uses64accumulatorregisters and onlyone chain forN128 (N32/64 retainexisting2/4chains), a matching fresh GPU plan/gather, and doubled gate/down itemcounts128/64 per tile. Gatepublisher scatters32FFNfeatures into onehalf of existing128B-swizzled intermediate rows. Downcolumns64/stagingpitch144. All gatehalf addresses CPU-bijective; new repack bytes CPU-verified exactlyequal old128-row block halves forbothw13/w2.

Frozen `public_wg1.py` twoCTA grid264 and `public_wg1_single.py` grid132 shareCUDA. Target13 resourcehelper `wg1_resources.py` PASS128regs,40Blocal,2activeCTAs/SM. All2seedcorrect. `research/public_wg1.log`: native303.2/306.6us;WG1single584.8/650.5;WG1double395.1/440.6. TwoCTA mechanism improves1.48x butstill30–44%slower than native. Target SASS32HGMMA/8ARRIVE/11DEPBAR means WG1 isnotgloballyserialized.

`public_wg1_p40c216.py` restores producer40regs and raisesconsumer216, exactly32768regs/CTA. CPUstack40→24B,spill stores56→24bytes,loads84→40bytes,still128staticregs. `build_public_wg1_order.py` rank-bijection interleaveswidest/narrowesttiles fororder2; `public_wg1_p40c216_mix.py` combinesregisterfix+mixedqueue, rootGPU pending. Tile has128gatejobs soqueuealternation shouldplacehot/cold work onco-residentCTAs.

`build_public_wg1_splitn.py` creates `public_wg1_splitn.py`: N128 exactdecomposition into2N64 column MMAs using same A, first32accumulatorregisters andnext32, Bdescriptor+512(8192bytes). Still64totalaccumulators andsameFLOPs/dequant, but2independentcolumnchains replaceWG1'ssingleN128Kdependencychain. Candidate derivesp40c216_mix. CPUcompile pending. Rootharnessowns WG1 traceadapter with192recordspertile; originalpublictrace96bound mustnotbeusedunchanged.


WG1 update: `research/public_wg1_mix.log` corrected p40/c216 plus mixed queue gives~396.7/438.9us vs WG1~397.6/442.7, stillwellbehindnative302.7/307.5. Split-columnWG1 target13healthy40HGMMA/8ARRIVE/11DEPBAR, seed0correct406.1us; no evidence of serialization or benefit. Combinedround2hang reproduced in separate four8 publictinycandidate; root/attention debugging unsupportedwidthplan mismatch. Do not blameWG1 for that hang.

`build_public_wg1_bulk.py` creates `public_wg1_bulk.py`: gateintermediate blocks become contiguous32features×64Brow, matchingWG1weightoutputtile, permitting originalasyncbulkS2G. DownWGMMAuses64Bswizzle (chunkxor=(token>>1)&3), reading two consecutiveK32blocks perK64 stage. Producerstillbulkloadssametotalbytes; no weight repack orarithmetic change. CPU mapping verifiedeveryelementN32/64/128. CPU nvcc reports128regs,8Bstack,ZEROspillstores/loads vs prior24Bstack/24stores/40loads; rootGPU correctness/time pending. Root askedholdfurtherWG1variantsuntil existing splitn/bulkisolated.

Cheap pendingprobe `public_fine_small.py` sets FINE_MIN_T1 (previous2048), reusesalreadycompiledhealthyFINE1kernel. M753seed0expertcounts150/150/133currentlyeach split128+32, whereasFINE1usesone160tile withsamepaddedFLOPs andoneweightpass. Estimated~20uspotentialwithoutnewCUDAcompile. Existingfine5/9testedonly3012/16384anddidnotcoverthis.

## Fine small and seven-width follow-up

`public_fine_small.py` is correct on all three M753 seeds: native 302.5/307.2/313.3us versus public fine 290.5/312.0/326.8us. Hot-tail merging helps seed0, but tiny experts still cost more. Parent requested a six-width layout replacing64 by8: `build_public_fine_tiny8.py` and frozen `public_fine_tiny8.py`, widths8/32/128/160/192/256, FINE_MIN_T=1. Target13 FINE1 stays healthy at52 HGMMA/22 ARRIVE/28 DEPBAR,168 registers and8B stack. All synthetic plan/gather checks and three real-route checks passed (`fine_tiny8_plan_debug.py`). Micro is correct but mixed: native302.8/308.0/314.1us; fine8 289.4/318.4/321.5; ordinary fine291.7/313.5/329.3. Missing64 hurts seed1.

`build_public_fine_tiny8_seven.py` restores64, retaining all seven widths8/32/64/128/160/192/256. Frozen `public_fine_tiny8_seven.py` plus optional `public_fine_tiny8_seven_kps8.py`. KPS8 variant batches eight K64 blocks only forN8 in both producer and consumer. SLOT45056 exactly covers8×(8×128+4608); worst total shared including1023B alignment is227327B, below existing227KiB (232448B). Source static_asserts check stage and total bounds. CPU12.1 both variants compile to168 registers,8B stack,zero spills for both FINE kernels. Target resources/correctness/time pending. Matching GPU checker `fine_tiny8_seven_plan_debug.py` supports these exact widths and sentinel-hole detection; root owns launches.

Attention peer found and fixed the earlier four-width N8 plan hang: r161..192 generated two128 tiles but sorted writer put both in the same index. Their `research/attention/public_tiny_four8_fixed.py` adds the correct second index. Our six/seven-width planner uses incrementingidx, so repeated widths already receive separate descriptors.

## WG1 consumer-first192-thread prototype

`build_public_wg1_192.py public_wg1_bulk.py public_moe_adapter.py OUT` creates `public_wg1_192.py`. Consumer threads0..127; producer leader128; publisher160..191. No setmaxnreg redistribution. Restores256-token tiles and original N128 two-chain arithmetic;64 weight rows perCTA retained. Launch192 threads with two-CTA bound and grid264. CPU12.1 reports168 registers, FINE0 stack32B/spills24B each, FINE1 stack40B/spills32B each. This register allocation uses32256 registers/CTA and can fit two subject to target occupancy confirmation. `wg1_resources.py` now checks registers×threads≤32768 and activeBlocks≥2 instead of a hard128register ceiling.

Ring is2×37888B. Largest stage footprint isN128,KPS2:37376B; N256,KPS1:35072B. Gate staging maximum16384B. Down staging reuses the128-token half-buffer, so maximum18416B including last16B store (not256×144); exhaustive active store-address audit covers all widths32/64/128/160/192/256. STG_BYTES20480, header1024, worst alignment1023 give98303B total below100KiB launch. `public_wg1_192_fine.py` changes only FINE_MIN_T to1 and shares the same CUDA cache. GPU validation pending. These depend on the bulk gate/64B down-input layout, whose GPU result was still pending at this note.

Subsequent parent logs confirm bulk correctness on all three M753 seeds: native302.8/307.4/313.7us; mixed WG1398.3/440.1/453.1; bulk392.4/433.7/447.4. Bulk improves~6us but remains negative. Mixed WG1 trace has94.6% dual-consumer overlap and26.7% main hot/cold overlap; mainloops still dominate. Target192 resource gate subsequently passes:166 registers,40B local,192 threads,100KiB dynamic shared,2 activeCTAs/SM. Independent attention-agent audit (`research/attention/public_wg1_192_audit.py`) clears192-thread participant counts and all large-width addresses. Target timing remains pending.

## CUDA13 serialization diagnosis and N128 register repair

Parent target resources round5 show seven-width public FINE0 52/52/52 and FINE1 60/60/60 HGMMA/ARRIVE/DEPBAR; N8KPS8 also serialized68/68/68 and76/76/76. WG192 serializes36/36/36 and44/44/44. All remain numerically correct where timed, but codegen invalidates performance conclusions about the geometry. `extract_cached_codegen.py` is a CPU-only cache PTX/ptxas/SASS extractor. Root fixed cachednvcc ccache prefix parsing and ninja empty-variable regex; do not overwrite those fixes. Saved target13 artifacts under `target_codegen/`: healthy fine8 has no C75 warnings; seven emits C7512 insufficient register resources for both kernels. This is not the scalar-zero C7515 seen on local12.1.

Diagnostic `public_seven_outline8.py` isolates N8 in a noinline device function. Local12.1 immediately reports C7510 (MMA pipeline crosses function call), plus312B caller stack. Reject without GPU; public lee101/openkernels docs independently describe the same failure. No code from that repository was copied.

Parent-requested `public_six_drop160.py` and `public_six_drop192.py` retain8/32/64/128 plus one fine width and256. Remainder mask and fused dispatch change together; seven-class planner retains one zero-count class. `public_six_plan_debug.py` handles their exact oracles. Target round6 both still serialize: FINE0 52/52/52; FINE1 56/56/56. Reject; adding64 is problematic even when total specialization count returns to six.

`build_public_n128_group2.py` repairs an actual peak: N128 keeps128 FP32 accumulator registers but its A pipeline shrinks from32 to16 registers. Two alternating pairs replace two alternating quartets; two MMA instructions per async group, preserving every accumulator's complete issue order and FP32 reduction order. The prior ring slot releases after the first new pair retires the prior pair. CPU issue-sequence/buffer-lifetime oracle passes. Public and WG192 down64B descriptors are handled explicitly.

Frozen `public_seven_n128g2.py`: CPU168 registers,8B stack,zero spills. **Target13 healthy**: FINE0 52/18/23; FINE1 60/26/33;168 registers, stack8/16B. Parent queued its unchanged seven-width plan/gather checker and microbenchmark. This demonstrates that A operand pressure caused the global fallback, rather than unsupported padding widths.

Frozen `public_wg192_n128g2.py` and `_fine.py`: CPU stack32/40B and spills24/32B drop to8B stack andzero spills, but target still globally serializes36/36/36 and44/44/44. Hold;192-thread static register budget remains tight. Attention owns a separate shared-A/producers-dequant prototype.

`public_seven_p24c240.py`: active fused producer24/consumer240 fits exactly128×24+256×240=384×168=64512 registers. CPU stack80B,180B spill stores/188B loads. Target FINE0 healthy52/16/21 but FINE1 still60/60/60. Do not time/integrate. Main public follow-up is only the healthy N128G2 variant until timing establishes a gain.

Seven N128G2 passes all three real M753 routings. Parent micro: native301.1/305.9/312.0us; seven285.7/305.0/318.2; fine baseline290.4/312.4/328.4. Seed0 improves5.4%, seed1~0.3%, seed2 still loses2%. The remaining padding includes tiny9–16 routes rounded32 and205 routes rounded256.

`build_public_more_widths.py` adds16 plus optional224/96 while retaining N128G2. Frozen `public_eight_n128g2.py` (N16), `public_nine_n128g2.py` (N16+224), `public_ten_n128g2.py` (N16+224+96). Complete fine/nonfine remainder oracles pass0–4096; N8/N16 gate writes cover every BF16 element exactly once; N16/96/224 stage footprints fit43008B. N96 stays on original GROUP4=false path. Width96 helps the66/67/77-route remainder/expert in seeds0/1/2; it does not shrink a100-route expert. Target resource gates pending. Local eight and ten compile to168 registers,8B stack,zero spills. Nine's initial local compile lost its tmpxft PTX under concurrent cleanup; rerun isolated TMPDIR pending. `public_ten_plan_debug.py` is the matching GPU plan/gather oracle for all8/9/10 variants.

Independent read-only audit of harness `bundle3_public_pg.py`, `public_pg.cu`, `public_pg_router.cu`: no blocker found. Full256 gate copies use32768B K64 stride; partial tiles copy each K64 plane with native publicN×128B ring packing. The nonfine second tile's+128 token offset is handled inside each source plane. All tile starts are multiples of8, so source/destination XOR phases agree. Metadata zeros each fresh final-slot padding region, including all chunks. Slot bound32+ceil(8*MAX_M/256) is sufficient, and PG_KMAX128 exceeds the64 slots any single expert can need atM16384 (top8 is unique). Existing release/acquire slot allocation protocol is preserved; ordinary kernel ordering/PDL wait covers router writes before metadata/expert reads. mbarrier transaction byte totals still match split bulk copies. Root/harness own GPU metadata probe and fresh-input correctness; source audit is not a GPU correctness claim.


## All-GROUP4 operand repair and ten-width result

Target13 eight/nine/ten N128-only G2 variants globally serialize again after adding N16: FINE0 68/68/68, FINE1 76/76/76,80/80/80,88/88/88. `build_public_global_g2.py` applies the same 16-register A pair pipeline to all GROUP4 widths, preserving each independent accumulator chain and reduction order. Frozen `public_seven_globalg2.py` is healthy at FINE0 52/28/33 and FINE1 60/36/43; `public_ten_globalg2.py` is healthy at FINE0 68/36/42 and FINE1 88/56/66. Nine_no16 with only N128 G2 still serializes FINE1 72/72/72; reject it.

All-G2 ten passes plan/gather and all three M753 real inputs. Native approximately301/306.2/312us; ten286.3/300.8/319.0. Seven-all approximately289/308.8/320.4. Small behavior remains routing-dependent, so native remains the integrated M<2048 path. Larger ten micro passes all four inputs: M3012 gains2.1% seed0 and0.5% seed1 versus public baseline, M16384 is a tie (+0.1%/-0.1%). Root builds `research/integrated/bundle4_public_ten.py` with ten only on the established public branch; root owns full-layer checks.

`public_source_gate.py` compiles extracted CUDA+trimmed MoE glue without importing Torch or launching a CUDA context, and inspects ptxas/SASS resources. Root validated the fine8 device-only source against the cached complete extension: exact fused SASS counts and registers, target compile7.5s and total8.3s, no C75 warning. Use this fast target CPU gate before full extension builds. Independent compile jobs need unique TMPDIRs to avoid tmpxft filename collisions. `public_cpu_source.py` supplies the required glue and supports both _CUDA_SRC and namespaced _pm__CUDA_SRC.

## Fused public-hot / native-style R256-cold prototype

Frozen `public_r256_cold.py` (SHA256 f2fb0b02d015e437bd6fa0cbab510b3b02ad29ffe0e8ac2d98b7dc58fb66a1d9), builder `build_public_r256_cold.py`, device source `public_r256_cold.cu`, and standalone math helper `r256_cold_consumer.cuh` extend ten-global-G2 with one persistent queue/gather/combine. N8/N16/N32 use R256 mma.sync with two m16 row blocks per warp; wider tiles retain the existing R128 WGMMA path. The cold loop reuses B fragments across the two row blocks and only executes ceil(nreal/8) active token groups. Uniform cold KPS2 fits26624B maximum inside the existing43008B slots. Gate cold jobs halve to32/tile and down to16/tile. Gate publishes two adjacent K64 planes in one bulk copy; down publisher kind2 copies256-column rows with528B staging pitch. The largest cold down staging footprint is16896B, within existing45056B.

A single owned compressed-weight tensor retains the original R128 prefix plus a losslessly interleaved R256 suffix; no expanded weight precision or cached outputs. Fresh metadata now has two ints [total_tiles, hot_tiles], allowing matching variable-job producer/consumer dispatch and gate done thresholds (64 hot or32 cold). Width-sorted order1 is required and enforced. CIT remains disabled. All buffers retain owners. **Use `public_r256_plan_debug.py`, not an older scalar-ntiles checker**. Existing trace decoders also need variable item-count adaptation before use.

Independent CPU `r256_cold_audit.py` passes compressed-pack mapping, all active token groups, N8 B x2 addresses, complete unique gate/down logical coverage, staging bounds, and end-of-pack bounds. Target13 source gate passes FINE1 40 HGMMA /32 ARRIVE /39 DEPBAR plus112 HMMA.16816,168 registers,24B stack, no C75 warning. Full extension build, two-element plan/gather checker, and real microbenchmark are queued by root; no timing claim yet. Frozen files must remain unchanged while root tests them.

## PG tail source audit and submission gate

Read-only audit of harness `build_public_pg_tail.py` and `bundle3_public_pg_tail.{cu,cpp}` finds no address/API blocker. Metadata compacts only sub256 remainder planes into a per-expert2MiB tail arena; row0=0 or128 preserves XOR phase, padding is written fresh, producer KPS stages regain one contiguous bulk copy. Full256 remains direct PG. Trailing FusedArgs pointer, C++ signatures, Python allocation owner, and producer source arithmetic agree. GPU live-byte probe and fresh-input checks remain required; harness/root own those.

Latest user constraint relayed by root: **do not submit to the leaderboard until an offboard check is below194ms**. Root alone performs official checks/submits and owns the one Modal GPU. This child performs CPU implementation/audit only.


R256 trace adapter `public_r256_trace.py` is ready if the micro result warrants profiling. Its decoder accepts [ntiles, nhot] and maps 64/32 gate plus32/16 down row jobs correctly; a CPU synthetic audit passes all560 total/hot combinations up to32 tiles. Raw records include weight_rows and row_job; buffer owners and ordinary reference-relative checks remain inherited from the public trace harness.

Read-only PG tail-skew audit also clears `build_public_pg_skew.py` / `bundle3_public_pg_tail_skew.py`: public slot pitch2097280B is consistent in router, metadata zero/read, producer full256 source, and BF16 allocation. All slot starts retain128B alignment. Compact tail arena remains2MiB per expert and native PG pitch is unchanged. No descriptors in this variant need updating. This is source validation only; root owns GPU live-byte/fresh-input checks.


R256 micro (`research/public_r256.log`) passes all three real inputs at identical approximately2.87e-3 relative error but is negative: native301.1/306.1/312.8us, ten285.8/302.0/320.2us, R256320.5/309.6/337.2us. Do not integrate. Root requests a seed0 trace before any changes. Cold midpoint instrumentation is present immediately after its K loop and before staging waits/epilogue, and all three cold dispatches pass the common valid leader timestamp pointer. No speculative tuning variant queued before trace evidence.


R256 seed0 trace (`research/public_r256_trace.json`) confirms main-loop regression: N8 gate/down18.3/19.5us, N16 25.7/26.6us, N32 45.7/45.9us. Cold tail0.32–0.77us is small; per-CTA average inter-item gaps1.36us total. N32 real18/24/30 all take about45.3–45.8us, so predicated HMMA preserves nearly the same issued instruction stream even for three active groups. Native gate_stsm MOE_WGMIN4 already sends25–32 routes to WGMMA, with an archived HMMA-throughput-bound comment. R256's three N32 tiles in seed0 and two in seed2 explain most route-dependent loss; seed1 has none.

Target SASS `r256_target_f1.sass` also confirms the new cold helper omitted tested opaque decode multipliers: scale PRMT is followed by LEA shift-add, and some weight×16 becomes SHF, whereas original public consumer loads p.mul16/p.mul64 to retain IMAD. Frozen `public_r256_imad.py` / `.cu` via `build_r256_imad.py` changes only these cold multiplier sources/calls. Queue, pack, geometry, stage sizes, and precision unchanged; target source gate and micro pending. Candidate original remains frozen.


R256 opaque decode source gate is healthy on target13: FINE1 40 HGMMA/32 ARRIVE/39 DEPBAR,112 HMMA.16816,168 registers,24B stack, no C75 warning. Root queues fullbuild and micro. `build_r256_threshold.py` additionally freezes `public_r256_imad16.py`: only N8/N16 stay R256; N32 returns to original WGMMA with its old compressed-prefix layout and public KPS. Sorted metadata hot count now includes the N32 class (first8 classes). `public_r256_threshold_plan_debug.py` / `public_r256_threshold_trace.py` read R256_COLD_MAX_N for correct oracles. Target gate pending; original candidates unchanged.


Next separate threshold16 stage probe: `public_r256_imad16_stage4.py` / `.cu` via `build_r256_stage4.py`. Cold KPS2→4 halves stage waits and bulk transactions; SLOT43008→45056 makes both N8(40960B) and N16(45056B) fit. Four ring slots,45056B staging,1024B header and worst1023B alignment total227327B, below unchanged227KiB=232448B launch. Hot stage sizes remain valid. Complete K64/MMA issue order matches KPS2 exactly; no precision/order change. Root has optional source for target gate; no GPU claim yet. Threshold checker/trace helpers still apply.


IMAD/threshold16 micro (`research/r256_imad_bench.log`) passes all three inputs: native301.2/306.0/312.6us; ten286.4/301.8/318.7; IMAD316.9/305.8/333.4; IMAD16 292.9/306.8/324.1. Returning N32 to WGMMA recovers most loss, but IMAD16 remains5.0–6.5us behind ten on every seed. Target `r256_imad_target_f1.sass` verifies the intended decode change exactly: scale-bias80 LEA+8 IMAD becomes32 LEA+56 IMAD (all48 cold conversions moved). HMMA/HGMMA/PRMT/LOP3/HMUL2 counts unchanged; overall weight-decode IMAD.SHL136→61, SHF.L113→92, ordinary IMAD192→322. No inactive-knob explanation.

Stage4 fast target gate passes FINE1 56 HGMMA/40 ARRIVE/48 DEPBAR plus96 HMMA.16816,168 registers,24B stack, no C75 warning. Root queues its full build/micro after PG-skew. Parent decision: if stage4 cannot beat ten across all three seeds, close the R256 hybrid branch rather than adding unsupported tuning variants.


R256 branch closed after stage4 micro: all correct; native301.2/305.6/312.0us, ten286.4/301.1/319.3us, stage4 293.4/306.2/321.7us. Stage4 still loses every seed; no more R256 variants planned.

Independent audit clears harness large-only ten+PG-tail integration `bundle4_public_ten_pg_large.py`, SHA b416d6ecaea8a1da7ff4f6c280657579be7ca955d0ddb139546847fa3b09023a. Active last kernel_fn selects public PG only8192..MAX_M; medium gets no PG; fallback passes pg_xs only when bucketed. Exact consumer/plan prefix matches bundle4 (SHA7cb7137f69055b2de1740fa5395b7ff7828ce718d0ecaee196fc264c01d0d0dd); native/attention sources unchanged. All new8/16/96/224 tail widths and coarse+128 offsets independently address-audited; full256 direct slots, owners and resets consistent. GPU checks remain root-owned.

## Remaining bounded combine and identity-gather probes

Latest user steering relayed by root: preserve and stack every repeatable small gain; no minimum gain-size cutoff. Negative variants stay closed. Publication gate still requires offboard score<194ms before leaderboard submission.

`public_combine_probe.py` compares the current bundle4 reset-bearing Triton combine at2/4/8 warps and the existing CUDA helper. The CUDA helper gets its required separate32-counter reset included. Real front/routing/MoE outputs are regenerated per seed; only combine is timed, with graph owners retained, cold AB/BA pairs, finite/error/reset checks. Upstream explicitly marks CUDA_COMBINE slower; no existing2/4-warps component sweep found. Parent command: `python research/moe/public_combine_probe.py --ms 753,3012,16384 --seeds 2 --reps 10 --json /tmp/public_combine.json`. Current measured combine~25–27us medium/~128us large; compulsory large read+write~380MB gives~79us floor at4.8TB/s, so~40–50us large is an optimistic ceiling, not a predicted saving.

Concrete independent cleanup `build_identity_gather.py` freezes `bundle4_public_identity.py` / `.cu`. The integrated public gather receives normalized BF16 and zero norm weights, but target-independent local12.1 SASS confirms it still emits16 FFMA,9 SHFL.BFLY,23 FADD,4 norm-weight64-bit loads,16 FMUL and8 BF16 pack operations per thread. Packed uint4 copy removes all of them. Both barriers remain, including the essential publication of route metadata; plan/padding/layout code and expert consumers remain unchanged. Local compilation passes; gather72regs/no spills unchanged, fused consumer resource/SASS counts identical. Original and copy SASS saved as identity_gather_original.sass / identity_gather_copy.sass. Whole gather30.5–33us medium/~135us large is the upper bound; single-digit medium savings are plausible but unmeasured. Builder applies to large-PG integration too, where ordinary gather remains only at medium M. Root owns target compile and fresh-input/timing checks.


Canonical identity stack frozen from latest `research/integrated/bundle5_ten_pg.py`: `bundle5_identity.py` SHA1114922d19ea14505fe24f23ae184c145a280b61448274801e6b0ccec2728ca9 and `bundle5_identity.device.cu`. Runtime invariant audited: fused PNR or fallback pre_expert_norm supplies normalized BF16; adapter gather weight is prepared zeros; explicit prenorm fallback retained. Large public PG bypass remains intact, so ordinary medium gather is the primary active gain target.

Independent source/lifetime audit clears harness RA-reuse candidate `bundle4_public_ten_pg_large_ra.py`: both MODE0 boundary kernels already store ra through p.v3; every active fast_bnd now allocates/holds it and the sole live _bnd2 caller passes it. _bnd2 always launches512 threads, matching16-warp csum_cached_ra. Its y shuffle/store/read reduction tree and synchronization match old csum2.y; other modes keep old math. Large UB already used the same scratch and stays unchanged. GPU freshness checks remain required.

`research/public_combine.log` closes the bounded combine sweep negatively. All outputs pass with relative differences<=1.55e-5. Medium w4's own adjacent controls are29.328/29.312us versus29.904/29.920us; paired speedups0.97831/0.97770, losing23/24 medium pairs. Earlier separate w8 medians30.4–30.5us are not the proper comparison because clocks drifted. W4 also loses small andlarge; w2 and CUDA lose every shape. Keep current8-warps combine; no purported tiny gain to stack from this sweep.


Separate exact combine cleanup: `build_combine_unit_scale.py` creates `bundle4_combine_unit.py`, removing only `(tl.load(gs_ptr+lid)*pw)`→`pw`. Builder verifies every public combine caller passes prepared ones32, because global expert scales are already applied by the MoE epilogue. ABI,8-warps, norm/reduction order, weighted sum, and reset stay unchanged. `public_combine_unit_probe.py` clones the previous real-input paired probe but compares only original8 vs unit8, two fresh seeds, no rejected warp/CUDA variants. CPU-only TritonSM90 compilation passes both; PTX global loads176→168 and mul.f32 instructions272→144 (static unrolled counts), confirming actual removal. Saved combine_original.ptx / combine_unit.ptx. Root owns GPU timing and correctness.


Unit-scale combine probe positive and bit-identical (`research/public_combine_unit.log`): medium paired1.007736/1.008068,12/16 and10/16 wins; large1.006642/1.005118,16/16 and15/16 wins, median adjacent savings0.832/0.656us. Frozen canonical `bundle5_combine_unit.py` SHA5daad9e2a49b39a4a3929c38d100ae633b52643a28775c1f214b7b15a8ac5ef8 changes only public combine; native<2048 unchanged. Root queues full-stack measurement rather than discarding small repeatable gains.

Identity gather full-layer comparison passes correctness but is within noise (mediumcase2+.0535%,case3-.0986%; unaffectedlarge+.0496%). `identity_gather_probe.py` now isolates the component with fresh real front/routing/normalized inputs per seed, both negative and positive>=32 invalid IDs, exact comparison of every full arena byte including untouched sentinels, all metadata, independent swizzled live-value checks, and all pad bytes. Only gather/plan is cold AB/BA graph-timed, owners retained. DefaultM3012,two seeds,24pairs per encoding; cached bundle5/bundle5_identity extensions need no full prepare. CPU API and2050 planner-boundary cases pass. Parent command: `python research/moe/identity_gather_probe.py --m 3012 --seeds 2 --reps 24 --json /tmp/identity_gather.json`. GPU result pending; no benefit claim from noisy full-layer medians alone.


## Accepted identity gather and current exact native cleanup

Identity gather component probe passed all four fresh-input tests, including both invalid-ID encodings, exact arena bytes and metadata. Parent reports paired speedups1.00393/1.00425 for seed0 and1.01265/1.01111 for seed1, with17/24,14/24,17/24,17/24 wins. Accepted and stacked with unit-scale combine, RA reuse and attention mask/EXPQ cleanup in canonical `research/integrated/bundle8_expq_stack.py`. No further repeat is needed.

Independent read-only audit of harness `boundary_ra_probe.py` passed source/lifetime and paired-measurement review: captured ra is retained in output dictionaries, independent cloned inputs, matching first/second split graphs, fresh AB/BA pairs, bitwise ra checks and final changed-input replay. Parent/harness own its GPU measurements.

Current native small investigation: the direct REPI gate and down arms cover every valid phase1/2 item when HASWG,T128,KW1,MW<=32,LMX,KS2=KSX=1. The compiler still emits the generic epilogue because it cannot infer the phase domain across producer shared memory. `build_native_direct_epi.py` makes that domain explicit in the existing down-arm condition, leaving every other configuration and all live arithmetic unchanged. `bundle8_native_direct_epi.py` is the isolated candidate; exact source extraction and branch-equivalence audit are saved alongside. CPU resource comparison pending. Prior `--clear-live` zero trim was never measured on target, but local12.1 spills increased248/452 to292/744, so it is not the first follow-up.

Native direct-epilogue candidate `bundle8_native_direct_epi.py` is frozen at SHA228265f5167d0dd1e8e22fdf8630a3a69e5c9a9e932dc5651c61e7f38fb6eadb. Audit covers1984 width/phase/tail combinations across all prepared configs; exactlyRW8/NT16/KS2=1/NS2/MW32 takes the exhaustive direct arm. AST comparison proves onlyCUDA_SRC changed. Local12.1 production-feature compile passes without C75 warnings:168registers,136Bstack,224spillstores/288spillloads unchanged. Native SASS20092→19500 instructions, eliminating592 generic epilogue instructions, including128shared stores,56FMUL,28BF16packs and4barriers; these were unreachable work, so do not infer a timing gain from their count. The live path loses the redundant second-phase dispatch check; code-layout benefit remains unmeasured. Same448HGMMA/112ARRIVE/126DEPBAR. Filesnative_epi_control/direct.{cu,ptx,cubin,sass,ptxas.log},native_epi_codegen.json preserve diagnostics. Parent has exact target NVRTC and3-seed cold paired commands; no GPU here.


## Native exact scalar follow-ups

Parent target13 direct-epilogue resource gate passes168regs/120Bstack/noC75. Three real-input micro timings all correct but mixed: base302.1→301.1us,306.3→306.8,311.9→312.4. Preserve as unaccepted artifact; do not stack from this evidence. Independent harness audit GO: all three StageInfo publishers produce only phases1/2, sentinels break before epilogue,442368 independent branch/config states agree. Best official checked result reported by parent is199955.328807us for bundle8compact.

`build_native_stage_desc.py` freezes `bundle8_native_stage_desc.py` SHA2df3cdca5b6f34af02a2484ea13e675ac64c86af789c7f40aa344252a06fc316. Active stage_compute_wg2 repeatedly formed wg_desc(xaddr+kk*T*128) four times. The candidate forms one base descriptor and addskk*T*8 to its encoded16-byte address. All96 descriptor forms across compiled WG configurations are bit-identical; maximum stage end209920<2^18 proves no address-field wrap. Local12.1 confirms descriptor mask/shift/OR counts56→14 each, instructions20092→19940, same448HGMMA/112ARRIVE/126DEPBAR. Registers168,stack136,spillstores224/spillloads288 unchanged; noC75. Target gate/timing pending.

At parent request, `build_native_clear_live.py` freezes current-source clear-only-live sibling `bundle8_native_clear_live.py` SHAa155d7391ce22a9b25c6483edb7f17b4f4501e19d0949cd342b594d7b8d5fe41, without direct-epilogue or descriptor changes. Initial full zeroing remains, each exact-width MMA writesb<ntl, and post-item clear restores those columns;256 adjacent-width sequences prove the induction. MOE_NTLPAD retains full clear explicitly. Local current-source12.1 resources are unchanged, so archived pre-no-combine spill increase does not recur. Compiler changes reset placement to128FSEL plus15 predicates; timing must decide. Parent requested target13 resource gate before any GPU run. Both candidates have frozen .cu/.audit.json, source AST checks prove other components unchanged.


## Clear-live accepted; low32 descriptor and raw paired timing

Parent accepted native clear-live after all18hard tests and full scored checks: small cases567.664→565.576us with18/20 positive pairs and763.664→762.644us with11/20; unchangedmedium/large tie/noise. Bundle10full8x2+18hard passed,.092%aggregate gain versusbundle8. Subsequent officialbundle11 boundary/attention stack passed199454.169947us.

`build_native_stage_desc32.py` froze `bundle10_native_stage_desc32.py` SHA9e81240a36f458e5d1d0912f0330e722cb0d14f8a3e027b00781d057ac93f296 and same-base64-bit comparator `bundle10_native_stage_desc64.py`. Low32 removes42 carry-adds, but introduces other moves: total19260instructions versus19236 for64-bit; both healthy. All96actual descriptorforms and106504 aligned addresses match. Target13 both168regs/120Bstack/noC75. Parent raw3seed microcorrect but negative/tied: base300.4/304.3/310.5us,low32300.4/304.7/310.7,desc64300.7/305.1/311.0. Both unaccepted; no repeated sweep absent new mechanism.

New `moe_real_ab_raw.py`, reproducibly generated by `build_raw_ab_harness.py`, copies unchangeddev/moe_bench.py pipeline and installs exactdev/moe_real_ab.py routing. It records full event samples, original fixed-cost subtraction, original per-replay ratios, adjacentAB/BA averaged ratios/savings/wins andsource hashes. timed()AST is identical tooriginal; preparation, inputs, captures,1GiBflush andorder unchanged. CPU synthetic statistics pass. Parent used it successfully; defaultcheck retains originalreferenceerror reporting.

Peer audits GO: `bundle10_boundary_keep.py` SHA92cbb900…3827 only hoists immutablepreparedv0/v1/v2loads inbody4;8192coefficientpositions unique,RA/barriers/math unchanged. `bundle10_publish_pg.py` SHA525ac286…6f2d5 only expandsDIRECT_PGpublisher to96threads. Shared252headerandbarrier3unusedelsewhere; all96publishersshareoneacquiredfrontier via twobarriers; copycompletionbarrierprecedesuniquestagingrelease. All256possibleamounts preserveexactvectoraddressbijection. Exact isolatedbuilderASTs verified andreport savedpeer_cleanup_audit.json.

Router probe review found nativePGoracle initiallyreadnaturalK despite within64-column weight permutation. Sent helperowner exactinverse q=16*((k>>2)&3)+2*(k>>4)+(k&1)+((k>>1)&1)*8; independentlychecked4096scalars. PublicPGXOR formula correct. Other scratchreset,ownership,freshABBA andpoisonreplay checks reviewed; waitingownerfix beforeGO.

## First-issue accumulator initialization

`build_native_first_issue.py` freezes `bundle10_native_first_issue.py` SHA37a34353cffc2409170de92a3dd1d098e3d462968c5d819dc31445368a19f0a1. Originalsingleaccarray andoneinitialfullclear remain, preservingdefinedoddSTSMpaddingreads. OnlycurrentHASWG/T128/MW32 normalXPIPE schedule changes: WGMMAfirstst0/kk0/jj0 perrowarray usesdynamicuniformscaleD0; allotherMMAs retain1. TinyMMA.sync livecolumns resetbeforetheiritem, preventinghot→tiny staleC. Post-item reset omittedonlyforselectedconfig; archivedXPIPE2/A3/nonWGfallbacks retainoriginal. Unlikeearliertinychains, noextraaccumulatororrecurrencyschedule. CPU1024adjacentwidth/phasestatespass; sourceASTonlynativechanged. Local12.1healthy448HGMMA/112ARRIVE/126DEPBAR,168regs/136stack/same224store288loadspills/noC75. FSEL128→24,instructions19388→19172. Target13gate andpeerreview pending; noGPUhere.


Router probe nativePGinverse fixedbyharness and verifiedondisk. Final independentGO: eachvariantownsmutablecounts/buckets/slots/nslot/PGandinputs,allthreeresetsinsidetimedgraph,finalchangedinput+poisoncheck,normalizedvaluescomparedfreshnoPGoracle,routeweightsalignedtoFP64sigmoid. Nootherblocker.

Root `bundle11_public_small.py` SHA fafe565de1e92a5d5b7c36a4663ff88f3dbe8ea3fd7defd5ac630a1db1d7ad3e sourceGO: exactlypublicminimumM2048→512 plusnativePGdisabledforthesamerange. PublicPGstill8192+,native<512fallbackpreserved. PNRnormalizeddense→identitygatherfreshcounts/positionsvalid;8M+32*256rowandtilearenaboundsadequate. `_pm_FINE_MIN_T`remains2048,soM753runsFINE0coarsewidths,unliketheearlierFINE1fine-smallmicro;parentnotifiedforcorrectinterpretation.

Nativefirst-issue has independentharnessGO andattentionasyncGO. HelperpredicateindexN/2+5correctforall16wrappers;eachacc[r]firstst/u/kk/jj=0uses scaleD0,remainingcontributions1. Predicateisconsumedatissue,notanasyncoperandrequiringlifetimeuntilwait. Previousitemdrainsbeforetinypreclear;initialfullclearretainsdefinedpadding. OddSTSMtrailinggroupstorestokens>=8*ntl,outsideliveglobalcopyrange. No source blocker;targetnumerics/timingpending.


## Bounded public N64 regrouping

Parent reports native first-issue's small isolated gains did not repeat in whole-layer measurements. Keep it unaccepted. Public fine-width routing from M512 now has two positive full-turn comparisons (1.00163/1.00259) and passes all18 hard tests; acceptance/full score work remains with root.

Reviewed earlier failures before creating one new configuration. Seven N128G2 retained G4 at N8/N32/N64 and was healthy. Adding N16 to that broad-G4 mix serialized the ten-width kernel. All-G2 repaired it. No previous candidate restored G4 at only N64 while keeping N8/N16/N32/N128 on G2. The N64 four-chain experiment changed accumulator count and is a different, closed mechanism.

`build_public_n64_g4.py` freezes `bundle11_public_n64g4.py` on `bundle11_public_small_fine.py`, SHA08c422255f98ec8a608d1168d8235a547cfbed57d416cdfd4aa358c874c9c939. Only consumer_item changes: N64 uses two alternating A quartets (32 registers) and four MMAs/group. Its original two accumulator chains, every per-chain K/j issue, reduction, descriptor and staging output remain unchanged. Other widths retain globalG2. N64 essential storage is64 accumulator+32 A registers, below unchanged N128's128+16. It halves N64 fence/commit/wait groups per KPS2 stage (4→2), saving64 groups per complete item. CPU sequence/lifetime oracle verifies256 MMAs and32 shared-slot releases, and AST proves all planner/gather/PG/host/epilogue code unchanged.

Target13 `research/public_n64g4_gate.log` is healthy: F0 68HGMMA/34ARRIVE/40DEPBAR and F1 88/54/64, no C75 warnings,168 registers. F1 stack rises8→32 bytes with extra local accesses; F0 stack remains8 bytes. Do not assume a timing win from async cadence alone. Root owns GPU and may fetch target SASS to locate new spills. Local12.1 emits the same C7515 serialization for both control and candidate, so it is not a comparative rejection. Frozen device sources, local reports and .audit.json are retained.

Target SASS spill audit is saved in `n64g4_target/spill_analysis.json`, reproducible with `analyze_spills.py`. Both FINE1 PG forms have 28 B spill stores / 72 B loads and no local accesses inside any width's tensor mainloop. The extra slots hold down-epilogue address/index terms and timing locals: +8 is the j2=1 stmatrix address, +18 is q*8+g, and +1c saves soff=4096+(q*8+g)*8 across the N32 epilogue's thread-ID use. There are no accumulator spills. Root full-layer measurement nevertheless loses at all three sizes: M753 +1.784 us, M3012 +5.689 us, M16384 +49.800 us. Original N64G4 remains unaccepted.

At root request, `build_public_epi_remat.py` freezes one bounded compile-only repair `bundle11_public_n64g4_remat.py`, SHA5c5e55e62edaf690cf69fb8f58b4489ecdb96551dedb209477d12a3873f88a29. It reads a fresh volatile thread ID only in the down epilogue, after the existing async drain/staging barrier, and derives that branch's store coordinates from it. Mainloop, gate epilogue, FP arithmetic/order, barriers and output copies remain unchanged. All39,424 addresses agree for both consumer bases and all ten widths. Target gate must materially remove extra local traffic while retaining async cadence before any further GPU measurement; no timing benefit is assumed.
