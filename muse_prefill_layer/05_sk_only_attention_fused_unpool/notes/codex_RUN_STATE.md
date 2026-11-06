# Prefill top-2 research, 2026-10-29

Goal remains active until an official submission verifies rank <= 2.
User authorizes parallel workers and exactly one Modal GPU instance.

## Current evidence

- Official incumbent `incumbent.py` is identical to `submits/submit4_full25a.py`, SHA256 `51b11619f1908f7164bbb25491ded16516379461b50e325c9a50258a2c514e49`.
- Job `0432ea6e152a48a3`, 209914.271737 us, speedup 1.845533878, H200, 5/5 tests.
- Live initial rank 24; second place 191287.237148 us, speedup 2.0252. Required reduction 8.874%.
- Initial daily quota: 20 checks and 10 scored attempts available; none used this session.
- Previous work copied from neighboring `muse-opt/muse_prefill_layer`; original remains untouched.

## Compute and tools

- Persistent Modal sandbox ID in `research/runtime.json`. Root exclusively owns GPU scheduling.
- One H200, 143771 MiB, 700W, max SM clock1980MHz. Official image pinned in `research/modal_h200.py`.
- `python research/modal_h200.py sync <relative files/dirs>` uploads a frozen tar snapshot.
- `python research/modal_h200.py exec "python3 ..."` executes serially, with local lock.
- `python research/modal_h200.py fetch /tmp/file.json research/file.json` retrieves artifacts.
- `python research/modal_h200.py stop` terminates owned sandbox when finished/paused.
- Max lifetime24h, idle timeout1h. Check state before replacements; never create concurrent GPUs.
- Modal SDK1.5.3 requires `sb.filesystem.write_bytes/read_bytes`, legacy `Sandbox.open` removed server-side.

## Owners

- root: integration, dense projections/boundaries, GPU queue, official checks/submits.
- moe: `research/moe/`; small-expert scheduling, lossless decode, output staging.
- attention: `research/attention/`; packed half max, grid/combine, qkv post layout.
- harness: `research/harness/`; frozen checker and paired benchmark, scoring analysis.

## Queue / resume

1. Running: MoE base/stsm/deq6/both on M753,3012,16384, seeds2,reps12,check; log `research/moe_round1.log`.
2. Next: sync harness + submitted profile JSON, run `python3 research/harness/bench.py --control incumbent.py --self-check --candidate full28=cand_full28.py --reps 6 --check-reps 2 --json /tmp/prefill_initial.json`.
3. MoE tiny-WGMMA variants `wg_small4.py`, `wg_small8.py`, `wg_small2.py` (worker command in mailbox).
4. Attention probe variants include maxpair, post4, post2, inline_comb, grid128, fullunits; test M753 first.

## Current session results

- Initial full-layer self/full28/SGLang benchmark passed all8 scored shapes twice. Independent self copies agree0.03% aggregate. `research/prefill_initial.json` and `initial_bench.log`.
- full28 gains0.474% vs incumbent locally, ~1.1–1.9% large shapes. Official check `b1033df7778e4586` PASSED,205343.699765us, speedup1.886612. Result `research/check_full28_status.json`. This is off-board; current rank remains24. One check used,zero submissions.
- MoE round1: all errors equal~2.87e-3. stsm0–0.4% kernel gain; deq6 regresses1% larger shapes. No meaningful whole-layer win yet.
- Routing compaction ruled out: zero-local-route fractions4.25%,3.29%,1.59%; tile rounding erases most savings.
- Concurrent hot/tiny threshold48 corrected probe passed; best339.6us versus321.8us control (slower). No runtime partition testing warranted yet.
- Attention round1b: epilogue (fast division+factored norm) wins~1.8us/2.8us/15.6us on sliding cases0/2/5; output errors<=2.13e-5. Other exact tweaks mostlynoise; cw1dualregresses fullcases12–15%.
- Running MoE round2: base,wg8,wg2,small_decode,pair_scales,pair_scales_diet, all3M/seeds2/reps12/check; `research/moe_round2.log`.
- Next boundary probe `python research/dense/probe.py --ms 753,3012,16384 --reps10 --json /tmp/boundary_probe.json` (space required before10).
- Attention worker building follow-up ablations and integration stack; pending actual layer validation.

Do not confuse archived gains, microbench gains, projections or official off-board checks with leaderboard rank.

## Later session update (supersedes queue above)

- Best combined local file `research/integrated/full28_epi_pair_stsm.py`: all5hard+4edgecases,2replays each passed; all8scored twice passed. Relative to officially checked full28, aggregate1.005053524. Projected official score204311us, explicitly unverified. `BEST_LOCAL.py` points here. One official check used, zero submissions.
- MoE WG4 avoids compiler serialization but still regresses~2%small,0.4%large; do not use. Pairdiet+stsm replicated~0.3–0.5% kernel gains with duplicate controls. Globaldeq6 and WG2/WG8 rejected.
- BF16 fresh expansion feasibility probe correct but slower: native_hot4411us, expansion535us, dense4929us,total5428us. No grouped rewrite justified.
- Attention lazy OR guard preserves probabilities exactly (all8probes bit-identical) but regresses7–20%; rejected.
- Large Oproj+bnd1 G4/cap32 saves25us standalone,22us full-layercase5; cases4/5combined1.002478. Builder `research/dense/build_oproj.py`, candidate `research/dense/bundle1_bk4_g4cap32.py`. Keep as pending small additive win.
- Boundary coefficient retention512/Keep1 exact, saves3uslarge bnd1; full-layer noisy0.05%large. Not yet best.
- GPU currently testing root `research/pipeline/build.py` whole FFN token chunk overlap: after full attention, Oproj/boundaries/pool/router/MoE/UB run2x8192 or4x4096, each stream independentmutable state, directly writesoutputviews. Candidate paths `research/pipeline/chunk8192_s1.py`, `chunk8192_s2.py`, `chunk4096_s2.py`; log `research/pipeline_ffn.log`.
- Pending GPU: widepairMoE `research/moe/pairdiet_wide.py`; smallOprojsweep using updatedresearch/dense/oproj_probe.py; optionalMoEchunkcostboundresearch/moe/chunk_bound.py. Root owns oneGPU exclusively.
- Attention agent investigating adding attention chunk overlap after prewriting allKV; harness auditingpipeline/smallprojectionfusion; MoE agent trimming padded accumulator liveness to repair tinyWGMMA serialization.

## Structural round update

- Latest live board: rank25, same209914us submission; rank2still191287us. `research/leaderboard_latest.json`.
- Root FFNchunking2x8192/4x4096, with1/2streams and64/88/132SMs, passedfreshchecks but lost2–6%. Reject.
- B4request attention+FFN pipeline passed but lost11%full,15%sliding. B1full16kchunkpipeline passed butlost~1%. Stagedattention/FFN streams alsofailed;64SMpartition loses24–48%. Do notrepeat these branches without newmechanism.
- Exact pooler/router cuBLAS overlap negative: unrestricted1552→1569us;116/16SMs1648us;100/32SMs1795us. No integration.
- Corrected historical gate-stmatrix builder: previousexperiment initializedonlydownexpert31! Corrected all32experts +tensor maps, now all6realMoEchecks pass andgains0.2–0.4%small/medium,0.1%large. `research/moe/build_gate_stsm.py`. Integrated gate+bk4+bundle1 at `research/integrated/bundle2.py` pendingwholelayer.
- Genuine bucket-order changes (alternatelarge/tiny, tinyfirst) passed butlost1–2%; reject. Widepairconversion also~1%large regression;reject.
- New Apache2publicrepo fromhttps://github.com/vjeux/okc-2026, commitf35a41ee1c13e316dabbf772830d47b23b453393; copiedat research/public_okc_2026. Itswholepublishedscore215434us(rank60) is slower, but isolatedMoE128weightrows×256tokens layoutandasyncpublisher beatnative256×128.
- PublicMoEadapter `research/moe/public_moe_adapter.py` compilesupstreamCUDAextensiononce~1min, excludesattention/densetuning; gathersalready-normalizedinputwithoutnormalizingagain. All6realroutingchecks pass ~.00287error. PairedM3012 gains2.9/4.6%;M16384 gains5.4/5.7%;M753loses3.7%. Logresearch/moe_public_bound.log. MoEagentintegratingM>=2048only, reusenativecounts/positions, disableduplicatedPGgather, resetcurinfusedcombine. Rootwillvalidate fullstack beforeofficialcheck.
- Root newexperiment `research/public_tuning/build_publish.py`: use96publisherthreads insteadof32 ondown-outputcopies, sharedqueue observation+barrier3beforestagingreuse. `publish96.py` pendingGPU, harnessauditing.
- CurrentGPU: attentionepilogue vsnewearlyvote-max vs publicattentioncomponent, logresearch/public_attention.log. Nextpublicpublish96, integratedpublicMoE, wholebundle2 validation. OneGPU,1officialcheck,0scoredsubmissionsstill.

## Lessons recovered

- starter_pack workflow: paired cold graph runs, source hashes, frozen deliverables, keep each verified gain, full-stack validation.
- Auto-research-agent is an orchestration app with public starter prompts, not a source of private tuned kernels.
- Previous full28 fused unpooler/boundary helped large shapes ~1–2% locally, never submitted. Needs revalidation.
- MoE incumbent equals v29 CUDA source; v45 output staging small old gain, v46 incorrect.
- Dense GEMMs were extensively tuned and often near clock-adjusted tensor throughput; avoid repeating old tests without a new mechanism.

## Public integration + attention vote validation

- bundle2_public vs bundle1: all8scored×2checks PASS, aggregate1.013469. Native retained below2048; public for2048..MAX_M.
- Full-only earlyattentionvote skips rowmax shuffles only when warp-wide vote proves no threshold update; exact all8attentionprobes, ~1.4–2.5% fullattentiongain. Sliding path unchanged.
- Combined `research/attention/bundle2_public_vote_full.py` passed5hard+4edges ×2replays=18checks; fresh8score×2checks passed; full10rep paired speedup1.021566317 vs officiallychecked full28. Projection201009us, NOTofficial. BEST_LOCAL updated.
- Officialcheck ee0692843fb94ded pending. Sessionchecks2,scoredsubmissions0. Boardstillrank25, rank2=191287.237us.
- Publicpublisher96 correct butgain~0.1–0.2%MoEonly, notintegrated. Trace proves large width256mainloopcloseclock-normalizedTCfloor; finetiles/tinyscheduling are nextpriority.
- Harness profile.py renamed layer_profile.py to avoid stdlibshadow (cProfile import failure). Remoteoldfile removed. Hardcheck rerunpassed.
- CurrentGPUhybrid_moe_bound fixed masks onlyinitialbound: publichot+nativetiny samegraph, allroutesmaintained. Ifboundpositive, dynamicGPUmasks and trueFP32partialcombine.
- NexttinyWGMMA corrected firstscale_d=0/outputonlyMMA chains2/4; attentionworker12.1C7515 fixed, target13awaitsGPU.

Official combined check ee0692843fb94ded PASSED:201438.401529us,1.923187918x,5/5hard. Submitted identical file, job455b4f71a2a940b9 pending. Checks2,submissions1. Native tinychains2/4 allcorrectbut~6%slower; reject. Serialhybrid optimisticfixedmasks10–11%slower753,3–4%slower3012; reject. Finewidth5/9marginal/noise,notintegrated. CurrentGPUcompactnativefinish; nextpublicstage2+chain64, thenunpoolsmall3012.

Checkpoint submission455b4f71a2a940b9 PASSED201763.307472us,1.920090946x. Live rank12 (formerly25), second191287.237148us. Goal NOTachieved. BEST_SUBMITTED updated. Checks2,submissions1. Moreexperiments: publicstage2negative0.5–2%,chain64noise; compactnativefinishnegative0.3–0.5%;128x256unpool3012fusionbest353.49vs344.59usnegative. Nativecurrenttrace753research/native_trace753.log mainhot44.4us,small19.4usgate,epi1–2.4us. CurrentGPUweightcachekeep1/2/all. PendingpublicN8/N16, inactivecombinecompileelimination, epiloguetrimming.

Small-MoE round: current math padding only3.948% (1393real/1448issued rows,145.760GF), TCfloor161.43us@1.67GHz versus303us actual; tiny phases streamweightsnear4.2TB/s. Cachekeep1/2/all all3seeds negative1–2%; reject. Public N8/N16 tiny correctbut15–20%slower; public MMA32 correctbut20–25%slower. Both rejected pendingcompiler-resource diagnosis. Root row-blockgroup order builderresearch/schedule/build_row_order.py (groups1/4 andgroup32sameorder+allgathercontrol) CPUbijection andharnessreadinessauditpassed; pendingGPU. Attentionworkerbuildingprefix-preserving group4 toretainfirstwavegatheroverlap. MoEworkerbuildingWG1/64weightrow layout with2CTA/SM target for753. CurrentGPU native nocombine+epilogue trim (research/native_cleanup.log).

## Small-MoE compiler/occupancy round

- Native row-order group1/4 and all-gather control all correct but negative; prefix4 tied/slightlyslower. Reject these variants.
- Native trim+nocombine cleanup and direct-small STSM epilogue all3seedscorrect; paired cleanup1.003–1.005, directsmall1.004–1.007 vs gate. `research/native_directsmall.log`. Integrated pendingfullvalidation `research/attention/bundle3_native_epi.py` sha8373d2f0405ba675af5659c949800559bfae13cf825a12769dfc86d8fa250772. All34_pm_publicsymbolsbyteidenticaltoBEST. NoBESTlinksupdated.
- TargetCPU NVRTC prewarm fixed usingASTdependencyclosure (candidate import initialized CUDA viaSGLang, guardcorrectlyrejected). `research/harness/native_nvrtc_prewarm.py` compiled2sourcesin45.9sparallel; normalpreparethen0.8seach. Use3workersforfutureCPU-onlywarmup. Exactmoe.cu/sm90a/options/source.
- Target13public tinyFINE0 is globallyserialized:68HGMMA/68ARRIVE/68DEPBAR vsbaseline36/12/16. Peeledfirststage output-onlyfixstill100/100/100, correctbut~15%slower. Baselineandtinyall168regs/8Bstack. FINE1unchanged44/20/26. `research/extension_resources{,2}.jsonlog`. Attentionfour-width andearlyclobberdiagnosticsnowtesting.
- PublicWG1/64weightrows/max128tokens passescorrectness andactualtargetoccupancy128regs/40Blocal/2CTAperSM at256threads110KiB. Healthy32HGMMA/8ARRIVE/11DEPBAR (NOTserialized). Single132CTAs584/650us; double264CTAs395/440us vsnative303/307. Mixedqueue+p40c216producregs saves~1–4usonly; stillnegative. SplitN128 into2N64independentcolumnchains is nextprobe. WG1traceworkeradaptingtrace192items/tile/264CTAs.
- CurrentGPU `research/public_small_round2.log`: native/public/splitn/four8/clobber; afterbenchcapturesresources3. Session44377. Next fullvalidationbundle3 andWG1trace.
- Attentionbuildingseparatetinyepiloguebuffer toreleaseheldlastslotearlier; upperboundonly~5us/CTA affectedepiwork, notfullringdrain. Mustinclude1024staticshared:32x264x2scratchoverflowscapby512;use24x264x2=12672onlyn<=24. NoGPUyet.
- Official state unchanged:rank12,201763.307us submit455b4f71a2a940b9; target191287.237us. Checks2/submissions1. Goalactive.

## Latest fine-width / occupancy follow-up

- Fine8 planner/gather passes synthetic and three real seeds, but timing mixed (289.4/318.4/321.5us vs native302.8/308.0/314.1). Seven widths restore64, but target13 globally serializes FINE0 52/52/52 and FINE1 60/60/60 HGMMA/ARRIVE/DEPBAR. KPS8 also serialized68/68/68 and76/76/76. WG192 actual occupancy2CTA but all36/36/36 and44/44/44 serialized; static166/168registers,40B stack. Source audit passes. Not timed while diagnosing compiler.
- PTX setmaxnreg requires all four warps of a complete warpgroup; partial producer warpgroup in192-thread layout cannot legally use redistribution.
- Native tiny_epibuf target13 compiles cleanly and passes all3 real seeds, but 302.7/307.6/313.3us versus direct301.4/306.1/312.4. Reject. Log native_tinybuf.log.
- Bundle3_native_epi passed18hard/edge checks and8scored×2; local aggregate1.00107667 vs submittedbundle2, BEST_LOCAL updated. No new official attempt.
- Board rechecked rank12,201763.307us; rank2 target191287.237us. Checks2,submissions1. Goal active.

- Target13 PTX extraction establishes C7512 for seven widths, no C75 warning for healthy fine8. Files research/moe/target_codegen/*. Dropping160/192 instead of64 still globally serializes; reject both. Seven N128 two-MMA groups halves A registers32→16 and restores healthy F1 cadence60/26/33. Plan sentinel checks pass all threshold counts. Micro all3correct: native direct301.1/305.9/312.0us;seven2 285.7/305.0/318.2. Mixed benefit, not integrated. Adding16/224 is next. WG192group2 remains serialized; P24C240 only fixesF0, notF1.
- New SS-WGMMA WG192 moves weightdequant to producerwarp, targetcompilepending. Exact CPU mapping and independent barrier audit passed; only fused entrypoint valid. New public-PG inputscatterfusesrouter norm/gather, removesactivationcopyfromexistinggatherlaunch; targetcompilepending, layout/audit in progress.

- SS prototype target13 healthy167regs/8stack/2CTAs, all3seedcorrect, but SSfine745/790/885us vs native300/307/311. Reject currentone-decoder-warp architecture. Three-decoder-warps cannotbeatnative even underoptimistic2×decode bound; instrumentation pending to bound4warp variant. SSguarded siblings forbidunsupportednonfused entrypoints.
- PublicPG target13 healthy baselineWGMMA cadence,168regs16stack. GPUlayout8checkspassedM3012/16384 withfreshinputs,localownership0/32,coarse/fine,zero live/paddingmismatches. Wholebenchcases2/5running research/public_pg_bench.log.
- Adding16 tosevenG2 makes8/9/10widths globallyserializedagain, includingF0. Fourfineextras96/224cannotbejudgedbecause16alreadycausesregression. NextglobalG2allgroupedwidths andnine_without16diagnostics. Allpreserveweight/inputprecisionandfullwork.

USER STEERING: no further leaderboard submissions until an official off-board check is strictly below194ms. Goal remains verifiedtop2. Root alone controls officialchecks/submissions.

Latest follow-up: bundle4_public_ten passed18hard/edge and all8scored x2 checks. Full paired aggregate1.0018062066 vsbundle3; medium cases repeat~0.4%gain, large tied/noisy. BEST_LOCAL nowbundle4 (SHA17e6b386f4e18a50cc84a0cd5729f26954a279b228cd9ac78fdee094a7d1962d). Official scores unchanged201438.4us offboard /201763.3us submitted. No new official attempts.

Q16/CW2 attention targetNVCC13 and exactNVRTC13 pass168regs,0stack/spills,48QGMMA/12ARRIVE/8DEPBAR,zeroC75. Source frozen research/attention/query16.py SHA847e4ab57fa6c441a2bc539315e3b34cb0068951b05ce3caf96111e5b8e9fda4; CPU43296stream-Kblocks auditPASS. Independent audit/GPUprobe pending. TensorPG full-layer negative; compactPGtail large-only+.2–.35%, mediumnegative; skewtail layout8pass, fullbenchpending. R256cold firsthybrid allcorrectbutnegative; IMAD repair and threshold16 targetgateshealthy, fullbuild underway. Ringwarp-leader release targetC7513 globallyserialized; rejected.

- Ten globalG2 fixesall10widths (targetF1 88/56/66). Small753 mixed(native~301/306/312 vs ten286/301/319); medium3012 improves2.1%/.5%two seeds,16384ties. Integratedonlypublicrangeonbundle3: research/integrated/bundle4_public_ten.py SHA17e6b386f4e18a50cc84a0cd5729f26954a279b228cd9ac78fdee094a7d1962d. Fullcases2/3/5 allchecksPASS; gains.446%/.369%/.028%. All8/hard suitepending; BEST_LOCALunchanged.
- PublicPGstatic specialization no gain;copy4hotspills(skip). PGtailcompact preservesoriginalcontiguousTMAforeverytile, 8CTAs/expert and64MiBtailarena. Fastgatehealthy168regs8stack;layout8checksPASS. Fullcase2−.72%,case5+.35%all6pairspositive. SHA44bc1431862b58eeaf56a11d51b76434983e0c07b8c3ccf01db6058a61e503a6. TensorTMAalternativeSRC3ef19858e22ba59ba05f2e0b2059cf696e65088bd775b35a456e70a2ed5531c7 targethealthy168regs24stack; rank4UINT8maps gatherstridedK64planeswithonetensorTMA. Fullbuilddone, fullbenchqueued research/public_pg_tensor_bench.log.
- SSproducertraceclosedbranch:46.3%decode,27.3%rawWait,10.1%issue,11.2%unaccounted; ideal4×decode still510us vsnative300. No furtherSSfourwarpprototype.
- SinglekernelR256coldMMA/R128hotWGMMA frozenresearch/moe/public_r256_cold.py SHA f2fb0b02d015e437bd6fa0cbab510b3b02ad29ffe0e8ac2d98b7dc58fb66a1d9. FulltargetcompilehealthyF1 40HGMMA/32ARRIVE/39DEPBAR,112HMMA;168regs24stack. CPU bytepack/job/epi auditsPASS; GPUtwo-integerplan checkerPASSthresholdsynthetic. Micro3seedrunningresearch/public_r256.log.
- FastCPUtargetsourcegate validatedfine8exactcachedSASS/regs,8.3s vs45–65sfullbuild. research/moe/public_source_gate.py uses .cuMoEonly; compile2fusedinstantiations. RootmayrunCPU-onlygateviaSDKconcurrentlywithserializedGPUwork; sameonesandbox. FullGPUcallsstillmodal_h200lock.

## Stacked checkpoint and next isolated gains

- User explicitly values small repeatable gains and asks to stack all; no minimum-size cutoff. Submission gate remains passing officialoffboard<194000us.
- bundle5_ten_pg.py SHA bf9b279a4c04c39f82bf13e7d6c859e513f43f0672d6d8cc1a8c0ef7010332d6 is canonicalBEST_LOCAL. It is testedbundle4_public_ten_pg_large plusproblem-tagcommentonly. Adds large-onlycompactPGtails toten+nativeepicleanup. Layout4cases(all10widths), hard18checks, all8scorex2PASS. Direct16repAB/BA vs submittedbundle2:1.0031583087 aggregate, projected200804.2us (NOT official). Offboardcheck7493a4491e4d4707running; sessionchecks3/submissions1.
- PGlarge vsbundle4 isolatedoverall1.0013176867, largecases+.32–.54%; medium/native pathsunchanged/noise. OriginalPGtail_skew large+.323%,medium−.82%, no demonstratedskewbenefit.
- Q16CW2 primary fullattention21–24%slower; grouped14–22%slower. Bothfiniteandrelativecontrolerror.16–1.21%; notfullcorrectnessvalidated/nointegration. AllQ8slidingexact.
- R256IMAD16 correctbut5–6.5usbehindten. Stage4correctstillbehindtenall3seeds(293.4/306.2/321.7 vs286.4/301.1/319.3). R256hybridbranchclosed.
- Q8CW4 targetC7512 globallyserializedandspills;112consumerregs insufficient. full-onlystaticQ8 dropscodehalfbutmixed/noisytiming, notaccepted. Packedhalfcausalmask exactsource targethealthy12/4/4,0spills,maxR143vs155; runnableauditpending.
- Newboundedlevers: publicgatherdirectBF16copyavoidsredundantzero-weightnormalize; publiccombine2/4/8warps sweep; boundaryra reuse fromearlierRMS computation withonefloat/row,noextra kernel. WorkerspreparingCPUonly, rootschedulesonesandboxGPU.

## Latest stacked exact cleanups

User reiteratedretainallrepeatabletinywins. bundle8_compact.py SHA f2d0fbb38214576316a2460adfb60f40d6ab5627c2e1c8464aac18f4c65f8c51 isnowBEST_LOCAL. Rawbundle8_expq_stack.py SHA cd6f45f5de505ae0502c85d1f21278b3efb59658e4b23624045be60703728c7f passedall18hard/edgeandall8scorex2,32rep paired1.0030175713vsbundle5. CompactonlyremovesunusedATT_EMPTY_MASK_CUDA_SRC literal (0ASTloads); allremainingASTexact. Raw1026168bytesexceeded1MBAPIlimitandcreatednojob; compact941587bytesacceptedoffboardjob23df26ab67904f5d. Sessionactualchecks4/scoredsubmissions1.

Bundle5offboard7493a4491e4d4707 PASSED201740.832392us,slowerthanbest201438.401529withinjudgeinterval. Keepstackbecausepairedlocalpositive. BEST_OFFICIAL_CHECK/BEST_SUBMITTEDstillbundle2. Latestliveboardrank12,target2still191287.237us. No submissionsuntilpassingofficialcheck<194000us.

Bundle8stacksall: nativeepicleanup,tenwidthMoE,largecompactPGtails,empty-prefixpackedmask,unit-scalecombinecleanup,identitygathercopy,RAreuse,andcachedfull-attentionEXPQbias. Mask+unitbundle6passedfull8at1.00032018vs5. Identitycomponentall4byteexacttestspositive1.0039–1.0127(.08–.43us). RAcomponentbitexact,splitcoldincludingextraRAwrite+.448us753(11/12pairs),+.608us3012(10/12); largeUBunchanged. CachedEXPQfullattentionall5bitexactand1.7–3.4%positive(4/20/13/55/132us); componentcacheavoids8instructionsperunchanged-maxKVtile.

Cumulativeestimatefromadjacentpairedruns200200.08us (NOT official), directbundle8vssubmittedpending. Currentnewunit1slidingcachepositive~.8usmedium/2.5uslarge,butunit2smallnegative~.6us; workerpreparingsinglemod_sk sourceargumentchangeoncompact8toretainunit1only. Alltargetgateshealthy156/157regs,noSTL/LDL,C75. NativeMoEworkerinvestigatesunreachablegenericepilogue underREPI; noGPUcandidateyet.

## New official best and next round

Officialcheck23df26ab67904f5d PASSED **199955.328806976us**,speedup1.937452242, onbundle8_compact.py SHA f2d0fbb38214576316a2460adfb60f40d6ab5627c2e1c8464aac18f4c65f8c51. BEST_OFFICIAL_CHECK nowpoints8compact; BEST_SUBMITTEDstillbundle2 at201763.307us, rank12. Useraskedscore; answered199.96ms officiallypassed,1.48msbetterthan201.44, noLBsubmissionbecause194gate. Direct32repagainstsubmittedconfirmsbundle8 relative1.0061255526.

Retainedunit1slidingcachebundle9_unsplit_shared.py SHA6242548366e259f24e64d9b4611a837109aef5a9fd11afe5ded1b71a1c1b7da9,941598bytes. Onlyself.mod_sk sourceargumentchanges toexistingcachedfullsource; originalunit2/fullcachedsourcesunchanged. Scoredall8/hard18pass. Threevariantfullrunwasnoisylargeunchangedcases, soisolated40rep8vs9cases0/2/5 confirmsmedium+.713us,large+3.692us; component+.816/+2.544us. Keptfornextstack, officialbestprotected8.

Packedthreshold-vote variantbundle9_packed_vote.py SHA33288571d05b43c1ca3c712f2e82a97b8f58ca1295c297e2b88aa8e09b0626ff (942734bytes) hasexhaustive65536half/7.9mcomparisonproof+independentGO; targethealthyfull24/8/8,empty12/4/4,0spills. GPUbitexactbutmixed:small−.5us,B4−2.1us,empty−3.7us,ctx32k+3.8us,ctx64k+3.2us. Notaccepted; attentionworkerinspectingtargetSASSandpredicatecomparevariant. TargetSASSdownloaded research/attention/{expq_cache_full,expq_cache_empty,packed_vote_full,packed_vote_empty}_target.sass.

Nativeunreachable-epi exactbutmixed302.1→301.1/306.3→306.8/311.9→312.4; notaccepted. Nativeclear-live+descriptor target13healthy168regs120stack; independentauditsGO. Microbase301.6/306.9/311.7,clear301.6/305.7/310.6,desc301.1/307.1/311.5. Clearbeingfull-layerchecked; descnotyetaccepted, workerpreparinglow32add. Composedbundle10_clear.py SHA c1163a90eeeea7843ba7e2bd38bf715f363389c495eea9b8f2b3403499141351,941791bytes, onlyCUDA_SRCchangesfrom9. CurrentGPUfourcase40repbenchresearch/bundle10_clear_bench.log thenhard18research/bundle10_hard.log. NativeworkerpreparingoptionalrawpairedJSONharness; rootGPUonly.


## Bundle10 local stack accepted

Bundle10_clear all18hard and all8scoredx2PASS. Direct40rep vs official8 aggregate1.0009243038 (~199770.68us projection, NOT official). Cases0/1 improve1.589/1.674us, large sliding3.177us, other unchanged paths noise. Previous isolatedclear vs9 cases0/1 improve2.088/1.020us; case0 18/20positivepairs. BEST_LOCAL nowbundle10_clear.py SHA c1163a90eeeea7843ba7e2bd38bf715f363389c495eea9b8f2b3403499141351. BEST_OFFICIAL_CHECK remainscompact8 at199955.328807us. All tinywins retained; noofficialsubmission. Next: native low32descriptor, scalarthresholdcache, recovered d5 coefficient retention, root kv_pack prefixscan reduction.


## Bundle11 threshold and boundary stack

All18 hard/edge and all8 scoredx2 pass. Full40rep speedup1.0009382184 vs10; BEST_LOCAL now research/integrated/bundle11_threshold_keep.py SHA68e0dc2bdb87e0d15ca9c734a8711aa80df4ee649b8db4a696feafe65b1429fe,942368bytes. Scalar attention threshold cache bitexact all5 full cases, saves1.552/6.544/4.784/13.952/30.608us,14/16 then16/16 pairs positive. Boundary coefficient retention current RA path bitexact, split-cold+1.584us,17/20 positive. Official offboard launch pending; no submission.

Root KV packing warp scan passes16 byte/prefix GPU cases and2 independent CPU audits, but mostly tied or0.1us slower. Preserve unaccepted candidate research/packing/bundle10_warp_scan.py; do not stack. New probes: publisher96 only DIRECT_PG (target13 healthy), native descriptor32 (healthy), unsplit scalar threshold (source GO), router FP32 fast sigmoid intrinsics (source only).


OFFICIAL BEST: check02dd9d6ca1c748a9 PASSED 199454.169947us, speedup1.942320384, onbundle11 SHA68e0dc2bdb87e0d15ca9c734a8711aa80df4ee649b8db4a696feafe65b1429fe. BEST_OFFICIAL_CHECK andBEST_LOCAL both11. Improvement501.158860us vs bundle8. Five checks, one scored submission; board remains201763.307us rank12 (last verified), no new submission before194000us. User told199.45ms official. Next GPU: first-zero attention, then reversed-order unsplit scalar, then routerfastsig pending helper peer audit. PublisherPG full correct but6.7–10.3us slower large; unaccepted. Native desc32 tie/slower across3seeds; unaccepted.


## Next bundle12 components

Unsplit scalar threshold repeat confirms medium+.296/.512us andlarge+1.480/1.864us withreversedvariantorder; bitexact. Router FP32 __expf/__fdividef saves.216/.616/1.624us at753/3012/16384 (12/16,14/16,16/16 positive); all18component checks pass, sameIDs, maxweightdelta5.96e-8, normalizedinputsbitexact.

First-zero attention initialaccumulator rescale cleanup passes exhaustive65536half proof, independentactual-preprocessing audit andall7componentcasesbitexact. Two fullattentionruns combinedpairedsavings.576/.928/1.096/.568/-.584us forcases1/3/4/6/7 (22/36,26/36,30/36,21/36,17/36positive). Lastcase ties atnoise; retain aggregate tinygain forstack.

Publicfine MoE small expansion: publicmin512 plus _pm_FINE_MIN_T512, disable nativePG in samepublicrange; allCUDA unchanged. research/moe/bundle11_public_small_fine.py SHAb15e606d69080b40be5d2ca913f5112ebd0d384e1ba7295153c38a0cf8c1cb0f. Fullcases0/1 at40freshreps +.599/+2.170us, turn1relative1.00163038; repeat48freshinputsoffset2000 +2.429/-.819us, turn1relative1.00259084. All18hard/edgepass. Coarsepublicsmall loses6–7us; leaveunaccepted.

Nativefirstissue scaleD0 all3seeds correct, micro+.816/-.032/+.160us (17/20,10/20,11/20) butfullsmallcases-.125/-.634us. Preserve unaccepted; finepublic beatsnativeinrepeatweightedload. Currentofficial11 protected.

Router sentinel eliminates taken bitmask bysettingselected st=-Inf; exactadversarial131074localpatterns/2048rows perFASTproof andpeerGO. Targetpnr2y48regs vs56,healthy. Componentlarge+5.552us16/16positive,medium+.072tie,small-.136negative. AttentionworkerpreparingPG-onlysentinel tokeepnonPGoriginal andlargegain. Rootwillcomposeallconfirmed additions into12,thenhard/full. HarnessworkercheckingpublicupstreamcurrentGitHub fornewideas.


## Bundle12 validated

BEST_LOCAL now research/integrated/bundle12_small_router.py, SHA4377ef923d2d1cf48fe812a914b44d44a2ddc09331b001a95534b1c5246367ba,945014 bytes. Target router/attention gates pass; full48 paired repetitions all8 scoredx2 and hard18 pass. Aggregate1.0031869230 vs official11. Case savings5.866/3.717/-1.049/3.822/7.130/7.077/7.196/8.050us. Projection198820.544180us is NOT official. Check launch pending; official11 remains best199454.169947us, no scored submission before194000us.

Bundle12 official offboard job8e6838c863164d3b accepted. Sessionchecks6/scored1. Status waiter research/bundle12_official_check_final.json. OneModal sb-<id> still root-only. N64G4target13 healthy butFINE1 extra spills32stack; independent sourceauditGO. Fullpublic extension build queued thenfullcases0/2/5 comparison onfinepublic11 baseline.


OFFICIAL NEW BEST: bundle12 check8e6838c863164d3b PASSED198050.455700us, speedup1.956086890. Improvement1403.714247us vs11. Useraskedscore, answered198.05ms official. BEST_LOCAL/BEST_OFFICIAL_CHECK both12; BEST_SUBMITTED unchanged201763.307472us/rank12. Six checks,one scored submission; no submission before194000us. N64G4 full0/2/5 allcorrect butslower1.784/5.689/49.800us,unaccepted. MoE prepared compile-only remat sibling bundle11_public_n64g4_remat.py targetingepilogueaddressspills; source/targetgatepending. MoEworker thenhitusage limit; noautomaticretry/modelchange. Harnesspublicupstreamresearchpending. Goal active, not complete.


User explicitly requested an official submission, authorizing this candidate despite the prior194ms gate. Submitted exact verified bundle12 (SHA4377ef923d2d1cf48fe812a914b44d44a2ddc09331b001a95534b1c5246367ba), scored jobcc229f30eb514d6a, pending. No code changes. Session counts:6 checks,2 scored submissions. Goal remains top2.


## Bundle12 official submission completed

User explicitly authorized this submission despite the previous194ms gate. Job `cc229f30eb514d6a` passed with checks OK, no flags:198669.742156us, speedup1.949989444. Live leaderboard rank9 (previous rank12); improvement3093.565316us versus previous201763.307472us. Exact bundle12 SHA4377ef923d2d1cf48fe812a914b44d44a2ddc09331b001a95534b1c5246367ba. `BEST_SUBMITTED.py` updated. Offboard check remains198050.455700us. Top-two target191287.237148us; goal remains active. Original194ms gate retained for unsolicited future submissions.


## Post-submission small-gain round (13/14)

See `research/round13_status.json` for exact sources, measurements, completed CPU gates and current pending validation. Bundle13 combines non-PG address rematerialization, disabled unused public item timers, PG-only sorted router candidates, and persistent work-queue scratch cleared in the existing combine. Bundle14 adds FP32 route-weight fast reciprocal. No official check or scored submission launched in this round yet. Do not overwrite best pointers until full checks finish.


## Bundle13 official result and bundle14 retained

Official13 f69dd0b96a124ca1 PASSED199015.197543us; best official12 remains198050.455700us. Local13 paired gain1.002121410 retained. Local14 fast normalization repeats component savings and full1.000257828; all8x2+18hard pass. BEST_LOCAL14, official pointers unchanged. See CURRENT/round13_status for running two-phase combine.

Round14 continuation: no new official runs. Closed-form packing exact but mixed; direct unit counts exact but scored sliding slower0.3–0.7us. Two-phase/zero-route combine negative. PG exp2 guard/compiler effects under investigation; dynamic combine1/2 currently running. All best pointers remain14local/12official andsubmitted. See round14_status.json.

Official14 f4dcd2496f2c439f PASSED196176.475646us (196.18ms), speedup1.974772453, noflags. Bestofficial14, submitted12unchanged198.67ms; checks8/submissions2. Useraskedfullend-to-endroofline: historicalfull5/6/10/12/17/24 reportsfound; initialNo corrected. Historicalweightedideal136.224ms/chain152.154ms basedoldcounts/peakrates, notcurrentachievablebound. Refreshcurrentroofline next. Fixedunitcomponentunit1positive~.5/.8us,unit2smallnegative.


Current roofline completed: all8 actual cases and routes, current attention math and MoE widths; ideal128.195–140.879ms, current-stage144.577–165.017ms analytical estimates, not attainable-score claims. Files bundle14_roofline.json/txt. NCU actual graph MoE cases0/2/5 pass correctness; tensor63.02/81.58/90.51%, DRAM39.54/23.01/10.51%, clocks1.68–1.69GHz. File bundle14_moe_roofline_summary.json. Best official remains196.176475646ms. Unit1 medium repeat gain retained as pending candidate bundle14_fixed_unit_medium.py, full48/hard validation running. Finer13 builds serialize, N80 spills16B/48B; no performance run.


Round15: latest explicit scored submission14 passed198928.464036us; best submitted12 remains198669.742156us/rank9. Best check14 remains196176.475646us. Local15 non-PG208 and16 coalesced gather retained, each all8x2+18hard pass. BEST_LOCAL16 SHA6638f9d21a1f3182987a141dccf3bcc80639ff3e1ae8471a526c9f6cd4babd6e. All111 kernel counter profiles and source-PC stalls collected; see round15_status.json. Metadata32-part bitexact and saves~18us; further comparison running before full validation. OneModal sb-<id>. No verified194ms score.


User tightened the judge gate: no off-board checks unless a passing full-layer local run measures below194000us. This is now recorded in progress.offboard_check_gate. Prior leaderboard restriction remains. No judge action authorized by this update.


## Resume after user asked why work stopped
Work resumed; no off-board checks permitted until passing measured full-layer local <194ms. Bundle18 direct64 vs14 passed,1.004039207 relative,198117.820953us local. Retained19 small matrix stores: three fresh component savings0.240/1.008/0.296us, exact outputs, all8x2+18hard; other size compiled kernels byte-identical. Its full-run measured198419.969423us is not a new best absolute score. Width24 WGMMA serializes; warpMMA healthy but measurednegative. Packed row-max attention candidate removes8staticSHFLs; component bitexact/no improvement, full0/2/5probe running. Root owns Modal sb-<id>. No worker-limit retries.


Aggressive round: QKV fusion implemented in research/dense/qkv_fusion.py, 6 transposed +2 matrix-store +6 direct-register configs all component checks within7.3e-5 butnegative(best57.54us vs48.2us). Same transposed math-only bound45.06us is diagnostic only. Four small-shape MoE ring/K-batch variants compilehealthy, component3x80 currentlyrunning. Two tiny warp-MMA alternatives compilehealthy/no spills, CPU exact-extensionprewarmrunning. Register P24C240 variants produce64-80B stack; deferred. K8 full8x2passed but hard suite didNOTrun because wrapper exec swallowed chainedcommand; hard remains pending. No offboardchecks;198.117821ms remainsbestmeasuredlocal.


Retained bundle20_small_warp_mma.py SHA45df0d1b41d3f50e5d52e2df39d3dd6e93ff1924c09d72d75b2dffe54440e3d3. New measured local best198098.920553us,64paired fullspeedup1.000762750vs19; all8x2scored+18hardpass. Independent96rep smallload1.002161695x. Nsight2freshinputs/2allocationcopies confirms MMA8gain0.616–1.256us; non-smallcompiledkernelinstructionwordsidentical. K8+MMA8stacknegative, MMA16mixed. Gateclosed;nojudge. AttentionNS4/12,PP1/2probe running,decode-reuseCPUgate/prewarmready.


Latest retained stack21: research/integrated/bundle21_decode_reuse.py, SHA 9cce96e7a1fd0d940364506da96c1a41bb087fded0734a0871174790a21b27a0. All16scored+18hard pass; paired1.001012340xvs20. Latestlocal198.187677910ms; bestabsolute remains20at198.098920553ms. No judge gate opened. See CURRENT.md for authoritative current state.

Retained22 research/integrated/bundle22_medium_decode_reuse.py; latestlocal198.170966809ms; bestabsolute198.098920553ms. All16scored+18hard passed; three fresh component gains and medium-load gain. Small/large binaries identical; gates closed.

## Stack23 gate queue retained

Best local197.708456489ms; full64 paired1.000321188x vs22; changed small-load1.001191428x. All16scored+18hard pass. Sourceresearch/integrated/bundle23_gate_queue.py,SHAbd39eec310c038765888ba0ad8938d7dee9aa9d16abb14b43a7a22e6ea45d6cc. Three component gains1.896/2.984/0.752us,116/120positivepairs. Four non-small expert binaries identical. BEST_LOCAL points23. Official198.669742156ms unchanged; no check or submission until local194ms gate.

Causal queue investigation: full interleaving lost15–32us; counters plus per-item traces identify20us added CTA finish imbalance and delayed gate dependencies (down max77vs40us). Reserving largest-first gate drain and leaving down order intact recovers this and yields retained gain. Artifacts queue_item_trace_summary.json/mixed_queue_counters.csv. Tiny-only12experts require339.74MB weights, observed347.57MB at3.368TB/s, not proof of optimality.

Width240nonPG small gain did not repeat; medium regresses, not retained. Width80nonPG mediumC7512, noGPU. AllwidthWG192globalG2 stillC7512; diagnostic width32 and256 individually healthy,128 serialized. Single-chain128alone does not cure wholekernel. NextCPUexperiment disables old trace instrumentation and rematerializes epilogue coordinates.

## Explicit official submission 2026-10-30T05:45:40.037715+00:00
User: "just submit this to official leaderboard." Submitted exact validated stack23 (`bd39eec310c038765888ba0ad8938d7dee9aa9d16abb14b43a7a22e6ea45d6cc`), job `fa9e150bebe74d1f`. No preliminary off-board check. This one scored submission explicitly overrides the194ms gate. Awaiting official result.

Official result: passed, score194138.214909752us, speedup1.995505625619, no flags. New leaderboard best; rank6 verified 2026-10-30T05:50:49.689819+00:00. Prior best198669.742156057us, improvement4531.527246304us. One-shot user override consumed; no pending judge job. Result `research/bundle23_official_submit_final.json`, board `research/bundle23_leaderboard_after.json`. Top2 not achieved.

## Rank1 target and sandbox restart 2026-10-30T06:37:55.618814+00:00
User raised goal to top of leaderboard. Official stack23 remains194138.21490975245us,rank6; leader190604.3542384025us. Prior one-instance and194ms submission/check gates persist. User stopped sb-<id>; poll confirmed0. Explicit instruction to start again: created sb-<id> as the sole replacement. Restoring local code/harness; old un-fetched epilogue diagnostics not counted as completed.

## Replacement instance diagnostics and dual consumers
Epilogue trace completed with bit-identical outputs on3fresh inputs pervariant. Width256 gate mean activation/store2.755us(oldqueue) vs1.413us(retainedqueue); staging wait~0.12usunchanged. The retained queue last32 gate tiles average~0.85usactivation/store. `epilogue_trace_summary.json`.
New dual-independent-consumer design shares oneCTA384threads,227KiB,fullproducerWG40regs+two232regconsumerWGs=64512regs. Separate rings/queues/barriers. RS and SS variants pass CPUasyncgate withoutC75/spills; resources verified1residentCTA,168launchregs. Initial component gather crashed because olderadapter lacked>=32nonlocalsentinelguard; fixedinbuilder. No valid new timing yet. `research/moe/build_dual_consumer.py`; `dual_consumer_gate.log`, `dual_consumer_repaired_gate.log`.


2026-10-30T07:32:05.710694+00:00 — Retained bundle24_wide_gate_fma.py, SHA bfc016171b7746110e896e237a0b7da9cd04b4232f893a3d799164d4da21b916. Wide (N>=128) small-input FP32 gate epilogue factoring/FMA; best complete local197.367280797ms, paired1.000200416x overall; small load1.001415327x and prior1.000310609x. All16scored+18hard/edge checks pass; four medium/large binaries identical. Six fresh component seeds show five positive and one -0.256us; mixed per-input magnitude, positive average. Official unchanged194.138214910ms/rank6 at last query. No judge action: local gate closed. Small-only80/112 tail component probe now running. Dual-consumer and weight-prefetch variants correctly computed but slower; archived evidence.


2026-10-30T08:18:31.150848+00:00 — Retained25 exact32-bit sliding descriptors; all16scored+18hardpass;6freshactualQKVpositivebitexact;fullrepeat2/5positive1.232/1.813us. Latestlocal197.391360873ms;bestabsolute24at197.367280797ms;official23at194.138214910ms. Nojudgegateopened. Early gate loads stackedprobe; unpooler full-ring waits underinvestigation.


2026-10-30T08:22:45.043065+00:00 — User urgency: finish current candidate, defer newexperiments. Full early+address32 run197.366009334ms, all16scored+18hardpass; best absolute effectively unchanged.128rep changedcases2/5 repeat running, beforepromotion. Official194.138ms unchanged.


2026-10-30T08:27:23.036716+00:00 — Final scoped attention validation completed, retained26 328cfc3a4f022ea8985a0810358dacb28a9fd761403d136461c7a74b8f7881e5. All16scored+18hardpass. Medium earlier gate-load gain two positivefullruns/thirdflat; large change not retained. Latest197.493077329ms;bestabsolute passing197.366009334ms(all-sizeearlycandidate);official194.138214910ms. No meaningful overall score improvement established. No GPU job or judge action pending. Rank1 goal unachieved. User requested urgency; checkpoint saved.
