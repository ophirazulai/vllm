# vLLM V1 × Evolutionary Code-Search: An Optimization-Candidate Map and Literature Review

> **Repository anchor.** All code-location claims in this report are anchored to `https://github.com/vllm-project/vllm` (default branch `main`). The local checkout used to verify paths and line numbers in this revision is at commit `e6ff3e9c8` (HEAD of the local working tree at the time of this audit, May 2026). The file `vllm/v1/core/sched/scheduler.py` is verified live at **2,295 lines** with the method line numbers listed below. Earlier drafts of this report referenced commit `c42ff4f` (from issue #27441 / PR #25903 line refs); those line numbers have been refreshed to the new HEAD. Any user reproducing this should re-run `git rev-parse HEAD` on the current `main` and re-verify line ranges, since this codebase changes weekly.

---

## Executive Summary (≤300 words)

vLLM V1 is a re-architected serving engine (alpha released Jan 27, 2025; default since v0.8.x) organized around a multi-process design: an **API server** process, an **EngineCore** process running the scheduler + KV-cache manager, and one **GPU worker** process per device. Stable models, kernels, and distributed plumbing were inherited from V0; the *new* code that matters for evolutionary search lives almost exclusively under `vllm/v1/`, plus the still-shared `vllm/model_executor/layers/` (kernels, MoE, quantization), `vllm/attention/ops/` and `vllm/compilation/`.

The highest-ROI evolve targets cluster around four families:

1. **Heuristic decision logic** with clear scalar rewards: the V1 scheduler's `Scheduler.schedule()` (vllm/v1/core/sched/scheduler.py), the prefix-cache block hashing/eviction in `vllm/v1/core/kv_cache_manager.py` and `vllm/v1/core/block_pool.py`, the speculative-decoding draft-length policy in `vllm/v1/spec_decode/eagle.py` and `ngram_proposer.py`, and CUDA-graph mode/size selection in `vllm/compilation/`.
2. **Triton kernels** with measurable latency: fused MoE (`vllm/model_executor/layers/fused_moe/fused_moe.py`), Marlin/Machete-adjacent quantized GEMMs, RoPE / RMSNorm / sampling kernels (`vllm/v1/sample/rejection_sampler.py`, `topk_topp_sampler.py`).
3. **`@triton.autotune` config search spaces** — the existing `configs/*.json` files for fused-MoE per-shape tuning are a natural drop-in target for FunSearch-style program search.
4. **Numerical–performance trade-offs**: KV-cache compression / FP8 quantization paths, prefix-cache hash function choice, structured-output mask construction.

Literature anchors include PagedAttention (SOSP'23), FlashAttention-1/2/3, FlashInfer (MLSys'25 best paper), Sarathi-Serve / DistServe, EAGLE-1/2/3, RadixAttention/SGLang, AlphaEvolve (DeepMind 2025), FunSearch (Nature 2024), KernelBench, AI CUDA Engineer (Sakana), Kevin, CUDA-L1/L2, and OpenEvolve. Frontier 2025/2026 work (Kernel-Smith, KernelFoundry, KernelEvolve, EvoEngineer, GEAK-OpenEvolve) directly demonstrates evolve-on-Triton viability.

---

## Recommended First Targets (5–10 highest-ROI evolve candidates)

These are ordered by expected ROI, weighting (a) isolation of the function, (b) cleanness of the fitness signal, (c) headroom suggested by the literature, and (d) maintenance risk for vLLM itself.

### 1. **Fused-MoE Triton kernel autotune configs** — `vllm/model_executor/layers/fused_moe/configs/*.json` + `fused_moe.py`

- **What to evolve**: The per-(num_experts E, intermediate-size N, GPU device) JSON config files that select `BLOCK_M / BLOCK_N / BLOCK_K / GROUP_M / num_warps / num_stages` for the fused MoE Triton kernel (see `vllm/model_executor/layers/fused_moe/fused_moe.py`). The structured space is small enough for evolutionary search and large enough that hand-tuned grids (cf. PR #19443, #31442 for B200 Qwen3MoE) miss optima.
- **Evaluator**: `benchmarks/kernels/benchmark_moe.py` (already in repo) — ms/iter at fixed (M, K, N, E, top-k); correctness via tolerance to a reference dense-then-gather PyTorch implementation.
- **Fitness**: minimize median latency over a workload distribution (M ∈ {1, 16, 64, 256, 1024, 4096}); enforce `max_relative_error < 1e-2` vs. reference fp32.
- **Expected gain (literature)**: GEAK-OpenEvolve and Kernel-Smith report 1.5–2.5× over vendor defaults on similar shape regimes; vLLM's own `configs/` directory shows ~1.2–2× swings between hand-tuned variants.
- **Risk**: low — purely additive; fall back to default config on any regression.

### 2. **Scheduler waiting-queue ordering policy** — `vllm/v1/core/sched/scheduler.py::Scheduler.schedule` (verified at line 352, full file 2,295 lines at HEAD `e6ff3e9c8`)

- **What to evolve**: The heuristic that selects which waiting request to admit next, the chunked-prefill chunk-sizing rule (`long_prefill_token_threshold` interaction with token_budget), the preemption victim selection (currently `max(running, key=(priority, arrival_time))` for PRIORITY policy), and the `skipped_waiting` re-queue logic (linked in issue #27441).
- **Evaluator**: `benchmarks/benchmark_serving.py` driving `vllm serve` against ShareGPT, Mooncake-Conversation, BurstGPT, and a synthetic mixed-prefill-decode trace. Metrics: TTFT p99, TPOT p99, completed-request goodput under SLO.
- **Fitness**: weighted sum of (TTFT p99 violation rate, TPOT p99 violation rate, throughput); evolve a small Python policy function with restricted API access (only `Request.priority`, `arrival_time`, `num_prompt_tokens`, `num_computed_tokens`, queue lengths, KV-cache utilization).
- **Expected gain**: Sarathi-Serve (OSDI'24) and Llumnix (OSDI'24) suggest 1.5–3× goodput from better scheduling; AlphaEvolve specifically reports a discovered datacenter scheduler that recovered 0.7% of Google's compute.
- **Risk**: medium — must run safety harness for fairness regressions (cf. VTC, Locality-aware Fair Scheduling).

### 3. **Prefix-cache block-hash strategy** — `vllm/v1/core/kv_cache_utils.py` + `block_pool.py`

- **What to evolve**: The block-hash function chain (`sha256` vs `hash`, mixing of multimodal extra keys, parent-hash chaining order) and the LRU/cached-block eviction priority. The `caching_hash_fn` choice in `KVCacheManager.__init__` is already a knob.
- **Evaluator**: synthetic workload mixing system prompts, few-shot prefixes, and multi-turn chats with controllable hit rates (0%, 30%, 70%, 95%); measure prefix-cache hit-rate AND end-to-end TTFT.
- **Fitness**: maximize cache_hit_rate × throughput, subject to "no false positives" (block-hash collisions must produce identical token sequences).
- **Literature**: RadixAttention (SGLang), Hydragen, ChunkAttention, BatchLLM, MemServe / Mooncake KV-centric design.
- **Risk**: medium-high — correctness bugs here corrupt outputs; require collision tests on 1B-token corpora.

### 4. **Speculative-decoding draft-length / acceptance policy** — `vllm/v1/spec_decode/eagle.py`, `ngram_proposer.py`

- **What to evolve**: (a) Per-request adaptive `num_speculative_tokens` (currently a static config); (b) ngram `prompt_lookup_min/max` heuristic; (c) the EAGLE chain-vs-tree topology and pruning rule. Acceptance-rate–aware draft length is a natural fit for evolutionary search because the objective (token-acceptance × throughput) is a simple scalar with a Bernoulli-mixture reward signal.
- **Evaluator**: `examples/offline_inference/spec_decode.py` on InstructCoder + MT-Bench (already in repo as PR #18847); record acceptance length AL and end-to-end tokens/sec.
- **Fitness**: tokens/sec (which already encodes the AL × draft-overhead trade-off); break ties by AL.
- **Literature**: Leviathan et al. ICML'23, Chen et al. 2023, EAGLE-1/2/3 (Li et al. 2024/2025), Medusa (Cai et al.), Lookahead (Fu et al.), REST, PLD, Online Speculative Decoding, Sequoia, SpecInfer, P-EAGLE.
- **Risk**: low–medium — wrong draft length only affects latency, never correctness, because the rejection sampler already preserves the target distribution.

### 5. **Top-k / top-p sampling kernel** — `vllm/v1/sample/ops/topk_topp_sampler.py::forward_cuda` and `apply_top_k_only`

- **What to evolve**: Replacement of the `forward_cuda` top-k/top-p path (currently mixes sort, FlashInfer rejection-sampling, and a reduce). FlashInfer's "sorting-free GPU sampling" blog (March 2025) shows there is real headroom; `apply_top_k_only` explicitly avoids full-vocab sort.
- **Evaluator**: a microbenchmark over (vocab_size ∈ {32k, 128k, 256k}, batch_size ∈ {1, 8, 64, 256}, k ∈ {1, 50, 1024}). Correctness via Kolmogorov-Smirnov test against a reference PyTorch sort+sample on 100k samples per config.
- **Fitness**: minimize per-token sample latency; reject any candidate failing K-S at p<0.01.
- **Literature**: FlashInfer (Ye et al. MLSys'25), Sorting-Free GPU Sampling (FlashInfer blog 2025), `min-p` sampling paper.
- **Risk**: low — kernel is functionally isolated and has a strong correctness oracle.

### 6. **Rejection-sampler Triton kernel** — `vllm/v1/sample/rejection_sampler.py::rejection_greedy_sample_kernel` and `expand_kernel`

- **What to evolve**: kernel block sizes, MAX_NUM_TOKENS unrolling, and the `apply_sampling_constraints` fused-vs-split strategy.
- **Evaluator**: spec-decode microbench measuring sampler-only ms; correctness against the existing PyTorch fallback.
- **Fitness**: ms/draft-token at fixed acceptance rate.
- **Risk**: low.

### 7. **CUDA-graph mode dispatcher** — `vllm/compilation/cuda_graph.py` + `vllm/compilation/piecewise_backend.py`

- **What to evolve**: (a) The set of `cudagraph_capture_sizes`, (b) the FULL vs PIECEWISE vs FULL_AND_PIECEWISE selection rule per (model_size, attention_backend, batch_shape) — currently a static mode flag. This is a textbook configuration-selection problem with expensive but bounded evaluation cost.
- **Evaluator**: `vllm bench latency` across {Llama-3-8B, 70B, Qwen3-30B-MoE, DeepSeek-V3} × {H100, B200} × representative TTFT/TPOT shape distributions.
- **Fitness**: end-to-end p50 + p99 latency; penalize capture memory overhead.
- **Literature**: vLLM CUDA Graphs design doc (`docs/design/cuda_graphs.md`); torch.compile / Inductor partitioning literature.
- **Risk**: low — mode selection can fall back to current default.

### 8. **FlashInfer plan() / metadata-builder dispatch heuristic** — `vllm/v1/attention/backends/flashinfer.py::FlashInferMetadataBuilder.build` and `plan()`

- **What to evolve**: The choice between cascade vs flat attention, the `prefill_fixed_split_size`, `disable_split_kv`, and the TRTLLM-vs-FlashAttention dispatch on Blackwell — currently a hand-coded if/elif lattice (see DeepWiki line refs `flashinfer.py:1148-1198` for cascade, `656-694` for CUDA graph capture).
- **Evaluator**: long-context attention benchmark (4k → 128k) against the existing reference output.
- **Fitness**: kernel-only latency; correctness via element-wise tolerance.
- **Risk**: medium — must not break MLA/Sparse paths.

### 9. **Quantized GEMM auto-selection (Marlin / Machete / cuBLAS-FP8)** — `vllm/model_executor/layers/quantization/utils/marlin_utils*.py` and Machete heuristics moved into C++

- **What to evolve**: The decision tree that maps (M, K, N, group_size, bit-width, dtype) → kernel implementation. The Machete PR (#7174) explicitly notes "Improve heuristic namely for 4096x4096" and "Improve batch size <32 performance" as open work.
- **Evaluator**: `benchmarks/kernels/benchmark_machete.py` and `benchmark_marlin.py` (already in repo).
- **Fitness**: per-shape ms; constrained correctness vs FP16 reference within bit-width-appropriate tolerance.
- **Risk**: low — decision logic is isolated and reversible.

### 10. **Block-aligned chunked-prefill split for Mamba/hybrid** — `vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split` (verified at line 302)

- **What to evolve**: The block-alignment policy for Mamba/hybrid models — currently a hand-coded snap-to-block rule. With more state-space models entering V1 (Qwen3-Next, Jamba), evolved rules can balance cache hit rate against unnecessarily small chunks.
- **Evaluator**: serving benchmark on a hybrid model with mixed prefill/decode loads.
- **Fitness**: throughput at fixed TTFT SLO.
- **Risk**: medium (correctness edge cases around state checkpointing).

---

## Deliverable 1 — vLLM V1 Architectural Map

The map below was assembled by combining: the live `scheduler.py` import block, the official architecture overview at `docs.vllm.ai/en/latest/design/arch_overview/`, the V1 blog post (Jan 27, 2025), the V1 user guide (`docs.vllm.ai/en/v0.8.1/getting_started/v1_user_guide.html`), DeepWiki's vLLM deep-graph, recent release notes (Q4 2025 / early 2026), and direct GitHub source for several files. **Where I have only one source for a path, I mark it (single-source) and recommend `git ls-tree main` to verify.**

### A. V1 Engine Core

- **Path**: `vllm/v1/engine/`
- **Key files**: `core.py` (EngineCore busy loop, request handshake, KV-cache config, ZMQ addresses), `async_llm.py` (AsyncLLM façade), `coordinator.py` (data-parallel coordinator), `utils.py` (`EngineHandshakeMetadata`, `EngineZmqAddresses`, `get_device_indices`).
- **Responsibility**: Owns the scheduler + executor; drives one tick = `schedule()` → `execute_model()` → `update_from_output()`. One engine-core process per data-parallel rank.
- **Performance criticality**: HIGH (single-threaded busy loop; CPU overhead here is the V1 design's critical path).

### B. V1 Scheduler

- **Path**: `vllm/v1/core/sched/`
- **Key files**: `scheduler.py` (`Scheduler` class, 2,295 LoC), `async_scheduler.py` (async-engine variant), `interface.py` (`SchedulerInterface`, `PauseState`), `output.py` (`SchedulerOutput`, `NewRequestData`, `CachedRequestData`, `GrammarOutput`), `request_queue.py` (`RequestQueue`, `SchedulingPolicy`, `create_request_queue`), `utils.py` (`check_stop`, `remove_all`).
- **Primary class**: `Scheduler` with `schedule()`, `update_from_output()`, `_preempt_request()`, `_try_schedule_encoder_inputs()`, `_make_cached_request_data()`, `_mamba_block_aligned_split()`, `_select_waiting_queue_for_scheduling()`, `_try_promote_blocked_waiting_request()`.
- **Responsibility**: Per-step token-budget allocation across running + waiting requests; handles chunked prefill, preemption (PRIORITY-policy `max((priority, arrival_time))` victim selection or LIFO under FCFS), encoder-input scheduling for VLMs, spec-decode token alignment, KV-connector hooks (P/D disaggregation, EC connector), structured-output flag propagation.
- **Performance criticality**: HIGH.

### C. V1 KV-Cache Manager

- **Path**: `vllm/v1/core/`
- **Key files**: `kv_cache_manager.py` (`KVCacheManager`), `kv_cache_utils.py` (block hashing, `PrefixCacheStats`), `kv_cache_coordinator.py` (`get_kv_cache_coordinator`), `block_pool.py` (`KVCacheBlock` allocator, doubly-linked free-queue, cached_block hash table), `encoder_cache_manager.py`.
- **Top-level interface**: `vllm/v1/kv_cache_interface.py` (`KVCacheConfig`, `AttentionSpec`, `KVCacheGroup`).
- **Methods of interest**: `KVCacheManager.{get_computed_blocks, allocate_slots, free, get_num_common_prefix_blocks, evict_blocks, take_events, reset_prefix_cache}`.
- **Responsibility**: Block-paged KV memory, automatic prefix caching with parent-chained hashes (sha256 or Python `hash`), LRU eviction, hybrid memory allocator for mamba+attention layers (RFC #11382), KV connector callbacks for disaggregated serving.
- **Performance criticality**: HIGH.

### D. V1 Worker / GPU Model Runner

- **Path**: `vllm/v1/worker/` (and the newer `vllm/v1/worker/gpu/` for Model Runner V2 / MRV2 — appearing in late-2025 releases)
- **Key files**: `gpu_model_runner.py` (`GPUModelRunner`, 7,124 LoC at HEAD `e6ff3e9c8`; "monolith" per `vllm-omni` RFC #1770), `gpu_worker.py` (`Worker` with `AsyncIntermediateTensors` for PP comm overlap), `worker_base.py`. MRV2 adds `vllm/v1/worker/gpu/model_runner.py` with `CudaGraphManager` and a `ModelState` interface; the MRV2 tree (`vllm/v1/worker/gpu/`) now contains parallel implementations for `attn_utils.py`, `block_table.py`, `cudagraph_utils.py`, `dp_utils.py`, `eplb_utils.py`, `input_batch.py`, `kv_connector.py`, `lora_utils.py`, `mm/`, `pool/`, `pp_utils.py`, `sample/`, `spec_decode/`, `states.py`, `structured_outputs.py`, `warmup.py`, `shutdown.py`.
- **Responsibility**: Persistent batch tensors, input ID/position/slot-mapping construction (Numpy-heavy on CPU), encoder-output cache, model forward, sampler invocation, CUDA-graph capture/replay.
- **Performance criticality**: HIGH (CPU side); HIGH (GPU side via attention/MLP).

### E. V1 Attention Backends

- **Path**: `vllm/v1/attention/backends/`
- **Key files (verified)**: `flash_attn.py` (FA2/3/4 dispatch; `_forward_encoder_attention`, `forward`, FP8 KV-cache, cascade attention `lines 741–767`), `flashinfer.py` (`FlashInferMetadataBuilder`, `plan()`, cascade `1148–1198`, CUDA-graph capture `656–694`, BatchPrefillWithPagedKVCacheWrapper for mixed prefill+decode), `triton_attn.py`, `mla/` (Multi-head Latent Attention for DeepSeek-V2/V3 — separate prefill/decode backends), `registry.py` (`AttentionBackendEnum`).
- **Default selection (per `docs.vllm.ai/en/latest/design/attention_backends/`)**: FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise; FlashInfer uses TRTLLM-GEN attention on Blackwell.
- **Performance criticality**: HIGH.

### F. V1 Sampling

- **Path**: `vllm/v1/sample/`
- **Key files**: `sampler.py` (`Sampler` nn.Module), `rejection_sampler.py` (Triton `rejection_greedy_sample_kernel`, `expand_kernel`, `apply_sampling_constraints`), `metadata.py` (`SamplingMetadata`), `ops/topk_topp_sampler.py` (`TopKTopPSampler`, `flashinfer_sample`, `random_sample`, `forward_cuda`, `apply_top_k_only`), `logits_processor.py` (or equivalent — exact filename uncertain from sources).
- **Responsibility**: Temperature, top-k, top-p, min-p, FlashInfer fast sampling, structured-output mask application, rejection sampling for spec-decode (greedy and probabilistic).
- **Performance criticality**: MEDIUM-HIGH (decode hot loop).

### G. V1 Speculative Decoding

- **Path**: `vllm/v1/spec_decode/` (legacy V1 entry points) and `vllm/v1/worker/gpu/spec_decode/` (MRV2 implementations).
- **Key files**: `llm_base_proposer.py` (`SpecDecodeBaseProposer` — actual base logic; legacy `eagle.py` is now a 22-line stub deriving from it with `pass_hidden_states_to_model=True`), `ngram_proposer.py` + `ngram_proposer_gpu.py` (PR #12193, NGram-GPU PR #29184), `medusa.py`, `dflash.py` (PR #32206), `draft_model.py`, `suffix_decoding.py`, `extract_hidden_states.py`, `metrics.py` (`SpecDecodingStats`). MRV2 path: `vllm/v1/worker/gpu/spec_decode/eagle/`, `rejection_sampler.py`, `probabilistic_rejection_sampler_utils.py`, `synthetic_rejection_sampler_utils.py`. Newer additions per release notes: "Unified Parallel Drafting" (PR #32887), "ngram-eagle hybrid" (PR #24344), "Eagle3 with CUDA graphs" (PR #35029, #35040).
- **Performance criticality**: MEDIUM.

### H. V1 Structured Output

- **Path**: `vllm/v1/structured_output/`
- **Key files (verified)**: `backend_xgrammar.py` (`has_xgrammar_unsupported_json_features`, `validate_xgrammar_grammar`, FSM step), `backend_outlines.py`, `backend_guidance.py`, `backend_lm_format_enforcer.py`, `backend_types.py`, `request.py`, `utils.py` — all confirmed present at HEAD. xgrammar is the default; the rest provide fallbacks per the V1 user guide.
- **Manager**: `StructuredOutputManager` (imported by scheduler).
- **Performance criticality**: MEDIUM (mask construction can dominate at high QPS).

### I. V1 Distributed / KV-Connector / EC-Connector

- **Path**: `vllm/distributed/kv_transfer/kv_connector/v1/` (`KVConnectorBase_V1`, `SupportsHMA`, factory) and `vllm/distributed/ec_transfer/ec_connector/` (Encoder Cache connector, factory).
- **Executor**: `vllm/v1/executor/multiproc_executor.py` (per-GPU `WorkerProc`), `vllm/v1/executor/abstract.py`.
- **Responsibility**: Tensor parallelism (Megatron-style), pipeline parallelism with `Cache Intermediate Tensors` (PR #13353), data-parallel coordinator, decode-context-parallel (DCP), prefill-context-parallel (PCP), expert parallelism (DP+EP for spec decoding PR #35294), P/D disaggregation (KV transfer via NCCL/RDMA/NIXL+UCX).
- **Performance criticality**: HIGH (collectives on hot path).

### J. Compilation / CUDA Graphs

- **Path**: `vllm/compilation/`
- **Key files (verified)**: `cuda_graph.py` (`CUDAGraphWrapper` with NONE/PIECEWISE/FULL runtime modes), `piecewise_backend.py` (`PiecewiseBackend`), `backends.py` (`VllmBackend`, `PiecewiseCompileInterpreter`), `decorators.py`, `partition_rules.py`, `passes/` (pass-manager directory), `caching.py`, `codegen.py`, `compiler_interface.py`, `wrapper.py`, `monitor.py`, `counter.py`, `base_static_graph.py`. New (per release notes): Inductor graph partition (`use_inductor_graph_partition=True`, torch≥2.9). Note: an earlier draft of this report named `cuda_piecewise_backend.py` — the actual filename is `piecewise_backend.py`.
- **Default mode in V1**: `FULL_AND_PIECEWISE` (full CUDA graph for uniform-decode, piecewise for prefill / mixed).
- **Performance criticality**: HIGH for low-latency/small-model paths.

### K. Quantization (V1-shared with V0)

- **Path**: `vllm/model_executor/layers/quantization/` (FP8 per-tensor/channel/token, INT8, AWQ, GPTQ, Marlin, Machete on SM90, MXFP4/MXFP8/NVFP4 via TRT-LLM-Gen on SM100, GGUF, compressed-tensors, ModelOpt, TorchAO).
- **Key files**: `fp8.py`, `gptq_marlin.py`, `awq_marlin.py`, `machete.py`, `compressed_tensors/*`, plus C++/CUDA in `csrc/quantization/` (Marlin templates, Machete CuTe).
- **Performance criticality**: HIGH.

### L. MoE Stack

- **Path**: `vllm/model_executor/layers/fused_moe/`
- **Key files**: `fused_moe.py` (Triton `fused_moe_kernel`, `moe_align_block_size`, JSON-config loader from `configs/`), `layer.py` (`FusedMoE`), `cutlass_moe.py`, `deep_gemm_moe.py`, `triton_deep_gemm_moe.py`, `modular_kernel.py` (`FusedMoEPrepareAndFinalizeModular` for EP/DP variants), `ibm_fused_moe/` (PR #19443 TMA-accelerated grouped-GEMM persistent kernel for Hopper), `routed_experts_capturer.py`, plus `configs/E=*,N=*,device_name=*.json`.
- **Performance criticality**: HIGH for MoE models.

### M. LoRA (multi-LoRA)

- **Path**: `vllm/lora/` (shared) plus `LoRAModelRunnerMixin` in `vllm/v1/worker/`.
- **Per release notes**: PR #13096 improving V1 LoRA performance; recent unpermute-aware fused MoE-LoRA path (#32655), reduced kernel overhead via multiple CUDA graphs (#32005).
- **Performance criticality**: MEDIUM.

### N. Multimodal

- **Path**: `vllm/multimodal/` + `vllm/v1/core/encoder_cache_manager.py` + `vllm/v1/core/sched/scheduler.py::_try_schedule_encoder_inputs`.
- **Recent**: ViT full CUDA graphs (#35963), VLM prefix caching (PR #11187).
- **Performance criticality**: MEDIUM-HIGH for VLM serving.

### O. Pooling / Embedding (V1)

- **Path**: V1 support is partial; per release notes pooling models gained MRV2 support (#35120). Still tracked under RFC #12249 ("hidden states processor") and #13360.
- **Performance criticality**: MEDIUM.

### P. Cascade Attention

- **Implementation**: Inside attention backends (`flash_attn.py:741–767`, `flashinfer.py:1148–1198`); driven by `Scheduler.get_num_common_prefix_blocks()`.
- **Performance criticality**: MEDIUM (only fires when shared prefix is large).

### Q. Custom Triton/CUDA Ops shared with V0

- **Path**: `vllm/attention/ops/` (Triton attention ops), `vllm/model_executor/layers/{rotary_embedding.py, layernorm.py, activation.py}` plus C++/CUDA in `csrc/` (RoPE, RMSNorm, fused activations, paged-attention V1/V2 kernels, sampling kernels, all-reduce custom ops).
- **Performance criticality**: HIGH.

---

## Deliverable 2 — Optimization-Candidate Inventory

Format per row: **File · Symbol · Approx. line range · What it does · Metric · Why evolve · Evaluator · Difficulty**.

(Where I cite a line range it is from a triangulated source — DeepWiki, a linked GitHub issue, or my live fetch — and noted as "verified" or "approx".)

### Scheduler heuristics

| File | Symbol | Lines | What | Metric | Why evolve | Evaluator | Risk |
|---|---|---|---|---|---|---|---|
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | line 352 (file: 2,295 LoC) | Per-step token budgeting + admission | TTFT/TPOT/throughput | Heuristic with scalar reward, complex multi-knob | `benchmarks/benchmark_serving.py` ShareGPT/Mooncake/BurstGPT | High |
| same | `_select_waiting_queue_for_scheduling` | line 1567 | Picks `waiting` vs `skipped_waiting` | Tail latency | Currently FIFO with prepend bug (issue #27441) | same | Medium |
| same | `_preempt_request` + `policy == PRIORITY` victim choice | line 952 | Eviction order | Preemption count, fairness | Currently `max((priority, arrival_time))` — replace with learned cost | Custom mixed-priority benchmark | Medium |
| same | `_try_schedule_encoder_inputs` | line 1103 | Encoder-input scheduling for VLMs | VLM throughput | Encoder-cache packing problem | `mmmu_bench.py` (in tree) | Medium |
| same | `_mamba_block_aligned_split` | line 302 | Block-aligned chunked prefill for Mamba/hybrid | Throughput | Snap-to-block rule with corner cases | Hybrid-model serving benchmark | Medium |
| same | `update_from_output` / `_make_cached_request_data` / `_try_promote_blocked_waiting_request` | lines 1290 / 1043 / 2061 | Per-step bookkeeping & promotion logic | Step latency | Hot-path data plumbing — small changes affect every step | Microbench + serving | Medium |
| `vllm/v1/core/sched/request_queue.py` | `SchedulingPolicy`, `create_request_queue` | full file | Queue datastructure | Insertion/peek cost | Small surface; affects every step | Microbenchmark | Low |

### KV-cache manager

| File | Symbol | What | Metric | Why evolve | Evaluator | Risk |
|---|---|---|---|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` | KV-block allocation w/ lookahead | OOM/preempt rate | Block-fit heuristic | KV-cache stress test | Medium |
| same | `get_computed_blocks`, `get_num_common_prefix_blocks` | Prefix-cache lookup + cascade-attn enabling | Cache hit, throughput | Hot per-step | Synthetic shared-prefix workload | Low |
| `vllm/v1/core/block_pool.py` | LRU eviction & free-queue | Block reclamation | Hit rate at low budget | Replaceable policy (LRU↔LFU↔Belady-approx) | Same | Medium |
| `vllm/v1/core/kv_cache_utils.py` | block-hash chain (sha256 vs `hash`) | Collision rate, hash latency | Cache lookup cost | Trade collision risk vs speed | Hashing microbench + 1B-token corpus | Medium-High |

### Triton kernels (priority targets)

| File | Symbol | What | Metric | Why evolve | Evaluator | Risk |
|---|---|---|---|---|---|---|
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | `fused_moe_kernel` `@triton.jit` + `configs/*.json` | Sparse expert MM | ms/token | Tunable BLOCK/warps/stages, per-shape JSON already exists | `benchmarks/kernels/benchmark_moe.py` | Low |
| same | `moe_align_block_size` | Token-routing prep | ms | CUDA-graph-compliant variant has knobs (PR #12036) | Microbench | Low |
| `vllm/model_executor/layers/fused_moe/cutlass_moe.py`, `deep_gemm_moe.py` | various | Backend dispatch | ms/token | Tunable choice + tile sizes | same | Low-Med |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | `forward_cuda`, `apply_top_k_only`, `flashinfer_sample` | Top-k/top-p sampling | ms | FlashInfer "sorting-free" headroom | Sampling microbench + KS test | Low |
| `vllm/v1/sample/rejection_sampler.py` | `rejection_greedy_sample_kernel`, `expand_kernel`, `apply_sampling_constraints` | Spec-decode sampling | ms | Triton, has MAX_NUM_TOKENS knob | Spec-decode benchmark | Low |
| `vllm/model_executor/layers/rotary_embedding.py` (and Triton variants in `vllm/attention/ops/`) | RoPE | RoPE apply | ms | Broad shape distribution | Microbench | Low |
| `vllm/model_executor/layers/layernorm.py` | RMSNorm fused | ms | Standard target in KernelBench | Microbench | Low |

### Attention metadata builders

| File | Symbol | What | Metric | Risk |
|---|---|---|---|---|
| `vllm/v1/attention/backends/flashinfer.py` | `FlashInferMetadataBuilder.build`, `plan` | Choose cascade vs flat, split sizes | Kernel ms | Medium |
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` builder, FA-version dispatch | FA2/3/4 selection per shape | Kernel ms | Medium |
| `vllm/v1/attention/backends/mla/*` | MLA prefill/decode dispatch | Kernel ms | High |

### Speculative decoding

| File | Symbol | What | Metric | Risk |
|---|---|---|---|---|
| `vllm/v1/spec_decode/eagle.py` | proposer chain, draft-length policy | Spec tokens per step | tokens/sec, AL | Low–Med |
| `vllm/v1/spec_decode/ngram_proposer.py` | `prompt_lookup_min/max` | n-gram match policy | AL | Low |
| `vllm/v1/sample/rejection_sampler.py` | acceptance kernel | acceptance throughput | ms | Low |

### CUDA graphs / compilation

| File | Symbol | What | Metric | Risk |
|---|---|---|---|---|
| `vllm/compilation/cuda_graph.py` | `CUDAGraphWrapper` runtime-mode dispatch | mode/size choice | latency | Low |
| `vllm/compilation/piecewise_backend.py` | `PiecewiseBackend.compile_sizes`, `cudagraph_capture_sizes` | Capture set | latency + capture mem | Low |
| `vllm/config/compilation.py` | `CompilationConfig` defaults | autotuned defaults | latency | Low |

### Quantization

| File | Symbol | Risk |
|---|---|---|
| `vllm/model_executor/layers/quantization/utils/marlin_utils*.py` | shape→kernel decision | Low |
| `csrc/quantization/machete/*` (CuTe heuristics) | tile-size templates | Med-High (C++) |
| `vllm/model_executor/layers/quantization/fp8.py` | per-tensor↔per-token↔block FP8 selection | Med |

### Sampling / structured output

| File | Symbol | Why evolve | Risk |
|---|---|---|---|
| `vllm/v1/structured_output/backend_xgrammar.py` | `has_xgrammar_unsupported_json_features` | grammar-acceptance heuristics | Low |
| (uncertain) `vllm/v1/structured_output/backend_outlines.py` / `backend_guidance.py` | backend selection | dispatch heuristic | Low |

> **Note**: All four backend files (`backend_xgrammar.py`, `backend_outlines.py`, `backend_guidance.py`, `backend_lm_format_enforcer.py`) plus `backend_types.py`, `request.py`, and `utils.py` are confirmed present at HEAD `e6ff3e9c8`. The earlier uncertainty caveat has been resolved.

---

## Deliverable 3 — Cross-Mapping: Module → Candidates → Papers

Each subsection maps one module group to the candidates above and to a curated 8–15-paper literature anchor. Citations use **Author(s) — Title — Venue Year — arXiv ID/URL** form. The "evolution direction" notes how the paper informs an evolve-target.

### A. V1 Scheduler & Continuous Batching

**Candidates**: 2, 10 above; queue policy, preemption victim, chunked-prefill chunk sizing, encoder-input packing.
**Papers**:

1. Yu et al. — *Orca: A Distributed Serving System for Transformer-Based Generative Models* — OSDI 2022. Iteration-level scheduling baseline. **Direction**: any evolved policy must remain iteration-level.
2. Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention* — SOSP 2023 — arXiv:2309.06180. Establishes the V0/V1 batching contract.
3. Agrawal et al. — *Sarathi-Serve: Taming Throughput-Latency Tradeoff* — OSDI 2024 — arXiv:2403.02310. Chunked prefill + stall-free schedules. **Direction**: evolve chunk size / mixing ratio.
4. Sun et al. — *Llumnix: Dynamic Scheduling for LLM Serving* — OSDI 2024 — arXiv:2406.03243. Live request migration. **Direction**: cross-instance migration policy.
5. Wu et al. — *FastServe: Fast Distributed Inference Serving* — arXiv:2305.05920. MLFQ scheduler.
6. Sheng et al. — *Fairness in Serving Large Language Models (VTC)* — OSDI 2024 — arXiv:2401.00588. Fairness-aware admission.
7. Cao et al. — *Locality-aware Fair Scheduling in LLM Serving (DLPM/D²LPM)* — arXiv:2501.14312.
8. Zhong et al. — *DistServe* — OSDI 2024 — arXiv:2401.09670. P/D disaggregation goodput.
9. Patel et al. — *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* — ISCA 2024 — arXiv:2311.18677.
10. Qin et al. — *Mooncake: A KVCache-Centric Disaggregated Architecture* — FAST 2025. KV pool & cache-aware routing.
11. Hu et al. — *MemServe* — arXiv:2401.11181. Co-located/disagg scheduling.
12. Novikov et al. — *AlphaEvolve* — DeepMind 2025 — arXiv:2506.13131. **Specifically reports an evolved datacenter scheduling algorithm** that recovered 0.7% of Google compute — direct precedent for evolving `Scheduler.schedule`.
13. Romera-Paredes et al. — *FunSearch / Mathematical Discoveries* — Nature 625 (2024) — doi:10.1038/s41586-023-06924-6. Bin-packing heuristics evolved by FunSearch are the same shape as queue-admission policies.

### B. KV-Cache Manager & Prefix Caching

**Candidates**: 3, KV utilities, block-pool eviction.
**Papers**:

1. Kwon et al. — PagedAttention/vLLM — SOSP 2023 (foundational).
2. Zheng et al. — *SGLang / RadixAttention* — arXiv:2312.07104; LMSYS blog 2024-01-17. **Direction**: evolve radix-tree eviction priorities.
3. Juravsky et al. — *Hydragen: High-Throughput LLM Inference with Shared Prefixes* — arXiv:2402.05099.
4. Brandon et al. — *Cascade Inference / Cascade Attention* — FlashInfer paper (Ye et al. 2025).
5. Ye et al. — *FlashInfer* — MLSys 2025 (best paper) — arXiv:2501.01005. Cascade and block-sparse formats.
6. Sheng et al. — *ChunkAttention* — arXiv:2402.15220.
7. Prabhu et al. — *vAttention: Dynamic Memory Management without PagedAttention* — ASPLOS 2025 — arXiv:2405.04437. **Direction**: evolve block-vs-virtual-memory hybrid heuristics.
8. Zhang et al. — *H2O: Heavy-Hitter Oracle for Efficient Generative Inference* — NeurIPS 2023. Eviction policy.
9. Xiao et al. — *Efficient Streaming LMs with Attention Sinks (StreamingLLM)* — ICLR 2024 — arXiv:2309.17453.
10. Li et al. — *SnapKV* — NeurIPS 2024 — arXiv:2404.14469.
11. Liu et al. — *Scissorhands* — NeurIPS 2023.
12. Ge et al. — *FastGen: Adaptive KV Compression* — ICLR 2024 — arXiv:2310.01801.
13. Hooper et al. — *KVQuant: Towards 10M Context Length KV Quantization* — arXiv:2401.18079.
14. Liu et al. — *KIVI: 2-bit KV Quantization* — ICML 2024 — arXiv:2402.02750.
15. Zhang et al. / Cai et al. — *PyramidKV / PyramidInfer* — arXiv:2406.02069 / arXiv:2405.12532. **Direction**: per-layer eviction budgets.

### C. Attention Backends (FlashAttention / FlashInfer / MLA)

**Candidates**: 8; backend dispatch logic.
**Papers**:

1. Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention* — NeurIPS 2022 — arXiv:2205.14135.
2. Dao — *FlashAttention-2* — arXiv:2307.08691.
3. Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao — *FlashAttention-3* — NeurIPS 2024 — arXiv:2407.08608. Hopper warp-specialization, FP8 incoherent processing. **Direction**: evolve tile-shape & pipeline-stage knobs for FA-style kernels.
4. Ye, Chen et al. — *FlashInfer* — MLSys 2025 — arXiv:2501.01005.
5. Hong et al. — *FlexAttention / Flex* — PyTorch blog & paper 2024.
6. DeepSeek-AI — *DeepSeek-V2: MLA + DeepSeekMoE* — arXiv:2405.04434.
7. NVIDIA — *TRT-LLM / TRTLLM-GEN attention* — technical reports 2024-25.
8. Liu et al. — *Lightning Attention-2* — arXiv:2401.04658.
9. Ainslie et al. — *GQA: Multi-Query → Grouped-Query Attention* — arXiv:2305.13245.
10. Pope et al. — *Efficiently Scaling Transformer Inference* — MLSys 2023.
11. Beltagy et al. — *Longformer / sliding-window attention* — arXiv:2004.05150.
12. Press et al. — *ALiBi* — arXiv:2108.12409.
13. Sun et al. — *POD-Attention: Fused Prefill+Decode* — arXiv:2410.18038.

### D. Triton/CUDA Kernels (general)

**Candidates**: 5, 6, kernel rows in inventory.
**Papers**:

1. Tillet, Kung, Cox — *Triton: An Intermediate Language and Compiler for Tiled Neural-Network Computations* — MAPL 2019.
2. Ouyang, Guo et al. — *KernelBench: Can LLMs Write Efficient GPU Kernels?* — ICML 2025 — arXiv:2502.10517. **Direction**: standard evaluation harness for evolved kernels.
3. Lange et al. — *AI CUDA Engineer / robust-kbench* — Sakana AI 2025 — pub.sakana.ai/static/paper.pdf.
4. Baronio et al. — *Kevin: Multi-Turn RL for Generating CUDA Kernels* — arXiv:2507.11948.
5. Li, Sun, Wang, Li, Shum — *CUDA-L1: Improving CUDA Optimization via Contrastive RL* — ICLR 2026 — arXiv:2507.14111.
6. *CUDA-L2* — arXiv:2512.02551 (2025/26). Multi-stage RL with NCU profiling.
7. Andrews & Witteveen — *GPU Kernel Scientist* — arXiv:2506.20807.
8. Chen et al. — *CUDA-LLM* — arXiv:2506.09092.
9. Tschand et al. — *SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance* — 2025.
10. Lange et al. — *EvoEngineer* — arXiv:2510.03760.
11. *Kernel-Smith: Unified Recipe for Evolutionary Kernel Optimization* — arXiv:2603.28342 (2026 anchor; uses OpenEvolve directly).
12. *KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization* — arXiv:2603.12440 (2026).
13. Meta — *KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators* — arXiv:2512.23236 (2025).
14. AMD — *GEAK-OptimAgentv2 / GEAK-OpenEvolve* — ROCm Blogs 2025 (cited even though target is AMD; methodology transfers cleanly to NVIDIA Triton).
15. Romera-Paredes et al. — *FunSearch* — Nature 2024.
16. Novikov et al. — *AlphaEvolve* — DeepMind 2025 — arXiv:2506.13131. AlphaEvolve specifically reports **+23% kernel-tiling and +32% FlashAttention speedups** internal to Google.
17. PyTorch / FAIR — *KernelLLM-8B* and KernelBook — 2025.

### E. MoE Stack (fused-MoE Triton, expert parallelism)

**Candidates**: 1.
**Papers**:

1. Shazeer et al. — *Sparsely-Gated MoE* — ICLR 2017.
2. Lepikhin et al. — *GShard* — ICLR 2021 — arXiv:2006.16668.
3. Fedus, Zoph, Shazeer — *Switch Transformer* — JMLR 2022 — arXiv:2101.03961.
4. Gale et al. — *MegaBlocks: Efficient Sparse Training with MoEs* — MLSys 2023 — arXiv:2211.15841. Block-sparse formulation.
5. Hwang et al. — *Tutel: Adaptive MoE at Scale* — MLSys 2023 — arXiv:2206.03382.
6. Rajbhandari et al. — *DeepSpeed-MoE* — arXiv:2201.05596.
7. He et al. — *FasterMoE* — PPoPP 2022.
8. DeepSeek-AI — *DeepSeekMoE: Towards Ultimate Expert Specialization* — arXiv:2401.06066. Fine-grained experts + shared experts.
9. DeepSeek-V3 Technical Report — arXiv:2412.19437.
10. *EPS-MoE: Expert Pipeline Scheduler* — arXiv:2410.12247.
11. *FinDEP: Fine-grained Disaggregated Expert Parallelism* — arXiv:2512.21487.
12. *X-MoE* — arXiv:2508.13337.
13. Jiang et al. — *Mixtral of Experts* — arXiv:2401.04088.

### F. Speculative Decoding

**Candidates**: 4.
**Papers**:

1. Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding* — ICML 2023 — arXiv:2211.17192.
2. Chen et al. — *Accelerating LLM Decoding with Speculative Sampling* — arXiv:2302.01318.
3. Cai et al. — *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* — ICML 2024 — arXiv:2401.10774.
4. Li, Wei, Zhang, Zhang — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — ICML 2024 — arXiv:2401.15077.
5. Li et al. — *EAGLE-2: Faster Inference with Dynamic Draft Trees* — EMNLP 2024.
6. Li et al. — *EAGLE-3: Scaling up Inference Acceleration via Training-Time Test* — NeurIPS 2025 — arXiv:2503.01840.
7. Ankner et al. — *Hydra: Sequentially-Dependent Draft Heads* — arXiv:2402.05109.
8. Fu et al. — *Lookahead Decoding* — ICML 2024 — arXiv:2402.02057.
9. He et al. — *REST: Retrieval-Based Speculative Decoding* — NAACL 2024 — arXiv:2311.08252.
10. Saxena — *Prompt Lookup Decoding (PLD)* — 2023 (GitHub).
11. Miao et al. — *SpecInfer: Accelerating LLM Serving with Tree-based Speculation* — ASPLOS 2024 — arXiv:2305.09781.
12. Liu et al. — *Online Speculative Decoding* — arXiv:2310.07177.
13. Chen et al. — *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding* — arXiv:2402.12374.
14. Svirschevski et al. — *SpecExec* — arXiv:2406.02532.
15. *P-EAGLE: Parallel-Drafting EAGLE* — arXiv:2602.01469 (2026 anchor).
16. *SpecVocab* — arXiv:2602.13836 (2026 anchor).
17. *SpecForge / FR-Spec / VocabTrim* — arXiv:2603.18567.

### G. Quantization (FP8/INT8/AWQ/GPTQ/Marlin/Machete)

**Candidates**: 9 + Marlin/Machete heuristic.
**Papers**:

1. Xiao et al. — *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs* — ICML 2023 — arXiv:2211.10438.
2. Lin et al. — *AWQ: Activation-Aware Weight Quantization* — MLSys 2024 — arXiv:2306.00978.
3. Frantar et al. — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — ICLR 2023 — arXiv:2210.17323.
4. Frantar & Alistarh — *Marlin: 4-bit Inference Kernel* — 2024 (vLLM PR #6612, NeurIPS 2024 Efficient ML Workshop).
5. Wilkinson et al. — *Machete: Hopper Optimized Mixed-Precision Linear Kernel* — vLLM PR #7174, 2024.
6. Lin et al. — *QServe: W4A8KV4 Quantization* — MLSys 2025 — arXiv:2405.04532.
7. Zhao et al. — *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving* — MLSys 2024 — arXiv:2310.19102.
8. Liu et al. — *KIVI: 2-bit KV Cache Quantization* — ICML 2024.
9. Hooper et al. — *KVQuant* — arXiv:2401.18079.
10. NVIDIA — *FP8 Formats for Deep Learning* — arXiv:2209.05433.
11. Rouhani et al. — *Microscaling (MX) Data Formats* — OCP 2023.
12. *QQQ: Quality Quattuor-Bit Quantization* — arXiv:2406.09904.
13. Dettmers et al. — *LLM.int8()* — NeurIPS 2022.

### H. CUDA Graphs & Compilation

**Candidates**: 7.
**Papers**:

1. Ansel et al. — *PyTorch 2: torch.compile / Inductor* — ASPLOS 2024.
2. Tillet et al. — *Triton* — MAPL 2019.
3. Jia, Padon et al. — *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions* — SOSP 2019.
4. *vLLM CUDA Graphs Design Doc* — `docs/design/cuda_graphs.md` (primary).
5. NVIDIA — *CUDA Graphs in CUDA 11/12* — technical blogs.
6. Chen et al. — *TVM: An Automated End-to-End Optimizing Compiler* — OSDI 2018.
7. Zheng et al. — *Ansor: Generating High-Performance Tensor Programs* — OSDI 2020.
8. Wang et al. — *FlashInfer JIT compilation* — MLSys 2025.

### I. Sampling & Structured Output

**Candidates**: 5, 6, structured-output dispatch.
**Papers**:

1. Holtzman et al. — *The Curious Case of Neural Text Degeneration (top-p)* — ICLR 2020 — arXiv:1904.09751.
2. Fan et al. — *Hierarchical Neural Story Generation (top-k)* — ACL 2018.
3. Minh et al. — *min-p Sampling: Balancing Creativity and Coherence* — arXiv:2407.01082.
4. Dong et al. — *XGrammar: Flexible and Efficient Structured Generation* — arXiv:2411.15100.
5. Willard & Louf — *Outlines: Efficient Guided Generation* — arXiv:2307.09702.
6. Microsoft — *Guidance / llguidance* — GitHub.
7. Ye et al. — FlashInfer (sorting-free sampling) — arXiv:2501.01005 + 2025-03-10 blog post.
8. Park et al. — *SGLang Compressed FSM* — arXiv:2312.07104.

### J. LoRA Serving

**Candidates**: LoRA mixin tuning targets (LoRA dispatcher in scheduler).
**Papers**:

1. Hu et al. — *LoRA: Low-Rank Adaptation of Large Language Models* — ICLR 2022 — arXiv:2106.09685.
2. Sheng et al. — *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* — MLSys 2024 — arXiv:2311.03285.
3. Chen, Ye et al. — *Punica: Multi-Tenant LoRA Serving* — MLSys 2024 — arXiv:2310.18547. SGMV kernel.
4. Wu et al. — *dLoRA* — OSDI 2024.
5. *CaraServe: CPU-Assisted LoRA Serving* — arXiv:2401.11240.
6. Sheng et al. — *PEFT-Aware Scheduling* — 2024.

### K. Distributed Inference & Disaggregation

**Candidates**: scheduler hooks for KV connector + EC connector; pipeline-bubble policies.
**Papers**:

1. Shoeybi et al. — *Megatron-LM: Training Multi-Billion Parameter Language Models* — arXiv:1909.08053.
2. Huang et al. — *GPipe: Pipeline Parallelism* — NeurIPS 2019.
3. Narayanan et al. — *PipeDream-2BW* — ICML 2021.
4. Korthikanti et al. — *Sequence Parallelism* — arXiv:2205.05198.
5. Liu et al. — *Ring Attention / Context Parallelism* — arXiv:2310.01889.
6. Zhong et al. — *DistServe* — OSDI 2024.
7. Patel et al. — *Splitwise* — ISCA 2024.
8. Qin et al. — *Mooncake* — FAST 2025.
9. Hu et al. — *MemServe* — 2024.
10. Shi, Cai et al. — *Nexus: Proactive Intra-GPU Disaggregation* — 2025.
11. *DuetServe* — arXiv:2511.04791 (2025).
12. *FlowKV* — arXiv:2504.03775.

### L. Multimodal & VLM

**Candidates**: encoder-budget heuristic, encoder-cache eviction.
**Papers**:

1. Liu et al. — *LLaVA* — NeurIPS 2023.
2. Bai et al. — *Qwen2-VL* — 2024.
3. *vLLM PR #11187* — Prefix caching for VLMs.
4. *PR #25903* — Encoder-cache memory consumption.
5. Cha et al. — *Honeybee: Locality-enhanced Projector* — CVPR 2024.

### M. LLM-Driven Code/Algorithm Optimization (the *meta* literature)

**Candidates**: all of the above (this is the methodology layer).
**Papers**:

1. Romera-Paredes et al. — *FunSearch* — Nature 625 (2024).
2. Novikov, Vũ, Eisenberger et al. — *AlphaEvolve* — Google DeepMind 2025 — arXiv:2506.13131.
3. *OpenEvolve* — codelion/openevolve, 2025 (open-source AlphaEvolve impl).
4. Ma, Liang et al. — *Eureka: Human-Level Reward Design via Coding LLMs* — ICLR 2024 — arXiv:2310.12931.
5. Lehman, Stanley — *Quality-Diversity / MAP-Elites* — 2015.
6. Mouret, Clune — *MAP-Elites* — arXiv:1504.04909.
7. Liu et al. — *DeepEvolve: Augmenting AlphaEvolve with Deep Research* — arXiv:2510.06056.
8. *CodeEvolve* — arXiv:2407.09876.
9. Lange et al. — *AI CUDA Engineer* — Sakana AI 2025.
10. Baronio et al. — *Kevin* — arXiv:2507.11948.
11. *CUDA-L1 / L2* — arXiv:2507.14111 / 2512.02551.
12. *KernelBench* — arXiv:2502.10517.
13. *KernelLLM-8B / KernelBook* — PyTorch / FAIR 2025.
14. *Kernel-Smith* — arXiv:2603.28342 (uses OpenEvolve directly on Triton).
15. *KernelEvolve* (Meta) — arXiv:2512.23236.
16. *KernelFoundry* — arXiv:2603.12440.
17. *EvoEngineer* — arXiv:2510.03760.
18. *AutoTriton* — RL for Triton programming, 2025.
19. *Discovering Multiagent Learning Algorithms with LLMs* — arXiv:2602.16928.

---

## Recommendations (staged plan)

### Stage 1 — Lowest-risk wins (week 1–2)

- Start OpenEvolve on **fused-MoE JSON configs** (target #1) for a single (E, N, GPU) tuple already used in production (e.g., DeepSeek-V3 on H100, Qwen3-MoE on B200). The benchmark already exists; the fitness function is a single ms number; falling back to `configs/E=...` is trivial.
- In parallel, run OpenEvolve on **top-k/top-p `forward_cuda`** (target #5) using a microbench. This pair gives one MoE-shaped and one sampler-shaped result quickly.
- **Threshold to escalate**: ≥10% median latency improvement over the current `configs/` baseline at iso-correctness.

### Stage 2 — Medium-risk algorithmic targets (week 3–6)

- Evolve the **scheduler waiting-queue ordering** (target #2) and **spec-decode draft-length policy** (target #4) inside a sandbox that exposes only a restricted Python API (`Request.priority`, `arrival_time`, `num_prompt_tokens`, `num_computed_tokens`, `kv_cache_utilization`, `running_lora_count`, `acceptance_history`). Use `benchmark_serving.py` against ShareGPT, Mooncake-Conversation, BurstGPT.
- **Threshold**: Pareto-dominate the current FCFS+chunked-prefill baseline on (TTFT p99, TPOT p99, throughput) on at least 2 of 3 traces.

### Stage 3 — High-risk / high-reward (month 2+)

- Evolve **prefix-cache hash function** (target #3) and **CUDA-graph mode/size dispatcher** (target #7). These require strong correctness oracles (collision tests; full vLLM unit-test suite + lm_eval accuracy parity). Use a 2-stage evaluator: (a) correctness gate, (b) performance scoring.
- Evolve **Marlin/Machete shape-dispatch heuristic** (target #9) — likely highest dollar-impact target because it sits in every quantized GEMM call in production.

### Methodological recommendations

- Use **OpenEvolve with MAP-Elites archive** keyed on (batch-size bucket, sequence-length bucket) — Kernel-Smith (arXiv:2603.28342) and KernelFoundry (arXiv:2603.12440) both report this is essential for kernel evolution.
- Inject **profiler feedback into prompts** (NCU for kernels, vLLM Prometheus metrics for scheduler) — CUDA-L2 and CudaForge (arXiv:2511.01884) report ~2× sample-efficiency gain from this.
- Always run a **multi-trace evaluator** to prevent overfitting to a single workload — robust-kbench (Sakana 2025) showed that single-input kernels often achieve illusory 50–120× speedups via benchmark exploits.
- Co-evaluate **accuracy preservation** (`lm-eval` on a held-out set) for any candidate that touches numerics (KV-cache hashing, sampling, quantization, attention).
- Use **vLLM's existing CI suite as a correctness gate** before any candidate is committed — this is the single most important safeguard.

---

## Caveats

- **Code anchor (refreshed)**: All paths and line numbers in this revision were re-verified against local HEAD `e6ff3e9c8` (May 2026). The file `vllm/v1/core/sched/scheduler.py` is now **2,295 lines** with method anchors at: `_mamba_block_aligned_split` (302), `schedule` (352), `_preempt_request` (952), `_make_cached_request_data` (1043), `_try_schedule_encoder_inputs` (1103), `update_from_output` (1290), `_select_waiting_queue_for_scheduling` (1567), `_try_promote_blocked_waiting_request` (2061). `gpu_model_runner.py` is 7,124 LoC. Re-run `git rev-parse HEAD` and `wc -l` before driving evolutionary search, since this codebase changes weekly.
- **MRV2 (Model Runner V2)**: vLLM is mid-migration from `vllm/v1/worker/gpu_model_runner.py` (the 7,124-LoC monolith at HEAD `e6ff3e9c8`) to the modular `vllm/v1/worker/gpu/` tree. Several release notes (Q4 2025, Q1 2026) indicate that **PP, DCP, EAGLE3 with CUDA graphs, pooling, piecewise+full mixed CUDA graph capture, DP+EP for spec decoding, and EPLB** are all landing in MRV2. Evolve targets should track MRV2 paths where possible to avoid bit-rot.
- **V1 features still maturing (per V1 user guide)**: LoRA performance lags V0; FP8 KV cache not yet supported; structured-output backends other than xgrammar:no_fallback were WIP; Mamba/Jamba SSM models partially supported. These should be lower priority for evolve targets until they stabilize.
- **Resolved path uncertainty**: `backend_outlines.py`, `backend_guidance.py`, and `backend_lm_format_enforcer.py` under `vllm/v1/structured_output/` are now all verified present at HEAD.
- **2026 papers (arXiv 2602.x, 2603.x, 2604.x, 2512.xxxxx)**: Several papers cited (Kernel-Smith, KernelFoundry, P-EAGLE, SpecVocab, SpecForge, FinDEP, KernelEvolve, DuetServe, Justitia) appear to be 2026 (or late-2025 with 2026 numbering). These were retrieved live; the user should re-verify final venue/version since some may still be preprints.
- **Speculation in cited blogs**: The `mlai.blog` and `huggingface.co/blog` posts cited for OpenEvolve evidence are author-driven and report specific numbers (e.g., "12.5% transformer-attention kernel speedup" on Apple Silicon) that may not transfer to NVIDIA Triton kernels. The Sakana AI CUDA Engineer paper itself documents that **benchmark exploits inflated reported speedups by ~2× on KernelBench**; treat all "12×–449×" headline numbers (CUDA-L1 v1) as upper bounds requiring robust-kbench-style validation.
- **vLLM-internal heuristics often live in C++/CUDA, not Python**: Marlin/Machete kernel selection logic is partially inside CMake-built native code (`csrc/quantization/`); these are evolvable, but the evaluator harness must rebuild C++ on each candidate, multiplying iteration cost ~10–100×. Prefer the Python-side dispatch heuristics first.
- **Fitness functions must include a guard against quality regressions**. Several papers cited (Kernel-Smith, EvoEngineer, robust-kbench) explicitly document LLMs discovering "kernels" that pass loose tolerance tests but produce incorrect outputs in production. A correctness gate using vLLM's full test suite + an `lm-eval` smoke test should be mandatory.
- **No experimental results in this report**: this is a *map*, not a measurement study. The expected-gain numbers cited (e.g., "1.5–3× goodput") come from the cited literature, not from running OpenEvolve on vLLM. The first concrete experiment (Stage 1, fused-MoE configs) is the cheapest way to ground these estimates in vLLM-specific measurements.
