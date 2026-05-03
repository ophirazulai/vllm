# Evolved CPU Offloading — Hot-Swap API for vLLM x CORAL

> **Status:** Design proposal. Not implemented. Branch: `feat/evolved_cpu_offloading`.
> **Author intent:** enable [CORAL](https://github.com/Human-Agent-Society/CORAL) to evolve vLLM's CPU-offloading logic by hot-swapping policy code into a running engine, without restarts.

## 1. Context and motivation

vLLM offloads cold KV-cache blocks (and optionally weights) from GPU to CPU to extend usable context length and batch size. The eviction/admission strategy directly governs hit rate, transfer volume, and tail latency — and there is no closed-form optimum: it depends on workload shape, model size, hardware, and request mix.

CORAL is an evolutionary search framework where LLM-driven agents propose source-code mutations, a grader scores each candidate, and the population improves over generations. Plugging CORAL into vLLM's offloading code requires that vLLM accept *new policy code at runtime*: spinning up a fresh engine per candidate is too slow (model load times dominate, masking the signal CORAL needs).

This document specifies the minimum API surface that lets CORAL — or any external evolutionary loop — hot-swap a CPU offloading policy into a live vLLM engine, drive a benchmark, and read out fitness.

## 2. Goals and non-goals

**Goals**
- Hot-swap a `CachePolicy` implementation in `CPUOffloadingManager` without restarting the engine.
- Three code-delivery modes: file path, dotted module name, raw source string.
- Atomic swap with rollback on failure; in-flight requests are not corrupted.
- Both Python (`LLM.swap_offload_policy(...)`) and HTTP (`POST /v1/swap_offload_policy`) entry points.
- Secure-by-default: feature is opt-in via `EngineArgs` / CLI / env; remote access is opt-in separately.
- Composable with existing `OffloadingSpecFactory` lazy-loading.
- Expose enough policy/offload stats for an external grader to score a candidate without scraping logs.
- Allow per-request opaque `policy_hints` (via `vllm_xargs`) to reach the active policy at insert/touch/evict time, so evolved policies can use request-level signal — priority, session id, expected reuse, tenant id, deadline — without changing vLLM's HTTP/Python API across CORAL generations. See §16.

**Non-goals (Phase 1)**
- Hot-swapping weight-prefetch logic (`PrefetchOffloader`) — sketched in §15 as Phase 2.
- Sandboxing untrusted code. Documenting threat model is enough; full isolation is out of scope.
- Mutating CORAL itself. Integration is one-sided: vLLM exposes; the grader consumes.
- Preserving fine-grained recency metadata across swaps (only `(key, BlockStatus)` is migrated).
- Data-parallel policy broadcast / two-phase commit. Phase 1 supports one `EngineCore`; DP support is called out in §13.

## 3. Primary recommendation

**Phase 1 ships hot-swap of the KV-cache `CachePolicy`** (LRU / ARC / evolved). Three reasons:
1. *The seam already exists.* `CPUOffloadingManager._policy` is one mutable pointer.
2. *Single-process scope.* The KV-offload manager lives in the scheduler. No `collective_rpc` to workers needed.
3. *Small surface.* Five cache operations plus the constructor, a known ABC, no torch.compile or cudagraph entanglement.

Weight-prefetch evolution (Phase 2) requires worker-side broadcast and likely cudagraph re-capture. We defer.

## 4. Interface — what an evolved policy must implement

Evolved policies subclass the existing ABC at [vllm/v1/kv_offload/cpu/policies/base.py](../vllm/v1/kv_offload/cpu/policies/base.py):

```python
class CachePolicy(ABC):
    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...
    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None: ...
    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus,
               req_context: ReqContext | None = None) -> None: ...
    @abstractmethod
    def remove(self, key: OffloadKey) -> None: ...
    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey],
              req_context: ReqContext | None = None) -> None: ...
    @abstractmethod
    def evict(self, n: int, protected: set[OffloadKey],
              req_context: ReqContext | None = None
              ) -> list[tuple[OffloadKey, BlockStatus]] | None: ...
```

The optional `req_context` parameter on `insert` / `touch` / `evict` carries per-request hints (priority, session id, etc.) sourced from the request's `vllm_xargs["policy_hints"]`. `get` and `remove` are intentionally context-free (`get` is content-addressed and may be invoked outside any request's scope; `remove` is a manager-internal cleanup). Built-in `LRUCachePolicy` / `ARCCachePolicy` accept and ignore the kwarg. The full hint channel — request → `ReqContext` → policy method — is specified in §16.

We extend `CachePolicy` with state-transfer hooks so swaps can carry resident blocks forward instead of cold-starting the cache:

- `export_state() -> Iterable[tuple[OffloadKey, BlockStatus]]` — **required** on every policy; called once on the **outgoing** policy.
- `import_state(items) -> None` — bulk-loaded into the **incoming** policy after construction. The base implementation calls `insert(key, block)` for each item.

Existing `LRUCachePolicy` and `ARCCachePolicy` must implement `export_state()` by returning their currently resident blocks. This is not optional: if the outgoing policy cannot export resident blocks, a swap must fail before touching `manager._policy`. Dropping the map while keeping `CPUOffloadingManager._num_allocated_blocks` / `_free_list` unchanged would leak CPU block slots.

Recency / ghost-list metadata is **not** preserved across swaps. Evolved policies must accept a brief warm-up window post-swap.

Required class-level metadata for telemetry and reproducibility:
- `POLICY_NAME: str`
- `POLICY_VERSION: str`

The request `name` / `version` fields may override these for experiment labeling, but the swapper should still require the resolved metadata to be non-empty.

Built-in `LRUCachePolicy` and `ARCCachePolicy` should define the same metadata (`"lru"` / `"arc"`, version `"builtin"`) so `GET /v1/offload_policy` can describe the initial generation before the first hot-swap.

Source-text SHA-256 (first 12 chars) is computed by the loader, not by the policy.

**Constructor signature widening.** The ABC's `__init__(self, cache_capacity: int)` is widened to `__init__(self, cache_capacity: int, **kwargs: Any)` so the swap request can pass `policy_kwargs` through without subclass churn. Existing `LRUCachePolicy` / `ARCCachePolicy` accept and ignore extra kwargs; evolved policies opt in by naming the kwargs they consume. This keeps the swap API parameterizable (e.g., ARC variants with a tunable `target_t1_ratio`) without breaking the ABC contract.

## 5. Loader — `vllm/v1/kv_offload/cpu/policies/loader.py` (new file)

Returns a `LoadedPolicy` dataclass: `{cls, source, source_hash, source_origin, module}`.

```python
class PolicyLoader:
    def load_from_path(self, path: str) -> LoadedPolicy: ...
    def load_from_module(self, dotted: str) -> LoadedPolicy: ...
    def load_from_source(
        self, src: str, module_name_hint: str = "evolved_policy"
    ) -> LoadedPolicy: ...
```

**Isolation rules**

- *Always fresh namespace.* Path and dotted-module loads resolve a source-backed `.py` file, read its bytes once, compute the hash, then execute that source under a synthetic module name such as `vllm._evolved.<source_hash>.<nonce>`. Raw-source loads use the same synthetic namespace path. No `importlib.reload` and no execution under the caller's dotted name — two loads of the same logical module produce distinct module objects with no shared globals.
- *Synthetic registration only.* The loaded module is inserted into `sys.modules` under the synthetic key so tracebacks and class `__module__` references work. User-controlled module names are never overwritten.
- *Module accumulation.* Synthetic entries in `sys.modules` are not collected automatically. The registry keeps only each engine's *active* policy module registered; on every successful swap it removes that engine's prior active module's `sys.modules` entry so long-running evolutionary loops do not leak namespaces over thousands of generations. Candidate modules that fail validation, fail swap, or are used only for `dry_run=True` are removed from `sys.modules` before returning.
- *Relative imports.* For dotted-module loads, keep `__package__` set to the original package so normal relative imports continue to work. **Caveat:** sibling modules reached via `from . import helpers` are imported under their original dotted names and therefore *shared* across swaps — they are not part of the fresh namespace. Authors of evolved policies should keep mutable state in the policy class itself, not in sibling modules. Path and raw-source loads should use absolute imports.
- *Engine-process filesystem.* Path-based and dotted-module loads read the file inside the engine process, not the API server. In containerized or remote deployments the path must be visible to the engine process; otherwise use the raw-source variant.
- *Class discovery.* `_pick_policy_class` walks `vars(mod).values()`, filters to non-abstract subclasses of `CachePolicy` whose `__module__ == mod.__name__`, requires exactly one. Zero, multiple, or abstract-only matches → `PolicyLoadError`.
- *Source-text capture.* For path/dotted variants, the loader reads the underlying file once and stores both source and hash on `LoadedPolicy` so the registry can log the bytes that were actually live. This matters because the agent's working tree may move on after the swap.
- *Failure semantics.* Loader entry points wrap ordinary exceptions in typed `PolicyLoadError(origin, traceback)`. `KeyboardInterrupt` / `SystemExit` are not swallowed.

**Threat model** (documented in module docstring): this `exec()`s untrusted Python in-process. The feature is disabled unless the operator passes `--enable-policy-hotswap`, sets `enable_policy_hotswap=True` in Python, or sets `VLLM_ENABLE_POLICY_HOTSWAP=1`. `EngineArgs.create_engine_config()` stores the resolved boolean in `VllmConfig.additional_config["enable_policy_hotswap"]` so the engine process can enforce the same gate. The HTTP route additionally requires a localhost client (`127.0.0.1` / `::1`) unless `VLLM_POLICY_HOTSWAP_ALLOW_REMOTE=1`.

## 6. Registry / swapper — `vllm/v1/kv_offload/cpu/policies/registry.py` (new file)

Process singleton living in the **scheduler** process (same process as `OffloadingConnectorScheduler`), keyed by `VllmConfig.instance_id` so multiple in-process engines and unit tests do not share a swap target by accident.

```python
from collections.abc import Callable

class ActivePolicy(msgspec.Struct):
    engine_id: str
    generation: int
    policy_name: str
    policy_version: str
    source_hash: str
    source_origin: str

class SwapResult(msgspec.Struct):
    ok: bool
    generation: int                 # current generation after the call
    policy_name: str | None = None  # of the now-active policy
    policy_version: str | None = None
    source_hash: str | None = None
    previous_generation: int = 0    # generation before this call (0 if never swapped)
    dry_run: bool = False
    latency_ms: float = 0.0
    error: str | None = None

@dataclass
class PolicyRegistryEntry:
    manager_ref: weakref.ref[CPUOffloadingManager]
    generation: int = 0
    active: ActivePolicy | None = None
    active_module_name: str | None = None
    # Installed only for evolved policies; used by §14.2 and stats.
    supervisor: SupervisedCachePolicy | None = None
    # Set by EngineCore after `_idle_state_callbacks` exists.
    enqueue_idle_callback: Callable[[Callable[[], None]], None] | None = None
    # Sticky-within-window stats (cleared only by `take_policy_stats(reset=True)`,
    # see §14.5). `cumulative_policy_errors` carries forward errors from rolled-back
    # supervisors so a snapshot taken after rollback still sees them, since
    # `entry.supervisor` has been cleared and its `_errors` field is no longer
    # reachable. The currently-active supervisor's `_errors` is added on top at
    # snapshot time.
    cumulative_policy_errors: int = 0
    policy_rolled_back: bool = False
    rollback_generation: int = 0

class PolicySwapRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, PolicyRegistryEntry] = {}

    def attach(
        self,
        engine_id: str,
        manager: CPUOffloadingManager,
    ) -> None: ...
    def attach_idle_callback(
        self,
        engine_id: str,
        enqueue_idle_callback: Callable[[Callable[[], None]], None],
    ) -> None: ...
    def swap(
        self,
        engine_id: str,
        loaded: LoadedPolicy,
        policy_kwargs: dict[str, object] | None = None,
        policy_name: str | None = None,
        policy_version: str | None = None,
        dry_run: bool = False,
    ) -> SwapResult: ...
    def current(self, engine_id: str) -> ActivePolicy | None: ...
```

`attach` is called from [`CPUOffloadingSpec.get_manager()`](../vllm/v1/kv_offload/cpu/spec.py), the lazy single-shot construction site, **gated on `self.vllm_config.additional_config.get("enable_policy_hotswap")`**. Keep the bare `CPUOffloadingManager` in a local variable; if the gate is set, attach that inner manager with `self.vllm_config.instance_id`; then optionally wrap it in `FilterReusedOffloadingManager`. The registry stores a weak ref to the inner `CPUOffloadingManager` (whose `_policy` is the swap target), not to the wrapper; the wrapper delegates `lookup`/`prepare_*` and has no policy state of its own. Hooking the spec rather than the manager's `__init__` means unit tests that construct `CPUOffloadingManager` directly do not pollute the singleton, and the registry sees exactly one attach per configured engine. Default-off deploys skip the registry entirely — no singleton entries, no idle callback, zero overhead — which matches the §5 "default-off" security posture.

The idle-rollback callback is attached separately from `EngineCore.__init__`, after `self._idle_state_callbacks` has been initialized, **gated on the same `additional_config["enable_policy_hotswap"]` flag** so default-off engines never create registry entries:

```python
if self.vllm_config.additional_config.get("enable_policy_hotswap"):
    PolicySwapRegistry.singleton().attach_idle_callback(
        self.vllm_config.instance_id,
        lambda cb: self._idle_state_callbacks.append(lambda _engine: cb()),
    )
```

The callback signature matches `_notify_idle_state_callbacks` at [core.py:1227-1228](../vllm/v1/engine/core.py#L1227-L1228), which calls `callback(self)` with the engine as the sole positional argument; the inner `lambda _engine: cb()` accepts and discards it.

This avoids threading an `EngineCore` reference through `KVConnectorFactory`, `OffloadingConnector`, `OffloadingConnectorScheduler`, and `OffloadingSpec`. The connector spec may be constructed *after* `EngineCore.__init__` finishes setting up `_idle_state_callbacks`, so `attach_idle_callback` is order-tolerant: it stores the enqueue function on the entry (creating a placeholder entry if `attach()` has not yet run), and the later `attach()` call preserves any enqueue function already present. Symmetrically, `attach()` arriving first leaves `enqueue_idle_callback=None` until `attach_idle_callback` fills it in. Either ordering yields a fully wired entry; only the absence of *both* calls leaves rollback to the next-swap path (§14.2 point 1).

**Saturation caveat.** `_notify_idle_state_callbacks` is drained inside `_process_input_queue` only when `not self.has_work()` ([core.py:1178-1180](../vllm/v1/engine/core.py#L1178-L1180)). On a permanently saturated busy loop the idle path may not fire for the duration of a CORAL candidate's run; rollback then waits for the next external swap (point 1 in §14.2). The supervisor still suppresses read-side errors during the wait, so saturation does not crash the engine — only delays recovery.

Different `engine_id` values coexist in the registry. If `attach` is called twice for the same `engine_id`, the registry logs a warning, replaces the weak ref, and drops any active synthetic module owned by the old entry.

`attach` initializes `ActivePolicy` at generation 0 from the built-in policy metadata (`source_origin="builtin"`, `source_hash="builtin"`) so `current(engine_id)` returns useful data before the first evolved policy is loaded.

**`swap()` algorithm** (under `self._lock`):

0. Resolve: find the registry entry for `engine_id` and dereference the manager. Missing entry → remove the candidate module from `sys.modules` and return `SwapResult(ok=False, generation=0, previous_generation=0, error="CPU offloading manager is not attached")`; no module becomes active. Entry present but weak ref dead (the manager has been GC'd — should not happen during normal operation, but defensively handled) → return the same shape but with `generation=previous_generation=entry.generation` so observers do not see an apparent reset.
0a. Pre-rollback if a previous candidate has tripped its supervisor: if `entry.supervisor is not None and entry.supervisor.tripped`, run the cold-recovery procedure from §14.2 *before* step 1. This ensures the snapshot in step 1 happens against a coherent built-in `LRUCachePolicy`, not the tripped supervisor (whose `inner.export_state()` may be the very thing that's broken). After this point `entry.supervisor is None` and `manager._policy` is the recovered LRU. The forward swap then proceeds normally; the rolled-back-from generation is reflected in the returned `SwapResult.previous_generation` (which now points at the recovery generation, not the tripped candidate).
1. Snapshot: `old_policy = manager._policy`; capture `state = list(old_policy.export_state())`. Export failure → return `SwapResult(ok=False, error=...)`; `manager._policy` untouched.
2. Construct: `new_policy = loaded.cls(cache_capacity=manager._num_blocks, **policy_kwargs)`. Constructor exception → return `SwapResult(ok=False, error=...)`; `manager._policy` untouched.
3. Migrate: `new_policy.import_state(state)`. Import failure aborts the swap and discards `new_policy`; no cold fallback in Phase 1. The block pool is only consistent if the new policy receives the same resident `BlockStatus` references.
4. Drain barrier: `CPUOffloadingManager` methods are called only from the scheduler thread (single-threaded inside `EngineCore.step`). The swap is **invoked from the scheduler thread itself** via the `call_utility` plumbing (§7), so we are between scheduler steps when we mutate. The lock guards only against concurrent *external* swap requests. In-flight transfers in the worker process are tracked by `BlockStatus.ref_cnt`, which is preserved across the migration.
5. Canary: exercise the new policy's full read+write path before the pointer flip. The canary runs on a **separate, empty `canary_policy = loaded.cls(cache_capacity=manager._num_blocks, **policy_kwargs)`** — never on `new_policy`. Running on `new_policy` after `import_state` would let `evict(...)` silently remove a real imported block without the manager's `_free_list` knowing, leaking a CPU slot when the pointer flip proceeds. Constructing a second instance is cheap (cache_capacity is just a sizing hint; data structures are empty until `insert`). With sentinel `OffloadKey` values (random bytes, length distinct from any real key) and fake `BlockStatus(block_id=-1)` objects:
   1. `canary_policy.touch([])` and `canary_policy.evict(0, set())` — empty-input edge cases.
   2. `canary_policy.evict(1, set())` — must return `None` on an empty cache (verifies the "cannot satisfy" branch and that protected-set iteration handles empty input).
   3. `canary_policy.insert(canary_key, fake_block)`; assert `canary_policy.get(canary_key) is fake_block`.
   4. Set `fake_block.ref_cnt = 0`; `canary_policy.touch([canary_key])`; assert `canary_policy.get(canary_key) is fake_block`.
   5. `canary_policy.evict(1, {canary_key})` — must return `None` (the only entry is protected); verifies protected-set short-circuit.
   6. Construct a second sentinel pair: `evict_key` from a different random byte string distinct from `canary_key`, and `evict_block = BlockStatus(block_id=-2)` with `evict_block.ref_cnt = 0` set explicitly (BlockStatus initializes `ref_cnt = -1`). Call `canary_policy.insert(evict_key, evict_block)`. Then assert `canary_policy.evict(1, set())` returns either `[(canary_key, fake_block)]` or `[(evict_key, evict_block)]` (length must be exactly 1; the choice depends on policy order), and assert the evicted key is no longer present via `canary_policy.get(...)`.
   7. Remove whichever sentinel key remains; assert both sentinel keys are absent.
   8. With-context probe: with `ctx = ReqContext(policy_hints={"_canary": True})`, construct a third sentinel pair (`ctx_key` from a new random byte string distinct from the prior two, `ctx_block = BlockStatus(block_id=-3)` with `ctx_block.ref_cnt = 0` set explicitly); call `canary_policy.insert(ctx_key, ctx_block, req_context=ctx)`; `canary_policy.touch([ctx_key], req_context=ctx)`; assert `canary_policy.get(ctx_key) is ctx_block`; `canary_policy.evict(1, set(), req_context=ctx)` must return a length-1 list; assert `canary_policy.get(ctx_key) is None` afterwards. This catches policies that crash when `req_context` is non-None — a class of bug the context-free substeps would miss (§16.7).

   If any step raises or any assertion fails, discard both `canary_policy` and `new_policy` and return failure before the pointer flip. The canary surfaces a broad class of bugs (missing returns, wrong types, mishandled empty inputs, stateful corruption from `insert→evict→remove`, broken protected-set handling) at swap time rather than mid-decode. The fake blocks / sentinel keys never enter `new_policy` or the manager's block pool — `canary_policy` is dropped on the floor in either branch.
6. Metadata: resolve `policy_name` / `policy_version` from the request override first, then from `loaded.cls.POLICY_NAME` / `POLICY_VERSION`. Empty values fail before the pointer flip.
7. Dry run: if `dry_run=True`, remove the candidate module from `sys.modules`, return success with the candidate metadata, and leave `manager._policy`, generation, and active policy unchanged.
8. Pointer flip: wrap the candidate in `supervisor = SupervisedCachePolicy(new_policy, on_trip=...)` (§14.1) and set `manager._policy = supervisor`. Increment this engine's generation, set `entry.supervisor = supervisor`, set `entry.active_module_name = loaded.module.__name__`, and update `entry.active` from the resolved metadata. After the new policy is active, remove the previous active synthetic module from `sys.modules` (if any). Phase 1 intentionally keeps only the current evolved module alive; auto-rollback recovers into a built-in LRU policy (§14.2), not into the previous evolved module.

**Rollback** is held in a stack-local. Any failure after step 1 restores `manager._policy = old_policy` before releasing the lock. We never lose the prior policy.

`ActivePolicy`, `SwapResult`, and stats objects should be msgpack-safe (`msgspec.Struct` or plain JSON-like dicts), because multiprocess utility results are serialized without insecure pickle by default.

## 7. Cross-process plumbing — `call_utility`

The KV-offload manager lives in the scheduler-side `EngineCore`. In multiprocess serving/offline mode, the API server / `LLM` talks to it through the established `call_utility(method, *args)` pattern at [vllm/v1/engine/core_client.py:812](../vllm/v1/engine/core_client.py#L812). LoRA add/remove/pin all use it.

We add:
- `EngineCore.swap_offload_policy(payload)` — utility method that receives the swap payload, verifies the engine-side feature gate before executing user code, runs the loader inside the engine process, calls `PolicySwapRegistry.singleton().swap(self.vllm_config.instance_id, loaded, ...)`, and returns the `SwapResult`.
- `EngineCore.get_offload_policy_stats(reset=False)` and `EngineCore.get_offload_policy()` — utility methods that resolve the current engine's registry entry and return msgpack-safe structs.
- `EngineCoreClient.swap_offload_policy(payload)`, `.get_offload_policy_stats(reset=False)`, `.get_offload_policy()`, and async counterparts — wrappers around direct `EngineCore` calls for `InprocClient`, and `call_utility` / `call_utility_async` for multiprocess clients.

Critically: **the loader runs in the engine process, not the API server**. The payload sent over IPC is the raw source / path / dotted name, never a class object. This avoids pickling user code and ensures the policy module is loaded into the same process that owns `CPUOffloadingManager`.

Phase 1 should fail fast when `data_parallel_size > 1`. The existing DP utility path is split: `DPLBAsyncMPClient.call_utility_async` ([core_client.py:1380](../vllm/v1/engine/core_client.py#L1380)) fans out to every `EngineCore` and returns only the first result, while the external-LB `DPAsyncMPClient` inherits `AsyncMPClient.call_utility_async` ([core_client.py:1038](../vllm/v1/engine/core_client.py#L1038)) and targets a single engine via `self.core_engine` — so a swap would only land on rank 0 and the other ranks would silently keep the prior policy. Either failure mode is unacceptable: hot-swap needs an aggregate result and all-or-nothing semantics across scheduler processes before it should support DP. The check fires at engine init (when policy hotswap is enabled and `data_parallel_size > 1`, raise from `EngineArgs.create_engine_config` so misconfiguration surfaces before any benchmark starts), not at swap call time — this prevents a CORAL run from silently scoring against only one of N engines.

## 8. Control plane — HTTP and Python

### 8.1 HTTP — `vllm/entrypoints/serve/policy_hotswap/{api_router.py, protocol.py}` (new)

Mirrors [vllm/entrypoints/serve/lora/](../vllm/entrypoints/serve/lora/). Three routes, registered only when either `app.state.args.enable_policy_hotswap` or `VLLM_ENABLE_POLICY_HOTSWAP` is true:

- `POST /v1/swap_offload_policy`
  ```json
  {
    "source_path": "/abs/path/to/policy.py",
    "module": "my_pkg.evolved_v3",
    "source": "class Policy(CachePolicy): ...",
    "name": "arc-tournament",
    "version": "v3",
    "policy_kwargs": {},
    "dry_run": false
  }
  ```
  Exactly one of `source_path` / `module` / `source` must be set. Response is the `SwapResult` (§6) serialized as JSON, e.g.:
  ```json
  {
    "ok": true, "generation": 42, "previous_generation": 41,
    "policy_name": "arc-tournament", "policy_version": "v3",
    "source_hash": "ab12cd34ef56", "dry_run": false, "latency_ms": 12.3,
    "error": null
  }
  ```
  Validation failures (more than one source, missing required field) return HTTP 400. Candidate load / swap failures also return a JSON body with `ok=false` and `error`; use HTTP 400 for candidate-caused failures and HTTP 500 only for unexpected internal errors.

- `GET /v1/offload_policy` → returns the current `ActivePolicy`.
- `GET /v1/offload_policy_stats?reset=false` → returns current `OffloadPolicyStats` (see §8.3).

The router is registered in [vllm/entrypoints/serve/__init__.py](../vllm/entrypoints/serve/__init__.py) next to the LoRA/profile/sleep/cache router attach calls. [vllm/entrypoints/openai/api_server.py](../vllm/entrypoints/openai/api_server.py) sets `app.state.args` before calling `register_vllm_serve_api_routers(app)`, so `attach_router` can consult either the CLI flag or the env var. On a default deployment the routes are not registered with FastAPI. Auth: feature gate + localhost default; reuse the loud-warning log line the LoRA router already prints.

### 8.2 Python API

```python
class LLMEngine:
    def swap_offload_policy(
        self,
        source_path: str | None = None,
        module: str | None = None,
        source: str | None = None,
        name: str | None = None,
        version: str | None = None,
        policy_kwargs: dict | None = None,
        dry_run: bool = False,
    ) -> SwapResult: ...

    def get_offload_policy(self) -> ActivePolicy | None: ...
    def get_offload_policy_stats(self, reset: bool = False) -> OffloadPolicyStats: ...
```

Add the same forwarding methods to the offline [`LLM`](../vllm/entrypoints/llm.py) wrapper because CORAL's grader uses `LLM(...)`, not `LLMEngine(...)`, directly:

```python
class LLM:
    def swap_offload_policy(...) -> SwapResult: ...
    def get_offload_policy(self) -> ActivePolicy | None: ...
    def get_offload_policy_stats(self, reset: bool = False) -> OffloadPolicyStats: ...
```

Parallel async variants live on `AsyncLLM` / `AsyncLLMEngine`. All paths forward to the matching `EngineCoreClient` swap/current-policy/stats methods.

### 8.3 Policy stats

Add a small scheduler-side stats object so graders do not depend on log scraping:

```python
class OffloadPolicyStats(msgspec.Struct):
    lookups: int = 0              # total `lookup()` calls
    hits: int = 0                 # `lookup()` calls that returned ready=True
    stores: int = 0               # blocks newly inserted via prepare_store
    evictions: int = 0            # blocks evicted via prepare_store
    loads: int = 0                # blocks loaded via prepare_load
    hit_rate: float = 0.0         # hits / max(lookups, 1)
    window_start_generation: int = 0
    active_generation: int = 0    # generation at the time this snapshot was taken
    hint_keys: tuple[str, ...] = ()  # union of top-level keys observed in
                                      # req_context.policy_hints since last reset (§16.6)
```

`CPUOffloadingManager` updates the raw counters in `lookup`, `prepare_load`, and `prepare_store`; `take_policy_stats(reset: bool, active_generation: int)` returns a snapshot with `hit_rate = hits / max(lookups, 1)`. `window_start_generation` and `active_generation` let the grader detect that stats span multiple policy versions (`window_start_generation != active_generation` means the score is not attributable to a single policy). The registry exposes the snapshot through `EngineCore.get_offload_policy_stats(reset=False)`. Resetting stats at the start of each CORAL candidate keeps scores comparable.

## 9. CORAL grader sketch

A reference grader (`coral_task/eval/grader.py` in the user's CORAL task repo, **not** in vLLM):

```python
import time, hashlib
from pathlib import Path
from vllm import LLM, SamplingParams
from coral.grader import TaskGrader

class Grader(TaskGrader):
    def evaluate(self):
        evolved = Path(self.codebase_path) / "policy.py"
        src = evolved.read_text()
        h = hashlib.sha256(src.encode()).hexdigest()[:12]

        llm = LLM(
            model=self.args["model"],
            kv_transfer_config={
                "kv_connector": "OffloadingConnector",
                "kv_connector_extra_config": {
                    "spec_name": "CPUOffloadingSpec",
                    "cpu_bytes_to_use": self.args["cpu_bytes"],
                    "eviction_policy": "lru",
                },
            },
            enable_policy_hotswap=True,  # forwarded through LLM(**kwargs) to EngineArgs
        )

        result = llm.swap_offload_policy(source_path=str(evolved))
        if not result.ok:
            return self.fail(f"swap failed: {result.error}")

        llm.get_offload_policy_stats(reset=True)
        prompts = (Path(self.private_dir) / "prompts.txt").read_text().splitlines()
        t0 = time.perf_counter()
        llm.generate(prompts, SamplingParams(max_tokens=256))
        elapsed = time.perf_counter() - t0
        stats = llm.get_offload_policy_stats()
        score = stats.hit_rate / max(elapsed, 1e-6)
        return self.score_bundle(
            value=score,
            explanation=f"hit_rate={stats.hit_rate:.3f} elapsed={elapsed:.2f}s hash={h}",
        )
```

The grader spins up `LLM` once per process (or reuses a daemonized one across attempts), bootstraps with `lru`, then hot-swaps the agent's `policy.py` and benchmarks. No CORAL-side changes are required; this fits the existing `TaskGrader` contract.

## 10. Files to add and modify

**New files** (all in vLLM):

| Path | Purpose |
|---|---|
| `vllm/v1/kv_offload/cpu/policies/loader.py` | `PolicyLoader`, `LoadedPolicy`, `PolicyLoadError`. Pure stdlib. |
| `vllm/v1/kv_offload/cpu/policies/registry.py` | `PolicySwapRegistry`, `ActivePolicy`, `SwapResult`. Singleton. |
| `vllm/v1/kv_offload/cpu/policies/supervisor.py` | `SupervisedCachePolicy` (§14.1) — try/except wrapper applied to evolved policies on install. |
| `vllm/v1/kv_offload/cpu/policies/metrics.py` | Prom counters: `vllm_policy_swap_count_total{result}`, `vllm_policy_swap_latency_ms`, `vllm_policy_errors_total`, `vllm_policy_rollbacks_total`, gauge `vllm_policy_active_generation`. |
| `vllm/entrypoints/serve/policy_hotswap/__init__.py` | Package marker. |
| `vllm/entrypoints/serve/policy_hotswap/api_router.py` | FastAPI router; mirrors LoRA. |
| `vllm/entrypoints/serve/policy_hotswap/protocol.py` | Pydantic request/response models. |
| `tests/v1/kv_offload/cpu/test_policy_loader.py` | Loader unit tests. |
| `tests/v1/kv_offload/cpu/test_policy_registry.py` | Registry/swap/rollback tests. |
| `tests/v1/kv_offload/cpu/test_policy_stats.py` | Manager policy-stats tests. |
| `tests/v1/kv_offload/cpu/test_policy_supervisor.py` | Supervisor unit tests: read-side suppression, error budget, rollback semantics (§14). |
| `tests/v1/kv_offload/cpu/test_policy_hotswap_e2e.py` | Engine-level swap mid-generation. |
| `tests/v1/kv_offload/cpu/test_policy_request_context.py` | End-to-end hint propagation: stub policy records `(method, key, hints)` tuples; verify `vllm_xargs["policy_hints"]` from an HTTP / offline request reaches `_policy.insert/touch/evict` (§16). |

**Modified files**:

| Path | Change |
|---|---|
| [vllm/v1/kv_offload/cpu/policies/base.py](../vllm/v1/kv_offload/cpu/policies/base.py) | Widen the ABC `__init__` to `(cache_capacity: int, **kwargs: Any)` per §4; widen `insert` / `touch` / `evict` to accept optional `req_context: ReqContext \| None = None` (§16); add an abstract `export_state` hook and a concrete `import_state` hook (default `import_state` loops through `insert`); add a no-op `record_write_error()` hook on `CachePolicy` that `SupervisedCachePolicy` overrides (§14.1) so the manager can notify the supervisor of write-side failures without isinstance checks. |
| [vllm/v1/kv_offload/cpu/policies/lru.py](../vllm/v1/kv_offload/cpu/policies/lru.py) | Implement `export_state()`, built-in policy metadata, widen `__init__` to accept and ignore `**kwargs` (per §4), and accept-and-ignore `req_context` on `insert` / `touch` / `evict` (per §16). |
| [vllm/v1/kv_offload/cpu/policies/arc.py](../vllm/v1/kv_offload/cpu/policies/arc.py) | Implement `export_state()` for resident T1/T2 blocks only, built-in policy metadata, widen `__init__` to accept and ignore `**kwargs` (per §4), and accept-and-ignore `req_context` on `insert` / `touch` / `evict` (per §16). |
| [vllm/v1/kv_offload/cpu/manager.py](../vllm/v1/kv_offload/cpu/manager.py) | Maintain `OffloadPolicyStats` (including `hint_keys` accumulation); expose `take_policy_stats(reset=False, active_generation=...)`. Widen `CPUOffloadingManager.touch` to accept `req_context` to match the new ABC. Thread `req_context` from `lookup` / `prepare_load` / `prepare_store` / `touch` into `_policy.insert` / `touch` / `evict` per §16, including the §14.3 hardened insert loop. Wrap the `prepare_store` insert loop to free pre-allocated blocks if `_policy.insert` raises (§14.3). |
| [vllm/v1/kv_offload/base.py](../vllm/v1/kv_offload/base.py) | Add `policy_hints: dict[str, Any] \| None = None` to `ReqContext` (§16.2); widen the `OffloadingManager.touch` ABC to take `req_context: ReqContext` so policy `touch` can receive request scope (§16.4). |
| [vllm/v1/kv_offload/reuse_manager.py](../vllm/v1/kv_offload/reuse_manager.py) | Update `FilterReusedOffloadingManager.touch` to delegate the widened `touch(keys, req_context)` signature (§16.4). No other behavior change; the wrapper has no policy state. |
| [vllm/v1/request.py](../vllm/v1/request.py) | Initialize `self.policy_hints: dict[str, Any] \| None = None` immediately after [request.py:101](../vllm/v1/request.py#L101) so the attribute exists for pooling-only requests too. Inside the existing `extra_args` block at lines 113-116, additionally lift `policy_hints`, validate via `dict(raw_hints)`, inject the reserved `_request_id` key, and assign to `self.policy_hints` (§16.1). |
| [vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) | In `RequestOffloadState.__post_init__` (line 141), populate `ReqContext.policy_hints=self.req.policy_hints` alongside `kv_transfer_params`. Inside `_touch` (lines 294 and 303), pass `req_status.req_context` into both widened `manager.touch(...)` calls (full-attention and sliding-window branches). |
| [vllm/v1/kv_offload/cpu/spec.py](../vllm/v1/kv_offload/cpu/spec.py) | In `get_manager()`, when `self.vllm_config.additional_config.get("enable_policy_hotswap")`, call `PolicySwapRegistry.singleton().attach(self.vllm_config.instance_id, inner)` on the bare `CPUOffloadingManager` *before* it is optionally wrapped by `FilterReusedOffloadingManager`. Skip the attach silently when the flag is unset (default-off). |
| [vllm/envs.py](../vllm/envs.py) | Add `VLLM_ENABLE_POLICY_HOTSWAP`, `VLLM_POLICY_HOTSWAP_ALLOW_REMOTE`. |
| [vllm/engine/arg_utils.py](../vllm/engine/arg_utils.py) | Add `enable_policy_hotswap: bool = False` field on `EngineArgs` and the matching `--enable-policy-hotswap` CLI flag; in `create_engine_config()` resolve the effective value (CLI flag OR `VLLM_ENABLE_POLICY_HOTSWAP`) and store it under `additional_config["enable_policy_hotswap"]` so the engine process sees the same gate. Raise from `create_engine_config()` if the flag is set together with `data_parallel_size > 1` (per §7). |
| [vllm/engine/protocol.py](../vllm/engine/protocol.py) | Add async `swap_offload_policy`, current-policy, and stats methods to the `EngineClient` protocol for serve routes. |
| [vllm/v1/engine/core.py](../vllm/v1/engine/core.py) | Add `swap_offload_policy(payload)`, `get_offload_policy()`, and `get_offload_policy_stats(reset=False)` utility methods on `EngineCore`; after `_idle_state_callbacks` is initialized, when `additional_config["enable_policy_hotswap"]` is set, register the policy registry idle-callback enqueue hook. |
| [vllm/v1/engine/core_client.py](../vllm/v1/engine/core_client.py) | Add sync/async swap, current-policy, and stats forwarding methods, including `InprocClient` direct calls. |
| [vllm/v1/engine/llm_engine.py](../vllm/v1/engine/llm_engine.py) | Public swap/current-policy/stats forwarding to client. |
| [vllm/v1/engine/async_llm.py](../vllm/v1/engine/async_llm.py) | Public async swap/current-policy/stats forwarding to client. |
| [vllm/entrypoints/llm.py](../vllm/entrypoints/llm.py) | Offline `LLM` forwarding methods used by CORAL. |
| [vllm/entrypoints/serve/__init__.py](../vllm/entrypoints/serve/__init__.py) | Attach the new router next to the other serve routers. |

## 11. Test plan

**Unit** (no GPU):
- `test_policy_loader.py`: load from path, dotted module, raw source. Hash determinism. Fresh-namespace isolation: mutate module-level state in load #1, verify load #2 sees pristine state. Negative cases (zero subclasses / multiple subclasses / syntax error / abstract subclass) → each raises `PolicyLoadError`.
- `test_policy_registry.py`: stub `CPUOffloadingManager` + dummy policy. Happy-path swap: verify pointer flip, generation increment, state migration. Failing `export_state`, `__init__`, or `import_state` → pointer untouched. Failing canary (each of `touch`, `evict(0)`, protected `evict`, successful `evict`, `insert`, `get`, `remove` synthetic-key cycle) → rollback restores prior pointer. `dry_run=True` validates but does not increment generation and cleans up the candidate module. Two attached `engine_id` values remain isolated.
- `test_policy_supervisor.py`: `get`/`evict`/`touch` raising → suppressed, error counter increments, manager sees safe defaults (None / no-op). `insert`/`remove` raising → re-raised. Error budget exceeded → supervisor trips. Cold rollback installs a coherent recovered policy from `inner.export_state()`; if inner export also raises, rollback installs an empty `LRUCachePolicy` and frees all CPU blocks (no leak). Block-pool reconciliation: when `inner.export_state()` succeeds but its residents are a strict subset of `_num_allocated_blocks` (simulating a buggy `insert` that silently dropped keys), assert that after rollback `_free_list` covers exactly the missing block_ids and that subsequent `prepare_store` can reuse those slots (no orphaned-slot leak).
- Manager hardening: `_policy.insert` raising mid-`prepare_store` → inserted keys are undone, all allocated blocks are freed, `prepare_store` returns `None` (no fatal exception), and `_num_allocated_blocks` / free-list invariants hold. Also cover `_policy.remove` failure in `complete_store(success=False)` (best-effort remove + block freed + error recorded) (§14.3).
- `test_policy_stats.py`: manager lookup/load/store counters, hit-rate calculation, reset behavior, and wrapper composition. Reset semantics for `policy_rolled_back` / `rollback_generation`: assert sticky-true within a window (visible to `get_offload_policy_stats(reset=False)` after a trip) and cleared only by `take_policy_stats(reset=True)`. Cover `hint_keys` accumulation and reset (§16.6).
- `test_policy_request_context.py`: built-in LRU/ARC accept and ignore `req_context` (signature compatibility, no behavior change). Stub `CachePolicy` records every `(method, key, hints)` it receives; (a) offline `LLM.generate` with `SamplingParams(extra_args={"policy_hints": {"priority": "high"}})` produces matching `insert` / `touch` / `evict` observations; (b) HTTP `POST /v1/completions` with `vllm_xargs={"policy_hints": {"priority": "high"}}` produces the same observations (§16).

**Integration** (1 GPU, small model):
- `test_policy_hotswap_e2e.py`: real `AsyncLLM` or HTTP server with `OffloadingConnector`, start ~50 prompts, then call `swap_offload_policy` to a hand-written counting policy while requests are active. Assert: generation completes, no token corruption (compare to reference run), `counter > 0` after swap. Offline `LLM` tests should swap between `generate()` calls, not from another thread.
- Rollback test: swap to a policy whose `evict()` always raises; assert subsequent generation succeeds (read-side exceptions are suppressed). Then escalate: swap to a policy whose `insert()` raises; assert the engine survives, the supervisor trips, and the next `get_offload_policy_stats()` reports `policy_rolled_back=True` with the correct `rollback_generation`.
- `FilterReusedOffloadingManager` composition: `attach()` must reach the inner manager. Verify with `store_threshold >= 2`.

**CORAL e2e smoke** (manual, not CI):
```bash
cd /Users/ophir/PycharmProjects/CORAL
.venv/bin/python -m coral.cli evaluate --task offload-policy --once
```

## 12. Verification — exact commands

```bash
# unit
.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_policy_loader.py \
    tests/v1/kv_offload/cpu/test_policy_registry.py \
    tests/v1/kv_offload/cpu/test_policy_stats.py -v

# integration (1 GPU, 1B model)
VLLM_ENABLE_POLICY_HOTSWAP=1 .venv/bin/python -m pytest \
    tests/v1/kv_offload/cpu/test_policy_hotswap_e2e.py -v

# server smoke
VLLM_ENABLE_POLICY_HOTSWAP=1 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  meta-llama/Llama-3.2-1B \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":4294967296,"eviction_policy":"lru"}}' \
  --enable-policy-hotswap &

curl -s localhost:8000/v1/offload_policy
curl -s -X POST localhost:8000/v1/swap_offload_policy \
  -H 'content-type: application/json' \
  -d '{"source_path":"/abs/path/to/evolved.py"}'
curl -s localhost:8000/v1/offload_policy   # generation incremented, name updated
curl -s localhost:8000/v1/offload_policy_stats
```

**Success looks like**: HTTP 200 with `ok: true`, `generation` strictly greater than the previous read, the next `/v1/completions` request returns normally, `vllm_policy_swap_count_total{result="ok"}` increments at `/metrics`.

## 13. Risks and open questions

- **Thread safety is shallow.** `CPUOffloadingManager` runs only on the scheduler thread; the registry's `RLock` exists only to serialize concurrent swap *requests*. If a future change makes the manager re-entrant from another thread (e.g. an async lookup callback), we must add a `_quiescent` event the scheduler sets between steps and the swap waits on. Document the assumption in the registry docstring.
- **State migration is lossy w.r.t. recency.** ARC and LRU keep different metadata; we transfer only `(key, BlockStatus)`. Evolved policies that need fine-grained recency must accept a cold first window post-swap. For evolutionary search this is fine — the grader runs long enough to amortize.
- **`exec()` of untrusted code.** Default-off, CLI/env flagged, localhost-only by default, source hash logged. Not sandboxed. For CORAL the threat model is "the agent owns its own worktree anyway"; for production servers this endpoint must remain disabled. Documented in route docstring and `--help` text.
- **Import side effects in evolved code.** Module-body code runs at swap time: a top-level `threading.Thread(...).start()` or `open(...)` will fire on every load. The loader does not (and cannot, in-process) sandbox this. The class-level `POLICY_VERSION` requirement gives the grader a way to reject candidates with obvious top-level work, but the contract has to be social: evolved policies should keep `__init__` and module body cheap and side-effect-free.
- **Buggy policy mid-decode.** A buggy `evict()` returning `None` is legitimate (means "can't satisfy"); a buggy one might raise, hang, or corrupt invariants. The canary in §6 step 5 catches obvious bugs at swap time, but evolutionary search will produce policies that pass the canary and only fail under load. §14 specifies a graceful-degradation layer (`SupervisedCachePolicy`) that wraps an evolved policy after the pointer flip, converts read-side exceptions into safe defaults, and triggers automatic recovery to a coherent built-in LRU policy when the error budget is exceeded. Built-in policies are not wrapped, so they pay no overhead.
- **`BlockStatus` is `ctypes.Structure`.** Migration is by reference, not value. Two policies briefly hold references during the swap — fine because nothing mutates `BlockStatus` outside the manager and we hold the lock.
- **Constructor args beyond `cache_capacity`.** Current ABC takes only `(cache_capacity)`. We add an optional `**kwargs` passthrough plumbed from the swap request body via `policy_kwargs: dict`.
- **`FilterReusedOffloadingManager` composition.** Defined at [vllm/v1/kv_offload/reuse_manager.py:23](../vllm/v1/kv_offload/reuse_manager.py#L23); applied conditionally in [vllm/v1/kv_offload/cpu/spec.py:78](../vllm/v1/kv_offload/cpu/spec.py#L78) when `store_threshold >= 2`. The wrapper holds no policy state — it only filters which keys reach `prepare_store`. `attach()` must be called on the **inner** `CPUOffloadingManager`, before wrapping (see §6). Integration test must cover `store_threshold >= 2` to confirm the swap still drives the inner policy through the wrapper's delegation.
- **Data parallelism.** Existing async DP utility paths are inconsistent: `DPLBAsyncMPClient` fans out to every `EngineCore` but returns only the first result, while external-LB `DPAsyncMPClient` only targets `self.core_engine`. Hot-swap needs an aggregate result and all-or-nothing semantics across scheduler processes; Phase 1 should reject `data_parallel_size > 1` rather than risk divergent policies.
- **`policy_hints` is untrusted user data.** Any HTTP client can populate `vllm_xargs["policy_hints"]`; an evolved policy that trusts it (e.g., `os.system(hints["cmd"])`) inherits the §5 threat model in full. Evolved policies must validate hint shape and types before use; the framework does not. See §16.9.
- **Cross-request hint merge.** `OffloadKey` is content-addressed, so a single key can receive `insert` from request A and `touch` from request B with different hints over its lifetime. The framework hands every call the *calling request's* `req_context`; the policy decides the merge convention (first-writer-wins, last-toucher-wins, weighted aggregate, etc.). Authors who don't think about this will get last-call semantics by default. See §16.5.

## 14. Buggy-policy protection — surviving bugs in swapped code

The canary in §6 step 5 catches policies that fail synchronously on benign inputs, but CORAL will produce candidates that pass the canary and only misbehave under real workload — exceptions on rare key shapes, `evict()` returning malformed tuples, stale references, missing returns, slow degenerate paths. The engine cannot afford to crash on these: each crash forces CORAL to rebuild a fresh `LLM`, which is precisely what hot-swap exists to avoid.

This section describes a layered defense applied **only to evolved policies** (built-in `LRUCachePolicy` / `ARCCachePolicy` are not wrapped). Goal: keep the engine alive across a buggy generation, give the grader a clear error signal, and roll back automatically when a candidate is unsafe.

### 14.1 `SupervisedCachePolicy` — wrap-on-install

A thin `CachePolicy` that delegates to a wrapped `inner` policy and intercepts every method. Unlike user-loadable policies, the supervisor is constructed inline by the registry (§6 step 8), never via `PolicyLoader`, so its `__init__` deliberately ignores the widened `(cache_capacity, **kwargs)` ABC signature — Python only requires the abstract method to be overridden, not signature-compatible.

```python
from collections.abc import Callable

class SupervisedCachePolicy(CachePolicy):
    def __init__(
        self,
        inner: CachePolicy,
        error_budget: int = 8,
        on_trip: Callable[[], None] | None = None,
    ):
        self._inner = inner
        self._errors = 0
        self._budget = error_budget
        self._on_trip = on_trip
        self._tripped = False  # True once budget exhausted; registry will recover

    @property
    def inner(self): return self._inner
    @property
    def errors(self): return self._errors
    @property
    def tripped(self): return self._tripped

    def _record(self):
        self._errors += 1
        if not self._tripped and self._errors >= self._budget:
            self._tripped = True
            if self._on_trip is not None:
                self._on_trip()

    # Called by manager hardening (§14.3) after write-side exceptions.
    # Overrides the no-op `record_write_error` defined on the `CachePolicy`
    # ABC so the manager can notify *any* current policy without isinstance
    # checks — built-ins inherit the no-op and pay nothing.
    def record_write_error(self): self._record()

    # Read-side / suggestion ops: degrade to safe defaults on exception.
    # Signatures mirror the widened ABC (§16.3) so the manager's
    # `_policy.evict/touch(..., req_context)` calls dispatch correctly
    # through the supervisor — the supervisor sits at `manager._policy`
    # after the §6 step 8 pointer flip, so a context-free signature here
    # would `TypeError` on every forwarded `req_context`. `get` and
    # `remove` stay context-free per §16.3.
    def get(self, key):
        try: return self._inner.get(key)
        except Exception: self._record(); return None
    def evict(self, n, protected, req_context=None):
        try: return self._inner.evict(n, protected, req_context)
        except Exception: self._record(); return None
    def touch(self, keys, req_context=None):
        try: self._inner.touch(keys, req_context)
        except Exception: self._record()

    # State-mutating ops: propagate exceptions to manager hardening
    # (see §14.3), which rolls back partial writes and reports a
    # recoverable store failure.
    def insert(self, key, block, req_context=None):
        self._inner.insert(key, block, req_context)
    def remove(self, key):
        self._inner.remove(key)

    # State-transfer hooks: forward to inner so the *next* swap (which
    # sees `manager._policy` as a SupervisedCachePolicy) can still
    # snapshot resident blocks in §6 step 1. import_state is included
    # for symmetry; in practice incoming policies are always unwrapped
    # at construction time and only get wrapped after the pointer flip.
    def export_state(self): return self._inner.export_state()
    def import_state(self, items): self._inner.import_state(items)
```

The two existing graceful-degradation contracts in `CPUOffloadingManager` make read-side suppression safe:
- `lookup` returns `False` when `_policy.get(key)` is `None` ([manager.py:87](../vllm/v1/kv_offload/cpu/manager.py#L87)) → degraded gets become misses.
- `prepare_store` returns `None` when `_policy.evict(...)` returns `None` ([manager.py:138](../vllm/v1/kv_offload/cpu/manager.py#L138)) → degraded evicts mean "we couldn't free a slot this step"; the offloading connector retries on the next step.
- `touch` is advisory; suppressing it only loses recency information.

`prepare_load` and `complete_load` assume the key exists ([manager.py:99](../vllm/v1/kv_offload/cpu/manager.py#L99), [manager.py:111](../vllm/v1/kv_offload/cpu/manager.py#L111)). A degraded `get` returning `None` for a key the policy *should* have known about would trip these asserts. That is the correct outcome — it indicates the policy lost data the manager believes is resident, and the candidate must be rolled back rather than silently masked. The error-budget mechanism (§14.2) catches this before it propagates far. (`complete_store` already tolerates `None` from `get` — see [manager.py:177-186](../vllm/v1/kv_offload/cpu/manager.py#L177-L186) — so a degraded read there is silent rather than fatal.)

### 14.2 Error budget and auto-rollback

The registry does **not** reinstall the previous policy in Phase 1. Once an evolved policy starts serving traffic, its `insert` / `remove` calls can change the manager's block pool, while the previous policy is no longer updated. Reinstalling that stale policy would risk pointing at freed or reused CPU block IDs. Instead, Phase 1 recovers into a fresh built-in `LRUCachePolicy` populated from the evolved policy's exported resident blocks.

On swap completion (§6 step 8), the registry installs `manager._policy = SupervisedCachePolicy(new_policy, on_trip=...)` and stores that supervisor on the registry entry. The supervisor's `_record()` increments `self._errors`; when the error budget is exhausted, it sets `_tripped = True` and invokes `on_trip()` exactly once.

Rollback is triggered at one of two safe points (never mid-call):
1. The next external `swap_offload_policy` call (registry checks `supervisor._tripped` first; if set, it performs the cold-rollback procedure below to install a coherent baseline policy before attempting the next forward swap).
2. An idle-state callback that runs between steps via the existing `EngineCore._idle_state_callbacks` list (drained by `_notify_idle_state_callbacks` at [core.py:1225](../vllm/v1/engine/core.py#L1225)). The callback runs on the busy-loop thread when the queue is empty, so the manager is quiescent by construction; the callback re-acquires the registry lock and performs the cold-rollback procedure.

Wiring point 2 uses the `attach_idle_callback()` hook from §6. The registry hands the supervisor a tiny `on_trip` callable at construction. **`on_trip` reads `entry.enqueue_idle_callback` at trip time, not at supervisor construction**: if `attach_idle_callback()` arrives after the swap that wraps the supervisor (legal under the order-tolerant contract in §6), the supervisor must still be able to enqueue rollback when it eventually trips. Concretely, `on_trip` is a closure over `(self_registry, engine_id)` that re-locks the registry, looks up the entry, and reads `entry.enqueue_idle_callback` then. If still unset, the trip is recorded (`supervisor._tripped` remains True) and the next external swap performs recovery before loading a new candidate. If the engine is permanently saturated and never reaches an idle tick, point 1 still fires on the next external swap, so rollback is at most one CORAL generation late.

**Inproc-engine caveat.** `InprocClient` (V0-style `LLMEngine` callers that do not run a busy loop) calls `step_fn()` directly without going through `_process_input_queue`, so `_idle_state_callbacks` is never drained in that mode. Point 2 is therefore unavailable under `InprocClient`; rollback falls back entirely to point 1 (next external swap). CORAL uses `LLM(...)` → `SyncMPClient`, which has a busy loop, so the idle path works for the primary use case.

Why the two-point design: rollback at idle is fast (sub-step); rollback at next-swap guarantees the next CORAL candidate sees a clean baseline. Both must hold the registry lock and check `_tripped` while the manager is quiescent.

**Cold recovery procedure (Phase 1 default).** Re-acquires the registry lock and is idempotent: if `entry.supervisor is None` on entry, the rollback has already run (e.g. via the next-swap path) and this invocation no-ops. This matters because point 1 (next-swap) and point 2 (idle callback) can both fire for the same trip — point 2's callback is queued from `on_trip` and may still run after point 1 has already recovered.

1. Capture `rolled_back_module = entry.active_module_name`, `rollback_generation = entry.generation`, and `supervisor = entry.supervisor`.
2. Try `residents = list(supervisor.inner.export_state())`.
3. Validate that every resident `BlockStatus.block_id` is unique and in `range(manager._num_allocated_blocks)`.
4. Build `recovered = LRUCachePolicy(cache_capacity=manager._num_blocks)` and call `recovered.import_state(residents)`.
5. Reconcile the block pool: `manager._free_list = sorted(set(range(manager._num_allocated_blocks)) - recovered_block_ids)`. **In-flight caveat:** if the buggy policy silently dropped a block whose `BlockStatus.ref_cnt > 0` (in-flight `prepare_load` whose `complete_load` has not yet arrived), that block_id is NOT in `recovered_block_ids` and reconciliation will return it to the free list. A subsequent `prepare_store` may then reallocate the same `block_id` and overwrite the worker's in-flight CPU buffer before the load finishes, corrupting the GPU read. The eventual `complete_load` will trip the `block is not None` assert at [manager.py:99](../vllm/v1/kv_offload/cpu/manager.py#L99) — loud failure, not silent — but corruption can occur in the window between reallocation and the assert. Phase 1 accepts this: a buggy policy that loses in-flight blocks is precisely the failure mode `SupervisedCachePolicy` exists to *contain*, not to *paper over*. CORAL graders should treat `policy_rolled_back=True` as a strong negative signal regardless of measured hit rate, since the in-flight window may have produced corrupt outputs.
6. Install `manager._policy = recovered`. Drop the rolled-back candidate's `sys.modules` entry using the `rolled_back_module` captured in step 1 — call `sys.modules.pop(rolled_back_module, None)` so the operation is no-op-safe if a concurrent path has already removed the entry. Then clear `entry.supervisor` and `entry.active_module_name` (set to `None`). Increment `entry.generation` and set `entry.active` to built-in LRU metadata with `source_origin=f"rollback:{rollback_generation}"` and `source_hash="builtin"`. The capture-then-clear order matters: the next forward swap's "remove the previous active synthetic module" step (§6 step 8) keys on `entry.active_module_name`, and a stale value would otherwise target a module already removed during rollback. (§6 step 8 should likewise use `sys.modules.pop(name, None)` rather than bare `del` for the same reason.)
7. Set sticky stats fields on the entry: `policy_rolled_back=True`, `rollback_generation=<rolled-back candidate generation>`, and `cumulative_policy_errors += supervisor._errors`. These persist across subsequent forward swaps and are cleared only by `take_policy_stats(reset=True)`.

If `export_state`, validation, or `import_state` fails, fall back to an empty `LRUCachePolicy` and rebuild `manager._free_list = list(range(manager._num_allocated_blocks))`. This empties the CPU cache but keeps the engine alive and avoids allocated-but-unreachable CPU block slots. `_num_allocated_blocks` is left unchanged so the manager can reuse all existing CPU buffer slots.

The one-shot reconciliation costs one set-difference at rollback time and zero ongoing overhead. A buggy `inner` may have silently dropped keys it told us it accepted via `insert`; without reconciliation those slots stay allocated but unreachable forever.

**Mirror rollback (Phase 1.5, optional).** A future variant can tee write-side calls (`insert` / `remove`) to both `inner` and a retained previous policy for a configurable warm-up window (e.g., first 1000 ops post-swap). Within the window, rollback could restore the previous policy with a coherent view. Cost is one extra dict op per `insert`/`remove`, negligible vs. the offload transfer itself, but it adds enough complexity that Phase 1 should ship cold recovery first.

### 14.3 Manager hardening — write-path failures in `prepare_store` / `complete_store`

[manager.py:153-159](../vllm/v1/kv_offload/cpu/manager.py#L153-L159) allocates blocks first, then calls `_policy.insert` in a loop. If `insert` raises mid-loop, we must not leak slots or leave partially-inserted not-ready keys behind. Guard this path and convert it to a recoverable `prepare_store -> None` result:

```python
inserted = 0
for key, block in zip(keys_to_store, blocks):
    try:
        # `req_context` is threaded through so evolved policies can
        # consult `req_context.policy_hints` on insert (§16.4).
        self._policy.insert(key, block, req_context)
        inserted += 1
    except Exception:
        # Undo successful inserts and free their blocks. `remove` is
        # context-free (§16.3) so it is called without `req_context`.
        for undo_key, undo_block in zip(keys_to_store[:inserted], blocks[:inserted]):
            try:
                self._policy.remove(undo_key)
            except Exception:
                pass
            self._free_block(undo_block)
        # Free blocks that were allocated but never inserted.
        for pending_block in blocks[inserted:]:
            self._free_block(pending_block)
        self._policy.record_write_error()  # no-op on built-ins; trips supervisor budget (§14.2)
        return None
```

`record_write_error` is defined as a no-op on the `CachePolicy` ABC and overridden on `SupervisedCachePolicy`, so the manager can call it unconditionally without an `isinstance` check or supervisor-aware branching. Built-in policies (LRU, ARC) inherit the no-op and pay nothing.

Likewise, guard `complete_store(success=False)` so a buggy `_policy.remove` cannot crash the engine: always free the block, best-effort remove from policy, record a policy write error, and continue. This keeps `EngineCore` alive long enough for the §14.2 rollback path to replace the candidate.

The fix is independent of hot-swap and worth landing on its own merit — even built-in policies could in principle raise (OOM during dict resize) and silently leak the block pool today.

### 14.4 Hang protection (out of scope, documented)

A policy that loops infinitely inside `evict()` blocks the entire `EngineCore` busy loop: no progress on requests, no shutdown. In-process Python cannot interrupt arbitrary user code without sending the signal to the main thread, and the busy loop runs on the main thread of the engine process. Options surveyed:
- `signal.SIGALRM` watchdog: works only on the main thread of the main process; brittle on macOS; will misfire if user code disables/reenables signals.
- Subinterpreter / subprocess isolation: out of scope per §2.
- Cooperative deadline: each policy method checks a `time.monotonic()` budget. Requires evolving policies to opt in; CORAL agents will not.

Phase 1 documents the hang risk in the swap route docstring and relies on the operator-side timeout the grader already applies (e.g., `kill` after the candidate's wall-clock budget). The supervisor protects against fast crashes, not slow hangs.

### 14.5 Stats — exposing protection signal

Extend `OffloadPolicyStats` (§8.3):

```python
class OffloadPolicyStats(msgspec.Struct):
    # ... existing fields ...
    policy_errors: int = 0          # policy exceptions recorded since last reset
    policy_rolled_back: bool = False  # True if any rollback occurred during the current window
    rollback_generation: int = 0      # generation that was rolled back from (0 if none)
```

The stats snapshot combines manager counters, active-supervisor errors, and registry-sticky recovery state. Concretely, `policy_errors = entry.cumulative_policy_errors + (entry.supervisor._errors if entry.supervisor else 0)`, while `policy_rolled_back` and `rollback_generation` are read directly from `entry`. **Reset semantics:** `policy_rolled_back`, `rollback_generation`, and `cumulative_policy_errors` are sticky-within-window: set on rollback, cleared only by `take_policy_stats(reset=True)`. They are not cleared by the rollback itself or by the next forward swap, because the grader may call `get_offload_policy_stats(reset=False)` after the candidate's run completes and must still see the trip. `take_policy_stats(reset=True)` additionally zeroes the *currently-active* supervisor's `_errors` (if any), so a per-candidate reset at run-start gives the next candidate a clean window even when the candidate is hot-swapped on top of a previously-active evolved policy that already accumulated some suppressed errors. The CORAL grader reads `policy_errors > 0` to penalize unstable candidates, and `policy_rolled_back` to know its score came from the recovered baseline rather than from the candidate. Without these signals, a candidate that crashed silently and was auto-recovered would receive a corrupted fitness signal.

### 14.6 Threat-model note

`SupervisedCachePolicy` is a robustness layer, not a security boundary. It does not protect against:
- A policy that calls `os._exit()`, `sys.exit()`, or `signal.kill(os.getpid(), ...)` from inside any method.
- A policy that mutates `manager.__dict__` directly.
- A policy that imports and invokes a Bash subprocess.

These remain governed by the §5 threat model (default-off, localhost-only, operator opts in to running untrusted code).

## 15. Phase 2 sketch — weight-prefetch evolution

Replace `PrefetchOffloader._start_prefetch` / `_wait_for_layer` with a pluggable `PrefetchSchedule` strategy. Hot-swap requires:
- Broadcasting new code to every worker via `collective_rpc("swap_prefetch_schedule", payload)`.
- Draining all in-flight torch streams before the swap (`sync_prev_onload` + `join_after_forward`).
- **Invalidating cudagraphs** if the schedule changes the per-layer call pattern — likely requires a one-time recapture, expensive.
- Restricting evolution to Python-level reordering of `start_prefetch`/`wait_prefetch` calls. New custom op definitions are forbidden (`prefetch_ops` is registered at import; evolved schedules must consume the existing op signatures).

Defer until Phase 1 is validated against a CORAL run. Phase 2 can plumb hint signal into prefetch decisions, but `PrefetchSchedule` operates per layer over a *batch* of requests, not per request, so the natural surface is an aggregated `BatchHintSummary` (set of priority classes seen, max deadline, etc.) — not a single `req_context`. Aggregation lives in the scheduler-side connector before the worker-side broadcast, so workers don't need to know about per-request hint plumbing. Design that aggregation in Phase 2.

## 16. Per-request policy hints

Hot-swappable policy code is only half of what an evolutionary loop wants to mutate. The other half is *what the policy gets to look at*. CORAL should be free to invent new request-level signal — priority class, session id, expected reuse, tenant id, deadline, "this is a one-shot toolcall" — and have the evolved policy read that signal at eviction time. This section adds an opaque pass-through channel that lets a candidate consist of two co-evolving artifacts: the policy code, plus a small client shim that decides what to put in `vllm_xargs["policy_hints"]` per request. vLLM's HTTP/Python surface stays unchanged across CORAL generations.

The plumbing is mostly free: every offloading-manager call site already receives a `ReqContext` ([vllm/v1/kv_offload/base.py:47](../vllm/v1/kv_offload/base.py#L47)) populated from the request at [scheduler.py:141](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L141). The chain that already carries `kv_transfer_params` ([completion/protocol.py:292-295](../vllm/entrypoints/openai/completion/protocol.py#L292-L295) → [v1/request.py:113-116](../vllm/v1/request.py#L113-L116) → `ReqContext`) carries `policy_hints` the same way. The only gap §16 closes is forwarding `req_context` from `CPUOffloadingManager` into `_policy.insert` / `touch` / `evict`.

### 16.1 Carrier — reuse `vllm_xargs["policy_hints"]`

Both `CompletionRequest` and `ChatCompletionRequest` already expose `vllm_xargs: dict[str, str|int|float] | None` as the documented user-extension field. The OpenAI-compat layer merges it into `SamplingParams.extra_args` at [vllm/entrypoints/openai/completion/protocol.py:292-295](../vllm/entrypoints/openai/completion/protocol.py#L292-L295). Offline `LLM` callers populate `SamplingParams.extra_args["policy_hints"]` directly. We pick the sub-key `policy_hints` and treat it as opaque `dict[str, Any]` end-to-end.

Mirror the existing `kv_transfer_params` extraction in `Request.__init__` at [vllm/v1/request.py:113-116](../vllm/v1/request.py#L113-L116). `__init__` is the right hook — `Request.from_engine_core_request` is a thin classmethod that forwards everything to `__init__`, and `self.request_id` is already set by the time we reach the extraction block. Two changes:

1. Initialize `self.policy_hints: dict[str, Any] | None = None` immediately after the existing `self.kv_transfer_params: dict[str, Any] | None = None` at [request.py:101](../vllm/v1/request.py#L101). This guarantees the attribute exists for *all* request kinds — pooling-only requests skip the sampling branch entirely, and a missing attribute would `AttributeError` later in the offloading scheduler.
2. Inside the existing `if sampling_params.extra_args is not None:` block (alongside the `self.kv_transfer_params = ...` assignment), add the hint extraction:

```python
# vllm/v1/request.py — append to the existing extra_args block at lines 113-116
raw_hints = sampling_params.extra_args.get("policy_hints")
if raw_hints is not None:
    # Validate at the carrier so malformed input fails the request, not the
    # policy. dict(raw_hints) raises TypeError on non-dict / non-mapping
    # input; the request entrypoint converts this into an HTTP 400 / Python
    # ValueError before the request reaches the engine. Shallow copy avoids
    # mutating caller-owned state; the reserved `_request_id` key is
    # injected for policies that need a stable per-request handle
    # (see "Reserved key" below).
    hints = dict(raw_hints)
    hints["_request_id"] = self.request_id
    self.policy_hints = hints
```

No HTTP schema change is required. Operators who don't enable hot-swap pay nothing — `policy_hints` is read but ignored when no evolved policy consumes it (built-ins discard `req_context`). Pooling-only requests cannot supply hints because `extra_args` is sampling-only today; this matches the existing `kv_transfer_params` restriction and is acceptable because offloading exists to extend KV-cache reuse for generative workloads.

**Reserved key.** Per the snippet above, `Request.__init__` shallow-copies the user-supplied `policy_hints` dict and injects `_request_id = self.request_id`. Policies that need a stable per-request handle (e.g. to record "request X inserted these blocks") read `hints.get("_request_id")` instead of inventing their own. The shallow copy avoids mutating user-supplied state. Names beginning with `_` are reserved for future framework use; evolved schemas must not collide.

### 16.2 `ReqContext` gains one field

```python
# vllm/v1/kv_offload/base.py
@dataclass
class ReqContext:
    kv_transfer_params: dict[str, Any] | None = None
    policy_hints: dict[str, Any] | None = None  # NEW
```

Populated alongside `kv_transfer_params` at [scheduler.py:141](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L141):

```python
self.req_context = ReqContext(
    kv_transfer_params=self.req.kv_transfer_params,
    policy_hints=self.req.policy_hints,
)
```

### 16.3 Widened `CachePolicy` signatures (additive, default-None)

`CachePolicy.insert` / `touch` / `evict` accept an optional `req_context`. `get` and `remove` are deliberately context-free: a `get` is a content-addressed lookup that may be triggered by any request (or none — e.g., supervisor cold-rollback validation), and `remove` is a manager-internal cleanup with no caller-meaningful request scope.

```python
class CachePolicy(ABC):
    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus,
               req_context: ReqContext | None = None) -> None: ...
    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey],
              req_context: ReqContext | None = None) -> None: ...
    @abstractmethod
    def evict(self, n: int, protected: set[OffloadKey],
              req_context: ReqContext | None = None
              ) -> list[tuple[OffloadKey, BlockStatus]] | None: ...
```

Defaulting to `None` means:
- Built-in `LRUCachePolicy` and `ARCCachePolicy` accept and ignore the kwarg — one-line widening, no behavior change.
- Manager paths that legitimately have no request scope (e.g. supervisor cold rollback in §14.2 calls `recovered.import_state(residents)` which loops `insert(key, block)`) can still call without supplying context.
- Evolved policies opt in by reading `req_context.policy_hints` only when they need it.

### 16.4 Manager forwards `req_context` to `_policy`

`CPUOffloadingManager` already receives `req_context` on `lookup` / `prepare_load` / `prepare_store`. The change is to forward it at three policy call sites:

- [manager.py:137](../vllm/v1/kv_offload/cpu/manager.py#L137) — `_policy.evict(num_blocks_to_evict, protected, req_context)` inside `prepare_store`.
- [manager.py:159](../vllm/v1/kv_offload/cpu/manager.py#L159) — `_policy.insert(key, block, req_context)` inside the `prepare_store` insert loop. The §14.3 hardened-loop variant must thread `req_context` through too.
- [manager.py:106](../vllm/v1/kv_offload/cpu/manager.py#L106) — `touch()`. The current `OffloadingManager.touch(keys)` ABC at [base.py:150](../vllm/v1/kv_offload/base.py#L150) takes no context; we widen it to `touch(self, keys, req_context: ReqContext)` and update the two `manager.touch(...)` call sites inside `OffloadingConnectorScheduler._touch` (the full-attention call at [scheduler.py:294](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L294) and the sliding-window call at [scheduler.py:303](../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L303)), both of which already have `req_status.req_context` in scope. `touch` is the only `OffloadingManager` method on the *forwarded-to-policy* path that lacks `req_context` today (`complete_load` / `complete_store` / `take_events` / `shutdown` also lack it but neither needs nor receives request scope). The wrapper [`FilterReusedOffloadingManager`](../vllm/v1/kv_offload/reuse_manager.py) is updated at the same time to delegate the new `touch(keys, req_context)` signature.

Other manager call sites of `_policy.get` / `_policy.remove` (e.g., `lookup`, `prepare_load`, `complete_load`, `complete_store`) keep their existing signatures.

### 16.5 Cross-request merge semantics

`OffloadKey` is content-addressed (block-hash + group-idx, [base.py:27](../vllm/v1/kv_offload/base.py#L27)). The same key can be `insert`-ed by request A and later `touch`-ed or seen during `evict` triggered by request B. The framework hands every call the *calling request's* `req_context`; the policy decides the merge convention (first-writer-wins, last-toucher-wins, weighted aggregate, anything else). Evolved policies that need to remember per-block hints across calls keep their own `dict[OffloadKey, ...]` table; the framework does not provide one.

Phase 1 deliberately does *not* tag `OffloadKey` with request-scoped data, because that would forfeit cross-request prefix-cache hits — the whole reason offloading helps. Tagging is a possible Phase 2 extension behind an explicit policy opt-in.

### 16.6 Telemetry — `hint_keys`

`OffloadPolicyStats` (§8.3) gains:

```python
hint_keys: tuple[str, ...] = ()  # union of top-level keys observed in
                                  # req_context.policy_hints since last reset
```

The manager samples `policy_hints.keys()` at every site where `req_context` is in scope at the manager level — `lookup`, `prepare_load`, `prepare_store`, and the widened `touch` (§16.4) — and unions them into a per-window set. `take_policy_stats(reset=True)` clears the set. Cost: one `set.update` per call where hints are non-empty; the union saturates quickly within a CORAL run, so steady-state cost is one membership check + a small set comparison per call. The grader uses this as a smoke signal that its client shim actually populated hints; values are not exposed because schema names tend to leak less than values (e.g., `"tenant_id"` is a key, the actual id is the value). Key names themselves can still be sensitive if a malicious client invents them, so operators should treat `hint_keys` as untrusted in any downstream logging.

### 16.7 Canary update (§6 step 5)

The §6 step 5 canary is extended with a final substep that exercises the with-context path. With `ctx = ReqContext(policy_hints={"_canary": True})`, the canary inserts a fresh sentinel pair via `canary_policy.insert(..., req_context=ctx)`, calls `touch(..., req_context=ctx)`, and `evict(1, set(), req_context=ctx)`. This catches policies that crash specifically when `req_context` is non-None — a class of bug a context-free canary would miss. Earlier substeps (1-7) keep using the default-None path so we cover both.

### 16.8 CORAL grader sketch — co-evolving artifact

The §9 grader sketch is unchanged on the vLLM side. The CORAL candidate's worktree now contains *two* files: `policy.py` (as before) and an optional `client_hints.py` exposing `make_hints(prompt: str, request_meta: dict) -> dict`. The grader, before each `llm.generate(...)` call, computes hints per prompt and threads them through:

```python
from coral_task.candidate import client_hints  # candidate-supplied module

hints_per_prompt = [client_hints.make_hints(p, meta) for p in prompts]
expected_keys = {k for h in hints_per_prompt for k in h}
outs = llm.generate(
    prompts,
    [SamplingParams(max_tokens=256, extra_args={"policy_hints": h})
     for h in hints_per_prompt],
)
stats = llm.get_offload_policy_stats()
# The canary's "_canary" hint runs on a separate canary_policy instance
# (§6 step 5 / §16.7) and never reaches manager-level sampling, so its
# absence here is incidental. The real check is that the candidate's
# keys made it through the carrier:
assert expected_keys & set(stats.hint_keys), (
    "client_hints output did not propagate to the policy"
)
```

CORAL's mutation operators are free to evolve both `policy.py` and `client_hints.py` together; vLLM never sees the hint schema. If `client_hints` is absent the grader passes no hints and the policy must still produce a usable score on raw content alone.

### 16.9 Threat model addendum

`policy_hints` is user-supplied untrusted data. The same gating from §5 (feature opt-in, localhost-only-by-default, source hash logging) applies — hot-swap remains the gate; hints are just one more thing that can be tampered with once an attacker is past the gate. Evolved policies must treat hints as untrusted inside their own logic: `dict.get` with type checks, bounded sizes, no `eval` / `exec` on hint values, no path traversal, no PII echoing into logs. Violations are policy bugs surfaced by the §14 supervisor's error budget — the framework does not validate hint contents.

What the framework guarantees:
- `req_context.policy_hints` is either `None` or a `dict[str, Any]`. Nested values are whatever the request body deserialized to.
- The framework does not mutate the hint dict; policies must not either (treat as read-only).
- The hint dict carries the request's `request_id` (or equivalent stable id) under a reserved `_request_id` key the carrier injects, so policies that need a stable request handle have one without inventing their own. (Policies must still tolerate its absence — e.g., during canary calls.)

What the framework does *not* guarantee:
- Hint key names, types, or sizes. Two CORAL generations may use entirely different schemas.
- Stability of values across calls. A misbehaving client could send different hints for the same `OffloadKey` on every call.
- Object identity of the hint dict across calls. The connector → manager hop is in-process, but stats / IPC paths may reconstruct dicts; policies must not rely on `id(hints)` being stable.
- Sanitization. Hints can contain arbitrary strings, including ones that look like log injection or path traversal — the policy is responsible for not interpreting them as control flow.
