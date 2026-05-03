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
    def insert(self, key: OffloadKey, block: BlockStatus) -> None: ...
    @abstractmethod
    def remove(self, key: OffloadKey) -> None: ...
    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey]) -> None: ...
    @abstractmethod
    def evict(self, n: int, protected: set[OffloadKey]
              ) -> list[tuple[OffloadKey, BlockStatus]] | None: ...
```

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

`attach` is called from [`CPUOffloadingSpec.get_manager()`](../vllm/v1/kv_offload/cpu/spec.py), the lazy single-shot construction site. Keep the bare `CPUOffloadingManager` in a local variable, attach that inner manager with `self.vllm_config.instance_id`, then optionally wrap it in `FilterReusedOffloadingManager`. The registry stores a weak ref to the inner `CPUOffloadingManager` (whose `_policy` is the swap target), not to the wrapper; the wrapper delegates `lookup`/`prepare_*` and has no policy state of its own. Hooking the spec rather than the manager's `__init__` means unit tests that construct `CPUOffloadingManager` directly do not pollute the singleton, and the registry sees exactly one attach per configured engine.

The idle-rollback callback is attached separately from `EngineCore.__init__`, after `self._idle_state_callbacks` has been initialized:

```python
PolicySwapRegistry.singleton().attach_idle_callback(
    self.vllm_config.instance_id,
    lambda cb: self._idle_state_callbacks.append(lambda _engine: cb()),
)
```

This avoids threading an `EngineCore` reference through `KVConnectorFactory`, `OffloadingConnector`, `OffloadingConnectorScheduler`, and `OffloadingSpec`. The connector spec may be constructed *after* `EngineCore.__init__` finishes setting up `_idle_state_callbacks`, so `attach_idle_callback` is order-tolerant: it stores the enqueue function on the entry (creating a placeholder entry if `attach()` has not yet run), and the later `attach()` call preserves any enqueue function already present. Symmetrically, `attach()` arriving first leaves `enqueue_idle_callback=None` until `attach_idle_callback` fills it in. Either ordering yields a fully wired entry; only the absence of *both* calls leaves rollback to the next-swap path (§14.2 point 1).

Different `engine_id` values coexist in the registry. If `attach` is called twice for the same `engine_id`, the registry logs a warning, replaces the weak ref, and drops any active synthetic module owned by the old entry.

`attach` initializes `ActivePolicy` at generation 0 from the built-in policy metadata (`source_origin="builtin"`, `source_hash="builtin"`) so `current(engine_id)` returns useful data before the first evolved policy is loaded.

**`swap()` algorithm** (under `self._lock`):

0. Resolve: find the registry entry for `engine_id` and dereference the manager. Missing entry → remove the candidate module from `sys.modules` and return `SwapResult(ok=False, generation=0, previous_generation=0, error="CPU offloading manager is not attached")`; no module becomes active. Entry present but weak ref dead (the manager has been GC'd — should not happen during normal operation, but defensively handled) → return the same shape but with `generation=previous_generation=entry.generation` so observers do not see an apparent reset.
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
   6. Construct a second sentinel pair (`evict_key` from a different random byte string distinct from `canary_key`, `evict_block = BlockStatus(block_id=-2)` with `ref_cnt = 0`); `canary_policy.insert(evict_key, evict_block)`. Then assert `canary_policy.evict(1, set())` returns either `[(canary_key, fake_block)]` or `[(evict_key, evict_block)]` (length must be exactly 1; the choice depends on policy order), and assert the evicted key is no longer present via `canary_policy.get(...)`.
   7. Remove whichever sentinel key remains; assert both sentinel keys are absent.

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

Phase 1 should fail fast when `data_parallel_size > 1`. The existing DP utility path is split: `DPLBAsyncMPClient.call_utility_async` ([core_client.py:1380](../vllm/v1/engine/core_client.py#L1380)) fans out to every `EngineCore` and returns only the first result, while the external-LB `DPAsyncMPClient` inherits `AsyncMPClient.call_utility_async` and targets a single engine via `self.core_engine` ([core_client.py:597](../vllm/v1/engine/core_client.py#L597)) — so a swap would only land on rank 0 and the other ranks would silently keep the prior policy. Either failure mode is unacceptable: hot-swap needs an aggregate result and all-or-nothing semantics across scheduler processes before it should support DP. The check fires at engine init (when policy hotswap is enabled and `data_parallel_size > 1`, raise from `EngineArgs.create_engine_config` so misconfiguration surfaces before any benchmark starts), not at swap call time — this prevents a CORAL run from silently scoring against only one of N engines.

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

**Modified files**:

| Path | Change |
|---|---|
| [vllm/v1/kv_offload/cpu/policies/base.py](../vllm/v1/kv_offload/cpu/policies/base.py) | Widen the ABC `__init__` to `(cache_capacity: int, **kwargs: Any)` per §4; add an abstract `export_state` hook and a concrete `import_state` hook (default `import_state` loops through `insert`); add a no-op `record_write_error()` hook on `CachePolicy` that `SupervisedCachePolicy` overrides (§14.1) so the manager can notify the supervisor of write-side failures without isinstance checks. |
| [vllm/v1/kv_offload/cpu/policies/lru.py](../vllm/v1/kv_offload/cpu/policies/lru.py) | Implement `export_state()`, built-in policy metadata, and widen `__init__` to accept and ignore `**kwargs` (per §4). |
| [vllm/v1/kv_offload/cpu/policies/arc.py](../vllm/v1/kv_offload/cpu/policies/arc.py) | Implement `export_state()` for resident T1/T2 blocks only, built-in policy metadata, and widen `__init__` to accept and ignore `**kwargs` (per §4). |
| [vllm/v1/kv_offload/cpu/manager.py](../vllm/v1/kv_offload/cpu/manager.py) | Maintain `OffloadPolicyStats`; expose `take_policy_stats(reset=False, active_generation=...)`. Wrap the `prepare_store` insert loop to free pre-allocated blocks if `_policy.insert` raises (§14.3). |
| [vllm/v1/kv_offload/cpu/spec.py](../vllm/v1/kv_offload/cpu/spec.py) | In `get_manager()`, call `PolicySwapRegistry.singleton().attach(self.vllm_config.instance_id, inner)` on the bare `CPUOffloadingManager` *before* it is optionally wrapped by `FilterReusedOffloadingManager`. |
| [vllm/envs.py](../vllm/envs.py) | Add `VLLM_ENABLE_POLICY_HOTSWAP`, `VLLM_POLICY_HOTSWAP_ALLOW_REMOTE`. |
| [vllm/engine/arg_utils.py](../vllm/engine/arg_utils.py) | Add `enable_policy_hotswap: bool = False` field on `EngineArgs` and the matching `--enable-policy-hotswap` CLI flag; in `create_engine_config()` resolve the effective value (CLI flag OR `VLLM_ENABLE_POLICY_HOTSWAP`) and store it under `additional_config["enable_policy_hotswap"]` so the engine process sees the same gate. Raise from `create_engine_config()` if the flag is set together with `data_parallel_size > 1` (per §7). |
| [vllm/engine/protocol.py](../vllm/engine/protocol.py) | Add async `swap_offload_policy`, current-policy, and stats methods to the `EngineClient` protocol for serve routes. |
| [vllm/v1/engine/core.py](../vllm/v1/engine/core.py) | Add `swap_offload_policy(payload)`, `get_offload_policy()`, and `get_offload_policy_stats(reset=False)` utility methods on `EngineCore`; after `_idle_state_callbacks` is initialized, register the policy registry idle-callback enqueue hook. |
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
- `test_policy_stats.py`: manager lookup/load/store counters, hit-rate calculation, reset behavior, and wrapper composition. Reset semantics for `policy_rolled_back` / `rollback_generation`: assert sticky-true within a window (visible to `get_offload_policy_stats(reset=False)` after a trip) and cleared only by `take_policy_stats(reset=True)`.

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
    def get(self, key):
        try: return self._inner.get(key)
        except Exception: self._record(); return None
    def evict(self, n, protected):
        try: return self._inner.evict(n, protected)
        except Exception: self._record(); return None
    def touch(self, keys):
        try: self._inner.touch(keys)
        except Exception: self._record()

    # State-mutating ops: propagate exceptions to manager hardening
    # (see §14.3), which rolls back partial writes and reports a
    # recoverable store failure.
    def insert(self, key, block): self._inner.insert(key, block)
    def remove(self, key):        self._inner.remove(key)

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
5. Reconcile the block pool: `manager._free_list = sorted(set(range(manager._num_allocated_blocks)) - recovered_block_ids)`.
6. Install `manager._policy = recovered`. Drop the rolled-back candidate's `sys.modules` entry using the `rolled_back_module` captured in step 1, then clear `entry.supervisor` and `entry.active_module_name` (set to `None`). Increment `entry.generation` and set `entry.active` to built-in LRU metadata with `source_origin=f"rollback:{rollback_generation}"` and `source_hash="builtin"`. The capture-then-clear order matters: the next forward swap's "remove the previous active synthetic module" step (§6 step 8) keys on `entry.active_module_name`, and a stale value would target a module already removed during rollback (`del sys.modules[name]` raises `KeyError` if absent).
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
        self._policy.insert(key, block)
        inserted += 1
    except Exception:
        # Undo successful inserts and free their blocks.
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

The stats snapshot combines manager counters, active-supervisor errors, and registry-sticky recovery state. Concretely, `policy_errors = entry.cumulative_policy_errors + (entry.supervisor._errors if entry.supervisor else 0)`, while `policy_rolled_back` and `rollback_generation` are read directly from `entry`. **Reset semantics:** `policy_rolled_back`, `rollback_generation`, and `cumulative_policy_errors` are sticky-within-window: set on rollback, cleared only by `take_policy_stats(reset=True)`. They are not cleared by the rollback itself or by the next forward swap, because the grader may call `get_offload_policy_stats(reset=False)` after the candidate's run completes and must still see the trip. The CORAL grader reads `policy_errors > 0` to penalize unstable candidates, and `policy_rolled_back` to know its score came from the recovered baseline rather than from the candidate. Without these signals, a candidate that crashed silently and was auto-recovered would receive a corrupted fitness signal.

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

Defer until Phase 1 is validated against a CORAL run.
