# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Policy code loader for the CPU-offloading hot-swap path.

THREAT MODEL — READ THIS FIRST
==============================
This module ``exec()``s untrusted Python in-process. It is gated behind the
``--enable-policy-hotswap`` flag (CLI / ``EngineArgs.enable_policy_hotswap``
/ ``VLLM_ENABLE_POLICY_HOTSWAP=1``). The HTTP route additionally requires a
localhost client unless ``VLLM_POLICY_HOTSWAP_ALLOW_REMOTE=1``.

This is robustness-friendly, not security-friendly. A loaded policy can:
  * `os._exit()` the engine process,
  * shell out via `subprocess`,
  * mutate `sys.modules` arbitrarily,
  * hold the GIL forever inside any `CachePolicy` method.

Operators MUST keep this disabled in production. See
`design/evolved_cpu_offloading.md` §5 for the full discussion.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy

logger = init_logger(__name__)


class PolicyLoadError(Exception):
    """Raised when policy code cannot be located, executed, or validated."""

    def __init__(self, origin: str, message: str, traceback_str: str = ""):
        super().__init__(f"{origin}: {message}")
        self.origin = origin
        self.message = message
        self.traceback_str = traceback_str


@dataclass
class LoadedPolicy:
    cls: type[CachePolicy]
    source: str
    source_hash: str  # SHA-256 of `source`, first 12 hex chars
    source_origin: str  # "path:/abs/x.py" / "module:pkg.evolved" / "source"
    module: ModuleType


def _hash_source(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def _pick_policy_class(mod: ModuleType) -> type[CachePolicy]:
    """Find the unique non-abstract `CachePolicy` subclass defined in `mod`.

    Subclasses imported from elsewhere (`__module__ != mod.__name__`) are
    skipped so that `from vllm.v1.kv_offload.cpu.policies.lru import
    LRUCachePolicy` in user code does not get picked up as the "evolved"
    class.
    """
    candidates: list[type[CachePolicy]] = []
    for value in vars(mod).values():
        if not inspect.isclass(value):
            continue
        if not issubclass(value, CachePolicy) or value is CachePolicy:
            continue
        if inspect.isabstract(value):
            continue
        if value.__module__ != mod.__name__:
            continue
        candidates.append(value)
    if not candidates:
        raise PolicyLoadError(
            mod.__name__,
            "no concrete CachePolicy subclass found in module",
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise PolicyLoadError(
            mod.__name__,
            f"expected exactly one CachePolicy subclass, found {len(candidates)}: "
            f"{names}",
        )
    return candidates[0]


def _exec_source(
    source: str,
    source_origin: str,
    module_name_hint: str,
    package: str | None = None,
    file_path: str | None = None,
) -> ModuleType:
    """Compile + exec `source` under a synthetic, content-addressed module name.

    Synthetic name shape:
        ``vllm._evolved.<source_hash>.<nonce>.<hint>``

    `nonce` is a monotonic timestamp so that two consecutive loads of the
    same source still produce distinct module objects (no shared globals,
    no `importlib.reload`).

    Failure semantics: any exception during `exec` is wrapped in
    `PolicyLoadError`; the partially-populated `sys.modules` entry is
    cleaned up before raising. `KeyboardInterrupt` and `SystemExit` are
    *not* caught.
    """
    source_hash = _hash_source(source)
    nonce = time.monotonic_ns()
    safe_hint = "".join(c if c.isalnum() else "_" for c in module_name_hint) or "policy"
    synthetic_name = f"vllm._evolved.{source_hash}.{nonce}.{safe_hint}"

    spec = importlib.util.spec_from_loader(synthetic_name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__name__ = synthetic_name
    if package is not None:
        module.__package__ = package
    if file_path is not None:
        module.__file__ = file_path

    sys.modules[synthetic_name] = module
    try:
        compiled = compile(source, file_path or f"<evolved:{source_hash}>", "exec")
        exec(compiled, module.__dict__)
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        sys.modules.pop(synthetic_name, None)
        raise
    except Exception as e:  # noqa: BLE001
        import traceback

        sys.modules.pop(synthetic_name, None)
        raise PolicyLoadError(
            source_origin, f"executing module: {e}", traceback.format_exc()
        ) from e
    return module


class PolicyLoader:
    """Resolve evolved policy source into a `LoadedPolicy`.

    The loader is intentionally stateless. The registry (§6) owns the
    lifecycle of synthetic `sys.modules` entries.
    """

    def load_from_path(self, path: str) -> LoadedPolicy:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise PolicyLoadError(f"path:{path}", f"file not found: {p}")
        try:
            source = p.read_text()
        except Exception as e:  # noqa: BLE001
            raise PolicyLoadError(f"path:{path}", f"read failed: {e}") from e
        origin = f"path:{p}"
        module = _exec_source(
            source,
            source_origin=origin,
            module_name_hint=p.stem,
            file_path=str(p),
        )
        try:
            cls = _pick_policy_class(module)
        except PolicyLoadError:
            sys.modules.pop(module.__name__, None)
            raise
        return LoadedPolicy(
            cls=cls,
            source=source,
            source_hash=_hash_source(source),
            source_origin=origin,
            module=module,
        )

    def load_from_module(self, dotted: str) -> LoadedPolicy:
        try:
            spec = importlib.util.find_spec(dotted)
        except (ImportError, ValueError) as e:
            raise PolicyLoadError(
                f"module:{dotted}", f"could not resolve spec: {e}"
            ) from e
        if spec is None or spec.origin is None or spec.origin == "built-in":
            raise PolicyLoadError(
                f"module:{dotted}",
                "module has no source-backed file (built-in or namespace?)",
            )
        path = spec.origin
        try:
            source = Path(path).read_text()
        except Exception as e:  # noqa: BLE001
            raise PolicyLoadError(f"module:{dotted}", f"read failed: {e}") from e
        # Preserve the original package so relative imports keep working.
        # CAVEAT: sibling modules reached via `from . import x` import under
        # their original dotted names and are therefore *shared* across swaps.
        package = spec.parent if spec.parent else None
        origin = f"module:{dotted}"
        module = _exec_source(
            source,
            source_origin=origin,
            module_name_hint=dotted.replace(".", "_"),
            package=package,
            file_path=path,
        )
        try:
            cls = _pick_policy_class(module)
        except PolicyLoadError:
            sys.modules.pop(module.__name__, None)
            raise
        return LoadedPolicy(
            cls=cls,
            source=source,
            source_hash=_hash_source(source),
            source_origin=origin,
            module=module,
        )

    def load_from_source(
        self, src: str, module_name_hint: str = "evolved_policy"
    ) -> LoadedPolicy:
        origin = "source"
        module = _exec_source(
            src, source_origin=origin, module_name_hint=module_name_hint
        )
        try:
            cls = _pick_policy_class(module)
        except PolicyLoadError:
            sys.modules.pop(module.__name__, None)
            raise
        return LoadedPolicy(
            cls=cls,
            source=src,
            source_hash=_hash_source(src),
            source_origin=origin,
            module=module,
        )

    def load(
        self,
        *,
        source_path: str | None = None,
        module: str | None = None,
        source: str | None = None,
    ) -> LoadedPolicy:
        """Single entry point used by the registry.

        Exactly one of the three keyword arguments must be set.
        """
        provided = [(k, v) for k, v in (
            ("source_path", source_path),
            ("module", module),
            ("source", source),
        ) if v is not None]
        if len(provided) != 1:
            raise PolicyLoadError(
                "request",
                "exactly one of source_path / module / source must be provided",
            )
        kind, value = provided[0]
        assert isinstance(value, str)
        if kind == "source_path":
            return self.load_from_path(value)
        if kind == "module":
            return self.load_from_module(value)
        return self.load_from_source(value)


def discard_module(name: str | None) -> None:
    """Best-effort removal of a synthetic module from `sys.modules`."""
    if name is None:
        return
    sys.modules.pop(name, None)


# Re-exported for type-checkers / callers that don't want to import `Any`.
__all__ = [
    "LoadedPolicy",
    "PolicyLoadError",
    "PolicyLoader",
    "discard_module",
]


def _annotate_loader_test_hooks() -> dict[str, Any]:
    """Hook bag exposed for unit tests; intentionally empty in production."""
    return {}
