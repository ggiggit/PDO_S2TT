"""Direct guards for owner-local CUDA graph signatures.

The implementation keeps every dynamic guard while returning the immutable
signature captured at plan construction on the unchanged hot path.
Any structural, tensor, route, or training-state change falls back to the
original signature function so the graph bank makes the same replay decision.
"""
from __future__ import annotations

from collections.abc import Callable
import threading
from types import MappingProxyType
from torch import nn

_INSPECTION = ("named_parameters", "named_buffers", "named_modules")
_STANDARD = MappingProxyType({name: getattr(nn.Module, name) for name in _INSPECTION})
_STANDARD_MEMBERS = nn.Module._named_members


def _standard_root(module: nn.Module) -> bool:
    return all(
        name not in module.__dict__ and getattr(type(module), name, None) is _STANDARD[name]
        for name in _INSPECTION
    )


class SignaturePlan:
    """One graph root with direct guards and exact original fallback."""

    def __init__(self, module: nn.Module, original: Callable[[nn.Module], tuple]):
        self.module = module
        self._original = original
        self.calls = self.fallbacks = 0
        self.closed = False
        self.generated_source = ""
        self.expected = None
        self.original_only = not _standard_root(module)

        nodes = []
        seen = set()
        todo = [module]
        while todo:
            child = todo.pop()
            if id(child) in seen:
                continue
            seen.add(id(child))
            nodes.append(child)
            todo.extend(value for value in child._modules.values() if value is not None)
        if any(
            "named_modules" in child.__dict__
            or type(child).named_modules is not _STANDARD["named_modules"]
            for child in nodes
        ):
            self.original_only = True
        if (
            "_named_members" in module.__dict__
            or getattr(type(module), "_named_members", None) is not _STANDARD_MEMBERS
        ):
            self.original_only = True
        if self.original_only:
            self.parameters = self.buffers = self.modules = self.layout = ()
            self._collect = self._fallback
            return

        self.parameters = tuple(module.named_parameters())
        self.buffers = tuple(module.named_buffers())
        self.modules = tuple(module.named_modules())
        self.layout = tuple(
            (
                child,
                tuple((key, id(value)) for key, value in child._modules.items()),
                tuple((key, id(value)) for key, value in child._parameters.items()),
                tuple((key, id(value)) for key, value in child._buffers.items()),
            )
            for _, child in self.modules
        )
        self.expected = original(module)

        namespace = {
            "_root": module,
            "_standards": _STANDARD,
            "_standard_named_members": _STANDARD_MEMBERS,
            "_expected": self.expected,
            "_fallback": self._fallback,
        }
        guards = []
        for name in _INSPECTION:
            guards.extend(
                [
                    f"{name!r} not in _root.__dict__",
                    f"getattr(type(_root), {name!r}, None) is _standards[{name!r}]",
                ]
            )
        guards.extend(
            [
                "'_named_members' not in _root.__dict__",
                "getattr(type(_root), '_named_members', None) is _standard_named_members",
            ]
        )

        for index, (child, _, _, _) in enumerate(self.layout):
            reference = f"_c{index}"
            namespace[reference] = child
            guards.extend(
                [
                    f"'named_modules' not in {reference}.__dict__",
                    f"type({reference}).named_modules is _standards['named_modules']",
                ]
            )
            for field in ("_modules", "_parameters", "_buffers"):
                values = getattr(child, field)
                access = f"{reference}.{field}"
                if not values:
                    guards.append(f"not {access}")
                else:
                    # Key order is part of named_* enumeration and therefore
                    # remains an explicit guard.  No tensor/route tuple is built.
                    guards.append(f"tuple({access}) == {tuple(values)!r}")
                    for item_index, (name, value) in enumerate(values.items()):
                        value_reference = f"{reference}_{field}_{item_index}"
                        namespace[value_reference] = value
                        guards.append(f"{access}[{name!r}] is {value_reference}")

        expected_parameters, expected_buffers, expected_route = self.expected
        for prefix, rows, expected_rows in (
            ("p", self.parameters, expected_parameters),
            ("b", self.buffers, expected_buffers),
        ):
            if len(rows) != len(expected_rows):
                raise RuntimeError("original signature tensor inventory mismatch")
            for index, ((name, value), expected_row) in enumerate(zip(rows, expected_rows)):
                if expected_row[0] != name:
                    raise RuntimeError("original signature tensor name mismatch")
                reference = f"_{prefix}{index}"
                namespace[reference] = value
                namespace[f"_{prefix}version{index}"] = expected_row[2]
                namespace[f"_{prefix}ptr{index}"] = expected_row[3]
                namespace[f"_{prefix}dtype{index}"] = expected_row[4]
                namespace[f"_{prefix}device{index}"] = expected_row[5]
                guards.extend(
                    [
                        f"{reference}._version == _{prefix}version{index}",
                        f"{reference}.data_ptr() == _{prefix}ptr{index}",
                        f"{reference}.dtype == _{prefix}dtype{index}",
                        f"{reference}.device == _{prefix}device{index}",
                    ]
                )

        if len(self.modules) != len(expected_route):
            raise RuntimeError("original signature route inventory mismatch")
        def sequence_guard(reference: str, attribute: str, expected: tuple, slot: str) -> str:
            if not hasattr(namespace[reference], attribute):
                # Adding even an equivalent empty attribute takes the exact
                # original fallback.  This is conservative but remains safe.
                return f"not hasattr({reference}, {attribute!r})"
            raw = getattr(namespace[reference], attribute)
            if isinstance(raw, list):
                namespace[slot] = list(expected)
                return f"getattr({reference}, {attribute!r}) == {slot}"
            if isinstance(raw, tuple):
                namespace[slot] = expected
                return f"getattr({reference}, {attribute!r}) == {slot}"
            namespace[slot] = expected
            return f"tuple(getattr({reference}, {attribute!r})) == {slot}"

        for index, ((name, child), expected_row) in enumerate(zip(self.modules, expected_route)):
            if expected_row[0] != name:
                raise RuntimeError("original signature route name mismatch")
            reference = f"_m{index}"
            namespace[reference] = child
            namespace[f"_md{index}"] = expected_row[2]
            namespace[f"_mt{index}"] = expected_row[5]
            active_guard = sequence_guard(reference, "active_adapters", expected_row[1], f"_ma{index}")
            merged_guard = sequence_guard(reference, "merged_adapters", expected_row[3], f"_mm{index}")
            if not hasattr(child, "scaling"):
                scaling_guard = f"not hasattr({reference}, 'scaling')"
            else:
                raw_scaling = getattr(child, "scaling")
                if isinstance(raw_scaling, dict):
                    namespace[f"_ms{index}"] = dict(expected_row[4])
                    scaling_guard = f"getattr({reference}, 'scaling') == _ms{index}"
                elif hasattr(raw_scaling, "items"):
                    namespace[f"_ms{index}"] = expected_row[4]
                    scaling_guard = (
                        f"tuple(sorted(getattr({reference}, 'scaling').items())) == _ms{index}"
                    )
                else:
                    namespace[f"_ms{index}"] = expected_row[4]
                    scaling_guard = f"getattr({reference}, 'scaling') == _ms{index}"
            guards.extend(
                [
                    active_guard,
                    f"bool(getattr({reference}, 'disable_adapters', False)) == _md{index}",
                    merged_guard,
                    scaling_guard,
                    f"bool({reference}.training) == _mt{index}",
                ]
            )

        source = (
            "def collect():\n"
            "    if "
            + " and ".join(guards)
            + ":\n"
            "        return _expected\n"
            "    return _fallback()\n"
        )
        if len(source) >= 200000:
            raise ValueError("metadata plan exceeds its configured source bound")
        exec(compile(source, "<pdo-owner-local-direct-signature-guards>", "exec"), namespace)
        self._collect = namespace["collect"]
        self.generated_source = source

    def _fallback(self):
        self.fallbacks += 1
        return self._original(self.module)

    def __call__(self):
        if self.closed:
            raise RuntimeError("metadata plan is closed")
        self.calls += 1
        return self._collect()

    def close(self):
        if self.closed:
            return
        if self.generated_source:
            self._collect.__globals__.clear()
        self._collect = None
        self.module = self._original = self.expected = None
        self.parameters = self.buffers = self.modules = self.layout = ()
        self.generated_source = ""
        self.closed = True


class SignatureRegistry:
    """At most 256 root plans owned and closed by one graph-resource owner."""

    def __init__(self, original: Callable[[nn.Module], tuple], max_plans: int = 256):
        if not 1 <= max_plans <= 256:
            raise ValueError("metadata plan bound must be in 1..256")
        self._original = original
        self.max_plans = max_plans
        self._plans = {}
        self._owner_thread = threading.get_ident()
        self.closed = False
        self.calls = self.cap_fallbacks = 0

    def __call__(self, module: nn.Module):
        if self.closed:
            raise RuntimeError("signature registry is closed")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("signature registry requires its owner thread")
        self.calls += 1
        key = id(module)
        plan = self._plans.get(key)
        if plan is None:
            if len(self._plans) >= self.max_plans:
                self.cap_fallbacks += 1
                return self._original(module)
            plan = SignaturePlan(module, self._original)
            if not plan.original_only and plan() != self._original(module):
                plan.close()
                raise RuntimeError("metadata plan differs from original signature")
            self._plans[key] = plan
        # Registry ownership guarantees plans cannot be individually closed
        # while present.  Avoid a second Python method dispatch on every graph
        # replay while retaining direct SignaturePlan calls for qualification.
        plan.calls += 1
        return plan._collect()

    def stats(self):
        return dict(
            plans=len(self._plans),
            max_plans=self.max_plans,
            calls=self.calls,
            structural_fallbacks=sum(plan.fallbacks for plan in self._plans.values()),
            cap_fallbacks=self.cap_fallbacks,
            closed=self.closed,
            generated_source_bytes=sum(
                len(plan.generated_source.encode()) for plan in self._plans.values()
            ),
        )

    def close(self):
        if self.closed:
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("signature registry close requires its owner thread")
        for plan in self._plans.values():
            plan.close()
        self._plans.clear()
        self.closed = True
