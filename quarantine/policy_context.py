"""Immutable bytes for exactly one helper scan's administrative policy inputs."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from .registry_protocol import POLICY_COMPONENTS, ProtocolError, policy_bundle

MAX_COMPONENT_BYTES = 1024 * 1024
MAX_POLICY_BYTES = 3 * MAX_COMPONENT_BYTES


@dataclass(frozen=True, slots=True)
class PolicyContext:
    components: tuple[tuple[str, bytes], ...]

    def __post_init__(self):
        if type(self.components) is not tuple or any(type(pair) is not tuple or len(pair) != 2 for pair in self.components):
            raise ProtocolError("policy container must be immutable")
        if tuple(name for name, _raw in self.components) != POLICY_COMPONENTS:
            raise ProtocolError("policy component names/order invalid")
        if any(type(raw) is not bytes or not 0 < len(raw) <= MAX_COMPONENT_BYTES for _name, raw in self.components):
            raise ProtocolError("policy component byte bound invalid")
        if sum(len(raw) for _name, raw in self.components) > MAX_POLICY_BYTES:
            raise ProtocolError("cumulative policy byte bound exceeded")
        for name in POLICY_COMPONENTS:
            self.document(name)

    @classmethod
    def from_components(cls, values: dict[str, bytes]) -> PolicyContext:
        if set(values) != set(POLICY_COMPONENTS):
            raise ProtocolError("policy component inventory invalid")
        return cls(tuple((name, values[name]) for name in POLICY_COMPONENTS))

    def document(self, name: str) -> dict:
        for component, raw in self.components:
            if component == name:
                try:
                    value = yaml.safe_load(raw.decode("utf-8", errors="strict")) or {}
                except (UnicodeError, yaml.YAMLError, RecursionError) as error:
                    raise ProtocolError("policy YAML invalid") from error
                if not isinstance(value, dict):
                    raise ProtocolError("policy root must be a mapping")
                # Parse afresh from immutable bytes: caller changes cannot alter
                # either later readers or the bound component digests.
                return value
        raise ProtocolError("policy component is not part of this scan")

    def evidence(self) -> dict:
        import hashlib

        return policy_bundle({name: hashlib.sha256(raw).hexdigest() for name, raw in self.components})
