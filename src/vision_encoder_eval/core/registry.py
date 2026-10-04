from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping


MethodRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner: MethodRunner
    description: str
    requires: tuple[str, ...] = ()
    execution_mode: str = 'in_process'


class MethodRegistry:
    def __init__(self) -> None:
        self._items: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec) -> None:
        name = spec.name.strip().lower()
        if not name or name != spec.name:
            raise ValueError("method names must be non-empty lowercase identifiers")
        if name in self._items:
            raise ValueError(f"method already registered: {name}")
        self._items[name] = spec

    def get(self, name: str) -> MethodSpec:
        key = name.strip().lower()
        try:
            return self._items[key]
        except KeyError as exc:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown method {name!r}; registered: {known}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __iter__(self) -> Iterator[MethodSpec]:
        for name in self.names():
            yield self._items[name]


METHODS = MethodRegistry()


def register_method(
    name: str,
    *,
    description: str,
    requires: tuple[str, ...] = (),
    execution_mode: str = 'in_process',
) -> Callable[[MethodRunner], MethodRunner]:
    def decorate(runner: MethodRunner) -> MethodRunner:
        METHODS.register(
            MethodSpec(
                name=name,
                runner=runner,
                description=description,
                requires=requires,
                execution_mode=execution_mode,
            )
        )
        return runner

    return decorate
