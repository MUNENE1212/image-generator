"""Backend registry. Picks the right `ComputeBackend` for a request."""

from __future__ import annotations

from image_generator.backends.base import BackendError, ComputeBackend
from image_generator.backends.fal import FalBackend
from image_generator.backends.replicate import ReplicateBackend
from image_generator.config import settings
from image_generator.models.enums import Backbone, BackendName, Strategy


class BackendRegistry:
    """Holds instantiated backends. Resolves a backend for a (strategy, backbone, preference)."""

    def __init__(self, backends: dict[BackendName, ComputeBackend]) -> None:
        self._backends = backends

    @property
    def available(self) -> list[BackendName]:
        return list(self._backends.keys())

    def get(self, name: BackendName) -> ComputeBackend:
        if name not in self._backends:
            raise BackendError(f"Backend {name} not configured (missing credentials?)")
        return self._backends[name]

    def resolve(
        self,
        strategy: Strategy,
        backbone: Backbone,
        preferred: BackendName | None = None,
    ) -> ComputeBackend:
        """Pick a backend that supports (strategy, backbone).

        Preference order: explicit `preferred` > configured default > first match.
        """
        order: list[BackendName] = []
        if preferred is not None:
            order.append(preferred)
        default = BackendName(settings.imggen_default_backend)
        if default not in order:
            order.append(default)
        for name in self._backends:
            if name not in order:
                order.append(name)

        for name in order:
            backend = self._backends.get(name)
            if backend is not None and backend.supports(strategy, backbone):
                return backend

        raise BackendError(
            f"No configured backend supports strategy={strategy.value} backbone={backbone.value}"
        )


_registry_singleton: BackendRegistry | None = None


def get_registry() -> BackendRegistry:
    """Build the process-global registry from `settings`.

    Backends missing credentials are silently omitted; UI pages check
    `registry.available` to decide what to show.
    """
    global _registry_singleton
    if _registry_singleton is not None:
        return _registry_singleton

    backends: dict[BackendName, ComputeBackend] = {}
    if settings.replicate_api_token is not None:
        backends[BackendName.REPLICATE] = ReplicateBackend(
            settings.replicate_api_token.get_secret_value()
        )
    if settings.fal_key is not None:
        backends[BackendName.FAL] = FalBackend(settings.fal_key.get_secret_value())

    _registry_singleton = BackendRegistry(backends)
    return _registry_singleton
