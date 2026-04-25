"""Adapter tests with mocked vendor SDKs.

Live-API tests are marked @pytest.mark.integration and skipped without credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from image_generator.backends.base import BackendError
from image_generator.backends.fal import FalBackend, _build_arguments, _extract_image_url
from image_generator.backends.replicate import ReplicateBackend, _build_input
from image_generator.models.enums import Backbone, Strategy
from image_generator.models.requests import GenerationRequest

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 100  # plausible PNG-shaped bytes


class FakeStorage:
    """In-memory Storage stub that records writes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.writes: list[tuple[Path, bytes]] = []

    def put_selfie(self, data: bytes, sha256: str) -> Path:
        path = self.root / f"selfie_{sha256[:8]}.png"
        path.write_bytes(data)
        self.writes.append((path, data))
        return path

    def put_output(self, data: bytes, run_id: str, ext: str = "png") -> Path:
        path = self.root / f"{run_id}.{ext}"
        path.write_bytes(data)
        self.writes.append((path, data))
        return path

    def put_lora(self, data: bytes, name: str) -> Path:
        path = self.root / f"{name}.safetensors"
        path.write_bytes(data)
        self.writes.append((path, data))
        return path

    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def exists(self, path: Path) -> bool:
        return path.exists()


@pytest.fixture
def storage(tmp_path: Path) -> FakeStorage:
    return FakeStorage(tmp_path)


# ----------------------------- Replicate input shape -----------------------------


class TestReplicateBuildInput:
    def test_prompt_only_omits_image(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.PROMPT_ONLY, "selfie_sha256": None})
        payload = _build_input(req, selfie_bytes=None)
        assert "image" not in payload
        assert payload["prompt"] == req.prompt

    def test_instant_id_requires_selfie(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.INSTANT_ID})
        with pytest.raises(BackendError, match="requires a selfie"):
            _build_input(req, selfie_bytes=None)

    def test_instant_id_includes_image_and_strength(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.INSTANT_ID, "identity_strength": 0.7})
        payload = _build_input(req, selfie_bytes=PNG_BYTES)
        assert payload["ip_adapter_scale"] == 0.7
        assert hasattr(payload["image"], "read")

    def test_lora_requires_name(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.LORA})
        with pytest.raises(BackendError, match="lora_name"):
            _build_input(req, selfie_bytes=PNG_BYTES)


# ----------------------------- Fal input shape -----------------------------


class TestFalBuildArguments:
    def test_prompt_only_omits_face(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.PROMPT_ONLY, "selfie_sha256": None})
        args = _build_arguments(req, selfie_url=None)
        assert "face_image_url" not in args
        assert args["image_size"] == {"width": req.width, "height": req.height}

    def test_instant_id_includes_face_url(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.INSTANT_ID})
        args = _build_arguments(req, selfie_url="https://fal.example/selfie.png")
        assert args["face_image_url"] == "https://fal.example/selfie.png"

    def test_lora_emits_loras_array(self, sample_request: GenerationRequest) -> None:
        req = sample_request.model_copy(update={"strategy": Strategy.LORA, "lora_name": "https://x.safetensors"})
        args = _build_arguments(req, selfie_url="https://fal.example/s.png")
        assert args["loras"] == [{"path": "https://x.safetensors", "scale": req.identity_strength}]


class TestFalExtractImageUrl:
    def test_images_array_form(self) -> None:
        assert _extract_image_url({"images": [{"url": "u"}], "seed": 1}) == "u"

    def test_image_object_form(self) -> None:
        assert _extract_image_url({"image": {"url": "u"}}) == "u"

    def test_missing_url_raises(self) -> None:
        with pytest.raises(BackendError, match="Could not find image URL"):
            _extract_image_url({"foo": "bar"})


# ----------------------------- Replicate.generate (mocked) -----------------------------


class FakeFileOutput:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def aread(self) -> bytes:
        return self._data


class _FakeReplicateClient:
    """Stand-in for replicate.Client. Tests inject the desired behaviour
    into `async_run_handler`."""

    def __init__(self, async_run_handler: Any) -> None:
        self._handler = async_run_handler

    async def async_run(self, ref: str, *, input: dict[str, Any], use_file_output: bool) -> Any:
        return await self._handler(ref, input=input, use_file_output=use_file_output)


class TestReplicateGenerate:
    @pytest.mark.asyncio
    async def test_happy_path_writes_output_and_returns_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        storage: FakeStorage,
        sample_request: GenerationRequest,
    ) -> None:
        backend = ReplicateBackend(api_token="fake", storage=storage)

        captured: dict[str, Any] = {}

        async def fake_async_run(ref: str, *, input: dict[str, Any], use_file_output: bool) -> Any:
            captured["ref"] = ref
            captured["input"] = input
            return FakeFileOutput(PNG_BYTES)

        # Patch the factory — the per-call client construction means we can't
        # monkeypatch a long-lived attribute.
        monkeypatch.setattr(backend, "_new_client", lambda: _FakeReplicateClient(fake_async_run))

        result = await backend.generate(sample_request, selfie_bytes=PNG_BYTES)

        assert captured["ref"].startswith("zsxkib/instant-id")
        assert captured["input"]["prompt"] == sample_request.prompt
        assert result.image_path.exists()
        assert result.image_path.read_bytes() == PNG_BYTES
        assert result.cost_usd == 0.005
        assert result.backend.value == "replicate"

    @pytest.mark.asyncio
    async def test_vendor_exception_wrapped_in_backenderror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        storage: FakeStorage,
        sample_request: GenerationRequest,
    ) -> None:
        backend = ReplicateBackend(api_token="fake", storage=storage)

        async def boom(ref: str, *, input: dict[str, Any], use_file_output: bool) -> Any:
            raise RuntimeError("upstream 503")

        monkeypatch.setattr(backend, "_new_client", lambda: _FakeReplicateClient(boom))

        with pytest.raises(BackendError, match="Replicate call failed"):
            await backend.generate(sample_request, selfie_bytes=PNG_BYTES)

    @pytest.mark.asyncio
    async def test_unsupported_pair_raises(
        self,
        storage: FakeStorage,
        sample_request: GenerationRequest,
    ) -> None:
        backend = ReplicateBackend(api_token="fake", storage=storage)
        # PuLID is FLUX-only; passing SDXL should be rejected.
        bad = sample_request.model_copy(update={"strategy": Strategy.PULID, "backbone": Backbone.SDXL})
        with pytest.raises(BackendError, match="Unsupported"):
            await backend.generate(bad, selfie_bytes=PNG_BYTES)


# ----------------------------- Fal.generate (mocked) -----------------------------


class _FakeFalAsyncClient:
    """Stand-in for fal_client.AsyncClient — instance methods upload + subscribe."""

    async def upload(self, data: bytes, content_type: str) -> str:
        return "https://fal.example/uploaded.png"

    async def subscribe(self, application: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"images": [{"url": "https://fal.example/result.png"}], "seed": arguments["seed"]}


class TestFalGenerate:
    @pytest.mark.asyncio
    async def test_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        storage: FakeStorage,
        sample_request: GenerationRequest,
    ) -> None:
        backend = FalBackend(api_key="fake", storage=storage)

        # Patch the per-call client factory (same pattern as ReplicateBackend tests).
        monkeypatch.setattr(backend, "_new_client", lambda: _FakeFalAsyncClient())

        # Patch httpx so the result-image download doesn't hit the network.
        class FakeResp:
            content = PNG_BYTES

            def raise_for_status(self) -> None: ...

        class FakeHttpxClient:
            def __init__(self, **kw: Any) -> None: ...
            async def __aenter__(self) -> FakeHttpxClient:
                return self
            async def __aexit__(self, *a: Any) -> None: ...
            async def get(self, url: str) -> FakeResp:
                return FakeResp()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", FakeHttpxClient)

        result = await backend.generate(sample_request, selfie_bytes=PNG_BYTES)

        assert result.image_path.exists()
        assert result.image_path.read_bytes() == PNG_BYTES
        assert result.backend.value == "fal"
        assert result.model_version == "fal-ai/instant-id"
