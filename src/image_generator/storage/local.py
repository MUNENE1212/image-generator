"""Local filesystem storage. Content-addressed under `data/`."""

from __future__ import annotations

from pathlib import Path

from image_generator.config import settings


class LocalStorage:
    """Default Storage implementation. All paths live under `settings.imggen_data_dir`."""

    def __init__(
        self,
        *,
        selfies_dir: Path | None = None,
        outputs_dir: Path | None = None,
        loras_dir: Path | None = None,
    ) -> None:
        self._selfies_dir = selfies_dir or settings.selfies_dir
        self._outputs_dir = outputs_dir or settings.outputs_dir
        self._loras_dir = loras_dir or settings.loras_dir
        for d in (self._selfies_dir, self._outputs_dir, self._loras_dir):
            d.mkdir(parents=True, exist_ok=True)

    def put_selfie(self, data: bytes, sha256: str) -> Path:
        # Shard by first 2 hex chars to avoid giant directories at scale.
        shard = self._selfies_dir / sha256[:2]
        shard.mkdir(parents=True, exist_ok=True)
        path = shard / f"{sha256}.png"
        if not path.exists():
            path.write_bytes(data)
        return path

    def put_output(self, data: bytes, run_id: str, ext: str = "png") -> Path:
        path = self._outputs_dir / f"{run_id}.{ext}"
        path.write_bytes(data)
        return path

    def put_lora(self, data: bytes, name: str) -> Path:
        path = self._loras_dir / f"{name}.safetensors"
        path.write_bytes(data)
        return path

    def read(self, path: Path) -> bytes:
        return Path(path).read_bytes()

    def exists(self, path: Path) -> bool:
        return Path(path).exists()
