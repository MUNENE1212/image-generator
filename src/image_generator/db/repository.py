"""Repository layer over DuckDB. No SQL escapes this file."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from image_generator.models.selfie import FaceEmbedding, Selfie

if TYPE_CHECKING:
    from image_generator.db.connection import Database
    from image_generator.models.results import RunRecord


class RunsRepository:
    """Insert and query `runs` rows. All mutations take the DB lock."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert(self, record: RunRecord) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO runs VALUES (
                    $run_id,
                    $strategy, $backbone, $prompt, $negative_prompt, $selfie_sha256,
                    $seed, $num_inference_steps, $guidance_scale, $width, $height,
                    $identity_strength, $lora_name,
                    $backend, $model_version, $image_path,
                    $started_at, $completed_at, $duration_seconds, $cost_usd,
                    $identity_arcface, $identity_adaface, $prompt_siglip,
                    $aesthetic_laion, $aesthetic_qalign, $diversity_lpips
                )
                """,
                record.model_dump(),
            )

    def update_metrics(
        self,
        run_id: str,
        *,
        identity_arcface: float | None = None,
        identity_adaface: float | None = None,
        prompt_siglip: float | None = None,
        aesthetic_laion: float | None = None,
        aesthetic_qalign: float | None = None,
        diversity_lpips: float | None = None,
    ) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                UPDATE runs SET
                    identity_arcface = COALESCE($identity_arcface, identity_arcface),
                    identity_adaface = COALESCE($identity_adaface, identity_adaface),
                    prompt_siglip    = COALESCE($prompt_siglip,    prompt_siglip),
                    aesthetic_laion  = COALESCE($aesthetic_laion,  aesthetic_laion),
                    aesthetic_qalign = COALESCE($aesthetic_qalign, aesthetic_qalign),
                    diversity_lpips  = COALESCE($diversity_lpips,  diversity_lpips)
                WHERE run_id = $run_id
                """,
                {
                    "run_id": run_id,
                    "identity_arcface": identity_arcface,
                    "identity_adaface": identity_adaface,
                    "prompt_siglip": prompt_siglip,
                    "aesthetic_laion": aesthetic_laion,
                    "aesthetic_qalign": aesthetic_qalign,
                    "diversity_lpips": diversity_lpips,
                },
            )

    def count(self) -> int:
        with self._db.lock:
            row = self._db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0]) if row else 0

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        with self._db.lock:
            cursor = self._db.conn.execute(
                "SELECT * FROM runs ORDER BY completed_at DESC LIMIT ?", [limit]
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]


class SelfiesRepository:
    """Persist selfie metadata + face-embedding cache."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, selfie: Selfie) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO selfies (sha256, path, width, height, uploaded_at)
                VALUES ($sha256, $path, $width, $height, $uploaded_at)
                ON CONFLICT (sha256) DO NOTHING
                """,
                {
                    "sha256": selfie.sha256,
                    "path": str(selfie.path),
                    "width": selfie.width,
                    "height": selfie.height,
                    "uploaded_at": selfie.uploaded_at,
                },
            )

    def get(self, sha256: str) -> Selfie | None:
        with self._db.lock:
            row = self._db.conn.execute(
                "SELECT sha256, path, width, height, uploaded_at FROM selfies WHERE sha256 = ?",
                [sha256],
            ).fetchone()
        if row is None:
            return None
        from pathlib import Path

        return Selfie(
            sha256=row[0],
            path=Path(row[1]),
            width=row[2],
            height=row[3],
            uploaded_at=row[4],
        )

    def get_embedding(self, sha256: str, model_name: str) -> FaceEmbedding | None:
        with self._db.lock:
            row = self._db.conn.execute(
                """
                SELECT selfie_sha256, model_name, vector, computed_at
                FROM face_embeddings
                WHERE selfie_sha256 = ? AND model_name = ?
                """,
                [sha256, model_name],
            ).fetchone()
        if row is None:
            return None
        return FaceEmbedding(
            selfie_sha256=row[0],
            model_name=row[1],
            vector=list(row[2]),
            computed_at=row[3],
        )

    def put_embedding(self, embedding: FaceEmbedding) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO face_embeddings (selfie_sha256, model_name, vector, computed_at)
                VALUES ($sha, $model, $vec, $ts)
                ON CONFLICT (selfie_sha256, model_name)
                DO UPDATE SET vector = $vec, computed_at = $ts
                """,
                {
                    "sha": embedding.selfie_sha256,
                    "model": embedding.model_name,
                    "vec": embedding.vector,
                    "ts": embedding.computed_at,
                },
            )


class SweepsRepository:
    """Manage Experiments-page sweeps and their links to individual runs."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert(self, *, sweep_id: str, name: str, config_json: str, created_at: datetime) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO sweeps (sweep_id, name, config_json, created_at)
                VALUES ($sweep_id, $name, $config_json, $created_at)
                """,
                {
                    "sweep_id": sweep_id,
                    "name": name,
                    "config_json": config_json,
                    "created_at": created_at,
                },
            )

    def link_run(self, sweep_id: str, run_id: str) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO sweep_runs (sweep_id, run_id) VALUES ($sweep_id, $run_id)
                ON CONFLICT (sweep_id, run_id) DO NOTHING
                """,
                {"sweep_id": sweep_id, "run_id": run_id},
            )

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        with self._db.lock:
            cursor = self._db.conn.execute(
                """
                SELECT s.sweep_id, s.name, s.created_at, COUNT(sr.run_id) AS run_count
                FROM sweeps s
                LEFT JOIN sweep_runs sr ON sr.sweep_id = s.sweep_id
                GROUP BY s.sweep_id, s.name, s.created_at
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                [limit],
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def get_runs(self, sweep_id: str) -> list[dict[str, object]]:
        """Return every run row in the sweep, with full request + metric columns."""
        with self._db.lock:
            cursor = self._db.conn.execute(
                """
                SELECT r.*
                FROM runs r
                JOIN sweep_runs sr ON sr.run_id = r.run_id
                WHERE sr.sweep_id = ?
                ORDER BY r.completed_at
                """,
                [sweep_id],
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]


class TrainingsRepository:
    """Per-subject LoRA training jobs initiated from the Training Studio.

    Lifecycle: insert (status='running') → update_status repeatedly during polling
    → final state is one of 'succeeded' | 'failed' | 'cancelled'.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert(
        self,
        *,
        training_id: str,
        method: str,
        selfie_shas: list[str],
        backend: str,
        status: str = "running",
        cost_usd: float = 0.0,
        started_at: datetime | None = None,
    ) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                INSERT INTO trainings (training_id, method, selfie_shas, status,
                                       backend, cost_usd, started_at)
                VALUES ($training_id, $method, $selfie_shas, $status,
                        $backend, $cost_usd, $started_at)
                """,
                {
                    "training_id": training_id,
                    "method": method,
                    "selfie_shas": selfie_shas,
                    "status": status,
                    "backend": backend,
                    "cost_usd": cost_usd,
                    "started_at": started_at or datetime.now(UTC),
                },
            )

    def update_status(
        self,
        training_id: str,
        *,
        status: str,
        lora_path: str | None = None,
        lora_name: str | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        with self._db.lock:
            self._db.conn.execute(
                """
                UPDATE trainings SET
                    status        = $status,
                    lora_path     = COALESCE($lora_path, lora_path),
                    lora_name     = COALESCE($lora_name, lora_name),
                    error         = COALESCE($error, error),
                    completed_at  = COALESCE($completed_at, completed_at)
                WHERE training_id = $training_id
                """,
                {
                    "training_id": training_id,
                    "status": status,
                    "lora_path": lora_path,
                    "lora_name": lora_name,
                    "error": error,
                    "completed_at": completed_at,
                },
            )

    def get(self, training_id: str) -> dict[str, object] | None:
        with self._db.lock:
            cursor = self._db.conn.execute(
                "SELECT * FROM trainings WHERE training_id = ?", [training_id]
            )
            columns = [d[0] for d in cursor.description]
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip(columns, row, strict=True))

    def recent(self, limit: int = 50) -> list[dict[str, object]]:
        with self._db.lock:
            cursor = self._db.conn.execute(
                "SELECT * FROM trainings ORDER BY started_at DESC LIMIT ?", [limit]
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def succeeded(self) -> list[dict[str, object]]:
        """All completed trainings with a usable lora_path. Powers the Lab/Home cell list."""
        with self._db.lock:
            cursor = self._db.conn.execute(
                """
                SELECT * FROM trainings
                WHERE status = 'succeeded' AND lora_path IS NOT NULL
                ORDER BY completed_at DESC
                """
            )
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]
