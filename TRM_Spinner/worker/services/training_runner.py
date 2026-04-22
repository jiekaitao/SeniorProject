from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from services.training_manager import TrainingManager

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Runs TRM training as a subprocess and monitors progress."""

    def __init__(self, manager: TrainingManager, redis: aioredis.Redis) -> None:
        self.manager = manager
        self.redis = redis
        self._process: Optional[asyncio.subprocess.Process] = None

    async def run_training(self, job_id: str, config: Dict[str, Any]) -> None:
        """Launch pretrain_web.py as a subprocess with Hydra CLI args.

        Streams stdout/stderr to the worker's own logs so docker compose logs
        show training output, and uses an async subprocess so nothing blocks
        the event loop.
        """
        # Build Hydra CLI args from config
        cli_args = self._build_cli_args(job_id, config)

        # Path to pretrain_web.py (in the trm/ directory copied at build time)
        script_path = os.path.join(os.path.dirname(__file__), "..", "trm", "pretrain_web.py")
        script_path = os.path.abspath(script_path)

        cmd = ["python", script_path] + cli_args

        # Store checkpoint path in Redis for later download
        checkpoint_path = config.get(
            "checkpoint_path",
            os.path.join("checkpoints", job_id[:8]),
        )
        if not os.path.isabs(checkpoint_path):
            checkpoint_abs = os.path.abspath(
                os.path.join(os.path.dirname(script_path), checkpoint_path)
            )
        else:
            checkpoint_abs = checkpoint_path
        await self.redis.hset(f"trm:jobs:{job_id}", "checkpoint_path", checkpoint_abs)

        logger.info("Launching training for job %s: %s", job_id, " ".join(cmd))

        try:
            # Publish start event
            await self.redis.publish(
                f"trm:jobs:{job_id}:metrics",
                json.dumps({"status": "started", "job_id": job_id}),
            )

            # Use asyncio subprocess so we don't block the event loop and
            # can stream stdout/stderr without deadlocking on a full pipe.
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.path.dirname(script_path),
                env={
                    **os.environ,
                    "DISABLE_COMPILE": "1",
                    # RTX 5090 (Blackwell sm_120) has no prebuilt adam_atan2
                    # kernel — force AdamW so step() doesn't crash.
                    "DISABLE_ADAM_ATAN2": "1",
                },
            )

            tail: list[str] = []

            async def pipe_to_logs() -> None:
                assert self._process is not None and self._process.stdout is not None
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace").rstrip()
                    if text:
                        sys.stdout.write(f"[train {job_id[:8]}] {text}\n")
                        sys.stdout.flush()
                        tail.append(text)
                        if len(tail) > 50:
                            tail.pop(0)

            await asyncio.gather(pipe_to_logs(), self._process.wait())
            return_code = self._process.returncode

            if return_code == 0:
                # Run inference + save predictions.json before marking the
                # job complete so the results page always has something to
                # show.
                try:
                    await self._run_predictions(job_id, config)
                except Exception:
                    logger.exception("Prediction run failed for job %s", job_id)

                await self.manager.complete_job(job_id)
                await self.redis.publish(
                    f"trm:jobs:{job_id}:metrics",
                    json.dumps({"status": "completed", "job_id": job_id}),
                )
            else:
                err = "\n".join(tail[-20:])
                await self.manager.fail_job(
                    job_id, f"Exit code {return_code}: {err[-500:]}"
                )
                await self.redis.publish(
                    f"trm:jobs:{job_id}:metrics",
                    json.dumps(
                        {
                            "status": "failed",
                            "job_id": job_id,
                            "error": err[-500:],
                        }
                    ),
                )

        except Exception as e:
            logger.exception("Training subprocess failed for job %s", job_id)
            await self.manager.fail_job(job_id, str(e))
            await self.redis.publish(
                f"trm:jobs:{job_id}:metrics",
                json.dumps({"status": "failed", "job_id": job_id, "error": str(e)}),
            )

    async def _run_predictions(self, job_id: str, config: Dict[str, Any]) -> None:
        """Invoke predict_web.py on the latest checkpoint.

        Writes `predictions.json` next to the data directory so the results
        endpoint can serve it.
        """
        data_paths = config.get("data_paths", [])
        if not data_paths:
            return
        data_root = data_paths[0]
        # Data lives under <root>/train/
        train_dir = os.path.join(data_root, "train") if not data_root.rstrip("/").endswith("train") else data_root

        # Resolve latest checkpoint from redis-stored path.
        raw = await self.redis.hget(f"trm:jobs:{job_id}", "checkpoint_path")
        checkpoint_dir = raw.decode() if isinstance(raw, bytes) else raw
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            logger.info("No checkpoint dir for job %s; skipping predictions", job_id)
            return
        # Only keep bare step_N files (exclude step_N_all_preds.* etc.).
        def _is_checkpoint(name: str) -> bool:
            if not name.startswith("step_"):
                return False
            rest = name[len("step_"):]
            return rest.isdigit()

        ckpts = sorted(
            (f for f in os.listdir(checkpoint_dir) if _is_checkpoint(f)),
            key=lambda s: int(s.split("_")[1]),
        )
        if not ckpts:
            logger.info("No step_N file in %s; skipping predictions", checkpoint_dir)
            return
        checkpoint = os.path.join(checkpoint_dir, ckpts[-1])

        predictions_path = os.path.join(data_root, "predictions.json")

        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "trm", "predict_web.py")
        )
        cmd = [
            "python",
            script_path,
            "--data-dir", train_dir,
            "--checkpoint", checkpoint,
            "--output", predictions_path,
            "--limit", "16",
        ]

        logger.info("Running inference for job %s: %s", job_id, " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.path.dirname(script_path),
            env={**os.environ, "DISABLE_COMPILE": "1", "DISABLE_ADAM_ATAN2": "1"},
        )

        async def pipe() -> None:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    sys.stdout.write(f"[predict {job_id[:8]}] {text}\n")
                    sys.stdout.flush()

        await asyncio.gather(pipe(), proc.wait())
        if proc.returncode == 0 and os.path.exists(predictions_path):
            await self.redis.hset(
                f"trm:jobs:{job_id}", "predictions_path", predictions_path
            )

    # Fields in PretrainConfig that are NOT in cfg_pretrain.yaml and need + prefix
    _APPEND_KEYS = frozenset({
        "job_id", "redis_url", "checkpoint_path", "project_name",
        "run_name", "load_checkpoint", "eval_save_outputs",
    })

    def _build_cli_args(self, job_id: str, config: Dict[str, Any]) -> list[str]:
        """Build Hydra CLI override args from config dict."""
        args = [
            f"+job_id={job_id}",
            f"+redis_url={os.environ.get('REDIS_URL', 'redis://localhost:6379')}",
        ]

        for key, value in config.items():
            prefix = "+" if key in self._APPEND_KEYS else ""
            if isinstance(value, (dict, list)):
                args.append(f"{prefix}{key}={json.dumps(value)}")
            else:
                args.append(f"{prefix}{key}={value}")

        return args
