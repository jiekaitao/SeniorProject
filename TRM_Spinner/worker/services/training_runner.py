from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from services.training_manager import TrainingManager


class TrainingRunner:
    """Runs TRM training as a subprocess and monitors progress."""

    def __init__(self, manager: TrainingManager, redis: aioredis.Redis) -> None:
        self.manager = manager
        self.redis = redis
        self._process: Optional[subprocess.Popen] = None

    async def run_training(self, job_id: str, config: Dict[str, Any]) -> None:
        """Launch pretrain_web.py as a subprocess with Hydra CLI args.

        Monitors the subprocess and updates Appwrite on completion/failure.
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
        checkpoint_abs = os.path.abspath(
            os.path.join(os.path.dirname(script_path), checkpoint_path)
        )
        await self.redis.hset(f"trm:jobs:{job_id}", "checkpoint_path", checkpoint_abs)

        try:
            # Publish start event
            await self.redis.publish(
                f"trm:jobs:{job_id}:metrics",
                json.dumps({"status": "started", "job_id": job_id}),
            )

            # Run subprocess
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(script_path),
                env={**os.environ, "DISABLE_COMPILE": "1"},
            )

            # Monitor in a non-blocking way
            return_code = await asyncio.to_thread(self._process.wait)

            if return_code == 0:
                await self.manager.complete_job(job_id)
                await self.redis.publish(
                    f"trm:jobs:{job_id}:metrics",
                    json.dumps({"status": "completed", "job_id": job_id}),
                )
            else:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(errors="replace")
                await self.manager.fail_job(job_id, f"Exit code {return_code}: {stderr[-500:]}")
                await self.redis.publish(
                    f"trm:jobs:{job_id}:metrics",
                    json.dumps({"status": "failed", "job_id": job_id, "error": stderr[-500:]}),
                )

        except Exception as e:
            await self.manager.fail_job(job_id, str(e))
            await self.redis.publish(
                f"trm:jobs:{job_id}:metrics",
                json.dumps({"status": "failed", "job_id": job_id, "error": str(e)}),
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
