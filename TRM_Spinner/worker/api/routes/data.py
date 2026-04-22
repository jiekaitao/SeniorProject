from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, File

from api.middleware.auth import verify_token
from schemas.data import DataPoint, DataUpload, DataValidation
from services.appwrite_db import AppwriteDB
from services.data_converter import convert_data
from services.file_parser import parse_file_to_grid_pairs

router = APIRouter()

UPLOAD_DIR = os.path.abspath(os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "uploads")))

# Redis key for transient session state
def _redis_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


@router.post("/api/data/upload", response_model=DataValidation)
async def upload_data(
    body: DataUpload,
    request: Request,
    user_id: str = Depends(verify_token),
) -> DataValidation:
    """Upload and validate training data, then convert to numpy format."""
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    # Verify session ownership
    try:
        session = await db.get_session(body.session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # Validate the data
    errors = []
    max_grid_size = 0
    max_cell_value = 0

    for i, point in enumerate(body.data):
        # Check grid dimensions
        in_rows = len(point.input)
        in_cols = max((len(row) for row in point.input), default=0)
        out_rows = len(point.output)
        out_cols = max((len(row) for row in point.output), default=0)

        max_grid_size = max(max_grid_size, in_rows, in_cols, out_rows, out_cols)

        # Check for negative values
        for row in point.input + point.output:
            for v in row:
                if v < 0:
                    errors.append(f"Example {i}: negative cell value {v}")
                max_cell_value = max(max_cell_value, v)

        if max_grid_size > 30:
            errors.append(f"Example {i}: grid dimension exceeds 30x30 limit")

    if errors:
        return DataValidation(
            valid=False,
            num_examples=len(body.data),
            max_grid_size=max_grid_size,
            vocab_size=0,
            errors=errors,
        )

    # Convert to dict format for data_converter
    data_dicts = [{"input": p.input, "output": p.output} for p in body.data]

    # Create output directory
    output_dir = os.path.join(UPLOAD_DIR, body.session_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = convert_data(data_dicts, output_dir=output_dir)
    except Exception as e:
        return DataValidation(
            valid=False,
            num_examples=len(body.data),
            max_grid_size=max_grid_size,
            vocab_size=0,
            errors=[str(e)],
        )

    # Store data_path in Redis (not Appwrite — it's not in the schema)
    await redis.hset(_redis_state_key(body.session_id), "data_path", output_dir)

    return DataValidation(
        valid=True,
        num_examples=result.num_examples,
        max_grid_size=max_grid_size,
        vocab_size=result.vocab_size,
    )


@router.post("/api/data/validate", response_model=DataValidation)
async def validate_data(
    body: DataUpload,
    request: Request,
    user_id: str = Depends(verify_token),
) -> DataValidation:
    """Validate training data without saving."""
    errors = []
    max_grid_size = 0
    max_cell_value = 0

    for i, point in enumerate(body.data):
        in_rows = len(point.input)
        in_cols = max((len(row) for row in point.input), default=0)
        out_rows = len(point.output)
        out_cols = max((len(row) for row in point.output), default=0)

        max_grid_size = max(max_grid_size, in_rows, in_cols, out_rows, out_cols)

        for row in point.input + point.output:
            for v in row:
                if v < 0:
                    errors.append(f"Example {i}: negative cell value {v}")
                max_cell_value = max(max_cell_value, v)

    vocab_size = max_cell_value + 3 if not errors else 0  # PAD + EOS + cell values

    return DataValidation(
        valid=len(errors) == 0,
        num_examples=len(body.data),
        max_grid_size=max_grid_size,
        vocab_size=vocab_size,
        errors=errors if errors else None,
    )


@router.post("/api/data/upload-file", response_model=DataValidation)
async def upload_file(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token),
) -> DataValidation:
    """Upload any text file and parse it into grid pairs via LLM."""
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    # Verify session ownership
    try:
        session = await db.get_session(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # Read file content (keep raw bytes so we can parse binary formats).
    raw = await file.read()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    filename = file.filename or "upload.txt"

    # Parse via file_parser (JSON fast-path, xlsx, or LLM)
    grid_pairs, error = await parse_file_to_grid_pairs(text, filename, raw_bytes=raw)

    if error or len(grid_pairs) == 0:
        return DataValidation(
            valid=False,
            num_examples=0,
            max_grid_size=0,
            vocab_size=0,
            errors=[error or "No grid pairs found in file"],
        )

    # Validate grid sizes (reuse logic from upload_data)
    errors = []
    max_grid_size = 0
    max_cell_value = 0

    for i, point in enumerate(grid_pairs):
        in_rows = len(point["input"])
        in_cols = max((len(row) for row in point["input"]), default=0)
        out_rows = len(point["output"])
        out_cols = max((len(row) for row in point["output"]), default=0)
        max_grid_size = max(max_grid_size, in_rows, in_cols, out_rows, out_cols)

        for row in point["input"] + point["output"]:
            for v in row:
                max_cell_value = max(max_cell_value, v)

        if max_grid_size > 30:
            errors.append(f"Example {i}: grid dimension exceeds 30x30 limit")

    if errors:
        return DataValidation(
            valid=False,
            num_examples=len(grid_pairs),
            max_grid_size=max_grid_size,
            vocab_size=0,
            errors=errors,
        )

    # Convert and save
    output_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = convert_data(grid_pairs, output_dir=output_dir)
    except Exception as e:
        return DataValidation(
            valid=False,
            num_examples=len(grid_pairs),
            max_grid_size=max_grid_size,
            vocab_size=0,
            errors=[str(e)],
        )

    await redis.hset(_redis_state_key(session_id), "data_path", output_dir)

    return DataValidation(
        valid=True,
        num_examples=result.num_examples,
        max_grid_size=max_grid_size,
        vocab_size=result.vocab_size,
    )
