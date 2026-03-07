from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator


class DataPoint(BaseModel):
    input: List[List[int]]
    output: List[List[int]]

    @field_validator("input", "output")
    @classmethod
    def validate_2d_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if not v or not isinstance(v[0], list):
            raise ValueError("Grid must be a 2D list of integers")
        return v


class DataUpload(BaseModel):
    session_id: str
    data: List[DataPoint]

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v: List[DataPoint]) -> List[DataPoint]:
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class DataValidation(BaseModel):
    valid: bool
    num_examples: int
    max_grid_size: int
    vocab_size: int
    errors: Optional[List[str]] = None
