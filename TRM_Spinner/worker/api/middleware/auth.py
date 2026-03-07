from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import httpx

from config.settings import settings

security = HTTPBearer(auto_error=False)


async def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify Appwrite JWT token and return user_id.

    Supports an API key bypass for testing/admin: send X-API-Key header
    matching settings.admin_password to authenticate as test-user-001.
    """
    # API key bypass for testing/admin
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == settings.admin_password:
        return "test-user-001"

    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization")

    token = credentials.credentials

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.appwrite_endpoint}/account",
                headers={
                    "X-Appwrite-Project": settings.appwrite_project_id,
                    "X-Appwrite-JWT": token,
                },
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid or expired token")

            user = resp.json()
            return user["$id"]

    except httpx.RequestError:
        raise HTTPException(status_code=401, detail="Failed to verify token")
