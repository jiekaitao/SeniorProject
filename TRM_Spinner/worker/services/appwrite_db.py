from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings


class AppwriteDB:
    """Appwrite REST API client using httpx for database operations."""

    def __init__(self) -> None:
        self.endpoint = settings.appwrite_endpoint
        self.project_id = settings.appwrite_project_id
        self.api_key = settings.appwrite_api_key
        self.database_id = settings.appwrite_database_id
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Appwrite-Project": self.project_id,
            "X-Appwrite-Key": self.api_key,
        }

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint, headers=self.headers, timeout=30.0
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    # --- Generic CRUD ---

    async def create_document(
        self, collection_id: str, document_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents"
        payload = {
            "documentId": document_id,
            "data": data,
        }
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_document(
        self, collection_id: str, document_id: str
    ) -> Dict[str, Any]:
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents/{document_id}"
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_documents(
        self,
        collection_id: str,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents"
        params: Dict[str, Any] = {}
        if queries:
            params["queries[]"] = queries
        resp = await self.client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def update_document(
        self, collection_id: str, document_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents/{document_id}"
        payload = {"data": data}
        resp = await self.client.patch(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def delete_document(
        self, collection_id: str, document_id: str
    ) -> bool:
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents/{document_id}"
        resp = await self.client.delete(url)
        resp.raise_for_status()
        return True

    # --- Collection-specific helpers ---

    # Sessions
    async def create_session(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.create_document("sessions", session_id, data)

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        return await self.get_document("sessions", session_id)

    async def update_session(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.update_document("sessions", session_id, data)

    async def list_sessions(self, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        return await self.list_documents("sessions", queries=queries)

    # Messages
    async def create_message(self, message_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.create_document("messages", message_id, data)

    async def list_messages(self, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        return await self.list_documents("messages", queries=queries)

    # Training Jobs
    async def create_job(self, job_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.create_document("training_jobs", job_id, data)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        return await self.get_document("training_jobs", job_id)

    async def update_job(self, job_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.update_document("training_jobs", job_id, data)

    async def list_jobs(self, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        return await self.list_documents("training_jobs", queries=queries)
