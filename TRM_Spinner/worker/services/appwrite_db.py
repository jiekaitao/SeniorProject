from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx
import redis.asyncio as aioredis

from config.settings import settings


class LocalDB:
    """Redis-backed DB with Appwrite-compatible document semantics.

    Used when APPWRITE_PROJECT_ID is empty. All documents are stored as
    JSON blobs under `localdb:{collection}:{id}`, with a set
    `localdb:{collection}:ids` tracking membership for listing.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _doc_key(collection_id: str, document_id: str) -> str:
        return f"localdb:{collection_id}:{document_id}"

    @staticmethod
    def _index_key(collection_id: str) -> str:
        return f"localdb:{collection_id}:ids"

    async def close(self) -> None:
        try:
            await self._redis.close()
        except Exception:
            pass

    async def create_document(
        self, collection_id: str, document_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        doc = {
            "$id": document_id,
            "$createdAt": now_ms,
            "$updatedAt": now_ms,
            **data,
        }
        await self._redis.set(self._doc_key(collection_id, document_id), json.dumps(doc))
        await self._redis.sadd(self._index_key(collection_id), document_id)
        return doc

    async def get_document(
        self, collection_id: str, document_id: str
    ) -> Dict[str, Any]:
        raw = await self._redis.get(self._doc_key(collection_id, document_id))
        if raw is None:
            raise KeyError(f"Document {collection_id}/{document_id} not found")
        return json.loads(raw)

    async def list_documents(
        self,
        collection_id: str,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ids = await self._redis.smembers(self._index_key(collection_id))
        # Parse Appwrite-style JSON query objects to filter in Python.
        filters: list[tuple[str, list[Any]]] = []
        for q in queries or []:
            try:
                parsed = json.loads(q)
                if parsed.get("method") == "equal":
                    attr = parsed.get("attribute")
                    vals = parsed.get("values", [])
                    filters.append((attr, vals))
            except Exception:
                continue

        docs: list[Dict[str, Any]] = []
        for doc_id in ids:
            raw = await self._redis.get(self._doc_key(collection_id, doc_id))
            if raw is None:
                continue
            doc = json.loads(raw)
            if all(doc.get(attr) in vals for attr, vals in filters):
                docs.append(doc)

        docs.sort(key=lambda d: d.get("created_at") or d.get("$createdAt") or 0)
        return {"documents": docs, "total": len(docs)}

    async def update_document(
        self, collection_id: str, document_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw = await self._redis.get(self._doc_key(collection_id, document_id))
        if raw is None:
            raise KeyError(f"Document {collection_id}/{document_id} not found")
        doc = json.loads(raw)
        doc.update(data)
        doc["$updatedAt"] = int(time.time() * 1000)
        await self._redis.set(self._doc_key(collection_id, document_id), json.dumps(doc))
        return doc

    async def delete_document(
        self, collection_id: str, document_id: str
    ) -> bool:
        await self._redis.delete(self._doc_key(collection_id, document_id))
        await self._redis.srem(self._index_key(collection_id), document_id)
        return True


class AppwriteDB:
    """Storage client with Appwrite-compatible CRUD semantics.

    If APPWRITE_PROJECT_ID is set, talks to Appwrite Cloud via httpx.
    Otherwise, falls back to a Redis-backed LocalDB so the stack can run
    end-to-end without any external services.
    """

    def __init__(self) -> None:
        self.endpoint = settings.appwrite_endpoint
        self.project_id = settings.appwrite_project_id
        self.api_key = settings.appwrite_api_key
        self.database_id = settings.appwrite_database_id
        self._client: Optional[httpx.AsyncClient] = None
        self._local: Optional[LocalDB] = None

        if not self.project_id:
            # Local fallback mode.
            self._local = LocalDB(settings.redis_url)

    @property
    def is_local(self) -> bool:
        return self._local is not None

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
        if self._local is not None:
            await self._local.close()

    # --- Generic CRUD ---

    async def create_document(
        self, collection_id: str, document_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self._local is not None:
            return await self._local.create_document(collection_id, document_id, data)
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
        if self._local is not None:
            return await self._local.get_document(collection_id, document_id)
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents/{document_id}"
        resp = await self.client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def list_documents(
        self,
        collection_id: str,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self._local is not None:
            return await self._local.list_documents(collection_id, queries)
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
        if self._local is not None:
            return await self._local.update_document(collection_id, document_id, data)
        url = f"/databases/{self.database_id}/collections/{collection_id}/documents/{document_id}"
        payload = {"data": data}
        resp = await self.client.patch(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def delete_document(
        self, collection_id: str, document_id: str
    ) -> bool:
        if self._local is not None:
            return await self._local.delete_document(collection_id, document_id)
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
