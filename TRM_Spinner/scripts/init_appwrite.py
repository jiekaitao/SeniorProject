#!/usr/bin/env python3
"""Initialize Appwrite database and collections for TRM Spinner."""

import httpx
import os
import sys
import time
from dotenv import load_dotenv

# Load from .env in parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ENDPOINT = os.environ['APPWRITE_ENDPOINT']
PROJECT_ID = os.environ['APPWRITE_PROJECT_ID']
API_KEY = os.environ['APPWRITE_API_KEY']
DATABASE_ID = os.environ.get('APPWRITE_DATABASE_ID', 'trm_spinner')

headers = {
    'X-Appwrite-Project': PROJECT_ID,
    'X-Appwrite-Key': API_KEY,
    'Content-Type': 'application/json',
}

client = httpx.Client(base_url=ENDPOINT, headers=headers, timeout=30)


def create_database():
    resp = client.post('/databases', json={'databaseId': DATABASE_ID, 'name': 'TRM Spinner'})
    if resp.status_code == 409:
        print(f"Database '{DATABASE_ID}' already exists")
    elif resp.is_success:
        print(f"Created database '{DATABASE_ID}'")
    else:
        print(f"Error creating database: {resp.text}")
        sys.exit(1)


def create_collection(collection_id, name, attributes, indexes=None):
    # Create collection
    resp = client.post(f'/databases/{DATABASE_ID}/collections', json={
        'collectionId': collection_id,
        'name': name,
        'documentSecurity': True,
        'permissions': ['read("any")', 'create("users")', 'update("users")', 'delete("users")']
    })
    if resp.status_code == 409:
        print(f"  Collection '{collection_id}' already exists")
    elif resp.is_success:
        print(f"  Created collection '{collection_id}'")
    else:
        print(f"  Error: {resp.text}")
        return

    # Create attributes
    for attr in attributes:
        attr_type = attr.pop('type')
        endpoint = f'/databases/{DATABASE_ID}/collections/{collection_id}/attributes/{attr_type}'
        resp = client.post(endpoint, json=attr)
        if resp.status_code == 409:
            pass  # Already exists
        elif resp.is_success:
            print(f"    + {attr['key']}")
        else:
            print(f"    Error on {attr['key']}: {resp.text}")
        attr['type'] = attr_type  # Restore

    # Wait for attributes to be available
    time.sleep(2)

    # Create indexes
    if indexes:
        for idx in indexes:
            resp = client.post(f'/databases/{DATABASE_ID}/collections/{collection_id}/indexes', json=idx)
            if resp.status_code == 409:
                pass
            elif resp.is_success:
                print(f"    Index: {idx['key']}")
            else:
                print(f"    Error on index {idx['key']}: {resp.text}")


def create_sessions_collection():
    print("\nSessions collection:")
    create_collection('sessions', 'Sessions', [
        {'key': 'user_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'title', 'type': 'string', 'size': 512, 'required': False, 'default': ''},
        {'key': 'status', 'type': 'string', 'size': 32, 'required': True},
        {'key': 'problem_type', 'type': 'string', 'size': 64, 'required': False, 'default': ''},
        {'key': 'is_suitable', 'type': 'boolean', 'required': False, 'default': False},
        {'key': 'created_at', 'type': 'string', 'size': 64, 'required': True},
        {'key': 'updated_at', 'type': 'string', 'size': 64, 'required': True},
    ], indexes=[
        {'key': 'idx_user_id', 'type': 'key', 'attributes': ['user_id'], 'orders': ['ASC']},
    ])


def create_messages_collection():
    print("\nMessages collection:")
    create_collection('messages', 'Messages', [
        {'key': 'session_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'user_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'role', 'type': 'string', 'size': 16, 'required': True},
        {'key': 'content', 'type': 'string', 'size': 10000, 'required': True},
        {'key': 'metadata', 'type': 'string', 'size': 5000, 'required': False, 'default': ''},
        {'key': 'created_at', 'type': 'string', 'size': 64, 'required': True},
    ], indexes=[
        {'key': 'idx_session_id', 'type': 'key', 'attributes': ['session_id'], 'orders': ['ASC']},
    ])


def create_training_jobs_collection():
    print("\nTraining Jobs collection:")
    create_collection('training_jobs', 'Training Jobs', [
        {'key': 'session_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'user_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'status', 'type': 'string', 'size': 32, 'required': True},
        {'key': 'variant', 'type': 'string', 'size': 64, 'required': False, 'default': ''},
        {'key': 'problem_type', 'type': 'string', 'size': 64, 'required': False, 'default': ''},
        {'key': 'config_json', 'type': 'string', 'size': 5000, 'required': False, 'default': ''},
        {'key': 'total_steps', 'type': 'integer', 'required': False, 'default': 0},
        {'key': 'current_step', 'type': 'integer', 'required': False, 'default': 0},
        {'key': 'best_accuracy', 'type': 'float', 'required': False, 'default': 0.0},
        {'key': 'latest_loss', 'type': 'float', 'required': False, 'default': 0.0},
        {'key': 'checkpoint_path', 'type': 'string', 'size': 512, 'required': False, 'default': ''},
        {'key': 'error_message', 'type': 'string', 'size': 1000, 'required': False, 'default': ''},
        {'key': 'started_at', 'type': 'string', 'size': 64, 'required': False, 'default': ''},
        {'key': 'completed_at', 'type': 'string', 'size': 64, 'required': False, 'default': ''},
        {'key': 'created_at', 'type': 'string', 'size': 64, 'required': True},
    ], indexes=[
        {'key': 'idx_session_id', 'type': 'key', 'attributes': ['session_id'], 'orders': ['ASC']},
        {'key': 'idx_user_id', 'type': 'key', 'attributes': ['user_id'], 'orders': ['ASC']},
        {'key': 'idx_status', 'type': 'key', 'attributes': ['status'], 'orders': ['ASC']},
    ])


def create_analytics_events_collection():
    print("\nAnalytics Events collection:")
    create_collection('analytics_events', 'Analytics Events', [
        {'key': 'job_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'user_id', 'type': 'string', 'size': 256, 'required': True},
        {'key': 'event_type', 'type': 'string', 'size': 64, 'required': True},
        {'key': 'metrics_json', 'type': 'string', 'size': 10000, 'required': False, 'default': ''},
        {'key': 'created_at', 'type': 'string', 'size': 64, 'required': True},
    ], indexes=[
        {'key': 'idx_job_id', 'type': 'key', 'attributes': ['job_id'], 'orders': ['ASC']},
    ])


if __name__ == '__main__':
    print("Initializing Appwrite database for TRM Spinner...")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Project:  {PROJECT_ID}")
    print(f"Database: {DATABASE_ID}")
    print()

    create_database()
    create_sessions_collection()
    create_messages_collection()
    create_training_jobs_collection()
    create_analytics_events_collection()

    print("\nDone! All collections initialized.")
