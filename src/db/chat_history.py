"""
Persistent PostgreSQL Chat History and User Management for Hermes.

Provides:
1. User registration, login, and profile lookup with salted PBKDF2 password hashing.
2. Conversation session tracking (session_id, user_id, title, timestamps).
3. Message history persistence (user messages, assistant responses, tool calls).
4. Auto-derivation of conversation titles from first messages.
5. In-memory graceful fallback for offline/test environments when PostgreSQL is unavailable.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import logging
import os
import threading
import uuid
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger("hermes.db.chat_history")


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 rounds and random 16-byte hex salt."""
    salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return f"pbkdf2:sha256:100000${salt}${pwd_hash}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify plain password against PBKDF2 hash using constant-time comparison."""
    try:
        algo, salt, expected_hash = hashed.split("$")
        parts = algo.split(":")
        iterations = int(parts[2]) if len(parts) >= 3 else 100_000
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        ).hex()
        return hmac.compare_digest(derived, expected_hash)
    except Exception:
        return False


class ChatHistoryDB:
    def __init__(self):
        self._pool = None
        self._is_postgres_ready = False
        self._lock = threading.Lock()

        # In-memory fallbacks when Postgres is not available
        self._mem_users: dict[str, dict[str, Any]] = {}
        self._mem_conversations: dict[str, dict[str, Any]] = {}
        self._mem_messages: dict[str, list[dict[str, Any]]] = {}

    def init_db(self) -> None:
        """Initialize connection pool and ensure required tables and indexes exist."""
        try:
            import psycopg
            from psycopg_pool import ConnectionPool

            pg_url = settings.postgres_url
            if not pg_url:
                logger.warning("No POSTGRES_URL configured; using in-memory chat store.")
                return

            self._pool = ConnectionPool(
                conninfo=pg_url,
                min_size=1,
                max_size=10,
                timeout=5.0,
                open=False,
            )
            self._pool.open()

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS hermes_users (
                            id VARCHAR(128) PRIMARY KEY,
                            username VARCHAR(64) UNIQUE NOT NULL,
                            email VARCHAR(255) UNIQUE,
                            password_hash VARCHAR(255) NOT NULL,
                            role VARCHAR(32) NOT NULL DEFAULT 'viewer',
                            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );

                        CREATE TABLE IF NOT EXISTS hermes_conversations (
                            session_id VARCHAR(128) PRIMARY KEY,
                            user_id VARCHAR(128) NOT NULL,
                            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
                            title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_hermes_conv_user ON hermes_conversations(user_id, updated_at DESC);

                        CREATE TABLE IF NOT EXISTS hermes_messages (
                            id BIGSERIAL PRIMARY KEY,
                            session_id VARCHAR(128) NOT NULL REFERENCES hermes_conversations(session_id) ON DELETE CASCADE,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            tool_calls JSONB,
                            metadata JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        ALTER TABLE hermes_messages ADD COLUMN IF NOT EXISTS metadata JSONB;
                        CREATE INDEX IF NOT EXISTS idx_hermes_msgs_session ON hermes_messages(session_id, created_at ASC);
                    """)
                conn.commit()

            self._is_postgres_ready = True
            logger.info("Hermes PostgreSQL Chat History & User tables successfully initialized.")
        except Exception as exc:
            logger.warning(
                f"Failed to connect to PostgreSQL for chat history ({exc.__class__.__name__}: {exc}). "
                "Operating with in-memory persistence fallback."
            )
            self._is_postgres_ready = False

    def close(self) -> None:
        """Close connection pool on shutdown."""
        if self._pool:
            try:
                self._pool.close()
            except Exception:
                pass

    # --- User Management ---

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: str = "viewer",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Create a new user with hashed password."""
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        pwd_hash = hash_password(password)
        now = datetime.datetime.now(datetime.timezone.utc)

        clean_username = username.strip()
        clean_email = email.strip() if email else None

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_users (id, username, email, password_hash, role, tenant_id, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id, username, email, role, tenant_id, created_at, updated_at
                            """,
                            (user_id, clean_username, clean_email, pwd_hash, role, tenant_id, now, now),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        return {
                            "id": row[0],
                            "username": row[1],
                            "email": row[2],
                            "role": row[3],
                            "tenant_id": row[4],
                            "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                            "updated_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                        }
            except Exception as e:
                # Handle unique constraint violations
                if "unique" in str(e).lower():
                    raise ValueError(f"Username '{clean_username}' or email already exists.")
                raise e

        # In-memory fallback
        with self._lock:
            for u in self._mem_users.values():
                if u["username"].lower() == clean_username.lower():
                    raise ValueError(f"Username '{clean_username}' already exists.")
                if clean_email and u.get("email") and u["email"].lower() == clean_email.lower():
                    raise ValueError(f"Email '{clean_email}' already exists.")

            user_obj = {
                "id": user_id,
                "username": clean_username,
                "email": clean_email,
                "password_hash": pwd_hash,
                "role": role,
                "tenant_id": tenant_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            self._mem_users[user_id] = user_obj
            return {k: v for k, v in user_obj.items() if k != "password_hash"}

    def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        """Retrieve user by username (case-insensitive)."""
        clean_username = username.strip()
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
                            FROM hermes_users
                            WHERE LOWER(username) = LOWER(%s)
                            """,
                            (clean_username,),
                        )
                        row = cur.fetchone()
                        if row:
                            return {
                                "id": row[0],
                                "username": row[1],
                                "email": row[2],
                                "password_hash": row[3],
                                "role": row[4],
                                "tenant_id": row[5],
                                "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                                "updated_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
                            }
                        return None
            except Exception as e:
                logger.error(f"Error fetching user by username: {e}")

        # In-memory fallback
        with self._lock:
            for u in self._mem_users.values():
                if u["username"].lower() == clean_username.lower():
                    return dict(u)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve public user profile by user_id."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, username, email, role, tenant_id, created_at, updated_at
                            FROM hermes_users
                            WHERE id = %s
                            """,
                            (user_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return {
                                "id": row[0],
                                "username": row[1],
                                "email": row[2],
                                "role": row[3],
                                "tenant_id": row[4],
                                "created_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                                "updated_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                            }
                        return None
            except Exception as e:
                logger.error(f"Error fetching user by ID: {e}")

        # In-memory fallback
        with self._lock:
            u = self._mem_users.get(user_id)
            if u:
                return {k: v for k, v in u.items() if k != "password_hash"}
        return None

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[dict[str, Any]]:
        """Authenticate user against stored password hash."""
        clean_input = username_or_email.strip()
        user = None

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
                            FROM hermes_users
                            WHERE LOWER(username) = LOWER(%s) OR (email IS NOT NULL AND LOWER(email) = LOWER(%s))
                            """,
                            (clean_input, clean_input),
                        )
                        row = cur.fetchone()
                        if row:
                            user = {
                                "id": row[0],
                                "username": row[1],
                                "email": row[2],
                                "password_hash": row[3],
                                "role": row[4],
                                "tenant_id": row[5],
                                "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                                "updated_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
                            }
            except Exception as e:
                logger.error(f"Error authenticating user: {e}")

        if not user:
            with self._lock:
                for u in self._mem_users.values():
                    if (
                        u["username"].lower() == clean_input.lower()
                        or (u.get("email") and u["email"].lower() == clean_input.lower())
                    ):
                        user = dict(u)
                        break

        if user and verify_password(password, user["password_hash"]):
            return {k: v for k, v in user.items() if k != "password_hash"}
        return None

    # --- Conversation & Message Management ---

    def get_or_create_conversation(
        self,
        session_id: str,
        user_id: str,
        title: str = "New Conversation",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Ensure conversation record exists for session_id and user_id."""
        now = datetime.datetime.now(datetime.timezone.utc)
        clean_title = title.strip()[:200]

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_conversations (session_id, user_id, tenant_id, title, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE
                            SET user_id = EXCLUDED.user_id, updated_at = EXCLUDED.updated_at
                            RETURNING session_id, user_id, tenant_id, title, created_at, updated_at
                            """,
                            (session_id, user_id, tenant_id, clean_title, now, now),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        return {
                            "session_id": row[0],
                            "user_id": row[1],
                            "tenant_id": row[2],
                            "title": row[3],
                            "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                            "updated_at": row[5].isoformat() if hasattr(row[5], "isoformat") else str(row[5]),
                        }
            except Exception as e:
                logger.error(f"Error get_or_create_conversation: {e}")

        # In-memory fallback
        with self._lock:
            if session_id in self._mem_conversations:
                conv = self._mem_conversations[session_id]
                conv["user_id"] = user_id
                conv["updated_at"] = now.isoformat()
                return dict(conv)

            conv = {
                "session_id": session_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "title": clean_title,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            self._mem_conversations[session_id] = conv
            return dict(conv)

    def update_conversation_title(self, session_id: str, title: str) -> bool:
        """Update conversation title."""
        clean_title = title.strip()[:200]
        now = datetime.datetime.now(datetime.timezone.utc)

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE hermes_conversations
                            SET title = %s, updated_at = %s
                            WHERE session_id = %s
                            """,
                            (clean_title, now, session_id),
                        )
                        conn.commit()
                        return cur.rowcount > 0
            except Exception as e:
                logger.error(f"Error updating conversation title: {e}")

        with self._lock:
            if session_id in self._mem_conversations:
                self._mem_conversations[session_id]["title"] = clean_title
                self._mem_conversations[session_id]["updated_at"] = now.isoformat()
                return True
        return False

    def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        tenant_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List conversations belonging to user_id, ordered by most recently updated."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        if tenant_id:
                            cur.execute(
                                """
                                SELECT session_id, user_id, tenant_id, title, created_at, updated_at
                                FROM hermes_conversations
                                WHERE user_id = %s AND tenant_id = %s
                                ORDER BY updated_at DESC
                                LIMIT %s
                                """,
                                (user_id, tenant_id, limit),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT session_id, user_id, tenant_id, title, created_at, updated_at
                                FROM hermes_conversations
                                WHERE user_id = %s
                                ORDER BY updated_at DESC
                                LIMIT %s
                                """,
                                (user_id, limit),
                            )
                        rows = cur.fetchall()
                        return [
                            {
                                "session_id": r[0],
                                "user_id": r[1],
                                "tenant_id": r[2],
                                "title": r[3],
                                "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
                                "updated_at": r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5]),
                            }
                            for r in rows
                        ]
            except Exception as e:
                logger.error(f"Error listing conversations: {e}")

        # In-memory fallback
        with self._lock:
            user_convs = [
                c
                for c in self._mem_conversations.values()
                if c.get("user_id") == user_id and (tenant_id is None or c.get("tenant_id") == tenant_id)
            ]
            user_convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return user_convs[:limit]

    def delete_conversation(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Delete conversation and its associated messages with tenant scoping."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        if user_id and tenant_id:
                            cur.execute(
                                "DELETE FROM hermes_conversations WHERE session_id = %s AND user_id = %s AND tenant_id = %s",
                                (session_id, user_id, tenant_id),
                            )
                        elif user_id:
                            cur.execute(
                                "DELETE FROM hermes_conversations WHERE session_id = %s AND user_id = %s",
                                (session_id, user_id),
                            )
                        elif tenant_id:
                            cur.execute(
                                "DELETE FROM hermes_conversations WHERE session_id = %s AND tenant_id = %s",
                                (session_id, tenant_id),
                            )
                        else:
                            cur.execute(
                                "DELETE FROM hermes_conversations WHERE session_id = %s",
                                (session_id,),
                            )
                        conn.commit()
                        return cur.rowcount > 0
            except Exception as e:
                logger.error(f"Error deleting conversation: {e}")

        # In-memory fallback
        with self._lock:
            conv = self._mem_conversations.get(session_id)
            if not conv:
                return False
            if user_id and conv.get("user_id") != user_id:
                return False
            if tenant_id and conv.get("tenant_id") != tenant_id:
                return False
            del self._mem_conversations[session_id]
            self._mem_messages.pop(session_id, None)
            return True

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        user_id: str = "guest_user",
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """Append message to conversation, auto-creating conversation and title if needed."""
        # Ensure conversation exists
        conv = self.get_or_create_conversation(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Auto-update title if it is still the default and this is the first user message
        if role == "user" and conv.get("title") in ("New Conversation", "New Chat", ""):
            auto_title = content.strip().split("\n")[0][:48]
            if auto_title:
                self.update_conversation_title(session_id, auto_title)

        now = datetime.datetime.now(datetime.timezone.utc)
        serialized_tools = None
        if tool_calls is not None:
            try:
                serialized_tools = json.dumps(tool_calls)
            except Exception:
                serialized_tools = str(tool_calls)

        serialized_meta = None
        if metadata is not None:
            try:
                serialized_meta = json.dumps(metadata)
            except Exception:
                serialized_meta = str(metadata)

        clean_content = content.replace("\x00", "") if isinstance(content, str) else str(content)
        if serialized_tools:
            serialized_tools = serialized_tools.replace("\x00", "")
        if serialized_meta:
            serialized_meta = serialized_meta.replace("\x00", "")

        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO hermes_messages (session_id, role, content, tool_calls, metadata, created_at)
                            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
                            RETURNING id, session_id, role, content, tool_calls, metadata, created_at
                            """,
                            (session_id, role, clean_content, serialized_tools, serialized_meta, now),
                        )
                        row = cur.fetchone()

                        # Touch conversation updated_at
                        cur.execute(
                            "UPDATE hermes_conversations SET updated_at = %s WHERE session_id = %s",
                            (now, session_id),
                        )
                        conn.commit()

                        return {
                            "id": row[0],
                            "session_id": row[1],
                            "role": row[2],
                            "content": row[3],
                            "tool_calls": row[4],
                            "metadata": row[5],
                            "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
                        }
            except Exception as e:
                logger.error(f"Error appending message: {e}")

        # In-memory fallback
        with self._lock:
            msg_id = len(self._mem_messages.get(session_id, [])) + 1
            msg_obj = {
                "id": msg_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "metadata": metadata,
                "created_at": now.isoformat(),
            }
            if session_id not in self._mem_messages:
                self._mem_messages[session_id] = []
            self._mem_messages[session_id].append(msg_obj)
            if session_id in self._mem_conversations:
                self._mem_conversations[session_id]["updated_at"] = now.isoformat()
            return msg_obj

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages in a conversation in chronological order."""
        if self._is_postgres_ready and self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, session_id, role, content, tool_calls, metadata, created_at
                            FROM hermes_messages
                            WHERE session_id = %s
                            ORDER BY created_at ASC, id ASC
                            """,
                            (session_id,),
                        )
                        rows = cur.fetchall()
                        return [
                            {
                                "id": r[0],
                                "session_id": r[1],
                                "role": r[2],
                                "content": r[3],
                                "tool_calls": r[4],
                                "metadata": r[5],
                                "created_at": r[6].isoformat() if hasattr(r[6], "isoformat") else str(r[6]),
                            }
                            for r in rows
                        ]
            except Exception as e:
                logger.error(f"Error fetching messages: {e}")

        # In-memory fallback
        with self._lock:
            return list(self._mem_messages.get(session_id, []))


chat_history_db = ChatHistoryDB()

