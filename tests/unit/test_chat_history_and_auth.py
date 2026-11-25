"""
Unit tests for Chat History Database, Password Hashing, User Auth, and Conversation APIs.
"""
import uuid
import pytest
from fastapi.testclient import TestClient

from src.db.chat_history import chat_history_db, hash_password, verify_password
from src.main import app
from src.security.auth import create_access_token


@pytest.fixture(autouse=True)
def ensure_db():
    chat_history_db.init_db()


def test_password_hashing_and_verification():
    raw = "MySuperSecretPassword#2026"
    hashed = hash_password(raw)
    assert hashed.startswith("pbkdf2:sha256:100000$")
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


def test_user_creation_and_authentication():
    uname = f"testuser_{uuid.uuid4().hex[:6]}"
    email = f"{uname}@example.com"
    pwd = "SecurePassword123!"

    user = chat_history_db.create_user(uname, pwd, email=email, role="operator")
    assert user["username"] == uname
    assert user["email"] == email
    assert user["role"] == "operator"
    assert "password_hash" not in user

    # Auth by username
    authed = chat_history_db.authenticate_user(uname, pwd)
    assert authed is not None
    assert authed["id"] == user["id"]

    # Auth by email
    authed_email = chat_history_db.authenticate_user(email, pwd)
    assert authed_email is not None
    assert authed_email["id"] == user["id"]

    # Auth with incorrect password
    failed = chat_history_db.authenticate_user(uname, "BadPassword")
    assert failed is None


def test_duplicate_user_rejection():
    uname = f"dup_{uuid.uuid4().hex[:6]}"
    chat_history_db.create_user(uname, "pwd12345")
    with pytest.raises(ValueError):
        chat_history_db.create_user(uname, "pwd12345")


def test_conversation_and_messages_lifecycle():
    user_id = f"usr_{uuid.uuid4().hex[:8]}"
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Create conversation
    conv = chat_history_db.get_or_create_conversation(session_id, user_id, "Initial Title")
    assert conv["session_id"] == session_id
    assert conv["user_id"] == user_id

    # Append user message
    m1 = chat_history_db.append_message(session_id, "user", "Hello world!", user_id=user_id)
    assert m1["role"] == "user"
    assert m1["content"] == "Hello world!"

    # Append assistant reply with tool_calls
    tool_steps = [
        {
            "id": "step_1",
            "tool": "web_scraper.search",
            "arguments": {"query": "proactive agent"},
            "thought": "Searching for proactive agents definition",
            "output": "Proactive agents take initiative automatically.",
        }
    ]
    telemetry_meta = {
        "model": "groq/qwen/qwen3.8-27b",
        "latency_ms": 1250.5,
        "ttft_ms": 350.2,
        "task_type": "search_synthesis",
    }
    m2 = chat_history_db.append_message(
        session_id,
        "assistant",
        "Hi there, I am Winnie.",
        tool_calls=tool_steps,
        metadata=telemetry_meta,
        user_id=user_id,
    )
    assert m2["role"] == "assistant"
    assert m2["tool_calls"] is not None
    assert m2["metadata"] is not None

    # Fetch messages
    msgs = chat_history_db.get_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Hello world!"
    assert msgs[1]["content"] == "Hi there, I am Winnie."
    assert msgs[1]["tool_calls"] is not None
    assert msgs[1]["metadata"] is not None

    # Verify metadata content
    meta = msgs[1]["metadata"]
    if isinstance(meta, str):
        import json
        meta = json.loads(meta)
    assert meta["model"] == "groq/qwen/qwen3.8-27b"
    assert meta["latency_ms"] == 1250.5
    assert meta["ttft_ms"] == 350.2
    # Verify tool_calls content
    tc = msgs[1]["tool_calls"]
    if isinstance(tc, str):
        import json
        tc = json.loads(tc)
    assert tc[0]["tool"] == "web_scraper.search"
    assert tc[0]["output"] == "Proactive agents take initiative automatically."

    # Update title
    ok = chat_history_db.update_conversation_title(session_id, "Updated Custom Title")
    assert ok is True

    # List conversations
    convs = chat_history_db.list_conversations(user_id)
    assert len(convs) >= 1
    matched = [c for c in convs if c["session_id"] == session_id]
    assert len(matched) == 1
    assert matched[0]["title"] == "Updated Custom Title"

    # Delete conversation
    del_ok = chat_history_db.delete_conversation(session_id)
    assert del_ok is True
    assert len(chat_history_db.get_messages(session_id)) == 0


def test_fastapi_auth_and_conversations_api():
    client = TestClient(app)
    suffix = uuid.uuid4().hex[:6]
    uname = f"api_user_{suffix}"
    email = f"{uname}@example.com"
    pwd = "ApiPassword123!"

    # 1. Register
    reg_resp = client.post(
        "/api/auth/register",
        json={"username": uname, "email": email, "password": pwd},
    )
    assert reg_resp.status_code == 200, reg_resp.text
    reg_data = reg_resp.json()
    token = reg_data["token"]
    user_id = reg_data["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Login
    login_resp = client.post(
        "/api/auth/login",
        json={"username": uname, "password": pwd},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["user"]["username"] == uname

    # 3. /api/auth/me
    me_resp = client.get("/api/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["authenticated"] is True
    assert me_resp.json()["user"]["id"] == user_id

    # 4. Create and retrieve conversation
    sess = f"sess_api_{suffix}"
    chat_history_db.append_message(sess, "user", "What is quantum computing?", user_id=user_id)
    chat_history_db.append_message(sess, "assistant", "Quantum computing uses qubits.", user_id=user_id)

    list_resp = client.get("/api/conversations", headers=headers)
    assert list_resp.status_code == 200
    conv_list = list_resp.json()["conversations"]
    assert any(c["session_id"] == sess for c in conv_list)

    # 5. Fetch messages
    msg_resp = client.get(f"/api/conversations/{sess}")
    assert msg_resp.status_code == 200
    assert len(msg_resp.json()["messages"]) == 2

    # 6. Patch title
    patch_resp = client.patch(f"/api/conversations/{sess}", json={"title": "Quantum Basics"})
    assert patch_resp.status_code == 200

    # 7. Delete conversation
    del_resp = client.delete(f"/api/conversations/{sess}", headers=headers)
    assert del_resp.status_code == 200

