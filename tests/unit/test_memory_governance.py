"""
Comprehensive Unit Tests for Memory Governance and Lifecycle Management.

Tests:
1. Memory Classification Taxonomy (preference, fact, project, temporary_context).
2. Lifecycle Expiration Policies (temporary_context TTL vs preference TTL).
3. Contradiction Detection & Automatic Superseding (version bumps, preference flips, DB swaps).
4. Validation & Deduplication (chatter rejection, duplicate skipping).
5. Strict Multi-Tenant Isolation (tenant A cannot retrieve, update, or delete tenant B's memories).
6. User-Controlled Memory Correction (CRUD: get, update, delete, purge).
7. Memory Governance REST API Endpoints (/api/memory GET, POST, PUT, DELETE).
"""
import datetime
import pytest
from starlette.testclient import TestClient

from src.main import app
from src.memory.manager import (
    classify_memory_text,
    detect_memory_contradiction,
    memory_manager,
)
from src.memory.models import (
    CATEGORY_DEFAULT_TTL_DAYS,
    MemoryCategory,
    MemoryItem,
)


@pytest.fixture(autouse=True)
def clean_memory_store():
    """Ensure in-memory store is clean before each test."""
    with memory_manager._lock:
        memory_manager._mem_store.clear()
    yield


client = TestClient(app)


# =========================================================================== #
# Pillar 1: Classification & Lifecycle Expiration
# =========================================================================== #

def test_memory_classification_taxonomy():
    """Verify that memories are categorized into preference, fact, project, and temporary_context."""
    cat, conf = classify_memory_text("I prefer dark mode and concise responses")
    assert cat == MemoryCategory.PREFERENCE.value
    assert conf >= 0.90

    cat, conf = classify_memory_text("Our primary database is PostgreSQL 16")
    assert cat == MemoryCategory.FACT.value
    assert conf >= 0.90

    cat, conf = classify_memory_text("Working on Hermes agent platform architecture repository")
    assert cat == MemoryCategory.PROJECT.value
    assert conf >= 0.90

    cat, conf = classify_memory_text("Currently debugging auth token refresh today")
    assert cat == MemoryCategory.TEMPORARY_CONTEXT.value
    assert conf >= 0.90


def test_memory_lifecycle_expiration_policy():
    """Verify that temporary_context expires quickly while preference context is durable."""
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Temporary context item: default TTL is 2 days
    temp_item = MemoryItem(
        id="mem_temp_1",
        user_id="user_1",
        text="Debugging token issue today",
        category="temporary_context",
    )
    assert temp_item.expires_at is not None
    # Still valid right now
    assert temp_item.is_expired(now) is False
    # Expired 5 days from now
    five_days_later = now + datetime.timedelta(days=5)
    assert temp_item.is_expired(five_days_later) is True

    # 2. Preference item: durable
    pref_item = MemoryItem(
        id="mem_pref_1",
        user_id="user_1",
        text="Always use Python with strict typing",
        category="preference",
    )
    assert pref_item.is_expired(now) is False
    # Valid 30 days later
    assert pref_item.is_expired(now + datetime.timedelta(days=30)) is False


# =========================================================================== #
# Pillar 2: Contradiction Detection & Automatic Superseding
# =========================================================================== #

def test_contradiction_detection_heuristics():
    """Verify contradiction detection identifies version updates, preference flips, and technology swaps."""
    # 1. Version update
    is_contra, reason = detect_memory_contradiction("Using Python 3.12", "Using Python 3.10")
    assert is_contra is True
    assert "3.10 to 3.12" in reason or "Version updated" in reason

    # 2. Preference flip
    is_contra, reason = detect_memory_contradiction("I prefer spaces for indent", "I prefer tabs for indent")
    assert is_contra is True
    assert "tabs and spaces" in reason.lower()

    # 3. Database swap
    is_contra, reason = detect_memory_contradiction("Switched database to MySQL", "Primary database is PostgreSQL")
    assert is_contra is True
    assert "database" in reason.lower()

    # 4. Non-contradictory complementary statements
    is_contra, _ = detect_memory_contradiction("Using FastAPI for web backend", "Using PostgreSQL for database")
    assert is_contra is False


def test_automatic_superseding_on_contradiction():
    """Verify that storing a contradictory memory deactivates and supersedes the older memory."""
    user_id = "user_dev"
    tenant_id = "tenant_main"

    # Step 1: Add initial preference
    r1 = memory_manager.add(
        user_id=user_id,
        messages=[{"role": "user", "content": "Our backend is running on Python 3.10"}],
        tenant_id=tenant_id,
    )
    assert len(r1) == 1
    assert r1[0].status == "created"
    old_id = r1[0].id

    # Verify active
    initial_memories = memory_manager.get_all(user_id=user_id, tenant_id=tenant_id)
    assert len(initial_memories) == 1
    assert initial_memories[0].is_active is True

    # Step 2: User provides updated version
    r2 = memory_manager.add(
        user_id=user_id,
        messages=[{"role": "user", "content": "We upgraded the backend to Python 3.12"}],
        tenant_id=tenant_id,
    )
    assert len(r2) == 1
    assert r2[0].status == "superseded"
    assert r2[0].superseded_memory_id == old_id
    new_id = r2[0].id

    # Step 3: Verify active memories only contains Python 3.12
    active_memories = memory_manager.get_all(user_id=user_id, tenant_id=tenant_id)
    assert len(active_memories) == 1
    assert active_memories[0].id == new_id
    assert "Python 3.12" in active_memories[0].text

    # Step 4: Verify older memory is archived as inactive and superseded
    all_memories = memory_manager.get_all(user_id=user_id, tenant_id=tenant_id, include_inactive=True)
    old_mem = [m for m in all_memories if m.id == old_id][0]
    assert old_mem.is_active is False
    assert old_mem.superseded_by == new_id


# =========================================================================== #
# Pillar 3: Validation & Deduplication
# =========================================================================== #

def test_transient_chatter_rejected():
    """Verify that trivial conversational chatter is not stored as permanent memory."""
    res = memory_manager.add(
        user_id="user_chatty",
        messages=[{"role": "user", "content": "Thank you! Sounds good."}],
    )
    assert len(res) == 1
    assert res[0].status == "rejected_transient"
    assert len(memory_manager.get_all(user_id="user_chatty")) == 0


def test_exact_duplicate_skipped():
    """Verify that exact duplicate memories are deduplicated rather than duplicated."""
    user_id = "user_dedup"
    msg = [{"role": "user", "content": "I like dark mode themes."}]

    r1 = memory_manager.add(user_id=user_id, messages=msg)
    assert r1[0].status == "created"

    r2 = memory_manager.add(user_id=user_id, messages=msg)
    assert r2[0].status == "skipped"

    memories = memory_manager.get_all(user_id=user_id)
    assert len(memories) == 1


# =========================================================================== #
# Pillar 4: Strict Multi-Tenant Isolation
# =========================================================================== #

def test_strict_multi_tenant_memory_isolation():
    """Verify that Tenant A's memories cannot be searched, retrieved, or deleted by Tenant B."""
    user_id = "shared_user_account"

    # Store memory for Tenant Alpha
    r_alpha = memory_manager.add(
        user_id=user_id,
        messages=[{"role": "user", "content": "Alpha project confidential secret server: alpha-1.corp"}],
        tenant_id="tenant_alpha",
    )
    id_alpha = r_alpha[0].id

    # Store memory for Tenant Beta
    r_beta = memory_manager.add(
        user_id=user_id,
        messages=[{"role": "user", "content": "Beta project internal database: beta-db.corp"}],
        tenant_id="tenant_beta",
    )
    id_beta = r_beta[0].id

    # 1. Tenant Alpha query cannot see Tenant Beta's data
    alpha_mems = memory_manager.get_all(user_id=user_id, tenant_id="tenant_alpha")
    assert len(alpha_mems) == 1
    assert alpha_mems[0].id == id_alpha
    assert "beta-db" not in alpha_mems[0].text

    # 2. Tenant Beta query cannot see Tenant Alpha's data
    beta_mems = memory_manager.get_all(user_id=user_id, tenant_id="tenant_beta")
    assert len(beta_mems) == 1
    assert beta_mems[0].id == id_beta
    assert "alpha-1" not in beta_mems[0].text

    # 3. Search isolation
    search_beta = memory_manager.search(user_id=user_id, query="confidential server", tenant_id="tenant_beta")
    assert len(search_beta) == 0  # Cannot find Alpha's server from Beta tenant!

    # 4. Cross-tenant deletion blocked
    deleted = memory_manager.delete(user_id=user_id, memory_id=id_alpha, tenant_id="tenant_beta")
    assert deleted is False
    assert len(memory_manager.get_all(user_id=user_id, tenant_id="tenant_alpha")) == 1


# =========================================================================== #
# Pillar 5: User-Controlled Memory Correction (CRUD API)
# =========================================================================== #

def test_user_controlled_memory_correction_and_crud():
    """Verify user can update, correct, delete, and purge memories."""
    user_id = "user_crud"

    # Create memory
    r = memory_manager.add(
        user_id=user_id,
        messages=[{"role": "user", "content": "I like tabs"}],
        category="preference",
    )
    mem_id = r[0].id

    # 1. User corrects memory
    updated = memory_manager.update(
        memory_id=mem_id,
        new_text="I prefer 4 spaces instead of tabs",
        user_id=user_id,
        category="preference",
    )
    assert updated is not None
    assert updated.text == "I prefer 4 spaces instead of tabs"
    assert updated.verification_status == "user_confirmed"

    # Verify retrieval reflects update
    fetched = memory_manager.get_by_id(mem_id, user_id=user_id)
    assert fetched is not None
    assert fetched.text == "I prefer 4 spaces instead of tabs"

    # 2. User deletes memory
    deleted = memory_manager.delete(user_id=user_id, memory_id=mem_id)
    assert deleted is True
    assert memory_manager.get_by_id(mem_id, user_id=user_id) is None


def test_memory_rest_api_endpoints():
    """Verify REST API endpoints for governed memory management."""
    user_id = "user_api_test"
    tenant_id = "tenant_prod"

    # 1. POST /api/memory
    add_payload = {
        "user_id": user_id,
        "content": "Our production Kubernetes cluster runs in us-east-1",
        "tenant_id": tenant_id,
        "category": "fact",
    }
    res_add = client.post("/api/memory", json=add_payload)
    assert res_add.status_code == 200
    data = res_add.json()
    assert len(data["results"]) == 1
    mem_id = data["results"][0]["id"]
    assert data["results"][0]["category"] == "fact"

    # 2. GET /api/memory
    res_get = client.get(f"/api/memory?user_id={user_id}&tenant_id={tenant_id}")
    assert res_get.status_code == 200
    mems = res_get.json()["memories"]
    assert len(mems) == 1
    assert mems[0]["id"] == mem_id

    # 3. PUT /api/memory/{id} (Correction)
    update_payload = {
        "user_id": user_id,
        "new_text": "Our production Kubernetes cluster runs in us-west-2 (migrated)",
        "tenant_id": tenant_id,
    }
    res_put = client.put(f"/api/memory/{mem_id}", json=update_payload)
    assert res_put.status_code == 200
    assert "us-west-2" in res_put.json()["memory"]["text"]

    # 4. DELETE /api/memory/{id}
    res_del = client.delete(f"/api/memory/{mem_id}?user_id={user_id}&tenant_id={tenant_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # Verify deletion
    res_after = client.get(f"/api/memory?user_id={user_id}&tenant_id={tenant_id}")
    assert len(res_after.json()["memories"]) == 0
