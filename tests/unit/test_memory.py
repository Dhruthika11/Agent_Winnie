"""Unit tests for the Mem0-backed memory manager (noop-backend safe)."""
from src.memory.manager import Mem0Manager


def test_search_with_no_backend_returns_empty_list():
    manager = Mem0Manager()
    manager._client = None
    assert manager.search(user_id="u1", query="anything") == []


def test_add_with_no_backend_does_not_raise():
    manager = Mem0Manager()
    manager._client = None
    manager.add(user_id="u1", messages=[{"role": "user", "content": "hi"}])  # no exception


def test_get_all_with_no_backend_returns_empty_list():
    manager = Mem0Manager()
    manager._client = None
    assert manager.get_all(user_id="u1") == []


def test_delete_with_no_backend_returns_false():
    manager = Mem0Manager()
    manager._client = None
    assert manager.delete(user_id="u1", memory_id="m1") is False
