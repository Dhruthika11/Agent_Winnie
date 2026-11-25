"""
Unit test verifying that stream_engine is a pure streaming adapter over agent_graph.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agent.stream_engine import stream_agent_workflow


@pytest.mark.anyio
async def test_stream_agent_workflow_delegates_to_agent_graph():
    # Mock agent_graph invocation
    mock_result = {
        "messages": [{"role": "assistant", "content": "Hello from LangGraph!"}],
        "iteration": 1,
        "done": True,
    }

    def fake_invoke(state, config=None):
        q = config.get("configurable", {}).get("event_queue") if config else None
        if q:
            q.put({"type": "token", "delta": "Hello from "})
            q.put({"type": "token", "delta": "LangGraph!"})
        return mock_result

    with patch("src.agent.stream_engine.agent_graph.invoke", side_effect=fake_invoke) as mock_graph_invoke, \
         patch("src.agent.stream_engine.chat_history_db.append_message") as mock_db:

        events = []
        async for sse in stream_agent_workflow(message="Hi", user_id="test_user", session_id="test_sess"):
            events.append(sse)

        # Verify LangGraph invoke was called
        assert mock_graph_invoke.call_count == 1
        args, kwargs = mock_graph_invoke.call_args
        assert args[0]["user_id"] == "test_user"
        assert args[0]["session_id"] == "test_sess"
        assert "event_queue" in kwargs["config"]["configurable"]

        # Verify SSE events were streamed from the queue
        combined_sse = "".join(events)
        assert "event: session_init" in combined_sse
        assert "event: token" in combined_sse
        assert "Hello from " in combined_sse
        assert "event: done" in combined_sse
        assert "Hello from LangGraph!" in combined_sse
