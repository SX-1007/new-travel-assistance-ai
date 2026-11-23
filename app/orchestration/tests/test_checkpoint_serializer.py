"""Checkpoints are loaded strictly: a stored value can't make the server
build an arbitrary class, only LangGraph's safe types (messages, dates)."""

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from app.orchestration.checkpoints import checkpoint_serializer


class Planted(BaseModel):
    """Stands in for a class an attacker names in tampered checkpoint data."""

    command: str = "rm -rf /"


def test_an_unknown_class_in_a_checkpoint_is_not_rebuilt():
    tampered = JsonPlusSerializer().dumps_typed(Planted())

    loaded = checkpoint_serializer().loads_typed(tampered)

    assert not isinstance(loaded, Planted)
    assert loaded == {"command": "rm -rf /"}


def test_the_state_the_graph_stores_round_trips():
    serializer = checkpoint_serializer()
    value = {
        "messages": [
            HumanMessage(content="Swap day 2 lunch"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "edit_itinerary", "args": {"day": 2}}]),
            ToolMessage(content='{"ok": true}', tool_call_id="c1"),
        ],
        "draft_itinerary": [{"day": 1, "activities": [{"name": "Senso-ji"}]}],
        "fetched_at": datetime(2026, 11, 1, 9, 30),
    }

    assert serializer.loads_typed(serializer.dumps_typed(value)) == value
