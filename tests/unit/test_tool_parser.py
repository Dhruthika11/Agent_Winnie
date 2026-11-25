import pytest
from src.agent.tool_parser import parse_model_tool_calls, strip_unwanted_tool_xml


def test_parse_xml_tool_call():
    text = (
        "Let me check that for you.\n"
        "<tool_call>\n"
        "<function=github.list_user_repositories>\n"
        "<parameter=username>\n"
        "srujansai\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    cleaned, calls = parse_model_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "github.list_user_repositories"
    assert calls[0]["arguments"] == {"username": "srujansai"}
    assert "Let me check that for you." in cleaned


def test_parse_github_repo_normalization():
    text = (
        "<tool_call>\n"
        "<function=github.get_repository>\n"
        "<parameter=name>\n"
        "Adaptoid-OS Srujan0798\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    cleaned, calls = parse_model_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "github.get_repository"
    assert calls[0]["arguments"] == {"repo": "Adaptoid-OS Srujan0798"}


def test_strip_unwanted_tool_xml():
    raw = "Adaptoid-OS Srujan0798 <tool_call><function=test></function></tool_call>"
    assert strip_unwanted_tool_xml(raw) == "Adaptoid-OS Srujan0798 "
    # Crucially verify that inter-word spacing is preserved across chunks
    tokens = ["I", " cannot", " browse", " the", " web"]
    assert "".join([strip_unwanted_tool_xml(t) for t in tokens]) == "I cannot browse the web"

