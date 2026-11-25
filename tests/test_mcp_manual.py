"""
Manual Test Harness for Real Anthropic MCP Servers.

Run this script directly to test any or all MCP servers:
    .venv\\Scripts\\python tests/test_mcp_manual.py
"""
import json
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.tools.mcp_client import mcp_registry


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_tool(tool_name: str, args: dict):
    print(f"\n▶ Calling Tool: \033[1;36m{tool_name}\033[0m")
    print(f"  Arguments: {json.dumps(args, indent=2)}")
    try:
        result = mcp_registry.call_tool(tool_name, args)
        print("  \033[1;32m✔ Success Response:\033[0m")
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2))
        else:
            print(f"  {result}")
        return True
    except Exception as exc:
        print(f"  \033[1;31m✘ Error:\033[0m {exc.__class__.__name__}: {exc}")
        return False


def run_all_tests():
    print_section("1. Testing Filesystem MCP Server")
    test_tool("filesystem.write_file", {"path": "manual_test.txt", "content": "Manual test content from MCP harness!"})
    test_tool("filesystem.read_file", {"path": "manual_test.txt"})
    test_tool("filesystem.list_directory", {"path": "."})

    print_section("2. Testing Terminal MCP Server")
    test_tool("terminal.run", {"command": "echo Real MCP Terminal Process is active!"})
    test_tool("terminal.delete_file", {"path": "manual_test.txt"})

    print_section("3. Testing GitHub MCP Server")
    test_tool("github.get_repository", {"owner": "octocat", "repo": "Hello-World"})
    test_tool("github.search_repositories", {"query": "language:python topic:agent", "limit": 2})

    print_section("4. Testing Web Scraper MCP Server")
    test_tool("web_scraper.search", {"query": "Model Context Protocol", "max_results": 2})
    test_tool("web_scraper.read_url", {"url": "https://example.com"})

    print_section("Summary: All MCP Server Tests Completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow testing a single tool from command line:
        # python tests/test_mcp_manual.py filesystem.read_file '{"path": "real_mcp_verified.txt"}'
        tool = sys.argv[1]
        raw_args = sys.argv[2] if len(sys.argv) > 2 else "{}"
        try:
            parsed_args = json.loads(raw_args)
        except Exception:
            parsed_args = {}
        test_tool(tool, parsed_args)
    else:
        run_all_tests()
