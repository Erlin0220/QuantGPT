"""Deprecated compatibility entrypoint.

QuantGPT no longer hosts or proxies a DeepSeek/OpenAI-compatible LLM. Generative
reasoning belongs to the connected ChatGPT client; QuantGPT exposes deterministic
MCP tools for expression validation, research, simulation, diagnosis, and storage.
"""


def main() -> None:
    raise SystemExit(
        "mcp_deepseek has been removed from QuantGPT. Use the connected ChatGPT client with QuantGPT MCP tools instead."
    )


if __name__ == "__main__":
    main()
