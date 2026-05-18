from competitive_intel.config import Config

_CLAUDE_DEFAULT = "claude-sonnet-4-20250514"
_OPENAI_DEFAULT = "gpt-4o"
_VALID_PROVIDERS = ("claude", "openai")


def get_llm(config: Config, temperature: float):
    """Return the appropriate LangChain chat model based on config.provider."""
    if config.provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model or _CLAUDE_DEFAULT,
            temperature=temperature,
        )
    elif config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model or _OPENAI_DEFAULT,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown provider '{config.provider}'. "
            f"Valid choices: {', '.join(_VALID_PROVIDERS)}"
        )
