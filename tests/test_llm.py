import pytest
from unittest.mock import patch, MagicMock
from competitive_intel.config import Config
from competitive_intel.llm import get_llm


class TestGetLlm:
    def test_claude_returns_chat_anthropic(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="claude")
        mock_cls = MagicMock()
        with patch("competitive_intel.llm.ChatAnthropic", mock_cls, create=True):
            # Re-import to pick up mock
            import importlib
            import competitive_intel.llm as llm_mod
            with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
                result = llm_mod.get_llm(cfg, 0.1)
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514", temperature=0.1)

    def test_openai_returns_chat_openai(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="openai")
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}):
            import competitive_intel.llm as llm_mod
            result = llm_mod.get_llm(cfg, 0.2)
        mock_cls.assert_called_once_with(model="gpt-4o", temperature=0.2)

    def test_model_override_claude(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="claude", model="claude-opus-4-5")
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
            import competitive_intel.llm as llm_mod
            llm_mod.get_llm(cfg, 0.1)
        mock_cls.assert_called_once_with(model="claude-opus-4-5", temperature=0.1)

    def test_model_override_openai(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="openai", model="gpt-4-turbo")
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_cls)}):
            import competitive_intel.llm as llm_mod
            llm_mod.get_llm(cfg, 0.4)
        mock_cls.assert_called_once_with(model="gpt-4-turbo", temperature=0.4)

    def test_unknown_provider_raises_value_error(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="gemini")
        with pytest.raises(ValueError, match="Unknown provider 'gemini'"):
            get_llm(cfg, 0.1)

    def test_unknown_provider_lists_valid_choices(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="bedrock")
        with pytest.raises(ValueError, match="claude"):
            get_llm(cfg, 0.1)

    def test_none_model_uses_claude_default(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="claude", model=None)
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_cls)}):
            import competitive_intel.llm as llm_mod
            llm_mod.get_llm(cfg, 0.1)
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514", temperature=0.1)
