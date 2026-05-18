import pytest
import sys
from competitive_intel.config import Config


class TestConfig:
    def test_required_fields(self):
        cfg = Config(industry="FinTech", companies=["Stripe", "Plaid"])
        assert cfg.industry == "FinTech"
        assert cfg.companies == ["Stripe", "Plaid"]

    def test_defaults(self):
        cfg = Config(industry="FinTech", companies=["Stripe"])
        assert cfg.niche == ""
        assert cfg.search_focus == []
        assert cfg.provider == "claude"
        assert cfg.model is None

    def test_niche_empty_string_not_none(self):
        cfg = Config(industry="FinTech", companies=["Stripe"])
        assert cfg.niche == ""
        assert cfg.niche is not None

    def test_single_company_is_list(self):
        cfg = Config(industry="FinTech", companies=["Stripe"])
        assert isinstance(cfg.companies, list)
        assert cfg.companies == ["Stripe"]

    def test_openai_provider(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="openai")
        assert cfg.provider == "openai"
        assert cfg.model is None

    def test_model_override(self):
        cfg = Config(industry="FinTech", companies=["Stripe"], provider="openai", model="gpt-4-turbo")
        assert cfg.model == "gpt-4-turbo"

    def test_search_focus_independent_per_instance(self):
        cfg1 = Config(industry="A", companies=["X"])
        cfg2 = Config(industry="B", companies=["Y"])
        cfg1.search_focus.append("pricing")
        assert cfg2.search_focus == []


class TestCLIParsing:
    def _parse(self, args):
        from competitive_intel.main import build_parser
        parser = build_parser()
        return parser.parse_args(args)

    def test_required_args(self):
        args = self._parse(["--industry", "FinTech", "--companies", "Stripe", "Plaid"])
        assert args.industry == "FinTech"
        assert args.companies == ["Stripe", "Plaid"]
        assert args.niche == ""
        assert args.provider == "claude"
        assert args.model is None

    def test_provider_openai(self):
        args = self._parse(["--industry", "FinTech", "--companies", "Stripe", "--provider", "openai"])
        assert args.provider == "openai"

    def test_model_override(self):
        args = self._parse(["--industry", "FinTech", "--companies", "Stripe", "--provider", "openai", "--model", "gpt-4-turbo"])
        assert args.model == "gpt-4-turbo"

    def test_single_company_produces_list(self):
        args = self._parse(["--industry", "FinTech", "--companies", "Stripe"])
        assert args.companies == ["Stripe"]

    def test_missing_industry_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["--companies", "Stripe"])

    def test_missing_companies_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["--industry", "FinTech"])

    def test_invalid_provider_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["--industry", "FinTech", "--companies", "Stripe", "--provider", "gemini"])
