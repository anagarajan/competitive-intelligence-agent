from dataclasses import dataclass, field


@dataclass
class Config:
    industry: str
    companies: list[str]
    niche: str = ""
    search_focus: list[str] = field(default_factory=list)
    provider: str = "claude"
    model: str | None = None
