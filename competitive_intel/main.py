import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from competitive_intel.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a competitive intelligence briefing for any industry."
    )
    parser.add_argument("--industry", required=True, help="Industry to monitor (e.g. 'B2B SaaS')")
    parser.add_argument("--companies", nargs="+", required=True, help="Companies to track")
    parser.add_argument("--niche", default="", help="Optional niche within the industry")
    parser.add_argument(
        "--provider",
        choices=["claude", "openai"],
        default="claude",
        help="LLM backend to use (default: claude)",
    )
    parser.add_argument("--model", default=None, help="Override the default model for the selected provider")
    return parser


def run_briefing(config: Config) -> dict:
    from competitive_intel.graph import build_graph

    graph = build_graph(config)
    initial_state = {
        "companies": config.companies,
        "researcher_output": [],
        "analyst_output": None,
        "final_briefing": "",
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    briefing = final_state.get("final_briefing", "")
    errors = final_state.get("errors", [])

    print(briefing)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"briefing_{date.today().isoformat()}.md"
    output_file.write_text(briefing, encoding="utf-8")

    if errors:
        print("\n--- ERRORS ---")
        for err in errors:
            print(f"  • {err}")

    return final_state


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = Config(
        industry=args.industry,
        companies=args.companies,
        niche=args.niche,
        provider=args.provider,
        model=args.model,
    )
    run_briefing(config)


if __name__ == "__main__":
    main()
