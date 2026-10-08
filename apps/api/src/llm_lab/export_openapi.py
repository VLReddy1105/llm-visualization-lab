import argparse
import json
from pathlib import Path

from llm_lab.config import Settings
from llm_lab.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export implemented Stage 1–3 OpenAPI without loading Qwen"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema = create_app(Settings(auto_load=False)).openapi()
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != text:
            raise SystemExit("OpenAPI snapshot differs; regenerate it and TypeScript types.")
        print("OpenAPI snapshot matches implemented Pydantic routes.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
