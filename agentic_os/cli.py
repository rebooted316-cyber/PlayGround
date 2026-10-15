"""Command-line interface: ``agentic-os <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from agentic_os import roadmap
from agentic_os.config import Settings


def _build_os(args: argparse.Namespace):
    from agentic_os.kernel import AgenticOS

    settings = Settings()
    if getattr(args, "provider", None):
        settings.provider = args.provider
    if getattr(args, "model", None):
        settings.model = args.model
    system = AgenticOS(settings)
    for path in getattr(args, "docs", None) or []:
        chunks = system.ingest(path)
        print(f"ingested {path} ({chunks} chunks)", file=sys.stderr)
    return system


def cmd_roadmap(args: argparse.Namespace) -> int:
    print(roadmap.render())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    with _build_os(args) as system:
        if args.goal:
            result = system.goal(args.task)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, default=str))
            else:
                for t in result.plan.tasks:
                    print(f"  [{t.status}] {t.id}: {t.description}", file=sys.stderr)
                print(result.output)
            return 0
        proc = system.run(args.task)
        if args.json:
            print(json.dumps(proc.to_dict(), indent=2, default=str))
        elif proc.status != "completed":
            print(f"error: {proc.error}", file=sys.stderr)
            return 1
        else:
            for call in proc.result["tool_calls"]:
                print(f"  -> {call['tool']}({json.dumps(call['arguments'])})", file=sys.stderr)
            print(proc.result["output"])
        return 0 if proc.status == "completed" else 1


def cmd_chat(args: argparse.Namespace) -> int:
    system = _build_os(args)
    agent = system.spawn("chat", conversational=True)
    print(f"Agentic OS chat ({system.status()['provider']}). Ctrl-D or /exit to quit.")
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if line in {"/exit", "/quit"}:
                break
            if line == "/status":
                print(json.dumps(system.status(), indent=2))
                continue
            if line:
                print(f"agent> {agent.run(line).output}")
    finally:
        system.shutdown()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocking
    from agentic_os.server import serve

    serve(_build_os(args), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-os", description="An operating system for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["auto", "anthropic", "mock"], help="LLM provider (default: auto)")
        p.add_argument("--model", help="model id (default: claude-opus-5-5)")
        p.add_argument("--docs", nargs="*", metavar="FILE", help="files to ingest into the knowledge base")

    p = sub.add_parser("roadmap", help="show how the 12 roadmap stages map to modules")
    p.set_defaults(func=cmd_roadmap)

    p = sub.add_parser("run", help="run one task")
    p.add_argument("task")
    p.add_argument("--goal", action="store_true", help="plan and execute as a multi-step goal")
    p.add_argument("--json", action="store_true", help="print the full JSON result")
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("chat", help="interactive multi-turn chat")
    common(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    common(p)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
