#!/usr/bin/env python3
"""
VoiceAppAgent — Voice-enabled smartphone agent.

Unified CLI entry point with subcommands for exploration, task execution,
and demo recording. Supports voice input for hands-free operation.

Usage:
    python main.py explore --app Twitter [--voice]
    python main.py run --app Twitter [--voice]
    python main.py demo --app Twitter [--voice]
    python main.py generate-docs --app Twitter --demo demo_Twitter_2024-01-01_12-00-00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_explore(args: argparse.Namespace) -> None:
    """Run autonomous exploration."""
    from agents.explorer import ExplorerAgent

    agent = ExplorerAgent(
        app_name=args.app,
        root_dir=args.root_dir,
        voice_enabled=args.voice,
        config_path=args.config,
    )
    agent.run()


def cmd_run(args: argparse.Namespace) -> None:
    """Run task execution (deployment phase)."""
    from agents.task_runner import TaskRunnerAgent

    agent = TaskRunnerAgent(
        app_name=args.app,
        root_dir=args.root_dir,
        voice_enabled=args.voice,
        config_path=args.config,
    )
    agent.run()


def cmd_demo(args: argparse.Namespace) -> None:
    """Run human demonstration recording."""
    from agents.demo_recorder import DemoRecorderAgent

    agent = DemoRecorderAgent(
        app_name=args.app,
        root_dir=args.root_dir,
        voice_enabled=args.voice,
        config_path=args.config,
    )
    agent.run()


def cmd_generate_docs(args: argparse.Namespace) -> None:
    """Generate documentation from a recorded demo."""
    from agents.doc_generator import DocGeneratorAgent

    agent = DocGeneratorAgent(
        app_name=args.app,
        demo_name=args.demo,
        root_dir=args.root_dir,
        voice_enabled=args.voice,
        config_path=args.config,
    )
    agent.run()


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="VoiceAppAgent",
        description=(
            "🎤 VoiceAppAgent — A voice-enabled AI agent that operates "
            "smartphone apps through natural language commands."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py explore --app Twitter --voice    # Voice-driven exploration
  python main.py run --app Twitter --voice        # Voice-driven task execution
  python main.py demo --app Twitter               # Record a demo (keyboard)
  python main.py generate-docs --app Twitter --demo demo_Twitter_2024-01-01
        """,
    )

    # Global arguments
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--root-dir", dest="root_dir", default="./",
        help="Root directory for app data (default: ./)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── Explore subcommand ──────────────────────────────────────────────
    explore_parser = subparsers.add_parser(
        "explore",
        help="Autonomous exploration to generate UI documentation",
    )
    explore_parser.add_argument("--app", required=True, help="Target app name")
    explore_parser.add_argument(
        "--voice", action="store_true", help="Enable voice input"
    )
    explore_parser.set_defaults(func=cmd_explore)

    # ─── Run subcommand ──────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a task on an app (deployment phase)",
    )
    run_parser.add_argument("--app", required=True, help="Target app name")
    run_parser.add_argument(
        "--voice", action="store_true", help="Enable voice input"
    )
    run_parser.set_defaults(func=cmd_run)

    # ─── Demo subcommand ─────────────────────────────────────────────────
    demo_parser = subparsers.add_parser(
        "demo",
        help="Record a human demonstration",
    )
    demo_parser.add_argument("--app", required=True, help="Target app name")
    demo_parser.add_argument(
        "--voice", action="store_true", help="Enable voice input"
    )
    demo_parser.set_defaults(func=cmd_demo)

    # ─── Generate Docs subcommand ────────────────────────────────────────
    gendocs_parser = subparsers.add_parser(
        "generate-docs",
        help="Generate documentation from a recorded demo",
    )
    gendocs_parser.add_argument("--app", required=True, help="Target app name")
    gendocs_parser.add_argument("--demo", required=True, help="Demo name to process")
    gendocs_parser.add_argument(
        "--voice", action="store_true", help="Enable voice input"
    )
    gendocs_parser.set_defaults(func=cmd_generate_docs)

    # Parse and dispatch
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Banner
    print(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║  🎤 VoiceAppAgent                               ║\n"
        "║  Voice-enabled AI smartphone agent               ║\n"
        "╚══════════════════════════════════════════════════╝\n"
    )

    args.func(args)


if __name__ == "__main__":
    main()
