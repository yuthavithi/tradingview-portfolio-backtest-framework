---
name: rtk
description: "Rust Token Killer (RTK) is a token-optimized CLI proxy that filters and compresses command outputs to minimize LLM token usage."
---

# RTK (Rust Token Killer) Skill

This skill integrates the RTK CLI proxy tool into the development environment to reduce token consumption during terminal command execution by 60–90%.

## Usage Guide

To optimize context window usage and avoid hitting LLM API rate limits, always prefix shell commands with `rtk` when running them.

### Common Commands

Instead of running raw shell commands, prepend them with `rtk`:

- **Git Operations**: `rtk git status`, `rtk git diff`, `rtk git log -n 5`
- **Testing**: `rtk pytest`, `rtk test`
- **File System**: `rtk ls`, `rtk tree`, `rtk find`
- **Search**: `rtk grep "pattern"`, `rtk rg "pattern"`
- **Package Management**: `rtk npm install`, `rtk pip install`, `rtk uv run`
- **Containerization**: `rtk docker ps`, `rtk docker logs`

### Meta & Diagnostic Commands

Use these commands to manage, inspect, or troubleshoot RTK performance:

- `rtk gain`: Show token savings summary.
- `rtk gain --history`: View detailed command execution history and savings.
- `rtk discover`: Scan execution history to discover missed RTK optimization opportunities.
- `rtk proxy <command>`: Run the command raw (bypassing filters) for debugging purposes while still tracking statistics.

## Project Setup

RTK is configured for the Google Antigravity agent in this repository. 
- Rules file: [.agents/rules/antigravity-rtk-rules.md](file:///c:/Documents/Development/TradingView Portfolio Backtest Framework/.agents/rules/antigravity-rtk-rules.md)
- Executable path: `C:\rtk\rtk.exe`
