# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SOTA v2 wave 1 (2026-08-28):
  - Append-only thread invariant (`ThreadLedger`, strict mode in CI) and
    LLM record/replay (`--record`, `ReplayProvider`, zero-API-key fixtures).
  - Fail-closed bash sandbox via macOS seatbelt (`sandbox` config block,
    read-only / workspace-write / danger-full-access policies, escalation
    prompt, `sandbox.decision` event).
  - Byte-stable system prompt (`agent.stable_prompt`) with split
    `cache_control` blocks; cache metrics on the streaming path
    (`llm.complete` now emitted for interactive sessions).
  - CodeRAG tools wired behind the `coderag` flag; LLM-as-judge (`--judge`)
    and per-task session metrics in `norn bench run`; config-driven bench
    model/tiers and `--n-runs`; `norn bench guard` regression safety net
    for self-improvement lessons (W3.3).
- SOTA roadmap execution (2026-06→08): prompt caching + transient-error
  retry, marker-based bash polling, `/plan` `/proof` `/reflect`, CodeRAG
  backend (BM25+vector hybrid, call-graph, token budget), Docker bench
  sandbox, Harbor terminal-bench adapter, `auto_verify` self-verification,
  `image_read` vision tool.
- Initial public release scaffolding (LICENSE, CHANGELOG).
- Design docs for Phase 10 (`obsidian_rag` tool), Phase 11 (internal benchmark),
  and a deferral note for Phase 12 (watch mode).

## [0.1.0] - TBD

Initial release. See `docs/plans/` for the implementation history of phases 1–9.

[Unreleased]: https://github.com/gregoire-petit1/norn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gregoire-petit1/norn/releases/tag/v0.1.0
