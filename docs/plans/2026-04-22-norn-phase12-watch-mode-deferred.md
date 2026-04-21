# Norn — Phase 12 (deferred) : `watch` mode

**Date:** 2026-04-22
**Status:** **Deferred** — concept noted, design not yet clear
**Branch:** `chore/planning-future-phases`

---

## Origin

Suggested during a Phase 9 v1 close-out review by an external code-scanner
agent: a "watch" mode that listens to file-system modifications and
auto-triggers Dream / Coordinator / arbitrary Norn actions.

## Why deferred

The user (project owner) explicitly chose to skip detailed design for
this phase: the use case is not yet clear enough to commit to an
architecture without risking scope creep, a wrong abstraction, or worse
— an auto-trigger that loops on its own outputs.

Concrete questions left unresolved:

1. **What triggers what?** Several candidate use cases were sketched
   (dream-after-inactivity, auto-fix on red tests, Obsidian
   marker-trigger, index maintenance, generic hook framework). None
   stood out as the obvious right one without more usage data.
2. **Loop safety** — any mode that lets Norn react to file changes by
   modifying files needs a robust loop-prevention mechanism (debouncing,
   self-modification detection, max iterations, manual kill switch).
3. **Cost containment** — auto-triggers can silently rack up LLM cost.
4. **Scope boundaries** — which directories, which file patterns, which
   gitignored paths to skip.
5. **Coexistence with editor / IDE / git hooks** — not stepping on
   existing tooling.

## When to revisit

Revisit when one of the following becomes true:

- Phase 10 (`obsidian_rag`) lands and the user feels a clear need for
  "auto-update my obsidian_rag index when I write a new note in
  Obsidian" — this is the lightest-risk first step.
- Phase 11 (mini-eval benchmark) lands and the user wants to "auto-bench
  on every commit to `main`".
- The Dream system gets enough adoption that "trigger Dream when I stop
  coding for 30 min" becomes a clearly desired ergonomic.
- The user files a concrete pain point that a watch mode would solve.

## Adjacent prior art to study before designing

Before committing to a design, scan how comparable tools solved this:

- **`entr`** (Unix utility) — minimalist, command-on-change.
- **`watchman`** (Facebook) — robust filesystem watcher with subscriptions.
- **`tox-dev/pytest-watch`** — debounced pytest re-run.
- **Aider's `--watch-files`** mode.
- **Cursor "background agents"** — their UX for async agent triggers.
- **VS Code tasks `runOptions.runOn`** — IDE-level event-driven actions.

## Non-decisions

This file deliberately does **not** sketch a target architecture, an
event model, a config schema, or a CLI surface. Doing so prematurely
would anchor future thinking on assumptions that may not survive
contact with real usage.

## Status

Reserved as Phase 12 to keep the numbering free. Re-open with a full
brainstorm + design when the trigger condition above fires.

---

*Approved by user (FR session, 2026-04-22).*
