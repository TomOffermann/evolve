# Claude Knowledge Base — Rules

## Purpose
This folder is Claude's persistent, version-controlled context for the `evolve` project. Written and maintained by Claude to preserve a clear mental model across long conversations and sessions.

## Structure

| File / Folder | Purpose |
|---|---|
| `rules.md` | This file. Meta-rules for the knowledge base. |
| `overview.md` | High-level project snapshot: vision, goals, current state, directory layout. |
| `context.md` | Current focus, open questions, short-term priorities. Rewritten frequently. |
| `decisions/` | Design and architectural decisions — *why* choices were made, not just *what*. |

Research notes (papers, problems, proposals, experiments) live in the **top-level `research/`
folder**, not here. One location, not two. `claude/` holds meta-context and decisions only.

## Rules

### 1. Truth over completeness
Only write what is actually known and verified. An incomplete file is better than a stale or incorrect one.

### 2. Why over what
When recording a decision or direction, always include the reason. The code shows *what* was built; the knowledge base explains *why*.

### 3. Update on change
When a significant decision is made, a research direction shifts, or the project structure changes — update the relevant file in the same session, not later.

### 4. Keep it lean
No summaries of things obvious from the code or git history. No transcripts. Only context that would otherwise be lost between sessions.

### 5. One file, one concern
Each file covers exactly one concern. If a file exceeds ~100 lines, consider splitting it or pruning stale content.

### 6. `context.md` is ephemeral
Tracks what is *currently* in focus. Expected to be rewritten frequently. Historical context belongs in `decisions/` or `research/`, not here.

### 7. Distinguish settled from open
Use **Settled** / **Open** labels (or equivalent) when writing about decisions or research directions, so the difference between "we decided this" and "we're still figuring this out" is always clear.
