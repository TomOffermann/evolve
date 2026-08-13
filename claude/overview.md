# Project Overview — Evolve

**Status:** Early-stage, experimental  
**Started:** ~May 2026  
**Author:** Tom Offermann (toffermann@ethz.ch)

## Core Thesis (from README)
Current AI/ML/DL systems are probability distribution approximators — nothing more at the conceptual level. The "intelligence" is in sampling from an approximated distribution. Tom's term for this: **Statistical Illusion (SI)**. He does not dismiss the engineering achievement, but takes this framing as the starting point for the project.

## What Is Evolve?
**Partly settled (2026-08-12).** The first concrete research program is defined: use
**EGGROLL**-style low-rank Evolution Strategies as the optimizer substrate, and port
diversity mechanisms from the GA / nature-inspired side (starting with **DEGA/DiPEC**) onto it.
The central open problem is finding a continuous diversity metric to replace DEGA's Hamming
distance. See `decisions/0001-substrate-and-scope.md` and `research/`.

Still **open**: whether Evolve is ultimately a framework, a paper, or a long-running notebook.
The README vision section is also still unfinished.

## Directory Layout
```
evolve/
  README.md         # Vision and introduction (in progress, unfinished)
  code/             # Not started. README.md holds the planned layout + design constraints.
  research/         # The research program. Start at research/README.md.
    literature/     #   Paper notes + annotated bibliography
    problems/       #   P1 diversity metric, P2 partitioned perturbation, P3 objectives/hardware
    ideas/          #   Wide unfiltered idea lists
    proposals/      #   Worked-out methods (A, B, C)
    experiments/    #   Experiment designs + results
  claude/           # This knowledge base
    decisions/      #   Why choices were made
```

## Tone & Voice
The README is written in a deliberately personal, opinionated, slightly irreverent voice (e.g., "Cookies expire on Stackoverflow", "yada yada"). Future writing recommendations should preserve this character.

## Open Questions
- Is the "Statistical Illusion" framing the central organizing concept, or a rhetorical opener?
- Target audience: academic, technical general, personal blog?
- Framework vs. paper vs. notebook — what is the artifact at the end?
- The immediate technical questions live in `research/problems/` and `claude/context.md`.
