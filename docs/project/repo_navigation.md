# Repository Navigation

## Purpose

This note tells reviewers which parts of the repository represent the current
mainline and which parts are historical or local-only.

## 1. Canonical Entry Points

Start here for the current repository line:

- [../../README.md](../../README.md)
- [../../docs/README.md](../../docs/README.md)
- [../../scripts/launchers/README.md](../../scripts/launchers/README.md)
- [../../data/output/current_release/README.md](../../data/output/current_release/README.md)
- [../../data/metrics/current_release/README.md](../../data/metrics/current_release/README.md)

## 2. Current Public Mainline

| area | role |
| --- | --- |
| `scripts/launchers` | stable outward-facing wrappers |
| `scripts/runtime_sh` | long-form runtime shell launchers used by wrappers |
| `scripts/09_*`, `scripts/10_*`, `scripts/11_*` | active stage logic |
| `docs/contracts` | variable and launcher contracts |
| `docs/stage11` | current public Stage11 design notes and case notes |
| `data/output/current_release` | compact current result surface |
| `data/metrics/current_release` | compact current metrics surface |

## 3. Historical But Still Useful

| area | why it still matters |
| --- | --- |
| `scripts/stage01_to_stage08` | shows project foundation, early ingest, and historical development path |
| `data/output/showcase_history` | selected historical result references |
| `data/metrics/showcase_history` | selected historical metric references |
| `docs/release` | closeout and upgrade notes explaining the current repository line |

## 4. Archive / Local-Only Surfaces

Treat these as background material rather than as the canonical demo path:

| area | interpretation |
| --- | --- |
| `docs/archive` | historical and internal notes |
| `scripts/archive` | older or one-off scripts kept for auditability |
| `data/output/_prod_runs` | provenance packs and larger audit surfaces |

## 5. Recommended Reviewer Mental Model

The simplest way to navigate the repository is:

1. read the root README
2. inspect the current release surface
3. use the launchers and validation tools
4. dip into archive material only when you need historical context
