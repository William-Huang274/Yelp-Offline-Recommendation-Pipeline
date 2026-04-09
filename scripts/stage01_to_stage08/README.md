# Stage01 to Stage08 Foundations

This directory keeps the earlier local recommendation pipeline stages visible
as part of the repository's main development history.

These scripts are **not** required to reproduce the current frozen
`stage09 -> stage10 -> stage11` line, but they remain part of the main
repository surface because they explain how the project evolved from the
earlier local pipeline into the current multi-stage ranking stack.

## Included Stages

- `01` to `06`: early data prep, analysis, and baseline evaluation scripts
- `07`: relabel / clustering / local pilot-era scripts
- `08`: cluster refinement and profile merge scripts
- `legacy_stage07/`: preserved stage07 helpers that are still useful for
  historical reference

## Relationship To Current Mainline

- Current reproducible release path:
  - `stage09 -> stage10 -> stage11`
- Historical foundation path:
  - `stage01 -> stage08`

Keeping both visible makes the repository easier to explain externally without
mixing the old pipeline into the active launch surface.
