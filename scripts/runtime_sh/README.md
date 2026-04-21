# Runtime Shell Launchers

[English](./README.md) | [中文](./README.zh-CN.md)

This directory holds the long-form shell launchers for the active
`Stage09 -> Stage11` runtime path.

These scripts are Linux / cloud / WSL-style launchers. They are not the local
Windows demo path; use the PowerShell and Python helpers in [../../tools](../../tools)
for local review runs.

They used to live directly under `scripts/`. They are now grouped here so that
the root script surface stays focused on Python entry scripts, audit helpers,
and the outward-facing wrapper layer.

Use [../launchers/](../launchers) as the public entry surface.

Use this directory when you need the underlying full shell launcher that a
wrapper forwards to.

Additional persona-SFT quality-first launchers are also grouped here:

- `run_stage11_persona_sft_v3_build_dataset.sh`
- `run_stage11_persona_sft_v3_quality_6144.sh`
