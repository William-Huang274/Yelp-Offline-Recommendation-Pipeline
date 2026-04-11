# Scripts Surface

[English](./README.md) | [中文](./README.zh-CN.md)

This directory keeps the active script surface for the current repository line.

Historical scripts are still kept locally, but they are intentionally excluded
from the public repository surface.

## Active Areas

### Stage01 to Stage08

- [stage01_to_stage08](./stage01_to_stage08)

These files remain part of the repository history and stay visible, but they
are not part of the current reproducible `stage09 -> stage11` release path.

### Stage09

Current active Stage09 surface includes:

- `09_candidate_fusion.py`
- `09_1_recall_audit.py`
- typed-intent and match builders
- Stage10 feature builders
- Stage11 source-material and semantic-asset builders

### Stage10

Current active Stage10 surface includes:

- `10_1_rank_train.py`
- `10_2_rank_infer_eval.py`
- `10_4_bucket5_focus_slice_eval.py`
- bucket-level launchers for `bucket5`, `bucket2`, and `bucket10`

### Stage11

Current active Stage11 surface includes:

- `11_1_qlora_build_dataset.py`
- `11_2_dpo_export_pairs.py`
- `11_2_rm_train.py`
- `11_3_qlora_sidecar_eval.py`
- compact export / train / eval / watcher launchers
- current Stage11 audit scripts

### Shared Support

- `pipeline/`
- [launchers/](./launchers): stable outward-facing wrappers
- [runtime_sh/](./runtime_sh): grouped long-form shell launchers used by the wrappers

## Launcher Rule

The long shell launchers are now grouped under:

- [runtime_sh/](./runtime_sh)

Outward-facing documentation and demos should point to:

- [launchers/](./launchers)

Launcher variable naming rules are documented here:

- [../docs/contracts/launcher_env_conventions.md](../docs/contracts/launcher_env_conventions.md)

## Cleanup Rule

A script should stay on the active surface only if it satisfies at least one of
the following:

1. it is called by an active launcher
2. it belongs to the current release scope
3. it is required by an active mainline helper path

Other scripts should stay in the local archive area rather than on the public
script surface.
