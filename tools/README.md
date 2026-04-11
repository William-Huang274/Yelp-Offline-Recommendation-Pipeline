## Tools

This folder contains lightweight repository-side utilities aligned with the current public mainline.

- `validate_public_surface.py`: validates the public README, docs, launchers, and curated result surface
- `validate_current_release.py`: validates the current curated release summaries under `data/output/current_release`, including the three-expert Stage11 release line
- `validate_stage_artifact.py`: validates local Stage09/Stage10/Stage11 run directories using repository validators
- `analyze_stage11_prompt_length.py`: reports prompt-length distribution for a Stage11 dataset run
- `monitor_stage11_training.py`: tails a Stage11 training or evaluation log and extracts recent metric lines
- `monitor_stage11_training.bat` / `monitor_stage11_training.sh`: thin wrappers for the monitor
- `check_local_resources.py`: local GPU/RAM/disk inspection before heavy Stage11 work

Legacy tools from the previous DPO-oriented surface were moved to `tools/archive/release_closeout_20260410/`.
