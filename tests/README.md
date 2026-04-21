## Test Surface

This folder holds the current smoke tests for the public mainline repository surface.

- `test_full_chain_smoke.py`: runs the safe `stage01 -> stage11` smoke path, including stage01-08 help entrypoints, local stage09/10 checks, and Stage11 demo / prompt checks
- `test_public_release_surface.py`: checks that the public result surface is complete
- `test_release_metrics_surface.py`: checks that the current Stage10 and Stage11 summary files are internally consistent, including the three-expert Stage11 release line
- `test_stage11_model_prompt_surface.py`: checks the Stage11 Qwen3.5-9B reward-model surface and smoke-case contract
- `test_launcher_surface.py`: checks that the current launcher surface and compatibility stage scripts are present
- `test_public_readme_links.py`: checks that the public README links resolve locally

Legacy tests from the previous repository surface were moved to `tests/archive/release_closeout_20260410/`.
