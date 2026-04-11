## Test Surface

This folder holds the current smoke tests for the public mainline repository surface.

- `test_public_release_surface.py`: checks that the public result surface is complete
- `test_release_metrics_surface.py`: checks that the current Stage10 and Stage11 summary files are internally consistent, including the three-expert Stage11 release line
- `test_launcher_surface.py`: checks that the current launcher surface and compatibility stage scripts are present
- `test_public_readme_links.py`: checks that the public README links resolve locally

Legacy tests from the previous repository surface were moved to `tests/archive/release_closeout_20260410/`.
