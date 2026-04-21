## Config Surface

This folder contains two different kinds of configuration files.

### 1. Active repository configuration

These files are part of the current public mainline:

- `labeling/food_service/v1/`: labeling rules and lexicons used by the
  repository-side food-service labeling flow
- `demo/batch_infer_demo_input.json`: small request payload used by the mock
  `Stage09 -> Stage10 -> Stage11` demo path
- `demo/stage11_model_prompt_smoke_case.json`: smoke-case contract for the
  Stage11 `Qwen3.5-9B` reward-model surface and prompt-only probe templates

### 2. Legacy compatibility training profiles

These files are kept for local reproducibility and historical compatibility:

- `dpo_low_memory.env`
- `dpo_optimized_640.env`
- `dpo_safe_512.env`
- `dpo_semantic_mode.env`
- `dpo_ultra_low_memory.env`

They belong to the earlier local DPO / QLoRA experimentation surface and are
not the default public Stage11 mainline.

The current public Stage11 line is the reward-model rerank path documented in:

- `README.md`
- `docs/stage11/stage11_31_60_only_and_segmented_fusion_20260408.md`
