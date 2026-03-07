from stage07_core import run_relabel_only


# A/B experiment mode:
# A = rule + llm (no embedding recall)
# B = rule + selective embedding + llm
EXPERIMENT_MODE = "A"  # "A" | "B"

RUN_PROFILE_OVERRIDE = "full"  # "sample" | "full"
RUN_TAG_SUFFIX = "relabel_only_ab"
USE_BGE_M3_OVERRIDE = "true"  # "true" | "false"
OLLAMA_MODEL_OVERRIDE = "qwen3:8b"

if EXPERIMENT_MODE.upper() == "A":
    RELABEL_STRATEGY_OVERRIDE = "selective"
    RELABEL_USE_EMBED_RECALL_OVERRIDE = "false"
    RELABEL_USE_LLM_OVERRIDE = "true"
elif EXPERIMENT_MODE.upper() == "B":
    RELABEL_STRATEGY_OVERRIDE = "selective"
    RELABEL_USE_EMBED_RECALL_OVERRIDE = "true"
    RELABEL_USE_LLM_OVERRIDE = "true"
else:
    raise ValueError(f"Unsupported EXPERIMENT_MODE: {EXPERIMENT_MODE}")


def main() -> None:
    run_relabel_only(
        run_profile_override=RUN_PROFILE_OVERRIDE,
        run_tag_suffix=f"{RUN_TAG_SUFFIX}_{EXPERIMENT_MODE.lower()}",
        use_bge_m3_override=USE_BGE_M3_OVERRIDE,
        relabel_strategy_override=RELABEL_STRATEGY_OVERRIDE,
        relabel_use_embed_recall_override=RELABEL_USE_EMBED_RECALL_OVERRIDE,
        relabel_use_llm_override=RELABEL_USE_LLM_OVERRIDE,
        ollama_model_override=OLLAMA_MODEL_OVERRIDE,
    )


if __name__ == "__main__":
    main()
