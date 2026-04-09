from stage07_core import run_cluster_only


RUN_PROFILE_OVERRIDE = "full"  # optional: "sample" | "full"
SOURCE_RELABEL_RUN_DIR = r"D:\5006 BDA project\data\output\07_embedding_cluster\20260211_165550_full_relabel_minilm_relabel_only_ab_b"  # optional explicit dir containing biz_relabels.csv
RUN_TAG_SUFFIX = "cluster_only"
CLUSTER_K_OVERRIDE = 24  # optional: >0 to override profile cluster_k


def main() -> None:
    run_cluster_only(
        run_profile_override=RUN_PROFILE_OVERRIDE,
        source_relabel_run_dir=SOURCE_RELABEL_RUN_DIR,
        run_tag_suffix=RUN_TAG_SUFFIX,
        cluster_k_override=CLUSTER_K_OVERRIDE,
    )


if __name__ == "__main__":
    main()
