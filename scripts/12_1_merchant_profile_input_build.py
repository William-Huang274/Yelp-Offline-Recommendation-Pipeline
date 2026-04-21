from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This stage script is configured by environment variables and starts a merchant profile input build job.")
    print("Set the required INPUT_/OUTPUT_ environment variables, then run without --help.")
    sys.exit(0)

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import (
    env_or_project_path,
    normalize_legacy_project_path,
    project_path,
    write_latest_run_pointer,
)


RUN_TAG = "stage12_merchant_profile_input_build"

INPUT_MERCHANT_STATE_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_STATE_V2_ROOT_DIR",
    "data/output/09_merchant_state_v2",
)
INPUT_MERCHANT_STATE_RUN_DIR = os.getenv("INPUT_09_MERCHANT_STATE_V2_RUN_DIR", "").strip()
RAW_BUSINESS_ROOT = env_or_project_path(
    "RAW_BUSINESS_ROOT_DIR",
    "data/parquet/yelp_academic_dataset_business",
)
RAW_REVIEW_ROOT = env_or_project_path(
    "RAW_REVIEW_ROOT_DIR",
    "data/parquet/yelp_academic_dataset_review",
)
RAW_TIP_ROOT = env_or_project_path(
    "RAW_TIP_ROOT_DIR",
    "data/parquet/yelp_academic_dataset_tip",
)
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_12_MERCHANT_PROFILE_INPUT_ROOT_DIR",
    "data/output/12_merchant_profile_inputs",
)

MAX_BUSINESSES = int(os.getenv("MERCHANT_PROFILE_MAX_BUSINESSES", "0").strip() or 0)
MAX_REVIEW_EVIDENCE = int(os.getenv("MERCHANT_PROFILE_MAX_REVIEW_EVIDENCE", "36").strip() or 36)
MAX_TIP_EVIDENCE = int(os.getenv("MERCHANT_PROFILE_MAX_TIP_EVIDENCE", "8").strip() or 8)
REVIEW_TEXT_MAX_CHARS = int(os.getenv("MERCHANT_PROFILE_REVIEW_TEXT_MAX_CHARS", "720").strip() or 720)
TIP_TEXT_MAX_CHARS = int(os.getenv("MERCHANT_PROFILE_TIP_TEXT_MAX_CHARS", "360").strip() or 360)
MIN_REVIEW_TEXT_CHARS = int(os.getenv("MERCHANT_PROFILE_MIN_REVIEW_TEXT_CHARS", "40").strip() or 40)
MIN_TIP_TEXT_CHARS = int(os.getenv("MERCHANT_PROFILE_MIN_TIP_TEXT_CHARS", "12").strip() or 12)
PANDAS_ROW_CAP = int(os.getenv("MERCHANT_PROFILE_PANDAS_ROW_CAP", "20000").strip() or 20000)
SAMPLE_ROWS = int(os.getenv("MERCHANT_PROFILE_SAMPLE_ROWS", "8").strip() or 8)
MAX_REVIEW_DATE = os.getenv("MERCHANT_PROFILE_MAX_REVIEW_DATE", "").strip()
MAX_TIP_DATE = os.getenv("MERCHANT_PROFILE_MAX_TIP_DATE", "").strip()
REVIEW_PART_FILE_LIMIT = int(os.getenv("MERCHANT_PROFILE_REVIEW_PART_FILE_LIMIT", "0").strip() or 0)
TIP_PART_FILE_LIMIT = int(os.getenv("MERCHANT_PROFILE_TIP_PART_FILE_LIMIT", "0").strip() or 0)
BUSINESS_UNIVERSE_PATH_RAW = os.getenv("MERCHANT_PROFILE_BUSINESS_UNIVERSE_PATH", "").strip()
APPLY_STAGE09_HARD_FILTER = os.getenv("MERCHANT_PROFILE_APPLY_STAGE09_HARD_FILTER", "false").strip().lower() == "true"
STAGE09_TARGET_STATE = os.getenv("MERCHANT_PROFILE_STAGE09_TARGET_STATE", "LA").strip() or "LA"
STAGE09_MIN_BUSINESS_STARS = float(os.getenv("MERCHANT_PROFILE_STAGE09_MIN_BUSINESS_STARS", "3.0").strip() or 3.0)
STAGE09_MIN_BUSINESS_REVIEW_COUNT = int(
    os.getenv("MERCHANT_PROFILE_STAGE09_MIN_BUSINESS_REVIEW_COUNT", "20").strip() or 20
)
STAGE09_STALE_CUTOFF_DATE = os.getenv("MERCHANT_PROFILE_STAGE09_STALE_CUTOFF_DATE", "2020-01-01").strip() or "2020-01-01"

# These queue cut points mirror the measured user-side vLLM budget instead of inventing
# a new local budget: regular <=16k, high <=32k, overflow above that.
REGULAR_PROMPT_TOKEN_BUDGET = int(os.getenv("MERCHANT_PROFILE_REGULAR_PROMPT_TOKENS", "16000").strip() or 16000)
HIGH_PROMPT_TOKEN_BUDGET = int(os.getenv("MERCHANT_PROFILE_HIGH_PROMPT_TOKENS", "32000").strip() or 32000)
CANDIDATE_RECALL_MODE = os.getenv("MERCHANT_PROFILE_CANDIDATE_RECALL_MODE", "true").strip().lower() == "true"
DIRECT_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_DIRECT_MAX_ITEMS", "5" if CANDIDATE_RECALL_MODE else "5").strip()
    or 5
)
FINE_GRAINED_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_FINE_GRAINED_MAX_ITEMS", "12" if CANDIDATE_RECALL_MODE else "8").strip()
    or (12 if CANDIDATE_RECALL_MODE else 8)
)
USAGE_SCENE_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_USAGE_SCENE_MAX_ITEMS", "6" if CANDIDATE_RECALL_MODE else "4").strip()
    or (6 if CANDIDATE_RECALL_MODE else 4)
)
RISK_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_RISK_MAX_ITEMS", "6" if CANDIDATE_RECALL_MODE else "4").strip()
    or (6 if CANDIDATE_RECALL_MODE else 4)
)
CAUTIOUS_MAX_ITEMS = int(
    os.getenv("MERCHANT_PROFILE_CAUTIOUS_MAX_ITEMS", "5" if CANDIDATE_RECALL_MODE else "3").strip()
    or (5 if CANDIDATE_RECALL_MODE else 3)
)
REFS_MAX_ITEMS = int(os.getenv("MERCHANT_PROFILE_REFS_MAX_ITEMS", "3").strip() or 3)
SNIPPETS_MAX_ITEMS = int(os.getenv("MERCHANT_PROFILE_SNIPPETS_MAX_ITEMS", "2").strip() or 2)

SPACE_RE = re.compile(r"\s+")


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = normalize_legacy_project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def loads_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def normalize_text(value: Any, max_chars: int = 0) -> str:
    text = SPACE_RE.sub(" ", safe_text(value).replace("\r", " ").replace("\n", " ")).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_block(title: str, value: Any) -> str:
    return f"{title}\n```json\n{json.dumps(value, ensure_ascii=False, indent=2, default=str)}\n```"


def queue_label_from_char4_est(tokens: int) -> str:
    if tokens <= REGULAR_PROMPT_TOKEN_BUDGET:
        return "regular"
    if tokens <= HIGH_PROMPT_TOKEN_BUDGET:
        return "high_budget"
    return "overflow"


def configure_spark_env() -> Path:
    tmp_dir = env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp/stage12_merchant_profile_inputs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    return tmp_dir


def create_spark(app_name: str):
    from pyspark.sql import SparkSession

    tmp_dir = configure_spark_env()
    master = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
    shuffle = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "16").strip() or "16"
    parallelism = os.getenv("SPARK_DEFAULT_PARALLELISM", "16").strip() or "16"
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.shuffle.partitions", shuffle)
        .config("spark.default.parallelism", parallelism)
        .config("spark.local.dir", str(tmp_dir))
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", os.getenv("SPARK_NETWORK_TIMEOUT", "800s"))
        .config("spark.executor.heartbeatInterval", os.getenv("SPARK_HEARTBEAT_INTERVAL", "60s"))
        .config("spark.sql.execution.arrow.pyspark.enabled", os.getenv("SPARK_ARROW_ENABLED", "true"))
        .getOrCreate()
    )


def parquet_read_paths(root: Path, part_file_limit: int) -> list[str]:
    if part_file_limit <= 0:
        return [str(root)]
    files = sorted(root.glob("*.parquet"))[:part_file_limit]
    if not files:
        raise FileNotFoundError(f"no parquet part files under {root}")
    return [str(path) for path in files]


def render_prompt(row: pd.Series) -> str:
    merchant_meta = loads_json(row.get("merchant_meta_json"), {})
    aggregate_notes = loads_json(row.get("aggregate_notes_json"), {})
    sampling_profile = loads_json(row.get("sampling_profile_json"), {})
    positive_reviews = loads_json(row.get("positive_review_evidence_json"), [])
    mixed_reviews = loads_json(row.get("mixed_review_evidence_json"), [])
    negative_reviews = loads_json(row.get("negative_review_evidence_json"), [])
    tips = loads_json(row.get("tip_evidence_json"), [])

    instructions = """
You are building a merchant-side structured profile for downstream user-merchant matching.

Use only the supplied merchant evidence. Do not infer any target user's preference, label, rank,
click, conversion, held-out truth, or future outcome.

Your job:
1. Extract merchant attributes, strengths, weaknesses, usage scenes, dish/service signals, and risk notes.
2. Preserve fine-grained details from review/tip language; do not collapse distinct signals too aggressively.
3. Separate directly evidenced facts from cautious inferences. If you infer something, cite the evidence ids
   and explain why the inference follows.
4. Business metadata may only identify the merchant and broad category. Do not convert category/name/city/rating
   metadata into dish, service, scene, ambience, price, or risk claims unless review/tip evidence supports it.
5. If a claim comes only from metadata or aggregate notes, put it in cautious_inferences with reasoning. Do not mix
   metadata-only claims into direct_evidence_signals.
6. Use direct_evidence_signals for top-line recurring themes, not as a menu-item inventory. Reserve it for the
   strongest repeated merchant takeaways that summarize what the evidence keeps returning to.
7. Do not introduce facts not present in the supplied sections. The summary must be evidence-supported, not a
   general web-style description.
8. Do not use outside knowledge. Even if the merchant is famous, do not mention history, founding year, awards,
   neighborhood reputation, or tourist status unless the supplied review/tip evidence says it.
9. Treat fine_grained_attributes as the main recall surface for concrete user-meaningful details: specific dishes,
   drink details, texture or flavor cues, service behaviors, ambience notes, operational constraints, wait patterns,
   seating details, timing cues, and value signals. Keep distinct fine-grained items separate when they capture
   different matching value.
10. Evidence ref rules: direct_evidence_signals, fine_grained_attributes, usage_scenes, and risk_or_avoidance_notes
   must cite only mrev_* or mtip_* ids. cautious_inferences may cite MERCHANT_META or AGGREGATE_NOTES when the
   support_basis is metadata or aggregate.
11. Do not cite adjacent refs mechanically. Each cited mrev_* or mtip_* must directly contain the claim it supports.
12. For every item in direct_evidence_signals, fine_grained_attributes, usage_scenes, and risk_or_avoidance_notes,
    include evidence_snippets: 1-2 short exact text spans copied character-for-character from the visible text field
    in the supplied review/tip JSON. Use short spans that are easy to verify; do not reconstruct hidden continuation
    text or quote from memory. If you cannot copy a direct supporting snippet, omit the item or move it to
    cautious_inferences.
13. For scenes and risks, do not turn broad popularity/category metadata into a direct claim. Use direct sections only
    when the scene/risk is explicitly stated by review/tip language.
14. Every direct claim must be a tight paraphrase of its evidence_snippets. Do not use snippets about one dish,
    service issue, ambience feature, or scene to support another. If the snippet says oysters, do not claim Po-Boys;
    if it says service, do not claim ambience unless ambience words are also visible in the snippet.
15. Keep direct claims no broader than the snippets. If the snippet only says "Chargrilled Oysters ... famous",
    write a narrow claim like "Chargrilled oysters are repeatedly highlighted"; do not add butter, cheese, price,
    wait, scene, or service details unless those words are visible in the snippet.
16. Prefer descriptors that can later align with user-side language, such as dish texture/flavor cues, service style,
    crowding or wait patterns, seating constraints, noise level, timing, or value perception. Avoid generic merchant
    praise that would not help user-merchant matching.
17. If a concrete dish/service/ambience detail appears inside a broader top-line theme, keep the broader theme in
    direct_evidence_signals and preserve the concrete detail separately in fine_grained_attributes when it adds
    matching value.
18. Do not repeat the same claim with minor wording changes within a section or across sections. Merge duplicates and
    keep only the strongest grounded version of each idea.
19. If the supplied review/tip evidence clearly contains concrete dish, service, ambience, operations, scene, or risk
    details, prefer grounded direct/fine_grained/usage_scenes/risk items over leaving those sections empty or falling
    back to metadata-only cautious_inferences.
20. Keep each claim atomic and close to the snippet wording. Do not bundle several traits into one item unless the
    same snippet explicitly states all of them. Prefer "Long lines form outside" over "Long wait times on weekends and
    peak hours" unless weekend/peak wording is visible in the snippet.
21. For usage_scenes, prefer source-anchored scene language that is explicitly visible in review/tip text, such as
    brunch, date night, after work, company dinner, group meal, patio, hotel dining, or bar seating. Avoid generic
    labels like tourist dining, special occasion, casual dining, or family dining unless those exact ideas are visible.
22. For risk_or_avoidance_notes, prefer explicit negative conditions from the snippet surface form, such as long line,
    loud, noisy, slow service, cold dish, parking trouble, reservation friction, cramped seating, or inconsistent
    service. Do not inflate a mild inconvenience into a broader abstract risk claim.
23. If direct_evidence_signals would otherwise be filled with vague summary claims, leave it sparse and preserve the
    concrete value in fine_grained_attributes, usage_scenes, or risk_or_avoidance_notes instead.

Output budget:
- merchant_summary: <= 60 words.
- direct_evidence_signals: <= __DIRECT_MAX_ITEMS__ items.
- fine_grained_attributes: <= __FINE_GRAINED_MAX_ITEMS__ items.
- usage_scenes: <= __USAGE_SCENE_MAX_ITEMS__ items.
- risk_or_avoidance_notes: <= __RISK_MAX_ITEMS__ items.
- cautious_inferences: <= __CAUTIOUS_MAX_ITEMS__ items.
- evidence_refs: <= __REFS_MAX_ITEMS__ refs per item.
- evidence_snippets: 1-__SNIPPETS_MAX_ITEMS__ exact visible snippets per direct item, each <= 14 words when possible.

Return strict JSON with keys:
{
  "merchant_summary": string,
  "direct_evidence_signals": [{"signal": string, "polarity": "positive|mixed|negative|neutral", "support_basis": "review|tip|review_and_tip", "evidence_refs": [string], "evidence_snippets": [string]}],
  "fine_grained_attributes": [{"attribute": string, "attribute_type": "dish|service|ambience|scene|price_value|operations|other", "evidence_refs": [string], "evidence_snippets": [string]}],
  "usage_scenes": [{"scene": string, "fit": "strong|moderate|weak", "evidence_refs": [string], "evidence_snippets": [string]}],
  "risk_or_avoidance_notes": [{"risk": string, "severity": "high|medium|low", "evidence_refs": [string], "evidence_snippets": [string]}],
  "cautious_inferences": [{"inference": string, "reasoning": string, "support_basis": "metadata|aggregate|metadata_and_evidence|aggregate_and_evidence", "evidence_refs": [string], "evidence_snippets": [string]}],
  "audit_notes": {"unsupported_or_weak_claims": [string], "evidence_coverage": string}
}
""".strip()
    instructions = (
        instructions.replace("__DIRECT_MAX_ITEMS__", str(DIRECT_MAX_ITEMS))
        .replace("__FINE_GRAINED_MAX_ITEMS__", str(FINE_GRAINED_MAX_ITEMS))
        .replace("__USAGE_SCENE_MAX_ITEMS__", str(USAGE_SCENE_MAX_ITEMS))
        .replace("__RISK_MAX_ITEMS__", str(RISK_MAX_ITEMS))
        .replace("__CAUTIOUS_MAX_ITEMS__", str(CAUTIOUS_MAX_ITEMS))
        .replace("__REFS_MAX_ITEMS__", str(REFS_MAX_ITEMS))
        .replace("__SNIPPETS_MAX_ITEMS__", str(SNIPPETS_MAX_ITEMS))
    )

    sections = [
        instructions,
        json_block("MERCHANT_META", merchant_meta),
        json_block("AGGREGATE_NOTES", aggregate_notes),
        json_block("SAMPLING_PROFILE", sampling_profile),
        json_block("POSITIVE_REVIEW_EVIDENCE", positive_reviews),
        json_block("MIXED_REVIEW_EVIDENCE", mixed_reviews),
        json_block("NEGATIVE_REVIEW_EVIDENCE", negative_reviews),
        json_block("TIP_EVIDENCE", tips),
    ]
    return "\n\n".join(sections).strip()


def read_business_universe_path(spark, raw_path: str):
    from pyspark.sql import functions as F

    if not raw_path:
        return None
    path = normalize_legacy_project_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"MERCHANT_PROFILE_BUSINESS_UNIVERSE_PATH not found: {path}")

    suffix = path.suffix.lower()
    if path.is_dir() or suffix == ".parquet":
        frame = spark.read.parquet(str(path))
    elif suffix == ".csv":
        frame = spark.read.option("header", "true").csv(str(path))
    elif suffix in {".json", ".jsonl"}:
        frame = spark.read.json(str(path))
    else:
        raise ValueError(
            "MERCHANT_PROFILE_BUSINESS_UNIVERSE_PATH must be parquet, csv, json, jsonl, or a parquet directory: "
            f"{path}"
        )

    if "business_id" not in frame.columns:
        raise ValueError(f"business universe file has no business_id column: {path}")
    return (
        frame.select(F.col("business_id").cast("string").alias("business_id"))
        .filter(F.col("business_id").isNotNull() & (F.col("business_id") != ""))
        .dropDuplicates(["business_id"])
    )


def build_stage09_hard_filter_universe(spark):
    from pyspark.sql import functions as F

    if REVIEW_PART_FILE_LIMIT > 0:
        raise RuntimeError(
            "MERCHANT_PROFILE_APPLY_STAGE09_HARD_FILTER=true requires full review parquet scan; "
            "unset MERCHANT_PROFILE_REVIEW_PART_FILE_LIMIT to avoid changing the business universe."
        )

    business = (
        spark.read.parquet(str(RAW_BUSINESS_ROOT))
        .select("business_id", "state", "categories", "is_open", "stars", "review_count")
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    biz = (
        business.filter(F.col("state") == F.lit(STAGE09_TARGET_STATE))
        .filter(cat.contains("restaurants") | cat.contains("food"))
        .filter(F.col("is_open") == 1)
        .filter(F.col("stars") >= F.lit(float(STAGE09_MIN_BUSINESS_STARS)))
        .filter(F.col("review_count") >= F.lit(int(STAGE09_MIN_BUSINESS_REVIEW_COUNT)))
    )
    reviews = (
        spark.read.parquet(str(RAW_REVIEW_ROOT))
        .select("business_id", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
    )
    last_review = reviews.groupBy("business_id").agg(F.max("ts").alias("last_review_ts"))
    return (
        biz.join(last_review, on="business_id", how="left")
        .filter(F.col("last_review_ts") >= F.to_timestamp(F.lit(STAGE09_STALE_CUTOFF_DATE)))
        .select("business_id")
        .dropDuplicates(["business_id"])
    )


def resolve_business_universe(spark):
    explicit = read_business_universe_path(spark, BUSINESS_UNIVERSE_PATH_RAW)
    if explicit is not None:
        return explicit, {
            "business_universe_mode": "explicit_path",
            "business_universe_path": str(normalize_legacy_project_path(BUSINESS_UNIVERSE_PATH_RAW)),
        }
    if APPLY_STAGE09_HARD_FILTER:
        return build_stage09_hard_filter_universe(spark), {
            "business_universe_mode": "stage09_hard_filter_rebuilt",
            "business_universe_path": "",
            "stage09_hard_filter_policy": {
                "target_state": STAGE09_TARGET_STATE,
                "require_restaurants_or_food": True,
                "require_is_open": True,
                "min_business_stars": STAGE09_MIN_BUSINESS_STARS,
                "min_business_review_count": STAGE09_MIN_BUSINESS_REVIEW_COUNT,
                "stale_cutoff_date": STAGE09_STALE_CUTOFF_DATE,
            },
        }
    return None, {"business_universe_mode": "merchant_state_v2_all", "business_universe_path": ""}


def build_base_business_frame(spark, merchant_run_dir: Path):
    from pyspark.sql import functions as F

    merchant = spark.read.parquet(str(merchant_run_dir / "merchant_state_v2.parquet"))
    universe, universe_meta = resolve_business_universe(spark)
    if universe is not None:
        merchant = merchant.join(universe, "business_id", "inner")
    business = (
        spark.read.parquet(str(RAW_BUSINESS_ROOT))
        .select(
            "business_id",
            F.col("categories").alias("raw_categories"),
            F.col("stars").alias("yelp_business_stars"),
            F.col("review_count").alias("yelp_business_review_count"),
            F.col("is_open").alias("yelp_is_open"),
        )
        .dropDuplicates(["business_id"])
    )
    base = merchant.join(business, "business_id", "left")
    if MAX_BUSINESSES > 0:
        base = base.orderBy(
            F.desc_nulls_last("audit_review_count"),
            F.desc_nulls_last("avg_review_stars"),
            F.asc("business_id"),
        ).limit(MAX_BUSINESSES)
    return base, universe_meta


def review_bucket_expr(stars_col, lower_text_col):
    from pyspark.sql import functions as F

    pos_pattern = r"(delicious|fresh|tasty|amazing|excellent|friendly|attentive|cozy|clean|great|favorite|recommend)"
    neg_pattern = r"(rude|slow|cold|bland|overpriced|dirty|wait|waiting|disappoint|bad|worst|avoid|greasy)"
    has_pos = lower_text_col.rlike(pos_pattern)
    has_neg = lower_text_col.rlike(neg_pattern)
    return (
        F.when(has_pos & has_neg, F.lit("mixed"))
        .when(stars_col <= F.lit(2.0), F.lit("negative"))
        .when((stars_col >= F.lit(4.0)) & (~has_neg), F.lit("positive"))
        .otherwise(F.lit("mixed"))
    )


def add_review_scores(reviews):
    from pyspark.sql import functions as F

    clean_text = F.trim(F.regexp_replace(F.regexp_replace(F.col("text"), r"[\r\n]+", " "), r"\s+", " "))
    truncated_text = F.when(
        F.length(clean_text) > F.lit(REVIEW_TEXT_MAX_CHARS),
        F.substring(clean_text, 1, REVIEW_TEXT_MAX_CHARS),
    ).otherwise(clean_text)
    lower_text = F.lower(truncated_text)
    specific_pattern = (
        r"(menu|dish|portion|price|parking|reservation|service|server|wait|line|ambience|atmosphere|"
        r"seating|outdoor|bar|drink|cocktail|dessert|coffee|breakfast|brunch|lunch|dinner|takeout)"
    )
    return (
        reviews.withColumn("review_text_clean", truncated_text)
        .filter(F.length("review_text_clean") >= F.lit(MIN_REVIEW_TEXT_CHARS))
        .withColumn("stars_d", F.col("stars").cast("double"))
        .withColumn("review_date", F.to_date("date"))
        .withColumn("lower_review_text", lower_text)
        .withColumn("rating_bucket", review_bucket_expr(F.col("stars_d"), F.col("lower_review_text")))
        .withColumn("text_len", F.length("review_text_clean"))
        .withColumn("vote_score", F.log1p(F.coalesce(F.col("useful"), F.lit(0)) + F.coalesce(F.col("funny"), F.lit(0)) + F.coalesce(F.col("cool"), F.lit(0))))
        .withColumn("specificity_bonus", F.when(F.col("lower_review_text").rlike(specific_pattern), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("mixed_bonus", F.when(F.col("rating_bucket") == F.lit("mixed"), F.lit(0.6)).otherwise(F.lit(0.0)))
        .withColumn("length_score", F.least(F.col("text_len") / F.lit(360.0), F.lit(2.0)))
        .withColumn("evidence_score", F.col("length_score") + F.col("vote_score") + F.col("specificity_bonus") + F.col("mixed_bonus"))
    )


def build_review_stats(scored_reviews):
    from pyspark.sql import functions as F

    stats = (
        scored_reviews.groupBy("business_id")
        .agg(
            F.count("*").alias("sample_review_rows"),
            F.avg("stars_d").alias("sample_avg_review_stars"),
            F.max("review_date").alias("latest_review_date"),
            F.sum(F.when(F.col("rating_bucket") == "positive", 1).otherwise(0)).alias("positive_review_rows"),
            F.sum(F.when(F.col("rating_bucket") == "mixed", 1).otherwise(0)).alias("mixed_review_rows"),
            F.sum(F.when(F.col("rating_bucket") == "negative", 1).otherwise(0)).alias("negative_review_rows"),
        )
        .withColumn("positive_review_share", F.col("positive_review_rows") / F.greatest(F.col("sample_review_rows"), F.lit(1)))
        .withColumn("mixed_review_share", F.col("mixed_review_rows") / F.greatest(F.col("sample_review_rows"), F.lit(1)))
        .withColumn("negative_review_share", F.col("negative_review_rows") / F.greatest(F.col("sample_review_rows"), F.lit(1)))
    )
    stats = stats.withColumn(
        "merchant_sampling_segment",
        F.when(F.col("sample_review_rows") < 10, F.lit("low_signal"))
        .when(
            (F.col("sample_avg_review_stars") >= 4.2)
            & (F.col("negative_review_share") <= 0.10)
            & (F.col("mixed_review_share") <= 0.20),
            F.lit("high_stable"),
        )
        .when(
            (F.col("sample_avg_review_stars") >= 4.0)
            & ((F.col("negative_review_share") >= 0.15) | (F.col("mixed_review_share") >= 0.25)),
            F.lit("high_controversial"),
        )
        .when(F.col("sample_avg_review_stars") >= 3.2, F.lit("mixed_reputation"))
        .otherwise(F.lit("low_reputation")),
    )

    pos_frac = (
        F.when(F.col("merchant_sampling_segment") == "high_stable", F.lit(0.70))
        .when(F.col("merchant_sampling_segment") == "high_controversial", F.lit(0.50))
        .when(F.col("merchant_sampling_segment") == "mixed_reputation", F.lit(0.40))
        .when(F.col("merchant_sampling_segment") == "low_reputation", F.lit(0.20))
        .otherwise(F.lit(0.34))
    )
    mixed_frac = (
        F.when(F.col("merchant_sampling_segment") == "high_stable", F.lit(0.20))
        .when(F.col("merchant_sampling_segment") == "high_controversial", F.lit(0.25))
        .when(F.col("merchant_sampling_segment") == "mixed_reputation", F.lit(0.30))
        .when(F.col("merchant_sampling_segment") == "low_reputation", F.lit(0.25))
        .otherwise(F.lit(0.33))
    )
    stats = stats.withColumn("positive_review_quota", F.floor(F.lit(MAX_REVIEW_EVIDENCE) * pos_frac).cast("int"))
    stats = stats.withColumn("mixed_review_quota", F.floor(F.lit(MAX_REVIEW_EVIDENCE) * mixed_frac).cast("int"))
    return stats.withColumn(
        "negative_review_quota",
        (F.lit(MAX_REVIEW_EVIDENCE) - F.col("positive_review_quota") - F.col("mixed_review_quota")).cast("int"),
    )


def build_review_evidence(spark, base):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    base_ids = base.select("business_id").distinct()
    reviews = spark.read.parquet(*parquet_read_paths(RAW_REVIEW_ROOT, REVIEW_PART_FILE_LIMIT)).join(base_ids, "business_id", "inner")
    if MAX_REVIEW_DATE:
        reviews = reviews.filter(F.to_date("date") <= F.to_date(F.lit(MAX_REVIEW_DATE)))
    reviews = reviews.select("business_id", "review_id", "user_id", "stars", "date", "useful", "funny", "cool", "text")
    scored = add_review_scores(reviews)
    stats = build_review_stats(scored).persist()

    ranked = scored.join(
        stats.select(
            "business_id",
            "merchant_sampling_segment",
            "positive_review_quota",
            "mixed_review_quota",
            "negative_review_quota",
        ),
        "business_id",
        "inner",
    )
    bucket_rank_w = Window.partitionBy("business_id", "rating_bucket").orderBy(
        F.desc("evidence_score"),
        F.desc_nulls_last("review_date"),
        F.asc("review_id"),
    )
    ranked = ranked.withColumn("bucket_rank", F.row_number().over(bucket_rank_w))
    primary = ranked.filter(
        ((F.col("rating_bucket") == "positive") & (F.col("bucket_rank") <= F.col("positive_review_quota")))
        | ((F.col("rating_bucket") == "mixed") & (F.col("bucket_rank") <= F.col("mixed_review_quota")))
        | ((F.col("rating_bucket") == "negative") & (F.col("bucket_rank") <= F.col("negative_review_quota")))
    )

    selected_keys = primary.select("review_id").distinct()
    used_counts = primary.groupBy("business_id").agg(F.count("*").alias("used_review_rows"))
    leftover = ranked.join(selected_keys, "review_id", "left_anti").join(used_counts, "business_id", "left")
    leftover = leftover.fillna({"used_review_rows": 0})
    fill_w = Window.partitionBy("business_id").orderBy(
        F.desc("evidence_score"),
        F.desc_nulls_last("review_date"),
        F.asc("review_id"),
    )
    fill = (
        leftover.withColumn("fill_rank", F.row_number().over(fill_w))
        .filter(F.col("fill_rank") <= (F.lit(MAX_REVIEW_EVIDENCE) - F.col("used_review_rows")))
        .select(primary.columns)
    )
    selected = primary.unionByName(fill)
    final_w = Window.partitionBy("business_id").orderBy(
        F.desc("evidence_score"),
        F.desc_nulls_last("review_date"),
        F.asc("review_id"),
    )
    selected = (
        selected.withColumn("evidence_rank", F.row_number().over(final_w))
        .filter(F.col("evidence_rank") <= F.lit(MAX_REVIEW_EVIDENCE))
        .withColumn("evidence_id", F.concat(F.lit("mrev_"), F.col("evidence_rank").cast("string")))
    )

    payload = F.struct(
        F.col("evidence_rank"),
        F.col("evidence_id"),
        F.lit("review").alias("source"),
        F.col("rating_bucket"),
        F.col("stars_d").alias("stars"),
        F.col("date").alias("event_time"),
        F.round("evidence_score", 4).alias("evidence_score"),
        F.col("review_id"),
        F.col("useful"),
        F.col("funny"),
        F.col("cool"),
        F.col("review_text_clean").alias("text"),
    )

    grouped = selected.groupBy("business_id").agg(
        F.to_json(F.array_sort(F.collect_list(payload))).alias("review_evidence_json"),
        F.count("*").alias("selected_review_evidence_count"),
    )
    return stats, grouped


def build_tip_evidence(spark, base):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    base_ids = base.select("business_id").distinct()
    tips = spark.read.parquet(*parquet_read_paths(RAW_TIP_ROOT, TIP_PART_FILE_LIMIT)).join(base_ids, "business_id", "inner")
    if MAX_TIP_DATE:
        tips = tips.filter(F.to_date("date") <= F.to_date(F.lit(MAX_TIP_DATE)))
    clean_text = F.trim(F.regexp_replace(F.regexp_replace(F.col("text"), r"[\r\n]+", " "), r"\s+", " "))
    tips = (
        tips.select("business_id", "date", "compliment_count", "text")
        .withColumn("tip_text_clean", F.when(F.length(clean_text) > TIP_TEXT_MAX_CHARS, F.substring(clean_text, 1, TIP_TEXT_MAX_CHARS)).otherwise(clean_text))
        .filter(F.length("tip_text_clean") >= F.lit(MIN_TIP_TEXT_CHARS))
        .withColumn("tip_date", F.to_date("date"))
        .withColumn("tip_len_score", F.least(F.length("tip_text_clean") / F.lit(180.0), F.lit(2.0)))
        .withColumn("tip_score", F.col("tip_len_score") + F.log1p(F.coalesce(F.col("compliment_count"), F.lit(0))))
    )
    w = Window.partitionBy("business_id").orderBy(F.desc("tip_score"), F.desc_nulls_last("tip_date"), F.asc("tip_text_clean"))
    selected = (
        tips.withColumn("evidence_rank", F.row_number().over(w))
        .filter(F.col("evidence_rank") <= F.lit(MAX_TIP_EVIDENCE))
        .withColumn("evidence_id", F.concat(F.lit("mtip_"), F.col("evidence_rank").cast("string")))
    )
    payload = F.struct(
        F.col("evidence_rank"),
        F.col("evidence_id"),
        F.lit("tip").alias("source"),
        F.col("date").alias("event_time"),
        F.col("compliment_count"),
        F.round("tip_score", 4).alias("evidence_score"),
        F.col("tip_text_clean").alias("text"),
    )
    return selected.groupBy("business_id").agg(
        F.to_json(F.array_sort(F.collect_list(payload))).alias("tip_evidence_json"),
        F.count("*").alias("selected_tip_evidence_count"),
    )


def attach_prompt_sections(base, review_stats, review_evidence, tip_evidence):
    from pyspark.sql import functions as F

    final = (
        base.join(review_stats, "business_id", "left")
        .join(review_evidence, "business_id", "left")
        .join(tip_evidence, "business_id", "left")
    )
    final = final.fillna(
        {
            "merchant_sampling_segment": "low_signal",
            "sample_review_rows": 0,
            "positive_review_rows": 0,
            "mixed_review_rows": 0,
            "negative_review_rows": 0,
            "positive_review_share": 0.0,
            "mixed_review_share": 0.0,
            "negative_review_share": 0.0,
            "positive_review_quota": 0,
            "mixed_review_quota": 0,
            "negative_review_quota": 0,
            "selected_review_evidence_count": 0,
            "selected_tip_evidence_count": 0,
            "review_evidence_json": "[]",
            "tip_evidence_json": "[]",
        }
    )

    merchant_meta_json = F.to_json(
        F.struct(
            F.col("business_id"),
            F.col("name"),
            F.col("city"),
            F.col("state"),
            F.col("primary_category"),
            F.col("raw_categories"),
            F.col("yelp_business_stars"),
            F.col("yelp_business_review_count"),
            F.col("yelp_is_open"),
            F.col("merchant_entity_scope_v2"),
        )
    )

    aggregate_notes_json = F.to_json(
        F.struct(
            F.round("avg_review_stars", 4).alias("merchant_state_avg_review_stars"),
            F.col("audit_review_count"),
            F.col("audit_tip_rows"),
            F.col("audit_checkin_rows"),
            F.round("review_high_vote_share", 4).alias("review_high_vote_share"),
            F.round("review_negative_pressure", 4).alias("review_negative_pressure"),
            F.round("tip_recommend_share", 4).alias("tip_recommend_share"),
            F.round("semantic_support_per_review", 4).alias("semantic_support_per_review"),
            F.col("sample_review_rows"),
            F.round("sample_avg_review_stars", 4).alias("sample_avg_review_stars"),
            F.col("latest_review_date").cast("string").alias("latest_review_date"),
            F.col("positive_review_rows"),
            F.round("positive_review_share", 4).alias("positive_review_share"),
            F.col("mixed_review_rows"),
            F.round("mixed_review_share", 4).alias("mixed_review_share"),
            F.col("negative_review_rows"),
            F.round("negative_review_share", 4).alias("negative_review_share"),
            F.col("photo_count"),
            F.round("photo_food_ratio", 4).alias("photo_food_ratio"),
            F.round("photo_inside_ratio", 4).alias("photo_inside_ratio"),
            F.round("photo_outside_ratio", 4).alias("photo_outside_ratio"),
            F.round("photo_drink_ratio", 4).alias("photo_drink_ratio"),
        )
    )

    sampling_profile_json = F.to_json(
        F.struct(
            F.lit("merchant_review_tip_stratified_raw_evidence_v1").alias("sampling_policy"),
            F.lit(
                "Quota is adjusted by merchant reputation segment: stable high-rated merchants favor representative positive evidence; controversial/mixed merchants receive more mixed and negative evidence; low-rated merchants favor representative negative evidence while retaining positive/mixed counterexamples."
            ).alias("policy_note"),
            F.col("merchant_sampling_segment"),
            F.lit(MAX_REVIEW_EVIDENCE).alias("max_review_evidence"),
            F.lit(MAX_TIP_EVIDENCE).alias("max_tip_evidence"),
            F.col("positive_review_quota"),
            F.col("mixed_review_quota"),
            F.col("negative_review_quota"),
            F.col("selected_review_evidence_count"),
            F.col("selected_tip_evidence_count"),
            F.lit(REVIEW_TEXT_MAX_CHARS).alias("review_text_max_chars"),
            F.lit(TIP_TEXT_MAX_CHARS).alias("tip_text_max_chars"),
            F.lit(MAX_REVIEW_DATE).alias("max_review_date"),
            F.lit(MAX_TIP_DATE).alias("max_tip_date"),
            F.lit("No user labels, candidate ranks, clicks, conversions, held-out truth, or future user outcomes are included.").alias("leakage_guardrail"),
        )
    )

    return final.select(
        "business_id",
        "name",
        "city",
        "state",
        "primary_category",
        "merchant_sampling_segment",
        "selected_review_evidence_count",
        "selected_tip_evidence_count",
        merchant_meta_json.alias("merchant_meta_json"),
        aggregate_notes_json.alias("aggregate_notes_json"),
        sampling_profile_json.alias("sampling_profile_json"),
        "review_evidence_json",
        "tip_evidence_json",
    )


def add_prompts(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()
    for bucket, column in (
        ("positive", "positive_review_evidence_json"),
        ("mixed", "mixed_review_evidence_json"),
        ("negative", "negative_review_evidence_json"),
    ):
        out[column] = out["review_evidence_json"].map(
            lambda raw, b=bucket: json.dumps(
                [item for item in loads_json(raw, []) if safe_text(item.get("rating_bucket")) == b],
                ensure_ascii=False,
            )
        )
    out["prompt_text"] = out.apply(render_prompt, axis=1)
    out["prompt_chars"] = out["prompt_text"].str.len().astype("int64")
    out["prompt_tokens_char4_est"] = ((out["prompt_chars"] + 3) // 4).astype("int64")
    out["queue_label_char4_est"] = out["prompt_tokens_char4_est"].map(queue_label_from_char4_est)
    return out


def write_outputs(run_dir: Path, pdf: pd.DataFrame, run_meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = run_dir / "merchant_profile_inputs.parquet"
    json_path = run_dir / "merchant_profile_inputs.json"
    jsonl_path = run_dir / "merchant_profile_inputs.jsonl"
    sample_path = run_dir / "merchant_profile_inputs_sample.json"
    summary_path = run_dir / "merchant_profile_inputs_summary.json"
    meta_path = run_dir / "run_meta.json"

    pdf.to_parquet(parquet_path, index=False)
    json_path.write_text(pdf.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in pdf.to_dict(orient="records"):
            fh.write(compact_json(row) + "\n")

    sample_records = pdf.head(SAMPLE_ROWS).to_dict(orient="records")
    safe_json_write(sample_path, sample_records)

    queue_counts = pdf["queue_label_char4_est"].value_counts().sort_index().to_dict()
    segment_counts = pdf["merchant_sampling_segment"].value_counts(dropna=False).sort_index().to_dict()
    token_summary = {
        "rows": int(len(pdf)),
        "prompt_tokens_char4_est_p50": float(pdf["prompt_tokens_char4_est"].quantile(0.50)) if len(pdf) else 0.0,
        "prompt_tokens_char4_est_p90": float(pdf["prompt_tokens_char4_est"].quantile(0.90)) if len(pdf) else 0.0,
        "prompt_tokens_char4_est_p95": float(pdf["prompt_tokens_char4_est"].quantile(0.95)) if len(pdf) else 0.0,
        "prompt_tokens_char4_est_p99": float(pdf["prompt_tokens_char4_est"].quantile(0.99)) if len(pdf) else 0.0,
        "prompt_tokens_char4_est_max": int(pdf["prompt_tokens_char4_est"].max()) if len(pdf) else 0,
    }
    summary = {
        **token_summary,
        "queue_counts": {str(k): int(v) for k, v in queue_counts.items()},
        "merchant_sampling_segment_counts": {str(k): int(v) for k, v in segment_counts.items()},
        "business_universe_mode": str(run_meta.get("business_universe_mode", "merchant_state_v2_all")),
        "business_universe_path": str(run_meta.get("business_universe_path", "")),
        "stage09_hard_filter_policy": run_meta.get("stage09_hard_filter_policy", {}),
        "max_review_evidence": MAX_REVIEW_EVIDENCE,
        "max_tip_evidence": MAX_TIP_EVIDENCE,
        "candidate_recall_mode": CANDIDATE_RECALL_MODE,
        "output_item_caps": {
            "direct_evidence_signals": DIRECT_MAX_ITEMS,
            "fine_grained_attributes": FINE_GRAINED_MAX_ITEMS,
            "usage_scenes": USAGE_SCENE_MAX_ITEMS,
            "risk_or_avoidance_notes": RISK_MAX_ITEMS,
            "cautious_inferences": CAUTIOUS_MAX_ITEMS,
            "evidence_refs": REFS_MAX_ITEMS,
            "evidence_snippets": SNIPPETS_MAX_ITEMS,
        },
        "output_files": {
            "parquet": str(parquet_path),
            "json": str(json_path),
            "jsonl": str(jsonl_path),
            "sample": str(sample_path),
        },
    }
    safe_json_write(summary_path, summary)
    safe_json_write(meta_path, {**run_meta, "summary": summary})
    write_latest_run_pointer(
        "12_merchant_profile_inputs",
        run_dir,
        {"run_tag": RUN_TAG, "rows": int(len(pdf)), "prompt_tokens_char4_est_max": token_summary["prompt_tokens_char4_est_max"]},
    )


def main() -> None:
    merchant_run_dir = resolve_run(
        INPUT_MERCHANT_STATE_RUN_DIR,
        INPUT_MERCHANT_STATE_ROOT,
        "stage09_merchant_state_v2_build",
    )
    run_dir = OUTPUT_ROOT / now_run_id()
    spark = create_spark(RUN_TAG)
    base = None
    review_stats = None
    universe_meta: dict[str, Any] = {}
    try:
        base, universe_meta = build_base_business_frame(spark, merchant_run_dir)
        base = base.persist()
        review_stats, review_evidence = build_review_evidence(spark, base)
        tip_evidence = build_tip_evidence(spark, base)
        final = attach_prompt_sections(base, review_stats, review_evidence, tip_evidence)
        final_count = final.count()
        if final_count > PANDAS_ROW_CAP:
            raise RuntimeError(
                f"final one-row-per-merchant table has {final_count} rows, over MERCHANT_PROFILE_PANDAS_ROW_CAP={PANDAS_ROW_CAP}; "
                "raise the cap intentionally or narrow MERCHANT_PROFILE_MAX_BUSINESSES."
            )
        pdf = final.orderBy("business_id").toPandas()
        pdf = add_prompts(pdf)
    finally:
        if base is not None:
            try:
                base.unpersist()
            except Exception:
                pass
        if review_stats is not None:
            try:
                review_stats.unpersist()
            except Exception:
                pass
        spark.stop()

    run_meta = {
        "run_tag": RUN_TAG,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merchant_state_run_dir": str(merchant_run_dir),
        "raw_business_root": str(RAW_BUSINESS_ROOT),
        "raw_review_root": str(RAW_REVIEW_ROOT),
        "raw_tip_root": str(RAW_TIP_ROOT),
        **universe_meta,
        "max_businesses": MAX_BUSINESSES,
        "max_review_evidence": MAX_REVIEW_EVIDENCE,
        "max_tip_evidence": MAX_TIP_EVIDENCE,
        "review_part_file_limit_debug_only": REVIEW_PART_FILE_LIMIT,
        "tip_part_file_limit_debug_only": TIP_PART_FILE_LIMIT,
        "regular_prompt_token_budget": REGULAR_PROMPT_TOKEN_BUDGET,
        "high_prompt_token_budget": HIGH_PROMPT_TOKEN_BUDGET,
        "candidate_recall_mode": CANDIDATE_RECALL_MODE,
        "output_item_caps": {
            "direct_evidence_signals": DIRECT_MAX_ITEMS,
            "fine_grained_attributes": FINE_GRAINED_MAX_ITEMS,
            "usage_scenes": USAGE_SCENE_MAX_ITEMS,
            "risk_or_avoidance_notes": RISK_MAX_ITEMS,
            "cautious_inferences": CAUTIOUS_MAX_ITEMS,
            "evidence_refs": REFS_MAX_ITEMS,
            "evidence_snippets": SNIPPETS_MAX_ITEMS,
        },
        "spark_master": os.getenv("SPARK_MASTER", "local[2]"),
        "spark_shuffle_partitions": os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "16"),
    }
    write_outputs(run_dir, pdf, run_meta)
    print(f"[OK] wrote {len(pdf)} merchant profile input rows to {run_dir}")


if __name__ == "__main__":
    main()
