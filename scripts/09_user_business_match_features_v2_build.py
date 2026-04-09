from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

INTERACTION_V2_WEIGHTED_ROOT = env_or_project_path(
    'INPUT_09_INTERACTION_V2_WEIGHTED_ROOT_DIR',
    'data/output/09_interaction_v2_weighted',
)
MERCHANT_CARD_ROOT = env_or_project_path(
    'INPUT_09_MERCHANT_CARD_ROOT_DIR',
    'data/output/09_merchant_semantic_card',
)
USER_INTENT_ROOT = env_or_project_path(
    'INPUT_09_USER_INTENT_PROFILE_V2_ROOT_DIR',
    'data/output/09_user_intent_profile_v2',
)

INTERACTION_V2_WEIGHTED_RUN_DIR = os.getenv('INPUT_09_INTERACTION_V2_WEIGHTED_RUN_DIR', '').strip()
MERCHANT_CARD_RUN_DIR = os.getenv('INPUT_09_MERCHANT_CARD_RUN_DIR', '').strip()
USER_INTENT_RUN_DIR = os.getenv('INPUT_09_USER_INTENT_PROFILE_V2_RUN_DIR', '').strip()
SPLIT_AWARE_HISTORY_ONLY = os.getenv('USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_HISTORY_ONLY', 'false').strip().lower() == 'true'
HISTORY_STAGE09_BUCKET_DIR = os.getenv('INPUT_09_STAGE09_BUCKET_DIR', '').strip()
INDEX_MAPS_ROOT = env_or_project_path(
    'INPUT_09_INDEX_MAPS_ROOT_DIR',
    'data/output/09_index_maps',
)
INDEX_MAPS_RUN_DIR = os.getenv('INPUT_09_INDEX_MAPS_RUN_DIR', '').strip()
SPLIT_AWARE_BUCKET = int(os.getenv('USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_BUCKET', '5').strip() or 5)

OUTPUT_ROOT = env_or_project_path(
    'OUTPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_ROOT_DIR',
    'data/output/09_user_business_match_features_v2',
)
RUN_TAG = 'stage09_user_business_match_split_aware_v2_build' if SPLIT_AWARE_HISTORY_ONLY else 'stage09_user_business_match_features_v2_build'

SCENE_COLS = [
    'family_scene_fit',
    'group_scene_fit',
    'date_scene_fit',
    'nightlife_scene_fit',
    'fast_casual_fit',
    'sitdown_fit',
]
USER_SCENE_COLS = [
    'family_scene_pref',
    'group_scene_pref',
    'date_scene_pref',
    'nightlife_scene_pref',
    'fast_casual_pref',
    'sitdown_pref',
]
MEAL_COLS = ['meal_breakfast_fit', 'meal_lunch_fit', 'meal_dinner_fit', 'late_night_fit']
USER_MEAL_COLS = ['breakfast_pref', 'lunch_pref', 'dinner_pref', 'late_night_pref']
PROPERTY_COLS = ['attr_delivery', 'attr_takeout', 'attr_reservations', 'open_weekend', 'open_late_any']
USER_PROPERTY_COLS = ['delivery_pref', 'takeout_pref', 'reservation_pref', 'weekend_pref', 'late_share_pref']


def now_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_full_' + RUN_TAG


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f'no run in {root} with suffix={suffix}')
    return runs[0]


def resolve_optional_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f'run dir not found: {p}')
        return p
    return pick_latest_run(root, suffix)


def resolve_history_only_pairs() -> pd.DataFrame | None:
    if not SPLIT_AWARE_HISTORY_ONLY:
        return None
    if not HISTORY_STAGE09_BUCKET_DIR:
        raise FileNotFoundError('INPUT_09_STAGE09_BUCKET_DIR is required when USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_HISTORY_ONLY=true')
    stage09_bucket_dir = Path(HISTORY_STAGE09_BUCKET_DIR)
    if not stage09_bucket_dir.is_absolute():
        stage09_bucket_dir = project_path(stage09_bucket_dir)
    if not stage09_bucket_dir.exists():
        raise FileNotFoundError(f'stage09 bucket dir not found: {stage09_bucket_dir}')
    history_path = stage09_bucket_dir / 'train_history.parquet'
    if not history_path.exists():
        raise FileNotFoundError(f'train_history.parquet missing: {history_path}')
    index_root = resolve_optional_run(INDEX_MAPS_RUN_DIR, INDEX_MAPS_ROOT, '_full_stage09_index_maps_build')
    bucket_dir = index_root / f'bucket_{int(SPLIT_AWARE_BUCKET)}'
    user_map_path = bucket_dir / 'user_index_map.parquet'
    item_map_path = bucket_dir / 'item_index_map.parquet'
    if not user_map_path.exists():
        raise FileNotFoundError(f'user_index_map.parquet missing: {user_map_path}')
    if not item_map_path.exists():
        raise FileNotFoundError(f'item_index_map.parquet missing: {item_map_path}')
    hist = pd.read_parquet(history_path, columns=['user_idx', 'item_idx'])
    user_map = pd.read_parquet(user_map_path, columns=['user_idx', 'user_id'])
    item_map = pd.read_parquet(item_map_path, columns=['item_idx', 'business_id'])
    hist = hist.merge(user_map, on='user_idx', how='inner').merge(item_map, on='item_idx', how='inner')
    hist = hist[['user_id', 'business_id']].drop_duplicates().reset_index(drop=True)
    if hist.empty:
        raise RuntimeError('split-aware history-only pair set is empty')
    return hist


def safe_json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.replace(0, np.nan)
    return (num / den_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def classify_field(col: pd.Series, name: str) -> str:
    name_l = name.lower()
    if name_l in {'event_id', 'user_id', 'business_id', 'event_type', 'merchant_primary_cuisine', 'merchant_secondary_cuisine', 'city', 'geo_cell_3dp', 'top_city', 'top_geo_cell_3dp'}:
        return 'id_or_text'
    non_null = col.dropna()
    if non_null.empty:
        return 'empty'
    uniq = set(pd.unique(non_null))
    if uniq.issubset({0, 0.0, 1, 1.0, True, False}):
        return 'binary_flag'
    if pd.api.types.is_numeric_dtype(non_null):
        return 'numeric'
    return 'other'


def summarize_fields(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in df.columns:
        kind = classify_field(df[name], name)
        if kind == 'id_or_text':
            continue
        row = {'field': name, 'kind': kind, 'non_null_rate': float(df[name].notna().mean())}
        if kind == 'binary_flag':
            row['positive_rate'] = float(pd.to_numeric(df[name], errors='coerce').fillna(0).mean())
        elif kind == 'numeric':
            num = pd.to_numeric(df[name], errors='coerce')
            row['non_zero_rate'] = float((num.fillna(0) != 0).mean())
            row['mean'] = float(num.mean()) if num.notna().any() else None
            row['p50'] = float(num.quantile(0.5)) if num.notna().any() else None
            row['p90'] = float(num.quantile(0.9)) if num.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['kind', 'field']).reset_index(drop=True)


def build_sample_book(df: pd.DataFrame) -> dict[str, object]:
    cols = [
        'event_id', 'event_type', 'user_id', 'business_id',
        'merchant_primary_cuisine', 'merchant_secondary_cuisine',
        'match_primary_cuisine_taste', 'match_primary_cuisine_negative',
        'match_secondary_cuisine_taste', 'match_secondary_cuisine_negative',
        'match_scene_dot', 'match_meal_dot', 'match_property_dot',
        'match_geo_alignment', 'match_recent_cuisine', 'match_recent_tip_context',
        'match_positive_evidence', 'match_negative_conflict', 'match_total_v1',
    ]
    out = {
        'highest_total_positive_events': df.loc[df['event_type'] == 'review_positive']
        .sort_values(['match_total_v1', 'match_positive_evidence'], ascending=[False, False])
        .head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'highest_conflict_negative_events': df.loc[df['event_type'] == 'review_negative']
        .sort_values(['match_negative_conflict', 'match_primary_cuisine_negative'], ascending=[False, False])
        .head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'strong_recent_tip_events': df.loc[df['event_type'] == 'tip_signal']
        .sort_values(['match_recent_tip_context', 'match_recent_cuisine'], ascending=[False, False])
        .head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
    }
    return out


def main() -> None:
    interaction_run = resolve_optional_run(
        INTERACTION_V2_WEIGHTED_RUN_DIR,
        INTERACTION_V2_WEIGHTED_ROOT,
        '_full_stage09_interaction_v2_weight_build',
    )
    merchant_run = resolve_optional_run(
        MERCHANT_CARD_RUN_DIR,
        MERCHANT_CARD_ROOT,
        '_full_stage09_merchant_semantic_card_build',
    )
    user_intent_run = resolve_optional_run(
        USER_INTENT_RUN_DIR,
        USER_INTENT_ROOT,
        '_full_stage09_user_intent_profile_split_aware_v2_build' if SPLIT_AWARE_HISTORY_ONLY else '_full_stage09_user_intent_profile_v2_build',
    )

    out_dir = OUTPUT_ROOT / now_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(interaction_run / 'interaction_v2_weighted.parquet')
    merchant = pd.read_parquet(merchant_run / 'merchant_semantic_card_v2.parquet')
    user_profile = pd.read_parquet(user_intent_run / 'user_intent_profile_v2.parquet')
    user_cuisine = pd.read_parquet(user_intent_run / 'user_cuisine_pref_v2.parquet')

    trainable_rows_before_history_filter = int(len(events))
    history_pairs = resolve_history_only_pairs()
    if history_pairs is not None:
        events = events.merge(history_pairs, on=['user_id', 'business_id'], how='inner')

    events = events.loc[
        events['user_id'].notna()
        & events['business_id'].notna()
        & pd.to_numeric(events['sample_weight_v2'], errors='coerce').fillna(0).gt(0)
    ].copy()
    if events.empty:
        raise RuntimeError('no trainable event rows for match build')

    merchant_keep = [
        'business_id',
        'city',
        'geo_cell_3dp',
        'merchant_primary_cuisine',
        'merchant_secondary_cuisine',
        *SCENE_COLS,
        *MEAL_COLS,
        *PROPERTY_COLS,
        'weekend_share',
        'late_share',
        'tip_recommend_share',
        'tip_dish_share',
        'tip_time_share',
        'evidence_positive_per_review',
        'evidence_negative_per_review',
    ]
    merchant = merchant[merchant_keep].copy()
    user_keep = [
        'user_id',
        'activity_weight_sum',
        'positive_weight_sum',
        'negative_weight_sum',
        'recent_short_weight_sum',
        'recent_short_tip_weight_sum',
        'has_recommend_cue_weight',
        'has_time_cue_weight',
        'has_dish_cue_weight',
        *USER_SCENE_COLS,
        *USER_MEAL_COLS,
        *USER_PROPERTY_COLS,
        'top_city',
        'top_geo_cell_3dp',
        'geo_concentration_ratio',
        'city_concentration_ratio',
        'negative_pressure',
        'has_recent_intent_view',
        'has_negative_avoidance_view',
    ]
    user_profile = user_profile[user_keep].copy()

    user_profile['recent_recommend_share'] = safe_ratio(
        user_profile['has_recommend_cue_weight'],
        user_profile['recent_short_tip_weight_sum'],
    )
    user_profile['recent_time_share'] = safe_ratio(
        user_profile['has_time_cue_weight'],
        user_profile['recent_short_tip_weight_sum'],
    )
    user_profile['recent_dish_share'] = safe_ratio(
        user_profile['has_dish_cue_weight'],
        user_profile['recent_short_tip_weight_sum'],
    )
    user_profile['positive_share'] = safe_ratio(
        user_profile['positive_weight_sum'],
        user_profile['activity_weight_sum'],
    )

    cuisine_keep = [
        'user_id', 'cuisine', 'positive_weight_sum', 'negative_weight_sum',
        'tip_weight_sum', 'taste_weight_sum', 'net_weight_sum', 'recent_short_weight_sum',
    ]
    user_cuisine = user_cuisine[cuisine_keep].copy()

    primary_cuisine = user_cuisine.rename(columns={
        'cuisine': 'merchant_primary_cuisine',
        'positive_weight_sum': 'primary_cuisine_positive_weight',
        'negative_weight_sum': 'primary_cuisine_negative_weight',
        'tip_weight_sum': 'primary_cuisine_tip_weight',
        'taste_weight_sum': 'primary_cuisine_taste_weight',
        'net_weight_sum': 'primary_cuisine_net_weight',
        'recent_short_weight_sum': 'primary_cuisine_recent_weight',
    })
    secondary_cuisine = user_cuisine.rename(columns={
        'cuisine': 'merchant_secondary_cuisine',
        'positive_weight_sum': 'secondary_cuisine_positive_weight',
        'negative_weight_sum': 'secondary_cuisine_negative_weight',
        'tip_weight_sum': 'secondary_cuisine_tip_weight',
        'taste_weight_sum': 'secondary_cuisine_taste_weight',
        'net_weight_sum': 'secondary_cuisine_net_weight',
        'recent_short_weight_sum': 'secondary_cuisine_recent_weight',
    })

    events = events.merge(merchant, on='business_id', how='left')
    events['city'] = events['city_y'].fillna(events['city_x']) if 'city_y' in events.columns else events['city']
    events['geo_cell_3dp'] = (
        events['geo_cell_3dp_y'].fillna(events['geo_cell_3dp_x'])
        if 'geo_cell_3dp_y' in events.columns
        else events['geo_cell_3dp']
    )
    for col in ['city_x', 'city_y', 'geo_cell_3dp_x', 'geo_cell_3dp_y']:
        if col in events.columns:
            events = events.drop(columns=[col])

    events = events.merge(user_profile, on='user_id', how='left')
    events = events.merge(primary_cuisine, on=['user_id', 'merchant_primary_cuisine'], how='left')
    events = events.merge(secondary_cuisine, on=['user_id', 'merchant_secondary_cuisine'], how='left')

    fill_zero_cols = [
        'primary_cuisine_positive_weight', 'primary_cuisine_negative_weight', 'primary_cuisine_tip_weight',
        'primary_cuisine_taste_weight', 'primary_cuisine_net_weight', 'primary_cuisine_recent_weight',
        'secondary_cuisine_positive_weight', 'secondary_cuisine_negative_weight', 'secondary_cuisine_tip_weight',
        'secondary_cuisine_taste_weight', 'secondary_cuisine_net_weight', 'secondary_cuisine_recent_weight',
    ]
    for col in fill_zero_cols:
        events[col] = pd.to_numeric(events[col], errors='coerce').fillna(0.0)

    activity_den = events['activity_weight_sum'].replace(0, np.nan)
    recent_den = events['recent_short_weight_sum'].replace(0, np.nan)
    neg_den = events['negative_weight_sum'].replace(0, np.nan)

    events['match_primary_cuisine_taste'] = safe_ratio(events['primary_cuisine_taste_weight'], events['activity_weight_sum'])
    events['match_primary_cuisine_negative'] = safe_ratio(events['primary_cuisine_negative_weight'], events['negative_weight_sum'])
    events['match_secondary_cuisine_taste'] = safe_ratio(events['secondary_cuisine_taste_weight'], events['activity_weight_sum'])
    events['match_secondary_cuisine_negative'] = safe_ratio(events['secondary_cuisine_negative_weight'], events['negative_weight_sum'])
    events['match_cuisine_net'] = safe_ratio(
        events['primary_cuisine_net_weight'] + 0.6 * events['secondary_cuisine_net_weight'],
        events['activity_weight_sum'],
    )
    events['match_recent_cuisine'] = safe_ratio(
        events['primary_cuisine_recent_weight'] + 0.6 * events['secondary_cuisine_recent_weight'],
        events['recent_short_weight_sum'],
    )

    scene_terms = []
    for merchant_col, user_col in zip(SCENE_COLS, USER_SCENE_COLS):
        scene_terms.append(pd.to_numeric(events[merchant_col], errors='coerce').fillna(0.0) * pd.to_numeric(events[user_col], errors='coerce').fillna(0.0))
    events['match_scene_dot'] = np.mean(np.column_stack(scene_terms), axis=1)

    meal_terms = []
    for merchant_col, user_col in zip(MEAL_COLS, USER_MEAL_COLS):
        meal_terms.append(pd.to_numeric(events[merchant_col], errors='coerce').fillna(0.0) * pd.to_numeric(events[user_col], errors='coerce').fillna(0.0))
    events['match_meal_dot'] = np.mean(np.column_stack(meal_terms), axis=1)

    property_pairs = list(zip(PROPERTY_COLS, USER_PROPERTY_COLS))
    property_terms = []
    for merchant_col, user_col in property_pairs:
        property_terms.append(pd.to_numeric(events[merchant_col], errors='coerce').fillna(0.0) * pd.to_numeric(events[user_col], errors='coerce').fillna(0.0))
    events['match_property_dot'] = np.mean(np.column_stack(property_terms), axis=1)

    events['match_geo_city'] = events['top_city'].fillna('').eq(events['city'].fillna('')).astype(float)
    events['match_geo_cell'] = events['top_geo_cell_3dp'].fillna('').eq(events['geo_cell_3dp'].fillna('')).astype(float)
    city_geo_align = events['city_concentration_ratio'].fillna(0.0) * events['match_geo_city']
    cell_geo_align = events['geo_concentration_ratio'].fillna(0.0) * events['match_geo_cell']
    events['match_geo_alignment'] = np.maximum(city_geo_align, cell_geo_align)

    events['match_recent_tip_context'] = (
        events['recent_recommend_share'].fillna(0.0) * pd.to_numeric(events['tip_recommend_share'], errors='coerce').fillna(0.0)
        + events['recent_time_share'].fillna(0.0) * pd.to_numeric(events['tip_time_share'], errors='coerce').fillna(0.0)
        + events['recent_dish_share'].fillna(0.0) * pd.to_numeric(events['tip_dish_share'], errors='coerce').fillna(0.0)
    ) / 3.0

    cuisine_support = events['match_primary_cuisine_taste'] + 0.6 * events['match_secondary_cuisine_taste']
    cuisine_conflict = events['match_primary_cuisine_negative'] + 0.6 * events['match_secondary_cuisine_negative']
    events['match_positive_evidence'] = pd.to_numeric(events['evidence_positive_per_review'], errors='coerce').fillna(0.0) * cuisine_support
    events['match_negative_conflict'] = (
        pd.to_numeric(events['evidence_negative_per_review'], errors='coerce').fillna(0.0)
        * (cuisine_conflict + 0.5 * events['negative_pressure'].fillna(0.0))
    )

    events['match_time_context'] = (
        events['weekend_pref'].fillna(0.0) * pd.to_numeric(events['weekend_share'], errors='coerce').fillna(0.0)
        + events['late_share_pref'].fillna(0.0) * pd.to_numeric(events['late_share'], errors='coerce').fillna(0.0)
    ) / 2.0

    events['match_total_v1'] = (
        0.30 * events['match_cuisine_net']
        + 0.15 * events['match_recent_cuisine']
        + 0.15 * events['match_scene_dot']
        + 0.10 * events['match_meal_dot']
        + 0.10 * events['match_property_dot']
        + 0.10 * events['match_geo_alignment']
        + 0.05 * events['match_recent_tip_context']
        + 0.05 * events['match_time_context']
    )

    out_cols = [
        'event_id', 'event_type', 'user_id', 'business_id',
        'merchant_primary_cuisine', 'merchant_secondary_cuisine',
        'city', 'geo_cell_3dp',
        'sample_weight_v2',
        'match_primary_cuisine_taste', 'match_primary_cuisine_negative',
        'match_secondary_cuisine_taste', 'match_secondary_cuisine_negative',
        'match_cuisine_net', 'match_recent_cuisine',
        'match_scene_dot', 'match_meal_dot', 'match_property_dot',
        'match_geo_city', 'match_geo_cell', 'match_geo_alignment',
        'match_recent_tip_context', 'match_time_context',
        'match_positive_evidence', 'match_negative_conflict',
        'match_total_v1',
    ]
    match_df = events[out_cols].copy()

    user_item_df = match_df.groupby(['user_id', 'business_id'], as_index=False).agg(
        event_rows=('event_id', 'count'),
        positive_event_rows=('event_type', lambda s: int((s == 'review_positive').sum())),
        negative_event_rows=('event_type', lambda s: int((s == 'review_negative').sum())),
        tip_event_rows=('event_type', lambda s: int((s == 'tip_signal').sum())),
        mean_match_total_v1=('match_total_v1', 'mean'),
        max_match_total_v1=('match_total_v1', 'max'),
        mean_match_recent_cuisine=('match_recent_cuisine', 'mean'),
        mean_match_geo_alignment=('match_geo_alignment', 'mean'),
        mean_match_positive_evidence=('match_positive_evidence', 'mean'),
        mean_match_negative_conflict=('match_negative_conflict', 'mean'),
    )

    event_type_summary = match_df.groupby('event_type', as_index=False).agg(
        n_rows=('event_id', 'count'),
        mean_match_total_v1=('match_total_v1', 'mean'),
        mean_match_cuisine_net=('match_cuisine_net', 'mean'),
        mean_match_recent_cuisine=('match_recent_cuisine', 'mean'),
        mean_match_scene_dot=('match_scene_dot', 'mean'),
        mean_match_geo_alignment=('match_geo_alignment', 'mean'),
        mean_match_positive_evidence=('match_positive_evidence', 'mean'),
        mean_match_negative_conflict=('match_negative_conflict', 'mean'),
    ).sort_values('event_type')

    field_summary = summarize_fields(match_df)
    sample_book = build_sample_book(match_df)

    match_df.to_parquet(out_dir / 'user_business_match_features_v2_event_rows.parquet', index=False)
    user_item_df.to_parquet(out_dir / 'user_business_match_features_v2_user_item.parquet', index=False)
    event_type_summary.to_csv(out_dir / 'user_business_match_features_v2_event_type_summary.csv', index=False)
    field_summary.to_csv(out_dir / 'user_business_match_features_v2_field_summary.csv', index=False)
    safe_json_write(out_dir / 'user_business_match_features_v2_sample.json', sample_book)

    run_meta = {
        'run_id': out_dir.name,
        'run_tag': RUN_TAG,
        'input_run_dirs': {
            'interaction_v2_weighted_run_dir': str(interaction_run),
            'merchant_semantic_card_run_dir': str(merchant_run),
            'user_intent_profile_v2_run_dir': str(user_intent_run),
        },
        'schema_rule_v1': {
            'scope': 'trainable events only',
            'split_aware_history_only': bool(history_pairs is not None),
            'split_aware_bucket': int(SPLIT_AWARE_BUCKET) if history_pairs is not None else None,
            'merchant_asset_required': 'merchant_semantic_card_v2',
            'user_asset_required': 'user_intent_profile_v2 + user_cuisine_pref_v2',
            'event_level_output': True,
            'user_item_agg_output': True,
            'checkin_direct_user_item_feedback_allowed': False,
        },
        'row_counts': {
            'trainable_event_rows_before_history_filter': trainable_rows_before_history_filter,
            'event_match_rows': int(len(match_df)),
            'user_item_rows': int(len(user_item_df)),
            'users': int(match_df['user_id'].nunique()),
            'businesses': int(match_df['business_id'].nunique()),
        },
        'event_type_summary': event_type_summary.to_dict(orient='records'),
        'output_files': {
            'event_rows_parquet': str(out_dir / 'user_business_match_features_v2_event_rows.parquet'),
            'user_item_parquet': str(out_dir / 'user_business_match_features_v2_user_item.parquet'),
            'event_type_summary_csv': str(out_dir / 'user_business_match_features_v2_event_type_summary.csv'),
            'field_summary_csv': str(out_dir / 'user_business_match_features_v2_field_summary.csv'),
            'sample_json': str(out_dir / 'user_business_match_features_v2_sample.json'),
        },
    }
    safe_json_write(out_dir / 'run_meta.json', run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
