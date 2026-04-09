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

INTERACTION_V2_WEIGHTED_RUN_DIR = os.getenv('INPUT_09_INTERACTION_V2_WEIGHTED_RUN_DIR', '').strip()
MERCHANT_CARD_RUN_DIR = os.getenv('INPUT_09_MERCHANT_CARD_RUN_DIR', '').strip()

OUTPUT_ROOT = env_or_project_path(
    'OUTPUT_09_USER_INTENT_PROFILE_V2_ROOT_DIR',
    'data/output/09_user_intent_profile_v2',
)
SPLIT_AWARE_HISTORY_ONLY = os.getenv('USER_INTENT_PROFILE_V2_SPLIT_AWARE_HISTORY_ONLY', 'false').strip().lower() == 'true'
HISTORY_STAGE09_BUCKET_DIR = os.getenv('INPUT_09_STAGE09_BUCKET_DIR', '').strip()
INDEX_MAPS_ROOT = env_or_project_path(
    'INPUT_09_INDEX_MAPS_ROOT_DIR',
    'data/output/09_index_maps',
)
INDEX_MAPS_RUN_DIR = os.getenv('INPUT_09_INDEX_MAPS_RUN_DIR', '').strip()
SPLIT_AWARE_BUCKET = int(os.getenv('USER_INTENT_PROFILE_V2_SPLIT_AWARE_BUCKET', '5').strip() or 5)
RUN_TAG = 'stage09_user_intent_profile_split_aware_v2_build' if SPLIT_AWARE_HISTORY_ONLY else 'stage09_user_intent_profile_v2_build'

TIP_AUX_WEIGHT = float(os.getenv('USER_INTENT_PROFILE_V2_TIP_AUX_WEIGHT', '0.35'))
NEUTRAL_AUX_WEIGHT = float(os.getenv('USER_INTENT_PROFILE_V2_NEUTRAL_AUX_WEIGHT', '0.15'))
NEGATIVE_PENALTY_WEIGHT = float(os.getenv('USER_INTENT_PROFILE_V2_NEGATIVE_PENALTY_WEIGHT', '0.75'))
RECENT_SHORT_DAYS = int(os.getenv('USER_INTENT_PROFILE_V2_RECENT_SHORT_DAYS', '180'))
RECENT_LONG_DAYS = int(os.getenv('USER_INTENT_PROFILE_V2_RECENT_LONG_DAYS', '365'))
RECENT_LATEST_EVENT_COUNT = int(os.getenv('USER_INTENT_PROFILE_V2_RECENT_LATEST_EVENT_COUNT', '3'))
TOPK_PER_USER = int(os.getenv('USER_INTENT_PROFILE_V2_TOPK_PER_USER', '5'))

SCENE_COLS = [
    'family_scene_fit',
    'group_scene_fit',
    'date_scene_fit',
    'nightlife_scene_fit',
    'fast_casual_fit',
    'sitdown_fit',
]
MEAL_COLS = [
    'meal_breakfast_fit',
    'meal_lunch_fit',
    'meal_dinner_fit',
    'late_night_fit',
]
PROPERTY_COLS = [
    'attr_delivery',
    'attr_takeout',
    'attr_reservations',
    'open_weekend',
    'open_late_any',
]
RECENT_PREF_GROUPS = {
    'meal': {
        'recent_breakfast_pref': 'meal_breakfast_fit',
        'recent_lunch_pref': 'meal_lunch_fit',
        'recent_dinner_pref': 'meal_dinner_fit',
        'recent_late_night_pref': 'late_night_fit',
    },
    'scene': {
        'recent_family_scene_pref': 'family_scene_fit',
        'recent_group_scene_pref': 'group_scene_fit',
        'recent_date_scene_pref': 'date_scene_fit',
        'recent_nightlife_scene_pref': 'nightlife_scene_fit',
        'recent_fast_casual_pref': 'fast_casual_fit',
        'recent_sitdown_pref': 'sitdown_fit',
    },
    'property': {
        'recent_delivery_pref': 'attr_delivery',
        'recent_takeout_pref': 'attr_takeout',
        'recent_reservation_pref': 'attr_reservations',
        'recent_weekend_pref': 'weekend_share',
        'recent_late_share_pref': 'late_share',
    },
}
LATEST_PREF_GROUPS = {
    'meal': {
        'latest_breakfast_pref': 'meal_breakfast_fit',
        'latest_lunch_pref': 'meal_lunch_fit',
        'latest_dinner_pref': 'meal_dinner_fit',
        'latest_late_night_pref': 'late_night_fit',
    },
    'scene': {
        'latest_family_scene_pref': 'family_scene_fit',
        'latest_group_scene_pref': 'group_scene_fit',
        'latest_date_scene_pref': 'date_scene_fit',
        'latest_nightlife_scene_pref': 'nightlife_scene_fit',
        'latest_fast_casual_pref': 'fast_casual_fit',
        'latest_sitdown_pref': 'sitdown_fit',
    },
    'property': {
        'latest_delivery_pref': 'attr_delivery',
        'latest_takeout_pref': 'attr_takeout',
        'latest_reservation_pref': 'attr_reservations',
        'latest_weekend_pref': 'weekend_share',
        'latest_late_share_pref': 'late_share',
    },
}


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
        raise FileNotFoundError('INPUT_09_STAGE09_BUCKET_DIR is required when USER_INTENT_PROFILE_V2_SPLIT_AWARE_HISTORY_ONLY=true')
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


def bounded_quantile(series: pd.Series, q: float, default: float, low: float, high: float) -> float:
    valid = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return float(default)
    try:
        value = float(valid.quantile(float(q)))
    except Exception:
        value = float(default)
    return float(min(max(value, low), high))


def classify_field(col: pd.Series, name: str) -> str:
    name_l = name.lower()
    if name_l in {
        'user_id',
        'top_city',
        'top_geo_cell_3dp',
        'long_term_top_cuisine',
        'recent_top_cuisine',
        'typed_intent_primary_cuisine',
        'typed_intent_primary_source',
        'negative_top_cuisine',
    }:
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


def top_label_map(df: pd.DataFrame, user_col: str, label_col: str, score_col: str) -> pd.DataFrame:
    work = df.loc[df[label_col].notna() & df[score_col].gt(0), [user_col, label_col, score_col]].copy()
    if work.empty:
        return pd.DataFrame(columns=[user_col, label_col, score_col, 'rank'])
    work = work.sort_values([user_col, score_col, label_col], ascending=[True, False, True])
    work['rank'] = work.groupby(user_col).cumcount() + 1
    return work


def top2_score_frame(df: pd.DataFrame, user_col: str, label_col: str, score_col: str, prefix: str) -> pd.DataFrame:
    work = df.loc[df[label_col].notna(), [user_col, label_col, score_col]].copy()
    if work.empty:
        return pd.DataFrame(columns=[user_col, f'{prefix}_top1_raw', f'{prefix}_top2_raw'])
    work[score_col] = pd.to_numeric(work[score_col], errors='coerce').fillna(0.0)
    work = work.loc[work[score_col].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=[user_col, f'{prefix}_top1_raw', f'{prefix}_top2_raw'])
    work = work.sort_values([user_col, score_col, label_col], ascending=[True, False, True])
    work = work.groupby(user_col).head(2).copy()
    work['rank'] = work.groupby(user_col).cumcount() + 1
    top1 = work.loc[work['rank'] == 1, [user_col, score_col]].rename(columns={score_col: f'{prefix}_top1_raw'})
    top2 = work.loc[work['rank'] == 2, [user_col, score_col]].rename(columns={score_col: f'{prefix}_top2_raw'})
    out = top1.merge(top2, on=user_col, how='left')
    out[f'{prefix}_top1_raw'] = pd.to_numeric(out[f'{prefix}_top1_raw'], errors='coerce').fillna(0.0)
    out[f'{prefix}_top2_raw'] = pd.to_numeric(out[f'{prefix}_top2_raw'], errors='coerce').fillna(0.0)
    return out


def expand_cuisine_rows(events: pd.DataFrame) -> pd.DataFrame:
    primary = events[['user_id', 'merchant_primary_cuisine', 'merchant_primary_cuisine_share',
                      'positive_weight', 'negative_weight', 'tip_weight', 'neutral_weight',
                      'taste_weight', 'net_weight', 'recent_short_taste_weight',
                      'recent_long_taste_weight']].copy()
    primary = primary.rename(columns={
        'merchant_primary_cuisine': 'cuisine',
        'merchant_primary_cuisine_share': 'cuisine_share',
    })

    secondary = events[['user_id', 'merchant_secondary_cuisine', 'merchant_secondary_cuisine_share',
                        'positive_weight', 'negative_weight', 'tip_weight', 'neutral_weight',
                        'taste_weight', 'net_weight', 'recent_short_taste_weight',
                        'recent_long_taste_weight']].copy()
    secondary = secondary.rename(columns={
        'merchant_secondary_cuisine': 'cuisine',
        'merchant_secondary_cuisine_share': 'cuisine_share',
    })

    out = pd.concat([primary, secondary], ignore_index=True)
    out = out.loc[out['cuisine'].notna() & out['cuisine'].astype(str).ne('') & out['cuisine_share'].gt(0)].copy()
    if out.empty:
        return out
    for base in [
        'positive_weight', 'negative_weight', 'tip_weight', 'neutral_weight',
        'taste_weight', 'net_weight', 'recent_short_taste_weight', 'recent_long_taste_weight',
    ]:
        out[base] = out[base] * out['cuisine_share']
    return out


def long_format_pref(events: pd.DataFrame, value_cols: list[str], label_map: dict[str, str], user_col: str = 'user_id') -> pd.DataFrame:
    out_rows: list[pd.DataFrame] = []
    for col, label in label_map.items():
        part = events[[user_col, *value_cols]].copy()
        part['label'] = label
        part['weight'] = part[value_cols[0]] * events[col]
        if len(value_cols) > 1:
            for extra in value_cols[1:]:
                part[extra] = part[extra] * events[col]
        keep = [user_col, 'label', 'weight', *value_cols[1:]]
        out_rows.append(part[keep])
    return pd.concat(out_rows, ignore_index=True)


def build_sample_book(user_df: pd.DataFrame, cuisine_df: pd.DataFrame, geo_df: pd.DataFrame) -> dict[str, object]:
    cols = [
        'user_id',
        'n_train_max',
        'n_events_trainable',
        'positive_weight_sum',
        'negative_weight_sum',
        'tip_weight_sum',
        'long_term_top_cuisine',
        'recent_top_cuisine',
        'latest_top_cuisine',
        'negative_top_cuisine',
        'typed_intent_primary_cuisine',
        'typed_intent_primary_source',
        'typed_intent_confidence',
        'typed_intent_confidence_band',
        'typed_intent_conflict_score',
        'typed_intent_conflict_band',
        'typed_intent_staleness_score',
        'typed_intent_staleness_band',
        'top_city',
        'top_geo_cell_3dp',
        'dinner_pref',
        'late_night_pref',
        'weekend_pref',
        'family_scene_pref',
        'nightlife_scene_pref',
        'geo_concentration_ratio',
        'cuisine_shift_flag',
        'has_latest_recent_fallback_view',
        'negative_pressure',
    ]
    sample = {
        'late_night_users': user_df.sort_values(['late_night_pref', 'nightlife_scene_pref'], ascending=[False, False]).head(8)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'family_users': user_df.sort_values(['family_scene_pref', 'group_scene_pref'], ascending=[False, False]).head(8)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'negative_heavy_users': user_df.sort_values(['negative_pressure', 'negative_weight_sum'], ascending=[False, False]).head(8)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'cuisine_shift_users': user_df.loc[user_df['cuisine_shift_flag'] == 1].sort_values(['recent_short_weight_sum', 'positive_weight_sum'], ascending=[False, False]).head(8)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'geo_concentrated_users': user_df.sort_values(['geo_concentration_ratio', 'positive_weight_sum'], ascending=[False, False]).head(8)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'top_cuisine_rows': cuisine_df.sort_values(['rank_net', 'net_weight_sum'], ascending=[True, False]).head(12).replace({np.nan: None}).to_dict(orient='records'),
        'top_geo_rows': geo_df.sort_values(['rank_taste', 'taste_weight_sum'], ascending=[True, False]).head(12).replace({np.nan: None}).to_dict(orient='records'),
    }
    return sample


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

    out_dir = OUTPUT_ROOT / now_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(interaction_run / 'interaction_v2_weighted.parquet')
    merchant = pd.read_parquet(merchant_run / 'merchant_semantic_card_v2.parquet')

    history_pairs = resolve_history_only_pairs()
    if history_pairs is not None:
        events = events.merge(history_pairs, on=['user_id', 'business_id'], how='inner')

    events = events.loc[
        events['user_id'].notna()
        & events['business_id'].notna()
        & pd.to_numeric(events['sample_weight_v2'], errors='coerce').fillna(0).gt(0)
    ].copy()
    if events.empty:
        raise RuntimeError('no trainable events after sample_weight_v2 > 0 filter')

    events['event_time'] = pd.to_datetime(events['event_time'], errors='coerce')
    max_event_time = events['event_time'].max()
    events['days_from_latest'] = (max_event_time - events['event_time']).dt.days.fillna(99999).astype(int)
    events['sample_weight_v2'] = pd.to_numeric(events['sample_weight_v2'], errors='coerce').fillna(0.0)
    events['n_train'] = pd.to_numeric(events['n_train'], errors='coerce').fillna(0).astype(int)

    merchant_keep = [
        'business_id',
        'city',
        'geo_cell_3dp',
        'merchant_primary_cuisine',
        'merchant_primary_cuisine_score',
        'merchant_secondary_cuisine',
        'merchant_secondary_cuisine_score',
        *MEAL_COLS,
        *SCENE_COLS,
        *PROPERTY_COLS,
        'weekend_share',
        'breakfast_share',
        'lunch_share',
        'dinner_share',
        'late_share',
        'tip_recommend_share',
        'tip_dish_share',
        'tip_time_share',
        'evidence_positive_per_review',
        'evidence_negative_per_review',
    ]
    merchant = merchant[merchant_keep].copy()
    for col in merchant.columns:
        if col in {'business_id', 'city', 'geo_cell_3dp', 'merchant_primary_cuisine', 'merchant_secondary_cuisine'}:
            continue
        merchant[col] = pd.to_numeric(merchant[col], errors='coerce').fillna(0.0)

    score_sum = (
        merchant['merchant_primary_cuisine_score'].fillna(0.0)
        + merchant['merchant_secondary_cuisine_score'].fillna(0.0)
    )
    merchant['merchant_primary_cuisine_share'] = np.where(
        score_sum > 0,
        merchant['merchant_primary_cuisine_score'] / score_sum,
        np.where(merchant['merchant_primary_cuisine'].notna(), 1.0, 0.0),
    )
    merchant['merchant_secondary_cuisine_share'] = np.where(
        score_sum > 0,
        merchant['merchant_secondary_cuisine_score'] / score_sum,
        0.0,
    )

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

    events['positive_weight'] = np.where(events['event_type'] == 'review_positive', events['sample_weight_v2'], 0.0)
    events['negative_weight'] = np.where(events['event_type'] == 'review_negative', events['sample_weight_v2'], 0.0)
    events['neutral_weight'] = np.where(events['event_type'] == 'review_neutral', events['sample_weight_v2'], 0.0)
    events['tip_weight'] = np.where(events['event_type'] == 'tip_signal', events['sample_weight_v2'], 0.0)
    events['activity_weight'] = (
        events['positive_weight']
        + events['negative_weight']
        + events['neutral_weight']
        + events['tip_weight']
    )
    events['taste_weight'] = (
        events['positive_weight']
        + TIP_AUX_WEIGHT * events['tip_weight']
        + NEUTRAL_AUX_WEIGHT * events['neutral_weight']
    )
    events['net_weight'] = events['taste_weight'] - NEGATIVE_PENALTY_WEIGHT * events['negative_weight']
    events['recent_short_taste_weight'] = np.where(events['days_from_latest'] <= RECENT_SHORT_DAYS, events['taste_weight'], 0.0)
    events['recent_long_taste_weight'] = np.where(events['days_from_latest'] <= RECENT_LONG_DAYS, events['taste_weight'], 0.0)
    events['recent_short_tip_weight'] = np.where(events['days_from_latest'] <= RECENT_SHORT_DAYS, events['tip_weight'], 0.0)
    events['recent_recommend_cue_weight'] = events['has_recommend_cue'].fillna(0.0) * events['recent_short_tip_weight']
    events['recent_time_cue_weight'] = events['has_time_cue'].fillna(0.0) * events['recent_short_tip_weight']
    events['recent_dish_cue_weight'] = events['has_dish_cue'].fillna(0.0) * events['recent_short_tip_weight']

    user_base = events.groupby('user_id', as_index=False).agg(
        n_events_trainable=('event_id', 'count'),
        n_businesses_trainable=('business_id', 'nunique'),
        n_train_max=('n_train', 'max'),
        activity_weight_sum=('activity_weight', 'sum'),
        positive_weight_sum=('positive_weight', 'sum'),
        negative_weight_sum=('negative_weight', 'sum'),
        neutral_weight_sum=('neutral_weight', 'sum'),
        tip_weight_sum=('tip_weight', 'sum'),
        taste_weight_sum=('taste_weight', 'sum'),
        net_weight_sum=('net_weight', 'sum'),
        recent_short_weight_sum=('recent_short_taste_weight', 'sum'),
        recent_long_weight_sum=('recent_long_taste_weight', 'sum'),
        recent_short_tip_weight_sum=('recent_short_tip_weight', 'sum'),
        has_recommend_cue_weight=('recent_recommend_cue_weight', 'sum'),
        has_time_cue_weight=('recent_time_cue_weight', 'sum'),
        has_dish_cue_weight=('recent_dish_cue_weight', 'sum'),
    )

    taste_den = user_base['taste_weight_sum'].replace(0, np.nan)
    total_signal_den = (user_base['positive_weight_sum'] + user_base['negative_weight_sum'] + user_base['tip_weight_sum']).replace(0, np.nan)
    user_base['negative_pressure'] = safe_ratio(user_base['negative_weight_sum'], user_base['positive_weight_sum'] + user_base['negative_weight_sum'] + 0.35 * user_base['tip_weight_sum'])
    user_base['tip_aux_share'] = safe_ratio(user_base['tip_weight_sum'], user_base['positive_weight_sum'] + user_base['tip_weight_sum'])

    pref_specs = {
        'breakfast_pref': 'meal_breakfast_fit',
        'lunch_pref': 'meal_lunch_fit',
        'dinner_pref': 'meal_dinner_fit',
        'late_night_pref': 'late_night_fit',
        'family_scene_pref': 'family_scene_fit',
        'group_scene_pref': 'group_scene_fit',
        'date_scene_pref': 'date_scene_fit',
        'nightlife_scene_pref': 'nightlife_scene_fit',
        'fast_casual_pref': 'fast_casual_fit',
        'sitdown_pref': 'sitdown_fit',
        'delivery_pref': 'attr_delivery',
        'takeout_pref': 'attr_takeout',
        'reservation_pref': 'attr_reservations',
        'weekend_pref': 'weekend_share',
        'breakfast_share_pref': 'breakfast_share',
        'lunch_share_pref': 'lunch_share',
        'dinner_share_pref': 'dinner_share',
        'late_share_pref': 'late_share',
    }

    pref_rows: list[pd.DataFrame] = []
    for out_col, src_col in pref_specs.items():
        part = events[['user_id', 'taste_weight', src_col]].copy()
        part['weighted_value'] = part['taste_weight'] * part[src_col]
        pref = part.groupby('user_id', as_index=False)['weighted_value'].sum().rename(columns={'weighted_value': out_col})
        pref_rows.append(pref)
    for pref in pref_rows:
        user_base = user_base.merge(pref, on='user_id', how='left')
    for out_col in pref_specs:
        user_base[out_col] = safe_ratio(user_base[out_col].fillna(0.0), user_base['taste_weight_sum'])

    recent_pref_rows: list[pd.DataFrame] = []
    for _, pref_group in RECENT_PREF_GROUPS.items():
        for out_col, src_col in pref_group.items():
            part = events[['user_id', 'recent_short_taste_weight', 'recent_long_taste_weight', src_col]].copy()
            part['recent_short_weighted_value'] = part['recent_short_taste_weight'] * part[src_col]
            part['recent_long_weighted_value'] = part['recent_long_taste_weight'] * part[src_col]
            pref = (
                part.groupby('user_id', as_index=False)[['recent_short_weighted_value', 'recent_long_weighted_value']]
                .sum()
                .rename(
                    columns={
                        'recent_short_weighted_value': f'{out_col}_short_raw',
                        'recent_long_weighted_value': f'{out_col}_long_raw',
                    }
                )
            )
            recent_pref_rows.append(pref)
    for pref in recent_pref_rows:
        user_base = user_base.merge(pref, on='user_id', how='left')

    recent_short_den = user_base['recent_short_weight_sum'].replace(0, np.nan)
    recent_long_den = user_base['recent_long_weight_sum'].replace(0, np.nan)
    for _, pref_group in RECENT_PREF_GROUPS.items():
        for out_col in pref_group:
            short_raw = f'{out_col}_short_raw'
            long_raw = f'{out_col}_long_raw'
            short_col = f'{out_col}_short'
            long_col = f'{out_col}_long'
            user_base[short_col] = safe_ratio(user_base[short_raw].fillna(0.0), recent_short_den)
            user_base[long_col] = safe_ratio(user_base[long_raw].fillna(0.0), recent_long_den)
            user_base[out_col] = np.where(
                user_base['recent_short_weight_sum'].fillna(0.0) > 0,
                user_base[short_col],
                user_base[long_col],
            )
            user_base = user_base.drop(columns=[short_raw, long_raw, short_col, long_col])

    latest_events = (
        events.loc[events['taste_weight'].gt(0)]
        .sort_values(['user_id', 'event_time', 'taste_weight', 'event_id'], ascending=[True, False, False, True])
        .groupby('user_id')
        .head(RECENT_LATEST_EVENT_COUNT)
        .copy()
    )
    latest_user_base = latest_events.groupby('user_id', as_index=False).agg(
        latest_event_count=('event_id', 'count'),
        latest_event_weight_sum=('taste_weight', 'sum'),
    )
    user_base = user_base.merge(latest_user_base, on='user_id', how='left')
    user_base['latest_event_count'] = pd.to_numeric(user_base['latest_event_count'], errors='coerce').fillna(0).astype(int)
    user_base['latest_event_weight_sum'] = pd.to_numeric(user_base['latest_event_weight_sum'], errors='coerce').fillna(0.0)

    latest_pref_rows: list[pd.DataFrame] = []
    for _, pref_group in LATEST_PREF_GROUPS.items():
        for out_col, src_col in pref_group.items():
            part = latest_events[['user_id', 'taste_weight', src_col]].copy()
            part['latest_weighted_value'] = part['taste_weight'] * part[src_col]
            pref = (
                part.groupby('user_id', as_index=False)['latest_weighted_value']
                .sum()
                .rename(columns={'latest_weighted_value': f'{out_col}_raw'})
            )
            latest_pref_rows.append(pref)
    for pref in latest_pref_rows:
        user_base = user_base.merge(pref, on='user_id', how='left')
    latest_den = user_base['latest_event_weight_sum'].replace(0, np.nan)
    for _, pref_group in LATEST_PREF_GROUPS.items():
        for out_col in pref_group:
            raw_col = f'{out_col}_raw'
            user_base[out_col] = safe_ratio(user_base[raw_col].fillna(0.0), latest_den)
            user_base = user_base.drop(columns=[raw_col])

    cuisine_rows = expand_cuisine_rows(events)
    cuisine_pref = cuisine_rows.groupby(['user_id', 'cuisine'], as_index=False).agg(
        positive_weight_sum=('positive_weight', 'sum'),
        negative_weight_sum=('negative_weight', 'sum'),
        tip_weight_sum=('tip_weight', 'sum'),
        neutral_weight_sum=('neutral_weight', 'sum'),
        taste_weight_sum=('taste_weight', 'sum'),
        net_weight_sum=('net_weight', 'sum'),
        recent_short_weight_sum=('recent_short_taste_weight', 'sum'),
        recent_long_weight_sum=('recent_long_taste_weight', 'sum'),
    )
    cuisine_pref = cuisine_pref.sort_values(['user_id', 'net_weight_sum', 'recent_short_weight_sum', 'cuisine'], ascending=[True, False, False, True])
    cuisine_pref['rank_net'] = cuisine_pref.groupby('user_id').cumcount() + 1
    cuisine_pref['rank_recent_short'] = cuisine_pref.groupby('user_id')['recent_short_weight_sum'].rank(method='first', ascending=False)
    cuisine_pref['rank_negative'] = cuisine_pref.groupby('user_id')['negative_weight_sum'].rank(method='first', ascending=False)
    cuisine_pref['rank_recent_short'] = cuisine_pref['rank_recent_short'].astype(int)
    cuisine_pref['rank_negative'] = cuisine_pref['rank_negative'].astype(int)

    top_long = top_label_map(cuisine_pref, 'user_id', 'cuisine', 'taste_weight_sum')
    top_long = top_long.loc[top_long['rank'] == 1, ['user_id', 'cuisine']].rename(columns={'cuisine': 'long_term_top_cuisine'})
    top_recent_short = top_label_map(cuisine_pref, 'user_id', 'cuisine', 'recent_short_weight_sum')
    top_recent_short = top_recent_short.loc[top_recent_short['rank'] == 1, ['user_id', 'cuisine']].rename(columns={'cuisine': 'recent_top_cuisine_short'})
    top_recent_long = top_label_map(cuisine_pref, 'user_id', 'cuisine', 'recent_long_weight_sum')
    top_recent_long = top_recent_long.loc[top_recent_long['rank'] == 1, ['user_id', 'cuisine']].rename(columns={'cuisine': 'recent_top_cuisine_long'})
    top_negative = top_label_map(cuisine_pref, 'user_id', 'cuisine', 'negative_weight_sum')
    top_negative = top_negative.loc[top_negative['rank'] == 1, ['user_id', 'cuisine']].rename(columns={'cuisine': 'negative_top_cuisine'})
    long_term_score_stats = top2_score_frame(cuisine_pref, 'user_id', 'cuisine', 'taste_weight_sum', 'long_term_cuisine')
    recent_short_score_stats = top2_score_frame(cuisine_pref, 'user_id', 'cuisine', 'recent_short_weight_sum', 'recent_short_cuisine')
    recent_long_score_stats = top2_score_frame(cuisine_pref, 'user_id', 'cuisine', 'recent_long_weight_sum', 'recent_long_cuisine')

    latest_cuisine_rows = expand_cuisine_rows(latest_events)
    latest_cuisine_pref = latest_cuisine_rows.groupby(['user_id', 'cuisine'], as_index=False).agg(
        latest_taste_weight_sum=('taste_weight', 'sum'),
    )
    top_latest = top_label_map(latest_cuisine_pref, 'user_id', 'cuisine', 'latest_taste_weight_sum')
    top_latest = top_latest.loc[top_latest['rank'] == 1, ['user_id', 'cuisine']].rename(columns={'cuisine': 'latest_top_cuisine'})
    latest_score_stats = top2_score_frame(latest_cuisine_pref, 'user_id', 'cuisine', 'latest_taste_weight_sum', 'latest_cuisine')

    geo_pref = events.groupby(['user_id', 'city', 'geo_cell_3dp'], as_index=False).agg(
        activity_weight_sum=('activity_weight', 'sum'),
        taste_weight_sum=('taste_weight', 'sum'),
        negative_weight_sum=('negative_weight', 'sum'),
        recent_short_weight_sum=('recent_short_taste_weight', 'sum'),
    )
    geo_pref = geo_pref.sort_values(['user_id', 'activity_weight_sum', 'taste_weight_sum', 'recent_short_weight_sum', 'city', 'geo_cell_3dp'], ascending=[True, False, False, False, True, True])
    geo_pref['rank_taste'] = geo_pref.groupby('user_id').cumcount() + 1
    top_geo = geo_pref.loc[geo_pref['rank_taste'] == 1, ['user_id', 'city', 'geo_cell_3dp', 'activity_weight_sum']].rename(columns={'city': 'top_city', 'geo_cell_3dp': 'top_geo_cell_3dp', 'activity_weight_sum': 'top_geo_weight_sum'})

    city_pref = events.groupby(['user_id', 'city'], as_index=False).agg(
        activity_weight_sum=('activity_weight', 'sum'),
        taste_weight_sum=('taste_weight', 'sum'),
        recent_short_weight_sum=('recent_short_taste_weight', 'sum'),
    ).sort_values(['user_id', 'activity_weight_sum', 'taste_weight_sum', 'recent_short_weight_sum', 'city'], ascending=[True, False, False, False, True])
    city_pref['rank_taste'] = city_pref.groupby('user_id').cumcount() + 1
    top_city = city_pref.loc[city_pref['rank_taste'] == 1, ['user_id', 'city', 'activity_weight_sum']].rename(columns={'city': 'top_city_only', 'activity_weight_sum': 'top_city_weight_sum'})

    geo_total = geo_pref.groupby('user_id', as_index=False).agg(
        geo_weight_total=('activity_weight_sum', 'sum'),
        geo_cell_count=('geo_cell_3dp', 'nunique'),
        city_count=('city', 'nunique'),
    )

    user_df = user_base.merge(top_long, on='user_id', how='left')
    user_df = user_df.merge(top_recent_short, on='user_id', how='left')
    user_df = user_df.merge(top_recent_long, on='user_id', how='left')
    user_df = user_df.merge(top_negative, on='user_id', how='left')
    user_df = user_df.merge(top_latest, on='user_id', how='left')
    user_df = user_df.merge(long_term_score_stats, on='user_id', how='left')
    user_df = user_df.merge(recent_short_score_stats, on='user_id', how='left')
    user_df = user_df.merge(recent_long_score_stats, on='user_id', how='left')
    user_df = user_df.merge(latest_score_stats, on='user_id', how='left')
    user_df = user_df.merge(top_geo[['user_id', 'top_geo_cell_3dp', 'top_geo_weight_sum']], on='user_id', how='left')
    user_df = user_df.merge(top_city[['user_id', 'top_city_only', 'top_city_weight_sum']], on='user_id', how='left')
    user_df = user_df.merge(geo_total, on='user_id', how='left')

    user_df['top_city'] = user_df['top_city_only'].fillna('')
    user_df['recent_top_cuisine'] = user_df['recent_top_cuisine_short']
    recent_long_mask = user_df['recent_top_cuisine'].isna() & user_df['recent_top_cuisine_long'].notna()
    user_df.loc[recent_long_mask, 'recent_top_cuisine'] = user_df.loc[recent_long_mask, 'recent_top_cuisine_long']
    user_df['recent_top_cuisine_source'] = 'none'
    user_df.loc[user_df['recent_top_cuisine_short'].notna(), 'recent_top_cuisine_source'] = 'recent_short'
    user_df.loc[recent_long_mask, 'recent_top_cuisine_source'] = 'recent_long'
    user_df['geo_concentration_ratio'] = safe_ratio(user_df['top_geo_weight_sum'].fillna(0.0), user_df['geo_weight_total'].fillna(0.0))
    user_df['city_concentration_ratio'] = safe_ratio(user_df['top_city_weight_sum'].fillna(0.0), user_df['geo_weight_total'].fillna(0.0))
    recent_signal_any = (
        user_df['recent_short_weight_sum'].fillna(0.0)
        + user_df['recent_long_weight_sum'].fillna(0.0)
        + user_df['recent_short_tip_weight_sum'].fillna(0.0)
    )
    user_df['has_recent_intent_view'] = ((recent_signal_any > 0) | user_df['recent_top_cuisine'].notna()).astype(int)
    user_df['has_latest_recent_fallback_view'] = (
        user_df['has_recent_intent_view'].eq(0)
        & user_df['latest_event_count'].fillna(0).gt(0)
        & user_df['latest_top_cuisine'].notna()
    ).astype(int)
    user_df['has_negative_avoidance_view'] = user_df['negative_top_cuisine'].notna().astype(int)
    user_df['cuisine_shift_flag'] = (
        user_df['long_term_top_cuisine'].notna()
        & user_df['recent_top_cuisine'].notna()
        & user_df['long_term_top_cuisine'].ne(user_df['recent_top_cuisine'])
    ).astype(int)
    user_df['is_negative_heavy'] = (user_df['negative_pressure'] >= 0.35).astype(int)
    user_df['is_geo_concentrated'] = (user_df['geo_concentration_ratio'] >= 0.6).astype(int)

    for raw_col in (
        'long_term_cuisine_top1_raw',
        'long_term_cuisine_top2_raw',
        'recent_short_cuisine_top1_raw',
        'recent_short_cuisine_top2_raw',
        'recent_long_cuisine_top1_raw',
        'recent_long_cuisine_top2_raw',
        'latest_cuisine_top1_raw',
        'latest_cuisine_top2_raw',
    ):
        user_df[raw_col] = pd.to_numeric(user_df.get(raw_col), errors='coerce').fillna(0.0)

    user_df['long_term_cuisine_top1_share'] = safe_ratio(user_df['long_term_cuisine_top1_raw'], user_df['taste_weight_sum'])
    user_df['long_term_cuisine_margin_share'] = safe_ratio(
        user_df['long_term_cuisine_top1_raw'] - user_df['long_term_cuisine_top2_raw'],
        user_df['taste_weight_sum'],
    )
    user_df['recent_short_cuisine_top1_share'] = safe_ratio(user_df['recent_short_cuisine_top1_raw'], user_df['recent_short_weight_sum'])
    user_df['recent_short_cuisine_margin_share'] = safe_ratio(
        user_df['recent_short_cuisine_top1_raw'] - user_df['recent_short_cuisine_top2_raw'],
        user_df['recent_short_weight_sum'],
    )
    user_df['recent_long_cuisine_top1_share'] = safe_ratio(user_df['recent_long_cuisine_top1_raw'], user_df['recent_long_weight_sum'])
    user_df['recent_long_cuisine_margin_share'] = safe_ratio(
        user_df['recent_long_cuisine_top1_raw'] - user_df['recent_long_cuisine_top2_raw'],
        user_df['recent_long_weight_sum'],
    )
    user_df['latest_cuisine_top1_share'] = safe_ratio(user_df['latest_cuisine_top1_raw'], user_df['latest_event_weight_sum'])
    user_df['latest_cuisine_margin_share'] = safe_ratio(
        user_df['latest_cuisine_top1_raw'] - user_df['latest_cuisine_top2_raw'],
        user_df['latest_event_weight_sum'],
    )

    user_df['typed_intent_primary_cuisine'] = user_df['long_term_top_cuisine']
    user_df['typed_intent_primary_source'] = 'long_term'
    recent_short_mask = user_df['recent_top_cuisine_source'].eq('recent_short') & user_df['recent_top_cuisine'].notna()
    recent_long_mask = user_df['recent_top_cuisine_source'].eq('recent_long') & user_df['recent_top_cuisine'].notna()
    latest_mask = user_df['has_latest_recent_fallback_view'].eq(1) & user_df['latest_top_cuisine'].notna()
    user_df.loc[recent_short_mask, 'typed_intent_primary_cuisine'] = user_df.loc[recent_short_mask, 'recent_top_cuisine']
    user_df.loc[recent_short_mask, 'typed_intent_primary_source'] = 'recent_short'
    user_df.loc[recent_long_mask, 'typed_intent_primary_cuisine'] = user_df.loc[recent_long_mask, 'recent_top_cuisine']
    user_df.loc[recent_long_mask, 'typed_intent_primary_source'] = 'recent_long'
    user_df.loc[latest_mask, 'typed_intent_primary_cuisine'] = user_df.loc[latest_mask, 'latest_top_cuisine']
    user_df.loc[latest_mask, 'typed_intent_primary_source'] = 'latest_fallback'
    no_intent_mask = user_df['typed_intent_primary_cuisine'].isna()
    user_df.loc[no_intent_mask, 'typed_intent_primary_source'] = 'unknown'

    user_df['typed_intent_top1_share'] = user_df['long_term_cuisine_top1_share']
    user_df['typed_intent_margin_share'] = user_df['long_term_cuisine_margin_share']
    user_df.loc[recent_short_mask, 'typed_intent_top1_share'] = user_df.loc[recent_short_mask, 'recent_short_cuisine_top1_share']
    user_df.loc[recent_short_mask, 'typed_intent_margin_share'] = user_df.loc[recent_short_mask, 'recent_short_cuisine_margin_share']
    user_df.loc[recent_long_mask, 'typed_intent_top1_share'] = user_df.loc[recent_long_mask, 'recent_long_cuisine_top1_share']
    user_df.loc[recent_long_mask, 'typed_intent_margin_share'] = user_df.loc[recent_long_mask, 'recent_long_cuisine_margin_share']
    user_df.loc[latest_mask, 'typed_intent_top1_share'] = user_df.loc[latest_mask, 'latest_cuisine_top1_share']
    user_df.loc[latest_mask, 'typed_intent_margin_share'] = user_df.loc[latest_mask, 'latest_cuisine_margin_share']
    user_df['typed_intent_top1_share'] = pd.to_numeric(user_df['typed_intent_top1_share'], errors='coerce').fillna(0.0)
    user_df['typed_intent_margin_share'] = pd.to_numeric(user_df['typed_intent_margin_share'], errors='coerce').fillna(0.0)
    user_df['typed_intent_confidence'] = np.clip(
        0.65 * user_df['typed_intent_top1_share'] + 0.35 * user_df['typed_intent_margin_share'],
        0.0,
        1.0,
    )
    user_df.loc[no_intent_mask, 'typed_intent_confidence'] = 0.0
    user_df['typed_intent_confidence_band'] = np.where(
        user_df['typed_intent_confidence'] >= 0.45,
        'high',
        np.where(user_df['typed_intent_confidence'] >= 0.25, 'medium', 'low'),
    )

    latest_recent_conflict = (
        user_df['latest_top_cuisine'].notna()
        & user_df['recent_top_cuisine'].notna()
        & user_df['latest_top_cuisine'].ne(user_df['recent_top_cuisine'])
    ).astype(float)
    latest_long_conflict = (
        user_df['latest_top_cuisine'].notna()
        & user_df['long_term_top_cuisine'].notna()
        & user_df['latest_top_cuisine'].ne(user_df['long_term_top_cuisine'])
    ).astype(float)
    negative_overlap = (
        user_df['negative_top_cuisine'].notna()
        & user_df['typed_intent_primary_cuisine'].notna()
        & user_df['negative_top_cuisine'].eq(user_df['typed_intent_primary_cuisine'])
    ).astype(float)
    user_df['typed_intent_conflict_score'] = np.clip(
        0.45 * user_df['cuisine_shift_flag'].astype(float)
        + 0.20 * latest_recent_conflict
        + 0.15 * latest_long_conflict
        + 0.20 * negative_overlap * user_df['negative_pressure'].fillna(0.0),
        0.0,
        1.0,
    )
    user_df['typed_intent_conflict_band'] = np.where(
        user_df['typed_intent_conflict_score'] >= 0.45,
        'high',
        np.where(user_df['typed_intent_conflict_score'] >= 0.15, 'medium', 'low'),
    )

    recent_short_signal = safe_ratio(
        user_df['recent_short_weight_sum'].fillna(0.0) + 0.50 * user_df['recent_short_tip_weight_sum'].fillna(0.0),
        user_df['taste_weight_sum'].fillna(0.0)
        + user_df['recent_short_weight_sum'].fillna(0.0)
        + 0.50 * user_df['recent_short_tip_weight_sum'].fillna(0.0),
    )
    recent_long_signal = safe_ratio(
        user_df['recent_long_weight_sum'].fillna(0.0),
        user_df['taste_weight_sum'].fillna(0.0) + user_df['recent_long_weight_sum'].fillna(0.0),
    )
    latest_signal = safe_ratio(
        user_df['latest_event_weight_sum'].fillna(0.0),
        user_df['taste_weight_sum'].fillna(0.0) + user_df['latest_event_weight_sum'].fillna(0.0),
    )
    recent_source_short = user_df['typed_intent_primary_source'].eq('recent_short').astype(float)
    recent_source_long = user_df['typed_intent_primary_source'].eq('recent_long').astype(float)
    latest_source = user_df['typed_intent_primary_source'].eq('latest_fallback').astype(float)
    long_term_source = user_df['typed_intent_primary_source'].eq('long_term').astype(float)
    unknown_source = user_df['typed_intent_primary_source'].eq('unknown').astype(float)
    recency_support = np.clip(
        0.55 * np.sqrt(np.clip(recent_short_signal, 0.0, 1.0))
        + 0.20 * np.sqrt(np.clip(recent_long_signal, 0.0, 1.0))
        + 0.10 * np.sqrt(np.clip(latest_signal, 0.0, 1.0))
        + 0.10 * user_df['has_recent_intent_view'].astype(float)
        + 0.05 * recent_source_short
        + 0.02 * recent_source_long,
        0.0,
        1.0,
    )
    staleness_source_penalty = (
        0.10 * latest_source
        + 0.18 * long_term_source
        + 0.28 * unknown_source
        - 0.12 * recent_source_short
        - 0.06 * recent_source_long
    )
    user_df['typed_intent_staleness_score'] = np.clip(
        0.78 * (1.0 - recency_support)
        + staleness_source_penalty
        + 0.10 * user_df['typed_intent_conflict_score'].fillna(0.0),
        0.0,
        1.0,
    )
    valid_staleness = user_df.loc[
        user_df['typed_intent_primary_cuisine'].notna() & user_df['typed_intent_primary_source'].ne('unknown'),
        'typed_intent_staleness_score',
    ]
    staleness_fresh_cut = bounded_quantile(valid_staleness, q=0.30, default=0.38, low=0.20, high=0.55)
    staleness_stale_cut = bounded_quantile(valid_staleness, q=0.70, default=0.62, low=staleness_fresh_cut + 0.10, high=0.85)
    user_df['typed_intent_staleness_band'] = np.where(
        user_df['typed_intent_staleness_score'] >= staleness_stale_cut,
        'stale',
        np.where(user_df['typed_intent_staleness_score'] >= staleness_fresh_cut, 'mixed', 'fresh'),
    )

    user_df = user_df.drop(columns=['top_city_only', 'recent_top_cuisine_short', 'recent_top_cuisine_long'], errors='ignore')

    field_summary = summarize_fields(user_df)
    sample_book = build_sample_book(
        user_df,
        cuisine_pref.loc[cuisine_pref['rank_net'] <= TOPK_PER_USER].copy(),
        geo_pref.loc[geo_pref['rank_taste'] <= TOPK_PER_USER].copy(),
    )

    user_df.to_parquet(out_dir / 'user_intent_profile_v2.parquet', index=False)
    cuisine_pref.to_parquet(out_dir / 'user_cuisine_pref_v2.parquet', index=False)
    geo_pref.to_parquet(out_dir / 'user_geo_pref_v2.parquet', index=False)
    field_summary.to_csv(out_dir / 'user_intent_profile_v2_field_summary.csv', index=False)
    safe_json_write(out_dir / 'user_intent_profile_v2_sample.json', sample_book)

    run_meta = {
        'run_id': out_dir.name,
        'run_tag': RUN_TAG,
        'input_run_dirs': {
            'interaction_v2_weighted_run_dir': str(interaction_run),
            'merchant_semantic_card_run_dir': str(merchant_run),
        },
        'schema_rule_v1': {
            'use_trainable_events_only': True,
            'split_aware_history_only': bool(history_pairs is not None),
            'split_aware_bucket': int(SPLIT_AWARE_BUCKET) if history_pairs is not None else None,
            'tip_aux_weight': TIP_AUX_WEIGHT,
            'neutral_aux_weight': NEUTRAL_AUX_WEIGHT,
            'negative_penalty_weight': NEGATIVE_PENALTY_WEIGHT,
            'recent_short_days': RECENT_SHORT_DAYS,
            'recent_long_days': RECENT_LONG_DAYS,
            'recent_latest_event_count': RECENT_LATEST_EVENT_COUNT,
            'typed_intent_staleness_band_fresh_cut': float(staleness_fresh_cut),
            'typed_intent_staleness_band_stale_cut': float(staleness_stale_cut),
            'merchant_side_required': 'merchant_semantic_card_v2',
            'checkin_user_item_direct_events_allowed': False,
        },
        'row_counts': {
            'trainable_event_rows_before_history_filter': int(len(pd.read_parquet(interaction_run / 'interaction_v2_weighted.parquet'))),
            'trainable_event_rows': int(len(events)),
            'user_rows': int(len(user_df)),
            'user_cuisine_rows': int(len(cuisine_pref)),
            'user_geo_rows': int(len(geo_pref)),
        },
        'coverage': {
            'share_with_long_term_top_cuisine': float(user_df['long_term_top_cuisine'].notna().mean()),
            'share_with_recent_top_cuisine': float(user_df['recent_top_cuisine'].notna().mean()),
            'share_with_latest_top_cuisine': float(user_df['latest_top_cuisine'].notna().mean()),
            'share_with_typed_intent_primary_cuisine': float(user_df['typed_intent_primary_cuisine'].notna().mean()),
            'share_with_latest_recent_fallback_view': float(user_df['has_latest_recent_fallback_view'].mean()),
            'share_with_negative_top_cuisine': float(user_df['negative_top_cuisine'].notna().mean()),
            'share_with_top_geo': float(user_df['top_geo_cell_3dp'].notna().mean()),
            'share_with_high_typed_intent_confidence': float(user_df['typed_intent_confidence_band'].eq('high').mean()),
            'share_with_high_typed_intent_conflict': float(user_df['typed_intent_conflict_band'].eq('high').mean()),
            'share_with_stale_typed_intent': float(user_df['typed_intent_staleness_band'].eq('stale').mean()),
            'mean_negative_pressure': float(user_df['negative_pressure'].mean()),
            'mean_geo_concentration_ratio': float(user_df['geo_concentration_ratio'].mean()),
            'mean_typed_intent_confidence': float(user_df['typed_intent_confidence'].mean()),
            'mean_typed_intent_conflict_score': float(user_df['typed_intent_conflict_score'].mean()),
            'mean_typed_intent_staleness_score': float(user_df['typed_intent_staleness_score'].mean()),
        },
        'output_files': {
            'user_intent_profile_v2_parquet': str(out_dir / 'user_intent_profile_v2.parquet'),
            'user_cuisine_pref_v2_parquet': str(out_dir / 'user_cuisine_pref_v2.parquet'),
            'user_geo_pref_v2_parquet': str(out_dir / 'user_geo_pref_v2.parquet'),
            'field_summary_csv': str(out_dir / 'user_intent_profile_v2_field_summary.csv'),
            'sample_json': str(out_dir / 'user_intent_profile_v2_sample.json'),
        },
    }
    safe_json_write(out_dir / 'run_meta.json', run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
