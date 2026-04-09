from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

MATCH_FEATURE_ROOT = env_or_project_path(
    'INPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_ROOT_DIR',
    'data/output/09_user_business_match_features_v2',
)
MERCHANT_CARD_ROOT = env_or_project_path(
    'INPUT_09_MERCHANT_CARD_ROOT_DIR',
    'data/output/09_merchant_semantic_card',
)
USER_INTENT_ROOT = env_or_project_path(
    'INPUT_09_USER_INTENT_PROFILE_V2_ROOT_DIR',
    'data/output/09_user_intent_profile_v2',
)

MATCH_FEATURE_RUN_DIR = os.getenv('INPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_RUN_DIR', '').strip()
MERCHANT_CARD_RUN_DIR = os.getenv('INPUT_09_MERCHANT_CARD_RUN_DIR', '').strip()
USER_INTENT_RUN_DIR = os.getenv('INPUT_09_USER_INTENT_PROFILE_V2_RUN_DIR', '').strip()
SPLIT_AWARE_HISTORY_ONLY = os.getenv('USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_HISTORY_ONLY', 'false').strip().lower() == 'true'
HISTORY_STAGE09_BUCKET_DIR = os.getenv('INPUT_09_STAGE09_BUCKET_DIR', '').strip()
INDEX_MAPS_ROOT = env_or_project_path(
    'INPUT_09_INDEX_MAPS_ROOT_DIR',
    'data/output/09_index_maps',
)
INDEX_MAPS_RUN_DIR = os.getenv('INPUT_09_INDEX_MAPS_RUN_DIR', '').strip()
SPLIT_AWARE_BUCKET = int(os.getenv('USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_BUCKET', '5').strip() or 5)

OUTPUT_ROOT = env_or_project_path(
    'OUTPUT_09_USER_BUSINESS_MATCH_CHANNELS_V2_ROOT_DIR',
    'data/output/09_user_business_match_channels_v2',
)
RUN_TAG = 'stage09_user_business_match_split_aware_v2_build' if SPLIT_AWARE_HISTORY_ONLY else 'stage09_user_business_match_channels_v2_build'

PREF_CORE_CUISINE_WEIGHT = float(os.getenv('MATCH_CHANNEL_PREF_CORE_CUISINE_WEIGHT', '0.7'))
PREF_CORE_SCENE_WEIGHT = float(os.getenv('MATCH_CHANNEL_PREF_CORE_SCENE_WEIGHT', '0.3'))
RECENT_CUISINE_WEIGHT = float(os.getenv('MATCH_CHANNEL_RECENT_CUISINE_WEIGHT', '0.75'))
RECENT_TIP_WEIGHT = float(os.getenv('MATCH_CHANNEL_RECENT_TIP_WEIGHT', '0.25'))
CONTEXT_MEAL_WEIGHT = float(os.getenv('MATCH_CHANNEL_CONTEXT_MEAL_WEIGHT', '0.7'))
CONTEXT_TIME_WEIGHT = float(os.getenv('MATCH_CHANNEL_CONTEXT_TIME_WEIGHT', '0.3'))
CONFLICT_NEG_EVIDENCE_WEIGHT = float(os.getenv('MATCH_CHANNEL_CONFLICT_NEG_EVIDENCE_WEIGHT', '0.7'))
CONFLICT_NEG_PREF_WEIGHT = float(os.getenv('MATCH_CHANNEL_CONFLICT_NEG_PREF_WEIGHT', '0.3'))
TYPED_TRUST_CONFIDENCE_WEIGHT = float(os.getenv('MATCH_CHANNEL_TYPED_TRUST_CONFIDENCE_WEIGHT', '0.5'))
TYPED_TRUST_FRESHNESS_WEIGHT = float(os.getenv('MATCH_CHANNEL_TYPED_TRUST_FRESHNESS_WEIGHT', '0.35'))
TYPED_TRUST_CONFLICT_WEIGHT = float(os.getenv('MATCH_CHANNEL_TYPED_TRUST_CONFLICT_WEIGHT', '0.15'))
TYPED_CORE_MIN_SCALE = float(os.getenv('MATCH_CHANNEL_TYPED_CORE_MIN_SCALE', '0.35'))
TYPED_RECENT_MIN_SCALE = float(os.getenv('MATCH_CHANNEL_TYPED_RECENT_MIN_SCALE', '0.45'))
TYPED_CONTEXT_MIN_SCALE = float(os.getenv('MATCH_CHANNEL_TYPED_CONTEXT_MIN_SCALE', '0.7'))
TYPED_CONTEXT_V2_MIN_SCALE = float(os.getenv('MATCH_CHANNEL_TYPED_CONTEXT_V2_MIN_SCALE', '0.55'))
BREAKFAST_LONGTERM_ONLY_SCALE = float(os.getenv('MATCH_CHANNEL_BREAKFAST_LONGTERM_ONLY_SCALE', '0.55'))

LOW_SPEC_CUISINES = {'other', 'unknown', 'restaurants', 'food'}
MID_SPEC_CUISINES = {'american', 'american_new', 'asian_other'}

CHANNEL_COLS = [
    'channel_preference_core_v1',
    'channel_preference_property_v1',
    'channel_recent_intent_v1',
    'channel_context_time_v1',
    'channel_context_geo_v1',
    'channel_evidence_support_v1',
    'channel_conflict_v1',
    'channel_uncertainty_confidence_v1',
    'channel_uncertainty_conflict_v1',
    'channel_uncertainty_freshness_v1',
    'channel_typed_prior_trust_v1',
    'channel_preference_core_uaware_v1',
    'channel_recent_intent_uaware_v1',
    'channel_context_time_uaware_v1',
    'channel_typed_specificity_v1',
    'channel_breakfast_guard_scale_v1',
    'channel_typed_prior_trust_v2',
    'channel_preference_core_uaware_v2',
    'channel_recent_intent_uaware_v2',
    'channel_context_time_uaware_v2',
    'channel_typed_exact_match_uaware_v1',
]


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
        raise FileNotFoundError('INPUT_09_STAGE09_BUCKET_DIR is required when USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_HISTORY_ONLY=true')
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


def activity_band(n_train: pd.Series) -> pd.Series:
    out = pd.Series('unknown', index=n_train.index, dtype='object')
    out.loc[n_train <= 7] = 'light'
    out.loc[(n_train >= 8) & (n_train <= 19)] = 'mid'
    out.loc[n_train >= 20] = 'heavy'
    return out


def proxy_summary(df: pd.DataFrame, proxy_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feat in CHANNEL_COLS:
        feat_s = pd.to_numeric(df[feat], errors='coerce')
        if feat_s.notna().sum() < 50:
            continue
        for proxy in proxy_cols:
            proxy_s = pd.to_numeric(df[proxy], errors='coerce')
            valid = feat_s.notna() & proxy_s.notna()
            if valid.sum() < 50:
                continue
            corr = feat_s[valid].corr(proxy_s[valid])
            rows.append({
                'feature': feat,
                'proxy': proxy,
                'pearson_corr': None if pd.isna(corr) else float(corr),
                'abs_corr': None if pd.isna(corr) else float(abs(corr)),
                'n_valid': int(valid.sum()),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['proxy', 'abs_corr'], ascending=[True, False]).reset_index(drop=True)


def build_sample_book(df: pd.DataFrame) -> dict[str, object]:
    cols = [
        'event_id', 'event_type', 'user_id', 'business_id',
        'channel_preference_core_v1',
        'channel_preference_property_v1',
        'channel_recent_intent_v1',
        'channel_context_time_v1',
        'channel_context_geo_v1',
        'channel_evidence_support_v1',
        'channel_conflict_v1',
        'channel_uncertainty_confidence_v1',
        'channel_uncertainty_conflict_v1',
        'channel_uncertainty_freshness_v1',
        'channel_typed_prior_trust_v1',
        'channel_preference_core_uaware_v1',
        'channel_recent_intent_uaware_v1',
        'channel_context_time_uaware_v1',
        'channel_typed_specificity_v1',
        'channel_breakfast_guard_scale_v1',
        'channel_typed_prior_trust_v2',
        'channel_preference_core_uaware_v2',
        'channel_recent_intent_uaware_v2',
        'channel_context_time_uaware_v2',
        'channel_typed_exact_match_uaware_v1',
    ]
    return {
        'strong_preference_events': df.sort_values(['channel_preference_core_v1', 'channel_recent_intent_v1'], ascending=[False, False]).head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'strong_recent_events': df.sort_values(['channel_recent_intent_v1', 'channel_context_time_v1'], ascending=[False, False]).head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
        'strong_conflict_events': df.sort_values(['channel_conflict_v1', 'channel_preference_core_v1'], ascending=[False, False]).head(10)[cols].replace({np.nan: None}).to_dict(orient='records'),
    }


def build_user_item_channels(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, object]] = {
        'event_rows': ('event_id', 'count'),
        'positive_event_rows': ('event_type', lambda s: int((s == 'review_positive').sum())),
        'negative_event_rows': ('event_type', lambda s: int((s == 'review_negative').sum())),
        'tip_event_rows': ('event_type', lambda s: int((s == 'tip_signal').sum())),
    }
    for c in CHANNEL_COLS:
        agg_spec[f'mean_{c}'] = (c, 'mean')
        agg_spec[f'max_{c}'] = (c, 'max')
    return (
        df.groupby(['user_id', 'business_id'], as_index=False)
        .agg(**agg_spec)
        .sort_values(['user_id', 'business_id'], ascending=[True, True])
        .reset_index(drop=True)
    )


def main() -> None:
    match_run = resolve_optional_run(
        MATCH_FEATURE_RUN_DIR,
        MATCH_FEATURE_ROOT,
        '_full_stage09_user_business_match_features_v2_build',
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

    df = pd.read_parquet(match_run / 'user_business_match_features_v2_event_rows.parquet')
    trainable_rows_before_history_filter = int(len(df))
    history_pairs = resolve_history_only_pairs()
    if history_pairs is not None:
        df = df.merge(history_pairs, on=['user_id', 'business_id'], how='inner')
    merchant = pd.read_parquet(merchant_run / 'merchant_semantic_card_v2.parquet', columns=[
        'business_id',
        'audit_review_count',
        'audit_tip_rows',
        'audit_checkin_rows',
    ])
    users = pd.read_parquet(user_intent_run / 'user_intent_profile_v2.parquet', columns=[
        'user_id',
        'n_train_max',
        'has_recent_intent_view',
        'has_negative_avoidance_view',
        'typed_intent_primary_cuisine',
        'typed_intent_primary_source',
        'long_term_top_cuisine',
        'recent_top_cuisine',
        'typed_intent_confidence',
        'typed_intent_conflict_score',
        'typed_intent_staleness_score',
    ])
    users['activity_band'] = activity_band(pd.to_numeric(users['n_train_max'], errors='coerce').fillna(0))

    df = df.merge(merchant, on='business_id', how='left')
    df = df.merge(users, on='user_id', how='left')

    df['channel_preference_core_v1'] = (
        PREF_CORE_CUISINE_WEIGHT * pd.to_numeric(df['match_cuisine_net'], errors='coerce').fillna(0.0)
        + PREF_CORE_SCENE_WEIGHT * pd.to_numeric(df['match_scene_dot'], errors='coerce').fillna(0.0)
    )
    df['channel_preference_property_v1'] = pd.to_numeric(df['match_property_dot'], errors='coerce').fillna(0.0)
    df['channel_recent_intent_v1'] = (
        RECENT_CUISINE_WEIGHT * pd.to_numeric(df['match_recent_cuisine'], errors='coerce').fillna(0.0)
        + RECENT_TIP_WEIGHT * pd.to_numeric(df['match_recent_tip_context'], errors='coerce').fillna(0.0)
    )
    df['channel_context_time_v1'] = (
        CONTEXT_MEAL_WEIGHT * pd.to_numeric(df['match_meal_dot'], errors='coerce').fillna(0.0)
        + CONTEXT_TIME_WEIGHT * pd.to_numeric(df['match_time_context'], errors='coerce').fillna(0.0)
    )
    df['channel_context_geo_v1'] = pd.to_numeric(df['match_geo_alignment'], errors='coerce').fillna(0.0)
    df['channel_evidence_support_v1'] = pd.to_numeric(df['match_positive_evidence'], errors='coerce').fillna(0.0)
    neg_pref = (
        pd.to_numeric(df['match_primary_cuisine_negative'], errors='coerce').fillna(0.0)
        + 0.6 * pd.to_numeric(df['match_secondary_cuisine_negative'], errors='coerce').fillna(0.0)
    )
    df['channel_conflict_v1'] = (
        CONFLICT_NEG_EVIDENCE_WEIGHT * pd.to_numeric(df['match_negative_conflict'], errors='coerce').fillna(0.0)
        + CONFLICT_NEG_PREF_WEIGHT * neg_pref
    )
    df['channel_uncertainty_confidence_v1'] = pd.to_numeric(df['typed_intent_confidence'], errors='coerce').fillna(0.0)
    df['channel_uncertainty_conflict_v1'] = pd.to_numeric(df['typed_intent_conflict_score'], errors='coerce').fillna(0.0)
    df['channel_uncertainty_freshness_v1'] = 1.0 - pd.to_numeric(df['typed_intent_staleness_score'], errors='coerce').fillna(1.0)
    df['channel_typed_prior_trust_v1'] = np.clip(
        TYPED_TRUST_CONFIDENCE_WEIGHT * df['channel_uncertainty_confidence_v1']
        + TYPED_TRUST_FRESHNESS_WEIGHT * df['channel_uncertainty_freshness_v1']
        + TYPED_TRUST_CONFLICT_WEIGHT * (1.0 - df['channel_uncertainty_conflict_v1']),
        0.0,
        1.0,
    )
    df['channel_preference_core_uaware_v1'] = (
        df['channel_preference_core_v1']
        * (TYPED_CORE_MIN_SCALE + (1.0 - TYPED_CORE_MIN_SCALE) * df['channel_typed_prior_trust_v1'])
    )
    df['channel_recent_intent_uaware_v1'] = (
        df['channel_recent_intent_v1']
        * (TYPED_RECENT_MIN_SCALE + (1.0 - TYPED_RECENT_MIN_SCALE) * df['channel_typed_prior_trust_v1'])
    )
    df['channel_context_time_uaware_v1'] = (
        df['channel_context_time_v1']
        * (TYPED_CONTEXT_MIN_SCALE + (1.0 - TYPED_CONTEXT_MIN_SCALE) * df['channel_typed_prior_trust_v1'])
    )
    typed_cuisine = df['typed_intent_primary_cuisine'].fillna('').astype(str)
    df['channel_typed_specificity_v1'] = 1.0
    df.loc[typed_cuisine.isin(LOW_SPEC_CUISINES), 'channel_typed_specificity_v1'] = 0.35
    df.loc[typed_cuisine.isin(MID_SPEC_CUISINES), 'channel_typed_specificity_v1'] = 0.60
    breakfast_longterm_only = (
        df['long_term_top_cuisine'].fillna('').eq('breakfast_brunch')
        & df['typed_intent_primary_source'].fillna('').isin(['long_term', 'latest_fallback'])
        & df['recent_top_cuisine'].fillna('').ne('breakfast_brunch')
    )
    df['channel_breakfast_guard_scale_v1'] = np.where(breakfast_longterm_only, BREAKFAST_LONGTERM_ONLY_SCALE, 1.0)
    df['channel_typed_prior_trust_v2'] = np.clip(
        df['channel_typed_prior_trust_v1']
        * df['channel_typed_specificity_v1']
        * df['channel_breakfast_guard_scale_v1'],
        0.0,
        1.0,
    )
    df['channel_preference_core_uaware_v2'] = (
        df['channel_preference_core_v1']
        * (TYPED_CORE_MIN_SCALE + (1.0 - TYPED_CORE_MIN_SCALE) * df['channel_typed_prior_trust_v2'])
    )
    df['channel_recent_intent_uaware_v2'] = (
        df['channel_recent_intent_v1']
        * (TYPED_RECENT_MIN_SCALE + (1.0 - TYPED_RECENT_MIN_SCALE) * df['channel_typed_prior_trust_v2'])
    )
    df['channel_context_time_uaware_v2'] = (
        df['channel_context_time_v1']
        * (TYPED_CONTEXT_V2_MIN_SCALE + (1.0 - TYPED_CONTEXT_V2_MIN_SCALE) * df['channel_typed_prior_trust_v2'])
    )
    typed_exact_match = (
        typed_cuisine.ne('')
        & (
            df['merchant_primary_cuisine'].fillna('').astype(str).eq(typed_cuisine)
            | df['merchant_secondary_cuisine'].fillna('').astype(str).eq(typed_cuisine)
        )
    ).astype(float)
    typed_match_support = np.clip(
        0.65 * df['channel_recent_intent_v1'] + 0.35 * df['channel_preference_core_v1'],
        0.0,
        1.0,
    )
    df['channel_typed_exact_match_uaware_v1'] = (
        typed_exact_match
        * df['channel_typed_prior_trust_v2']
        * typed_match_support
    )

    channel_event_summary = df.groupby('event_type', as_index=False).agg(
        n_rows=('event_id', 'count'),
        **{f'mean_{c}': (c, 'mean') for c in CHANNEL_COLS},
    )
    channel_band_summary = df.groupby(['activity_band', 'event_type'], as_index=False).agg(
        n_rows=('event_id', 'count'),
        **{f'mean_{c}': (c, 'mean') for c in CHANNEL_COLS},
    )
    view_summary = df.groupby(['has_recent_intent_view', 'has_negative_avoidance_view', 'event_type'], as_index=False).agg(
        n_rows=('event_id', 'count'),
        mean_channel_recent_intent_v1=('channel_recent_intent_v1', 'mean'),
        mean_channel_conflict_v1=('channel_conflict_v1', 'mean'),
        mean_channel_preference_core_v1=('channel_preference_core_v1', 'mean'),
        mean_channel_uncertainty_confidence_v1=('channel_uncertainty_confidence_v1', 'mean'),
        mean_channel_uncertainty_conflict_v1=('channel_uncertainty_conflict_v1', 'mean'),
        mean_channel_uncertainty_freshness_v1=('channel_uncertainty_freshness_v1', 'mean'),
        mean_channel_typed_prior_trust_v1=('channel_typed_prior_trust_v1', 'mean'),
        mean_channel_preference_core_uaware_v1=('channel_preference_core_uaware_v1', 'mean'),
        mean_channel_recent_intent_uaware_v1=('channel_recent_intent_uaware_v1', 'mean'),
        mean_channel_context_time_uaware_v1=('channel_context_time_uaware_v1', 'mean'),
        mean_channel_typed_prior_trust_v2=('channel_typed_prior_trust_v2', 'mean'),
        mean_channel_preference_core_uaware_v2=('channel_preference_core_uaware_v2', 'mean'),
        mean_channel_recent_intent_uaware_v2=('channel_recent_intent_uaware_v2', 'mean'),
        mean_channel_context_time_uaware_v2=('channel_context_time_uaware_v2', 'mean'),
        mean_channel_typed_exact_match_uaware_v1=('channel_typed_exact_match_uaware_v1', 'mean'),
    )
    proxy = proxy_summary(
        df,
        proxy_cols=['audit_review_count', 'audit_tip_rows', 'audit_checkin_rows'],
    )
    user_item_df = build_user_item_channels(df)

    out_cols = [
        'event_id', 'event_type', 'user_id', 'business_id',
        *CHANNEL_COLS,
        'activity_band', 'has_recent_intent_view', 'has_negative_avoidance_view',
    ]
    out_df = df[out_cols].copy()
    out_df.to_parquet(out_dir / 'user_business_match_channels_v2_event_rows.parquet', index=False)
    user_item_df.to_parquet(out_dir / 'user_business_match_channels_v2_user_item.parquet', index=False)
    channel_event_summary.to_csv(out_dir / 'channel_event_summary.csv', index=False)
    channel_band_summary.to_csv(out_dir / 'channel_activity_band_summary.csv', index=False)
    view_summary.to_csv(out_dir / 'channel_view_summary.csv', index=False)
    proxy.to_csv(out_dir / 'channel_proxy_summary.csv', index=False)
    safe_json_write(out_dir / 'channel_sample.json', build_sample_book(df))

    run_meta = {
        'run_id': out_dir.name,
        'run_tag': RUN_TAG,
        'input_run_dirs': {
            'match_feature_run_dir': str(match_run),
            'merchant_card_run_dir': str(merchant_run),
            'user_intent_profile_run_dir': str(user_intent_run),
        },
        'row_counts': {
            'trainable_event_rows_before_history_filter': trainable_rows_before_history_filter,
            'event_rows': int(len(df)),
            'user_item_rows': int(len(user_item_df)),
            'users': int(df['user_id'].nunique()),
            'businesses': int(df['business_id'].nunique()),
        },
        'schema_rule_v1': {
            'separate_preference_core': True,
            'separate_recent_intent': True,
            'separate_context_time': True,
            'separate_context_geo': True,
            'separate_conflict': True,
            'no_recombined_total_feature': True,
            'split_aware_history_only': bool(history_pairs is not None),
            'split_aware_bucket': int(SPLIT_AWARE_BUCKET) if history_pairs is not None else None,
        },
        'outputs': {
            'event_rows_parquet': str(out_dir / 'user_business_match_channels_v2_event_rows.parquet'),
            'user_item_parquet': str(out_dir / 'user_business_match_channels_v2_user_item.parquet'),
            'channel_event_summary_csv': str(out_dir / 'channel_event_summary.csv'),
            'channel_activity_band_summary_csv': str(out_dir / 'channel_activity_band_summary.csv'),
            'channel_view_summary_csv': str(out_dir / 'channel_view_summary.csv'),
            'channel_proxy_summary_csv': str(out_dir / 'channel_proxy_summary.csv'),
            'channel_sample_json': str(out_dir / 'channel_sample.json'),
        },
    }
    safe_json_write(out_dir / 'run_meta.json', run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
