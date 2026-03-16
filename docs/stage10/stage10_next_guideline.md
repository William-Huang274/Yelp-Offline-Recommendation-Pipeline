# Stage10 Next Guideline (2026-02-25)

This file is the working memory for the next iteration. It should be updated after each major run.

## 1) Current Status Against Problem Table

### Recall cap / pretrim cut loss
- Status: **partially solved**
- Latest evidence (`data/metrics/stage09_recall_audit_summary_latest.csv`, run `20260225_130949`, source `20260225_125623_full_stage09_candidate_fusion`):
  - b2: `truth_in_all=0.6756`, `truth_in_pretrim=0.6351`, `pretrim_cut_loss=0.0405`
  - b5: `truth_in_all=0.6773`, `truth_in_pretrim=0.6453`, `pretrim_cut_loss=0.0321`
  - b10: `truth_in_all=0.6277`, `truth_in_pretrim=0.6144`, `pretrim_cut_loss=0.0133`
- Compared with old baseline (~16-18pp cut loss), this is a major fix. Remaining issue is route hard miss.

### Route coverage hard miss
- Status: **not solved**
- Latest hard miss:
  - b2: `0.3244`
  - b5: `0.3227`
  - b10: `0.3723`
- Meaning: even before pretrim, large user share has no truth hit in any route.

### Route diversity
- Status: **partially solved**
- b5/b10 have profile `vector+shared+bridge_user` enabled; b2 is still `vector` only in `run_meta`.
- Unique route hit is still weak for non-ALS routes (cluster/profile unique hit remains low).

### User structure (heavy users)
- Status: **not solved**
- Segment stats show heavy users are materially harder in b10 and still under-covered versus light/mid.

### Candidate fusion strategy consistency
- Status: **mostly solved**
- Bucket-aware pretrim is now active (`250/300/350`) and auditable from run meta + audit CSV.
- Remaining risk: route quality is not yet strong enough, so larger pool does not fully convert to hit gains.

### Rank learnability (XGB)
- Status: **partially solved**
- Latest online-style results (`data/metrics/recsys_stage10_results.csv`, run `20260225_143330`):
  - b2: LearnedBlendXGBCls@10 vs PreScore@10: `NDCG +0.000386`, `Recall +0.001309`
  - b5: flat
  - b10: flat (and previous run had negative drift)
- Conclusion: b2 is learnable, b5/b10 are still weak.

### Train/eval distribution mismatch
- Status: **not solved**
- Latest model (`data/output/10_rank_models/20260225_135103_stage10_1_rank_train/rank_model.json`):
  - train positive rate is around `4.76%` in all buckets
  - calibration positive rate is around `0.25%` (b2), `0.23%` (b5), `0.18%` (b10)
- This mismatch still pushes unstable alpha and weak transfer to online ranking.

### Label construction
- Status: **not solved**
- `label_stats` still shows `train_pos_hist_only=0` and `train_pos_valid_only=0`; time-window positives are not adding supervised positives.

## 2) Next Execution Order (strict)

1. **Finish recall-side fixes first (no long XGB loops yet)**
   - Goal-A1: reduce hard miss:
     - b2/b5 hard miss < 0.30
     - b10 hard miss < 0.35
   - Goal-A2: improve all-route truth:
     - b2/b5 `truth_in_all >= 0.70`
     - b10 `truth_in_all >= 0.65`
   - Priority actions:
     - strengthen b10 profile/cluster quota and heavy-user route quota;
     - enable controlled shared/bridge routes for b2 with purity guard;
     - keep bucket pretrim policy fixed (`250/300/350`) while tuning route coverage.

2. **Then align rank training label/negative strategy**
   - Add weak-supervised positives from time-window history with low weight (not hard positives).
   - Move some easy/random negatives to route-aware hard negatives for b5/b10.
   - Keep calibration split strictly time-safe (`train < calib < test` by user-time policy).

3. **Only after 1+2 are stable, run longer XGB/Transformer rounds**
   - Entry gate for long training:
     - b2/b5/b10 reach Goal-A1 and Goal-A2 in latest audit.
   - Success gate:
     - LearnedBlend beats PreScore on at least 2 buckets with stable repeats.

## 3) Guardrails

- Do not treat single-run small gain as done; require repeated gains.
- Any route/policy change must be logged in run meta and recall audit outputs.
- If b5/b10 alpha repeatedly collapses to 0, stop tuning model params and go back to recall/label quality.

