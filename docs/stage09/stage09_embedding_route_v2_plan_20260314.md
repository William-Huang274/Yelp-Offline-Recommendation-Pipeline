# Stage09 Embedding / Route V2 Audit And Plan (2026-03-14)

## 1. Conclusion Up Front

The current bucket10 issue is not only a head-ranking problem.

- `truth_in_all` has already shown that big-pool replay can bring recall back to
  `94%+`
- but `truth_in_top150` still stalls around `40%`
- this indicates that the bottleneck has moved from ?was the truth recalled at
  all? to ?the embedding / route evidence is not fine-grained enough to separate
  challengers from incumbents near the pre-rank boundary?

More concretely, the current `stage09` embedding and route design has three
limits:

1. the user side only has a single merged `profile_text` vector instead of a
   true multi-view user representation
2. the merchant side is still centered on a fixed tag taxonomy instead of a
   reusable merchant dense-embedding space
3. multiple profile routes exist, but they are merged and truncated too early,
   which destroys route identity and route diversity

## 2. Limitations In The Current Implementation

### 2.1 User-profile embedding is still a single-vector v1 design

Code evidence:

- [09_user_profile_build.py:103](../../scripts/09_user_profile_build.py#L103)
- [09_user_profile_build.py:941](../../scripts/09_user_profile_build.py#L941)
- [09_user_profile_build.py:942](../../scripts/09_user_profile_build.py#L942)
- [09_user_profile_build.py:950](../../scripts/09_user_profile_build.py#L950)
- [09_user_profile_build.py:1120](../../scripts/09_user_profile_build.py#L1120)
- [09_user_profile_build.py:1124](../../scripts/09_user_profile_build.py#L1124)

Current behavior:

- the script already builds `profile_text_short` and `profile_text_long`
- but the final embedding path only allows `EMBED_SCOPE="profile_text"`
- so short-term and long-term evidence are concatenated into one text and then
  encoded into one vector

Direct consequences:

- recent interests and long-term taste are collapsed into the same vector
- positive preferences and negative dislikes are not represented separately
- different facets such as cuisine, taste, service, and scene do not get their
  own vectors
- route logic can only use one `user_profile_vectors.npz` for a single-path
  similarity score

This design is clearly tuned for conservative local stability, not for a richer
cloud-side representation.

### 2.2 Merchant semantics still follow a tag-prototype system, not a merchant dense representation

Code evidence:

- [09_item_semantic_build.py:135](../../scripts/09_item_semantic_build.py#L135)
- [09_item_semantic_build.py:470](../../scripts/09_item_semantic_build.py#L470)
- [09_item_semantic_build.py:717](../../scripts/09_item_semantic_build.py#L717)
- [09_item_semantic_build.py:718](../../scripts/09_item_semantic_build.py#L718)
- [09_item_semantic_build.py:849](../../scripts/09_item_semantic_build.py#L849)
- [09_item_semantic_build.py:859](../../scripts/09_item_semantic_build.py#L859)

Current behavior:

- merchants are first filtered by `state/categories`
- review sentences are matched against fixed `TAG_DEFS` prototype text with
  embeddings
- the output is still `semantic_score`, `semantic_confidence`, `top_pos_tags`,
  and `top_neg_tags`

The problem is that this remains:

- `sentence -> fixed tag prototypes -> tag aggregate`

and not:

- `merchant -> dense semantic representation`

As a result, merchant attributes such as `name`, `city`, `attributes`, `hours`,
`price`, and `stars` barely enter the dense semantic space. The current design
is strong for explainable tag evidence, but weak as a true embedding route.

### 2.3 Profile routes are merged too early

Code evidence:

- [09_candidate_fusion.py:47](../../scripts/09_candidate_fusion.py#L47)
- [09_candidate_fusion.py:48](../../scripts/09_candidate_fusion.py#L48)
- [09_candidate_fusion.py:49](../../scripts/09_candidate_fusion.py#L49)
- [09_candidate_fusion.py:50](../../scripts/09_candidate_fusion.py#L50)
- [09_candidate_fusion.py:1455](../../scripts/09_candidate_fusion.py#L1455)
- [09_candidate_fusion.py:1634](../../scripts/09_candidate_fusion.py#L1634)
- [09_candidate_fusion.py:1678](../../scripts/09_candidate_fusion.py#L1678)
- [09_candidate_fusion.py:1746](../../scripts/09_candidate_fusion.py#L1746)
- [09_candidate_fusion.py:1844](../../scripts/09_candidate_fusion.py#L1844)
- [09_candidate_fusion.py:1845](../../scripts/09_candidate_fusion.py#L1845)

Current behavior:

- the main profile routes are `vector`, `shared`, `bridge_user`, and
  `bridge_type`
- `bridge_type` is still disabled by default
- `shared` relies on exact typed-tag overlap
- `bridge_user` relies on typed-tag user similarity
- all routes are concatenated into one stream
- everything is then sorted by a unified `source_score/source_confidence`
- everything is truncated into the same `profile_top_k`

That means:

- route identity is lost too early
- weaker but complementary routes can be buried by the dominant vector route
- there is no per-route quota or keepalive mechanism
- there are no short/long, facet, or negative-specific routes yet

## 3. Why The Current Head Feature Surface Is Stuck

Recent boundary repair and pairwise boundary probes already showed a clear
signal:

- under the current feature surface, quota and weight sweeps top out around
  `top150 +11`
- the best pairwise probe ultimately degenerates into ?all extra weights are
  nearly zero?

That usually means the head grid is not the real limit. The real limit is that
challengers and incumbents are not separable enough based on the current input
evidence.

In other words:

- the truth candidates around `151-200` already have some profile evidence
- but the evidence is not rich enough and not route-specific enough
- so those candidates cannot reliably beat incumbent items in the `131-150`
  region during boundary comparisons

## 4. New Features Needed In V2

### 4.1 User side: move from one vector to multiple vectors

Recommended user embeddings:

1. `user_vec_short`
   - encode recent-window preference only
   - used for short-term interest routes
2. `user_vec_long`
   - encode long-term stable taste only
   - used for long-horizon preference routes
3. `user_vec_pos`
   - encode only strongly positive sentences or tag evidence
   - represent what the user actively likes
4. `user_vec_neg`
   - encode negative sentences or negative tag evidence
   - represent explicit dislike or rejection evidence
5. `user_facet_vec_{facet}`
   - start with `cuisine`, `taste`, `service`, and `scene`

Instead of concatenating everything into one long text and one vector, export a
small family of vectors directly.

### 4.2 Merchant side: add merchant dense embeddings instead of only tag aggregation

Recommended merchant text views:

1. `merchant_text_core`
   - `name + categories + city + state`
2. `merchant_text_review_summary`
   - top high-confidence review sentences
   - separated into positive and negative evidence
3. `merchant_text_tag_summary`
   - a structured summary built from `item_tag_profile_long` and
     `item_semantic_features`

Recommended output vectors:

1. `merchant_vec_core`
2. `merchant_vec_review`
3. `merchant_vec_pos`
4. `merchant_vec_neg`
5. optional `merchant_facet_vec_{facet}`

That would allow user-to-merchant dense ANN retrieval instead of relying only on
fixed tag overlap.

### 4.3 Boundary-aware features

Even before route changes, the feature audit should add:

- `short_long_cos_gap`
- `pos_neg_margin`
- `route_agreement_count`
- `best_profile_route_score`
- `second_best_profile_route_score`
- `profile_route_entropy`
- `merchant_semantic_density`
- `facet_overlap_count`
- `facet_conflict_count`
- `negative_match_penalty`
- `vector_shared_consensus`
- `vector_bridge_consensus`

These are better suited to the `131-150 vs 151-200` boundary comparator.

## 5. New Routes Needed In V2

### 5.1 Add true dense retrieval routes first

At minimum, add:

1. `profile_short_vector_route`
   - `user_vec_short -> merchant_vec_review`
2. `profile_long_vector_route`
   - `user_vec_long -> merchant_vec_core / merchant_vec_review`
3. `profile_pos_route`
   - `user_vec_pos -> merchant_vec_pos`
4. `profile_negative_filter_route`
   - `user_vec_neg -> merchant_vec_neg`
   - this can start as a penalty or filter instead of direct recall

### 5.2 Add facet routes

Examples:

- `cuisine_route`
- `taste_route`
- `service_route`
- `scene_route`

These routes are useful because they can protect highly personalized but less
popular candidates near the top150 boundary.

### 5.3 Preserve route identity instead of flattening too early

One of the largest structural problems today is the early concat-plus-truncate
pattern. V2 should instead:

- keep a separate top-k per route first
- rank inside each route first
- carry `route_name`, `route_score`, `route_rank`, and `route_confidence`
- preserve signals showing that the same candidate is supported by multiple
  routes
- avoid mixing all routes into one stream before `profile_top_k`

Otherwise, even stronger embeddings will lose their value at merge time.

## 6. What To Improve In The Embedding Layer Once Cloud Resources Are Better

Cloud resources are now materially stronger than the local machine, so the
embedding layer should not stay locked to local-safe parameters forever.

Recommended upgrades:

1. increase `bge-m3` batch size
   - the user-profile recovery run already showed GPU headroom was not saturated
2. prefer the `sentence-transformers` path when possible
   - item semantic build already supports it alongside HF mean-pool fallback
3. produce multiple vector artifacts instead of one
   - `user_profile_vectors_short.npz`
   - `user_profile_vectors_long.npz`
   - `user_profile_vectors_pos.npz`
   - `user_profile_vectors_neg.npz`
   - and the merchant-side equivalents
4. start with audit-only route exports
   - do not rewrite production `stage09` all at once
   - first measure `truth_in_all`, `truth_in_pretrim`, and `truth_in_top150`

## 7. Recommended Rollout Order

### Phase 1: lowest risk, add evidence first without changing main semantics

1. make user-profile build export multiple vectors
2. make item-semantic build export merchant dense embeddings
3. add audit-only new-route exports in stage09

Goal:

- verify whether the new routes can lift big-pool `truth_in_top150`
- do not modify the main fusion contract yet

### Phase 2: preserve route identity

1. stop forcing vector/shared/bridge into one immediate `profile_top_k`
2. switch to per-route top-k with route keepalive
3. keep the strongest route-specific evidence inside fusion / pre-rank

Goal:

- see whether route diversity improves the `131-150` boundary

### Phase 3: build head-v3 afterwards

Only once phases 1 and 2 prove that the feature surface is actually stronger
should the project move to:

- a boundary-specific comparator
- lane allocation v3
- or a formal learned pre-ranker

## 8. The Three Highest-Priority Actions

If only three things are worth doing first, the order should be:

1. multi-vector user representation
   - `short / long / pos / neg`
2. merchant dense embeddings
   - stop relying on tag prototypes alone
3. per-route preservation in profile routing
   - stop merging and truncating too early

Without these three changes, it is unlikely that `top150` can move from the
current `40%` level to a materially stronger position.
