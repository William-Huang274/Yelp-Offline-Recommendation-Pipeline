# Model Card

## Model Lines

| line | role | public status |
| --- | --- | --- |
| PreScore baseline | simple candidate-order baseline | reference only |
| Stage10 `LearnedBlendXGBCls@10` | structured rerank fallback | public fallback |
| Stage11 Qwen3.5-9B reward-model rerank | bounded rescue rerank champion | public champion |

## Intended Use

This stack is intended for offline restaurant recommendation and reranking experiments on Yelp-style data. The public serving layer demonstrates how the ranking stack would be exposed through a small API, release config, fallback policy, and local load test.

## Not Intended For

- real production traffic
- user-facing deployment without privacy, safety, and monitoring work
- full-list LLM reranking
- claiming online A/B lift

## Inputs

The mock serving request accepts:

- user profile fields such as activity bucket, preferred cuisines, price tier, and zone
- candidate business fields such as tags, price tier, zone, prescore, quality, novelty, popularity, route hints, and reward-model score

## Outputs

The mock service returns:

- Top-K ranked businesses
- Stage09 route trace
- baseline rank
- Stage10 rank
- Stage11 rescue band and bonus
- strategy requested / strategy used
- latency and fallback counters

## Evaluation Summary

Current public headline:

- Stage09: `truth_in_pretrim150 = 0.7451`, `hard_miss = 0.1190`
- Stage10 bucket5: `Recall@10 = 0.1261`, `NDCG@10 = 0.0581`
- Stage11 v124 bucket5: `Recall@10 = 0.1857`, `NDCG@10 = 0.0838`

Detailed evaluation protocol is in [eval_protocol.md](./eval_protocol.md).

## Serving And Fallback

Serving config:

- [../config/serving.yaml](../config/serving.yaml)

Fallback ladder:

1. reward-rerank champion
2. XGBoost-style structured rerank
3. baseline ranking

The service reports fallback usage instead of silently changing strategy.

## Limitations

- The public API is a local mock, not a real online serving cluster.
- Latency numbers are local demo measurements, not production SLA.
- Stage11 checkpoints are not committed to git; cloud inventory documents what must be pulled when model files are required.
- Stage12 and A3B experiments are intentionally excluded from this public model card.
