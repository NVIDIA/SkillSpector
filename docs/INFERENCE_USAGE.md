# Inference usage telemetry

SkillSpector exposes provider-reported LLM usage in JSON reports so CI
consumers can calculate cost without scraping logs or estimating tokens. The
contract is intentionally raw: SkillSpector normalizes token counters and
model provenance, but it does not attach prices or calculate currency values.
This lets downstream systems apply an effective-dated pricing catalog without
rerunning a security scan.

## JSON contract

Run a scan with machine-readable output:

```bash
skillspector scan ./my-skill --format json
```

Each successfully observed provider response contributes one entry to
`metadata.inference_usage`:

```json
{
  "metadata": {
    "llm_requested": true,
    "llm_available": true,
    "inference_usage": [
      {
        "node": "semantic_security_discovery",
        "request_kind": "structured_output",
        "provider": "nv_inference",
        "model": "azure/anthropic/claude-opus-4-6",
        "model_source": "provider_response",
        "usage_source": "provider_response",
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_tokens": 400,
        "cache_write_tokens": 50,
        "reasoning_tokens": 25,
        "total_tokens": 1100
      }
    ]
  }
}
```

| Field | Meaning |
|---|---|
| `node` | SkillSpector analyzer that made the request. |
| `request_kind` | Invocation shape, such as `structured_output` or `chat_completion`. |
| `provider` | Sanitized provider identifier; it never contains an endpoint or credential. |
| `model` | Provider-returned model identity when available, otherwise the exact requested model. |
| `model_source` | `provider_response` when the response unambiguously identified a different resolved model; `requested_model` when identity is absent or indistinguishable from a client-configured fallback. |
| `usage_source` | Always `provider_response`. SkillSpector does not emit estimated usage records. |
| `prompt_tokens` | Total normalized input tokens, inclusive of cache reads and cache writes. |
| `completion_tokens` | Provider-reported output tokens. |
| `cached_tokens` | Cache-read input tokens; a subset of `prompt_tokens`. |
| `cache_write_tokens` | Cache-creation input tokens; a subset of `prompt_tokens`. |
| `reasoning_tokens` | Provider-reported reasoning-token partition, normally a subset of completion usage. |
| `total_tokens` | Provider total, normalized to `prompt_tokens + completion_tokens` when both partitions are known. |

Counter fields are optional because providers and transports expose different
levels of detail. A present zero is an observed zero. A missing field means the
provider did not expose that counter; it must not be treated as zero.

## Model provenance

`model_source` and `usage_source` answer different questions:

- `usage_source=provider_response` means all token counters in the record came
  from the completed provider response. SkillSpector never derives billing
  counters from prompt length, local tokenizers, or analyzer token budgets.
- `model_source=provider_response` means the provider returned a valid model
  identity distinguishable from the requested value. This is the strongest
  identity for pricing because a gateway can route an alias to a different
  deployed model.
- `model_source=requested_model` means the response had usage counters but no
  independently verifiable model identity. This includes LangChain clients that
  copy their configured model into response metadata when the provider omits
  the field. `model` is then the exact model SkillSpector requested; downstream
  pricing can use it, but should retain the weaker provenance.

The configured model is resolved independently for each analyzer slot. The
general precedence is:

1. `SKILLSPECTOR_MODEL_<SLOT>`
2. `SKILLSPECTOR_MODEL`
3. the active provider's default for that slot
4. the active provider's general default

For example, `SKILLSPECTOR_MODEL_META_ANALYZER` affects only the
`meta_analyzer` slot, while `SKILLSPECTOR_MODEL` overrides every slot that has
no slot-specific override. A configured slot is not proof that a request ran.
Only a corresponding `inference_usage` record proves that SkillSpector received
a provider response with usage counters.

## Cache and total-token semantics

SkillSpector normalizes provider differences into one additive pricing shape:

```text
uncached prompt = prompt_tokens - cached_tokens - cache_write_tokens
total tokens    = prompt_tokens + completion_tokens
```

OpenAI-compatible responses generally report cache-read tokens as a partition
already included in prompt tokens. Raw Anthropic responses report ordinary
input, cache reads, and cache creation separately. SkillSpector adds the raw
Anthropic cache partitions exactly once so `prompt_tokens` is inclusive for
both response shapes.

Anthropic cache-creation TTL details, when present, are combined into
`cache_write_tokens`. SkillSpector does not currently send prompt-cache
controls, so it does not choose between the separate 5-minute and 1-hour cache
write tiers. Downstream pricing must not infer a TTL that the provider response
did not preserve.

`reasoning_tokens` is a diagnostic partition and must not be added to
`completion_tokens` a second time. Likewise, cache reads and cache writes must
not be added to `prompt_tokens` after normalization.

## Missing usage and fail-closed integrations

`metadata.inference_usage` is always a list in JSON output. An empty list means
usage was not observable. It does **not** mean that no LLM ran, that the request
was free, or that the token count was zero. Typical causes include a provider or
CLI transport that does not expose counters, an LLM call that failed before a
response, or a static-only scan.

Cost observability and security-gate validity are separate decisions. A JSON
consumer should:

1. require a parseable top-level JSON object;
2. treat a fatal process exit or `execution_successful: false` as a blocking
   validation error;
3. surface `analysis_completeness.ledger_exceptions` for diagnosis;
4. apply its security policy to `risk_assessment.recommendation`; and
5. ingest every valid `inference_usage` record, including records preserved in
   a failed LLM attempt, because a failed scan can still incur provider cost.

Malformed telemetry must be discarded without turning an otherwise valid scan
into a failure. Conversely, valid usage telemetry must never make an incomplete
security scan pass. When an integrating tool retries a failed LLM scan in
static-only mode, it should ingest the failed attempt's usage once and avoid
double-counting the retry payload.

## Privacy and trust boundary

The report uses an explicit allowlist. Usage records contain only bounded
labels and non-negative provider counters. They do not contain prompts,
completions, analyzed skill content, credentials, headers, endpoint URLs,
provider request IDs, or raw provider metadata. Records with unknown sources,
invalid labels, negative or unbounded counters, or no counters are omitted.

Treat the JSON report as untrusted input at every downstream boundary. Validate
the allowlisted fields and counter ranges again before appending metrics or
applying prices.

## Downstream handoff

The intended handoff is:

```text
SkillSpector provider response
  -> metadata.inference_usage in the SkillSpector JSON report
  -> integrating evaluator validates and projects raw usage
  -> CI publishes a versioned metrics artifact
  -> dashboard applies an effective-dated pricing catalog
```

The evaluator should preserve `provider`, `model`, `model_source`,
`usage_source`, the analyzer/request identity, and every observed token
partition. Currency calculation belongs downstream so historical usage can be
repriced when a catalog is corrected without rewriting the original scan
artifact.
