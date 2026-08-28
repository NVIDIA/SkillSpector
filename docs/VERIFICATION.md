# Verification boundary

SkillSpector combines deterministic analyzers with optional model inference. Verification
claims must keep those surfaces separate:

| Stratum | Surface | Contract | Mechanism |
| --- | --- | --- | --- |
| S1 | risk scoring and banding | the implementation matches the executable reference model | differential property tests |
| S2 | deterministic findings crossing the LLM meta-analyzer | arbitrary valid model outcomes cannot remove, invent, or weaken deterministic evidence | adversarial-oracle property tests |
| S3 | semantic conclusions produced by an LLM | no universal correctness claim | evaluation datasets and provider tests |

Run the fast S1/S2 contract with:

```bash
make verify
```

## Phase 1 invariants

- Risk scores are integers in `[0, 100]` and map exhaustively to the documented bands.
- Diminishing returns apply independently per rule with weights `1`, `0.5`, and `0.25`.
- Higher severity receives priority within a rule. Equal-severity findings are ordered by
  confidence and then executable status, making the score permutation-invariant and
  preventing weaker weighted evidence from displacing stronger evidence solely because an
  analyzer returned it first.
- Blocking floors for proven SC8, BH2, and BH3 conditions survive ordinary weighted scoring.
- Executable-file multiplication is scoped by both source provenance and path.
- The LLM meta-analyzer may enrich a deterministic finding, but cannot remove it, change its
  identity or severity, lower its confidence, or create a new deterministic finding.

The reference model in `tests/verification/risk_reference.py` intentionally imports no
production scoring constants or helpers. A scoring-policy change must update that readable
specification deliberately. The suite does not claim to verify static detector recall, LLM
truthfulness, provider availability, or arbitrary malformed data rejected before the typed
meta-analyzer boundary.
