# Expected test failures (fork: exaforce schema pruning)

These upstream tests are kept at upstream parity on purpose and therefore
assert the *un-pruned* schema, which the exaforce runtime patch removes. They
are expected to FAIL. A failure here is only a problem if the failure is NOT an
assertion about a pruned key (e.g. an import/collection error).

Captured from:
`uv run pytest tests/nodes/test_llm_analyzer_base.py tests/nodes/test_semantic_quality_policy.py -q`

- tests/nodes/test_llm_analyzer_base.py::TestLLMAnalysisResult::test_to_finding
- tests/nodes/test_llm_analyzer_base.py::TestLLMAnalysisResult::test_model_dump
- tests/nodes/test_llm_analyzer_base.py::TestMetaAnalyzerResult::test_intent_validation
- tests/nodes/test_semantic_quality_policy.py::TestFixtureMaliciousSkill::test_malicious_skill_findings_preserve_metadata

All four fail with an `AssertionError` (or `KeyError`) about a pruned key
(`explanation`, `intent`) being absent — not an import/collection error.
Confirmed bounded to these two files via `uv run pytest -q -rf`:

```
4 failed, 1261 passed, 13 skipped, 34 deselected, 6 xfailed
```
