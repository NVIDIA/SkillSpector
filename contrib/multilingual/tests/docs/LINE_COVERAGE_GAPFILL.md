# Line Coverage — `GapFillAnalyzer.parse_response()` (gap_fill.py:206-257)

> Fifth-round audit, file #3. 22 tests targeting this function.

---

## `parse_response()` Source with Coverage

```python
213: text = str(response).strip()
```
| Coverage | ✅ All 22 tests — str(), int, Pydantic model, BOM |
|------|------|
| 🔴 Edge | `response = GapFillResult(...)` (Pydantic model): `str()` gives repr, not JSON. json.loads fails → returns []. Graceful but Q9 docstring is wrong. |

```python
216: if text.startswith("```"):
```
| ✅ True | Fence tests (3) |
| ✅ False | All other tests (19) |

```python
217:     first_nl = text.find("\n")
218:     if first_nl != -1:
219:         text = text[first_nl + 1:]
```
| ✅ True  | All fence tests (have newline) |
| 🔴 False | **No test**: `text = "```"` (only backticks, no newline). Uncovered branch. |

```python
220:     if text.rstrip().endswith("```"):
221:         text = text.rstrip()[:-3].rstrip()
```
| ✅ True  | All fence tests (have closing fence) |
| 🔴 False | **No test**: `text = "```json\ndata"` (opening fence, no closing fence). Uncovered branch. |

```python
225: try:
226:     data = json.loads(text)
227: except json.JSONDecodeError:
233:     return []
```
| ✅ Success | Valid JSON tests |
| ✅ Exception | test_not_json, test_empty_string, test_utf8_bom (BOM causes decode fail → exception caught) |
| Note | Line 228-232: logger.warning — never asserted, incidental coverage |

```python
235: try:
236:     result = GapFillResult.model_validate(data)
237: except Exception:
243:     return []
```
| ✅ Success | Valid schema tests |
| ✅ Exception | test_findings_is_not_list, test_integer, test_severity_not_in_literal |
| Note | Line 238-242: logger.warning — never asserted |

```python
246: for item in result.findings:
247:     if item.rule_id not in _GAP_FILL_RULE_IDS:
253:         continue
```
| ✅ True | test_unknown_rule_id_filtered_out |
| ✅ False | All valid-rule tests |

```python
254:     if item.confidence < 0.7:
255:         continue
```
| ✅ True | test_low_confidence_filtered_out |
| ✅ False (==0.7) | test_confidence_at_threshold_kept |
| ✅ False (>0.7) | All valid-finding tests |

```python
256:     findings.append(item.to_finding(batch.file_path))
257: return findings
```
| ✅ Empty list | Various invalid-input tests |
| ✅ Populated list | Valid JSON tests |
| ✅ 100-item list | test_parses_one_hundred_findings_within_one_second |

---

## 🔴 Uncovered Branches (New Findings)

### #Q26 `text.startswith("```")` True but `first_nl == -1`
**Trigger:** `text = "```"` — only backticks, no newline character.
**Behavior:** `first_nl = -1`, line 218 is False, line 219 skipped. `text` stays as `"```"`. Then line 220: `text.rstrip().endswith("```")` → True. Line 221: `text = "```".rstrip()[:-3].rstrip() = ""`. Then `json.loads("")` → JSONDecodeError → returns []. No crash, but the fence-stripping path never tested with this input.

### #Q27 `text.startswith("```")` True but `text.rstrip().endswith("```")` False
**Trigger:** `text = "```json\ndata"` — opening fence, no closing fence, no trailing backticks.
**Behavior:** Line 220 is False, line 221 skipped. `text` stays as `"data"` (after line 219 slice). Then `json.loads("data")` → JSONDecodeError → returns []. Fence NOT stripped. This might be valid behavior (malformed output) or a bug — either way, untested.

### #Q28 Fence stripping + leading whitespace WITHOUT strip() first
**What if:** `text = "  ```json\ndata\n```"` — leading spaces before fence. `startswith("```")` is False! `str(response).strip()` on line 213 handles this. But the test `test_json_with_leading_trailing_whitespace` verifies this. ✅ Covered.

---

## Summary

| # | Line(s) | Status | Trigger |
|---|---------|--------|---------|
| Q26 | 218 (False) | 🔴 Uncovered | Fence with no newline |
| Q27 | 220 (False) | 🔴 Uncovered | Fence with no closing ``` |
| Q9 | 213 (Pydantic model) | 🟢 Covered but misleading | Docstring says "delegates" but actually graceful degradation |
