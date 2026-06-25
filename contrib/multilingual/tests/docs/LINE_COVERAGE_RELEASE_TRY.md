# Line Coverage — `try_acquire()` + `release()` (api_pool.py:239-300)

> Fifth-round audit, file #2. Test-by-test trace into source.

---

## `try_acquire()` (lines 239-258)

```python
246: with self._lock:
247:     self._recover_expired_keys(time.monotonic())
248:     available = [k for k in self._keys if k.available]
```
| Coverage | ✅ All try_acquire tests |
|------|------|

```python
249:     if not available:
250:         return None
```
| Coverage | ✅ test_try_acquire_returns_none_when_slots_exhausted |
|------|------|
| Note | This line makes try_acquire non-blocking — key difference from acquire() |

```python
251:     key = min(available, key=lambda k: k.active_requests)
252:     key.active_requests += 1
253:     key.total_requests += 1
254:     self._total_requests_served += 1
255:     _now_active = sum(k.active_requests for k in self._keys)
256:     if _now_active > self._peak_active_requests:
257:         self._peak_active_requests = _now_active
258:     return key
```
| Coverage | ✅ test_try_acquire_returns_none |
|------|------|
| 🔴 Issue | **Line 255-257 (peak tracking):** No test verifies peak is updated by try_acquire() specifically. Covered incidentally by snapshot tests but never isolated. |
| 🔴 Issue | **Line 254 (total_requests_served):** Same — never verified that try_acquire increments this. Covered incidentally. |
| 🔴 Issue | **Line 247 (recover):** try_acquire calls _recover_expired_keys. No test verifies that a rate-limited key becomes available through try_acquire after manual expiry. Acquire() has this test (#C1), try_acquire() doesn't. |

---

## `release()` (lines 260-300)

```python
272: with self._condition:
273:     key.active_requests = max(0, key.active_requests - 1)
```
| Coverage | ✅ All release tests |
|------|------|
| Note | `max(0, ...)` guard: tested incidentally by Q1 (double-release without re-acquire). Guard works but test doesn't verify it explicitly. |

```python
275:     if success:
276:         key.consecutive_429 = 0
277:         logger.debug(...)
```
| Coverage | ✅ test_active_requests_tracks_correctly, test_release_after_success_resets |
|------|------|
| Note | Line 277-282: debug log — never asserted |

```python
283:     else:
284:         key.consecutive_429 += 1
285:         backoff = min(
286:             _BACKOFF_BASE_S * (2 ** (key.consecutive_429 - 1)),
287:             _BACKOFF_CAP_S,
288:         )
289:         key.rate_limited_until = time.monotonic() + backoff
290:         key.rate_limited = True
291:         self._rate_limits_hit += 1
292:         logger.warning(...)
```
| Coverage | ✅ test_release_with_failure_marks, test_consecutive_429, test_backoff_timestamp |
|------|------|
| 🔴 Issue | **Line 285-287 (backoff formula):** Tests verify output (rate_limited_until) but never feed specific consecutive_429 values to verify intermediate formula results for n=3,4. Only n=1,2 tested. n=3 → 120s, n=4 → 240s, n=5 → 300s(cap) — untested. |
| 🔴 Issue | **Line 291 (rate_limits_hit):** Incremented but only verified via snapshot (incidental). No test directly asserts `pool.rate_limits_hit == N` after N failures. |

```python
300: self._condition.notify_all()
```
| Coverage | ⚠️ Implicitly tested by concurrent test (C7): waiting threads wake up when release calls notify_all. But if notify_all were removed, the test would still pass (threads would eventually timeout instead of deadlocking). The test proves "no deadlock" but not "notify_all specifically worked." |
|------|------|

---

## Summary — try_acquire + release

| Line(s) | Status | Gap |
|----------|--------|-----|
| 247 (try_acquire recover) | ⚠️ | No test for rate-limited key recovery via try_acquire |
| 254-257 (try_acquire counters) | ⚠️ | Peak/total from try_acquire never isolated |
| 273 (max guard) | ⚠️ | Works but never explicitly tested |
| 285-287 (backoff n=3,4,5) | 🔴 | Only n=1,2 tested |
| 291 (rate_limits_hit) | ⚠️ | Never directly asserted |
| 300 (notify_all) | ✅ | Implicit coverage — accepted limitation |

**New findings for audit:**

- **#Q22**: try_acquire recovery path untested (parallel to #C1 which tests acquire recovery)
- **#Q23**: backoff formula n=3,4,5 never exercised
- **#Q24**: rate_limits_hit counter never directly asserted
- **#Q25**: ✅ notify_all behavior implicit — accepted limitation (concurrent test validates overall correctness)
