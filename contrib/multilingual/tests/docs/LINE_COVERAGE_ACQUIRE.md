# Line Coverage — `ApiKeyPool.acquire()` (api_pool.py:165-237)

> Fifth-round audit: test-by-test trace into source, line by line.
> 26 tests in test_api_pool.py. Below is every executable line in acquire() and which tests reach it.

---

## Acquire Source with Coverage Annotations

```python
192: deadline = time.monotonic() + timeout if timeout is not None else None
```
| Tests | 15 tests call acquire() with timeout=None; 1 test (timeout) with timeout=0.1; concurrent test with timeout=5.0 |
|------|------|
| Coverage | ✅ Full |

```python
194: with self._condition:
```
| Tests | All tests calling acquire() or try_acquire() |
|------|------|
| Coverage | ✅ Full |

```python
195: while True:
```
| Tests | All acquire tests |
|------|------|
| Coverage | ✅ Full |
| Note | First iteration exits in all tests. Loop runs >1 iteration ONLY in concurrent test (waiting threads wake and re-check). |

```python
196:     now = time.monotonic()
```
| Tests | All acquire tests |
|------|------|
| Coverage | ✅ Full |

```python
199:     self._recover_expired_keys(now)
```
| Tests | All acquire tests (called every iteration) |
|------|------|
| Coverage | ✅ Full |
| Note | Tests that verify recovery: test_recover_expired_keys_restores_availability, test_recovered_key_can_be_acquired_again |

```python
202:     available = [k for k in self._keys if k.available]
```
| Tests | All acquire tests |
|------|------|
| Coverage | ✅ Full |

```python
203:     if available:
204:         key = min(available, key=lambda k: k.active_requests)
205:         key.active_requests += 1
206:         key.total_requests += 1
207:         self._total_requests_served += 1
208:         _now_active = sum(k.active_requests for k in self._keys)
209:         if _now_active > self._peak_active_requests:
210:             self._peak_active_requests = _now_active
211:         logger.debug(...)
217:         return key
```
| Tests | 22 tests with available slots |
|------|------|
| Coverage | ✅ Full for lines 204-210, 217 |
| Coverage | ⬜ Line 211-216: debug log — never asserted, covered only incidentally |
| Note | Line 204 (min): test_released_slot_returns_least_loaded_key specifically verifies least-loaded behavior |
| Note | Line 208-210 (peak): test_snapshot_reflects_peak_and_total_after_usage verifies |

```python
219:     # Step 3: no capacity
220:     wait_for = self._next_available_in(now)
```
| Tests | Called when `available` is empty |
|------|------|
| Coverage | ⚠️ Called in timeout test + concurrent test |
| Note | Return value never influences behavior in any test: timeout test raises before reaching line 228; concurrent test has no rate-limited keys (wait_for=None) |

```python
221:     remaining = self._remaining_timeout(deadline)
```
| Tests | Timeout test (deadline set), concurrent test (deadline set) |
|------|------|
| Coverage | ✅ |

```python
222:     if remaining is not None and remaining <= 0:
223:         raise RuntimeError(
224:             "ApiKeyPool: timed out waiting for available slot "
225:             f"({self._capacity_summary()})"
226:         )
```
| Tests | `test_acquire_with_timeout_raises_runtime_error_when_pool_full` |
|------|------|
| Coverage | ✅ Line 222-226 |
| Note | Lines 224-225 (`_capacity_summary()`): called but string content never asserted |

```python
228:     if wait_for is None:
229:         self._condition.wait(timeout=remaining)
```
| Tests | Concurrent test (all keys at capacity, none rate-limited → wait_for=None) |
|------|------|
| Coverage | ✅ |

```python
230:     else:
231:         wait = min(wait_for, remaining or wait_for)
232:         logger.debug(
233:             "Pool: at capacity, waiting %.1fs (%s)",
234:             wait,
235:             self._capacity_summary(),
236:         )
237:         self._condition.wait(timeout=wait)
```
| Tests | 🔴 **NONE. Zero coverage.** |
|------|------|
| Coverage | ❌ |
| Trigger condition | All non-rate-limited keys at capacity AND at least one key rate-limited with future recovery time |
| Required scenario | 1-key 1-slot pool: acquire → use → 429 → release(fail) → try to acquire again (key is rate-limited, no other keys) |

---

## Summary

| Lines | Status | Tests |
|-------|--------|-------|
| 192-210, 217 | ✅ Happy path | 22 tests |
| 211-216 | ⬜ Debug log (incidental) | All happy-path tests |
| 220-221 | ⚠️ Called but return unused | Timeout, concurrent |
| 222-226 | ✅ Timeout | 1 test |
| 228-229 | ✅ Pure wait | Concurrent test |
| 230-237 | 🔴 **ZERO** | **No test** |
| 199 (recovery) | ⚠️ Manually expired only | 2 tests |
