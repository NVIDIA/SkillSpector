# Line Coverage — Context Manager + Patches (runner.py:300-590)

> Fifth-round audit, file #4. _verify_patch_targets, _check_signature, _apply_patches, _restore_patches, deepseek_compat.
> ⚠️ runner.py grew from ~530 to 789 lines; all line numbers verified against current version.

---

## `_apply_patches()` (lines 474-507)

```python
449: if _patches_depth > 0:
450:     _patches_depth += 1
451:     return
```
| ✅ True | Nesting tests (double, triple) |
| ✅ False | First entry in all context manager tests |

```python
453: _verify_patch_targets()
```
| ✅ | All context manager enter tests |
| 🔴 | Q13: tests only verify this doesn't CRASH. No test verifies it actively catches a broken upstream. |

```python
455-460: LLMAnalyzerBase.__init__ = _patched_base_init  (+4 more)
```
| ✅ | All apply tests |

```python
462-467: try: import httpx; _ChatOpenAI.__init__ = _patched_chatopenai_init
468-469: except ImportError: logger.debug(...)
```
| ✅ try | All apply tests (httpx always installed in dev) |
| 🔴 except | **Zero coverage.** ImportError path never triggered. If httpx is removed from dependencies, Patch 6 silently skips with no test catching the behavior change. |

```python
471: _asyncio.run = _patched_asyncio_run
473: _patches_depth = 1
```
| ✅ | All apply tests |

---

## `_restore_patches()` (lines 508-550)

```python
484: if _patches_depth == 0: return
```
| ✅ True | Called outside any context (should no-op) — test_patches_restored checks this implicitly |
| ✅ False | Normal context exit |

```python
486: _patches_depth -= 1
487: if _patches_depth > 0: return
```
| ✅ True | Nested context exit (double, triple tests) |
| ✅ False | Outermost exit |

```python
490-495: LLMAnalyzerBase.__init__ = _original_base_init (+4 more)
```
| ✅ | All restore tests |

```python
497-502: if _original_chatopenai_init is not None: restore ChatOpenAI
```
| ✅ True | All restore tests (Patch 6 was applied, so original is not None) |
| 🔴 except ImportError | **Zero coverage.** Same as apply — langchain_openai always available in dev. |

```python
504: _asyncio.run = _original_asyncio_run
```
| ✅ | All restore tests |

---

## `_check_signature()` (lines 440-473)

```python
426: sig = inspect.signature(func)
```
| ✅ | All _verify_patch_targets calls |

```python
428: except (ValueError, TypeError) as exc: raise RuntimeError(...)
```
| 🔴 | **Zero coverage.** No test passes an uninspectable function. |
| Note | This would only trigger if upstream replaced a method with a C extension or non-callable. Extremely rare. |

```python
434-438: for param in expected_params: if param not in sig.parameters: raise
```
| ✅ False | All 17 current checks pass (params exist) |
| 🔴 True | **Zero coverage.** No test verifies what happens when a param IS missing — the core purpose of this function. Q13. |

---

## `deepseek_compat()` (lines 551-590)

```python
520: _apply_patches()
try: yield
finally: _restore_patches()
```
| ✅ yield | All context manager tests |
| ✅ finally on exception | test_patches_restored_on_exception |
| ✅ finally on normal exit | All restore tests |

**All branches covered.** ✅

---

## Summary

| # | Line(s) | Status | Issue |
|---|---------|--------|-------|
| Q13 | 434-438 (param missing) | 🔴 | Guard's raise path never triggered |
| Q28 | 468-469 (ImportError) | 🔴 | Patch 6 skip path zero coverage |
| Q29 | 428 (uninspectable) | 🔴 | _check_signature except path zero coverage |
| - | All other lines | ✅ | Covered |
