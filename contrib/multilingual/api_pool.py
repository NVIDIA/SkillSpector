# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API Key Pool — multi-key load-balancer with per-key concurrency slots.

Each key has a configurable number of concurrent slots (default 5).  The pool
distributes requests across keys using least-loaded scheduling — it *never*
blocks unless every non-rate-limited key is at capacity.  A single key can
serve multiple callers simultaneously; rate-limit (HTTP 429) is the only
signal that removes a key from rotation.

Contrast with the previous mutex-per-key design where :meth:`acquire` blocked
as soon as every key had *one* active request, coupling worker count to key
count.  In the new design, throughput scales with workers independently of
how many keys are configured — keys just need enough aggregate slots.

Integration point
-----------------
Wrap a LangChain ``BaseChatModel`` with :class:`PooledChatModel` to give
it transparent access to the key pool.  The wrapper is API-compatible with
the models returned by :func:`skillspector.llm_utils.get_chat_model` and
can be used wherever a standard ``BaseChatModel`` is expected.

Configuration
-------------
Multi-key mode (recommended for batch scans)::

    export SKILLSPECTOR_API_KEYS="
      sk-or-xxx1|https://api.openai.com/v1|gpt-5.4
      sk-or-xxx2|https://api.openai.com/v1|gpt-5.4
    "

Single-key mode (backward-compatible — no pool needed)::

    export OPENAI_API_KEY=sk-or-xxx1

When ``SKILLSPECTOR_API_KEYS`` is not set, :func:`create_api_key_pool_from_env`
returns ``None`` and the caller should fall back to the single-key provider path.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEYS_ENV = "SKILLSPECTOR_API_KEYS"
_DEFAULT_MAX_CONCURRENT_PER_KEY = 5
_MAX_RATE_LIMIT_RETRIES = 5
_BACKOFF_BASE_S = 30.0
_BACKOFF_CAP_S = 300.0


# ---------------------------------------------------------------------------
# ApiKey — single key tracked by the pool
# ---------------------------------------------------------------------------


@dataclass
class ApiKey:
    """A single API key with concurrency and rate-limit metadata.

    Attributes
    ----------
    key :
        API key string (e.g. ``"sk-or-xxx"``).
    base_url :
        Optional base URL override for the provider endpoint.
    model :
        Model label to use with this key.
    rate_limited :
        ``True`` when this key is cooling down after a 429 response.
    rate_limited_until :
        Monotonic timestamp when this key becomes eligible again after a
        429.  Only meaningful when *rate_limited* is ``True``.
    consecutive_429 :
        Count of consecutive rate-limit hits.  Used to compute the next
        backoff duration via :math:`30 \\times 2^n` seconds, capped at 300.
    total_requests :
        Cumulative request count served by this key.  Used for
        least-loaded scheduling.
    active_requests :
        Number of callers currently using this key.
    max_concurrent :
        Maximum number of simultaneous callers allowed on this key
        (default 5).  One key serves up to this many concurrent LLM calls.
    """

    key: str
    base_url: str | None
    model: str
    rate_limited: bool = False
    rate_limited_until: float = 0.0
    consecutive_429: int = 0
    total_requests: int = 0
    active_requests: int = 0
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT_PER_KEY

    @property
    def available(self) -> bool:
        """``True`` when this key can accept at least one more caller."""
        return not self.rate_limited and self.active_requests < self.max_concurrent


# ---------------------------------------------------------------------------
# ApiKeyPool — multi-key load-balancer
# ---------------------------------------------------------------------------


class ApiKeyPool:
    """Thread-safe pool of API keys with per-key concurrency slots.

    Each key has *max_concurrent* slots (default 5).  :meth:`acquire` picks
    the least-loaded available key — multiple callers can share the same key
    as long as slots remain.  Only rate-limited keys (HTTP 429) are taken
    out of rotation; the pool only blocks when every non-rate-limited key
    is at capacity.

    Usage::

        pool = ApiKeyPool([ApiKey("sk-a", ...), ApiKey("sk-b", ...)])
        key = pool.acquire()          # blocks only if all keys full
        try:
            llm_call(key)
            pool.release(key, success=True)
        except RateLimitError:
            pool.release(key, success=False)
            key = pool.acquire()
    """

    def __init__(self, keys: list[ApiKey]) -> None:
        if not keys:
            raise ValueError("ApiKeyPool requires at least one key")
        self._keys = list(keys)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._rate_limits_hit: int = 0
        self._retry_successes: int = 0
        self._total_requests_served: int = 0
        self._peak_active_requests: int = 0

    # -- Public API -----------------------------------------------------------

    def acquire(self, timeout: float | None = None) -> ApiKey:
        """Acquire a slot on the least-loaded available key.

        Scheduling priority:

        1. **Recovered keys** — rate-limited keys whose backoff has expired
           become available again.
        2. **Least-loaded key** — among available keys, pick the one with
           the fewest ``active_requests``.
        3. **Block** — if every non-rate-limited key is at capacity, wait
           for a slot to free up or a rate-limited key to recover.

        Parameters
        ----------
        timeout :
            Maximum seconds to wait.  ``None`` means wait indefinitely.

        Returns
        -------
        ApiKey
            A key with at least one available slot.

        Raises
        ------
        RuntimeError
            If *timeout* expires before a slot becomes available.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        with self._condition:
            while True:
                now = time.monotonic()

                # Step 1: recover rate-limited keys whose backoff has expired
                self._recover_expired_keys(now)

                # Step 2: find available keys (not rate-limited, slots open)
                available = [k for k in self._keys if k.available]
                if available:
                    key = min(available, key=lambda k: k.active_requests)
                    key.active_requests += 1
                    key.total_requests += 1
                    self._total_requests_served += 1
                    _now_active = sum(k.active_requests for k in self._keys)
                    if _now_active > self._peak_active_requests:
                        self._peak_active_requests = _now_active
                    logger.debug(
                        "Pool: slot on key …%s (%d/%d active)",
                        key.key[-8:],
                        key.active_requests,
                        key.max_concurrent,
                    )
                    return key

                # Step 3: no capacity — compute wait time
                wait_for = self._next_available_in(now)
                remaining = self._remaining_timeout(deadline)
                if remaining is not None and remaining <= 0:
                    raise RuntimeError(
                        "ApiKeyPool: timed out waiting for available slot "
                        f"({self._capacity_summary()})"
                    )

                if wait_for is None:
                    self._condition.wait(timeout=remaining)
                else:
                    wait = min(wait_for, remaining or wait_for)
                    logger.debug(
                        "Pool: at capacity, waiting %.1fs (%s)",
                        wait,
                        self._capacity_summary(),
                    )
                    self._condition.wait(timeout=wait)

    def try_acquire(self) -> ApiKey | None:
        """Non-blocking acquire — returns a key immediately or ``None``.

        Unlike :meth:`acquire`, this never blocks.  If a slot is available
        right now, return the least-loaded key; otherwise return ``None``.
        Useful in async contexts where blocking would stall the event loop.
        """
        with self._lock:
            self._recover_expired_keys(time.monotonic())
            available = [k for k in self._keys if k.available]
            if not available:
                return None
            key = min(available, key=lambda k: k.active_requests)
            key.active_requests += 1
            key.total_requests += 1
            self._total_requests_served += 1
            _now_active = sum(k.active_requests for k in self._keys)
            if _now_active > self._peak_active_requests:
                self._peak_active_requests = _now_active
            return key

    def release(self, key: ApiKey, *, success: bool = True) -> None:
        """Release a slot on *key* back to the pool.

        Parameters
        ----------
        key :
            The key previously obtained from :meth:`acquire`.
        success :
            ``True`` if the API call succeeded; ``False`` if it failed with
            a rate-limit error (HTTP 429).  On failure the key is marked
            rate-limited with exponential backoff.
        """
        with self._condition:
            key.active_requests = max(0, key.active_requests - 1)

            if success:
                key.consecutive_429 = 0
                logger.debug(
                    "Pool: released slot on key …%s (%d/%d active)",
                    key.key[-8:],
                    key.active_requests,
                    key.max_concurrent,
                )
            else:
                key.consecutive_429 += 1
                backoff = min(
                    _BACKOFF_BASE_S * (2 ** (key.consecutive_429 - 1)),
                    _BACKOFF_CAP_S,
                )
                key.rate_limited_until = time.monotonic() + backoff
                key.rate_limited = True
                self._rate_limits_hit += 1
                logger.warning(
                    "Pool: key …%s rate-limited for %.0fs "
                    "(consecutive=%d)",
                    key.key[-8:],
                    backoff,
                    key.consecutive_429,
                )

            self._condition.notify_all()

    def record_retry_success(self) -> None:
        """Increment the retry-success counter for reporting.

        Only call this when a retry (after a key switch due to 429)
        actually succeeds, not on every attempt.
        """
        with self._lock:
            self._retry_successes += 1

    @property
    def rate_limits_hit(self) -> int:
        """Total number of 429 responses encountered across all keys."""
        with self._lock:
            return self._rate_limits_hit

    @property
    def retry_successes(self) -> int:
        """Total number of successful retries after a key switch."""
        with self._lock:
            return self._retry_successes

    @property
    def keys_configured(self) -> int:
        """Total number of keys in the pool."""
        return len(self._keys)

    @property
    def total_capacity(self) -> int:
        """Sum of ``max_concurrent`` across all keys."""
        return sum(k.max_concurrent for k in self._keys)

    @property
    def active_requests(self) -> int:
        """Total active requests across all keys."""
        with self._lock:
            return sum(k.active_requests for k in self._keys)

    def snapshot(self) -> dict[str, object]:
        """Return a snapshot dict suitable for report metadata."""
        with self._lock:
            rate_limited = sum(1 for k in self._keys if k.rate_limited)
            active = sum(k.active_requests for k in self._keys)
            return {
                "keys_configured": len(self._keys),
                "total_capacity": sum(k.max_concurrent for k in self._keys),
                "active_requests": active,
                "peak_active_requests": self._peak_active_requests,
                "total_requests_served": self._total_requests_served,
                "keys_rate_limited": rate_limited,
                "keys_available": len(self._keys) - rate_limited,
                "rate_limits_hit": self._rate_limits_hit,
                "retry_successes": self._retry_successes,
            }

    # -- Internal -------------------------------------------------------------

    def _recover_expired_keys(self, now: float) -> None:
        """Promote rate-limited keys whose backoff has expired."""
        for k in self._keys:
            if k.rate_limited and now >= k.rate_limited_until:
                k.rate_limited = False
                k.consecutive_429 = 0
                logger.info(
                    "Pool: key …%s recovered (backoff expired)", k.key[-8:]
                )

    def _next_available_in(self, now: float) -> float | None:
        """Seconds until the earliest rate-limited key recovers, or ``None``."""
        rate_limited = [k for k in self._keys if k.rate_limited]
        if not rate_limited:
            return None
        earliest = min(k.rate_limited_until for k in rate_limited)
        return max(0.0, earliest - now)

    def _capacity_summary(self) -> str:
        active = sum(k.active_requests for k in self._keys)
        total = sum(k.max_concurrent for k in self._keys)
        rate_limited = sum(1 for k in self._keys if k.rate_limited)
        return (
            f"{active}/{total} slots active, "
            f"{rate_limited} key(s) rate-limited"
        )

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())


# ---------------------------------------------------------------------------
# PooledChatModel — transparent key-switching wrapper
# ---------------------------------------------------------------------------


class PooledChatModel:
    """LangChain-compatible chat model wrapper with transparent key switching.

    Each :meth:`invoke` / :meth:`ainvoke` call acquires a key from the pool,
    builds a :class:`~langchain_openai.ChatOpenAI` instance on the fly, and
    releases the key when done.  On rate-limit errors the wrapper releases
    the key with ``success=False``, picks a different key, and retries.

    Parameters
    ----------
    pool :
        An :class:`ApiKeyPool` with at least one configured key.
    max_tokens :
        ``max_completion_tokens`` passed to each ``ChatOpenAI`` instance.
    timeout :
        Request timeout in seconds passed to each ``ChatOpenAI`` instance.
    max_retries :
        Maximum number of key-switch retries on rate-limit errors before
        giving up.
    """

    def __init__(
        self,
        pool: ApiKeyPool,
        *,
        max_tokens: int = 4096,
        timeout: float = 30.0,
        max_retries: int = _MAX_RATE_LIMIT_RETRIES,
    ) -> None:
        self._pool = pool
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

    # -- Public API -----------------------------------------------------------

    def invoke(self, prompt: str) -> object:
        """Synchronous invoke with automatic key switching on rate-limit."""
        return self._invoke_with_retry(prompt)

    async def ainvoke(self, prompt: str) -> object:
        """Async invoke with automatic key switching on rate-limit."""
        return await self._ainvoke_with_retry(prompt)

    # -- Internal -------------------------------------------------------------

    def _invoke_with_retry(self, prompt: str) -> object:
        """Sync retry loop — acquire slot, call LLM, release, retry on 429."""
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            key = self._pool.acquire()
            llm = self._build_llm(key)
            try:
                result = llm.invoke(prompt)
                self._pool.release(key, success=True)
                if attempt > 0:
                    self._pool.record_retry_success()
                return result
            except Exception as exc:
                if self._is_rate_limit(exc) and attempt < self._max_retries:
                    self._pool.release(key, success=False)
                    logger.debug(
                        "PooledChatModel: rate-limited, retrying "
                        "(attempt %d/%d)",
                        attempt + 1,
                        self._max_retries,
                    )
                    continue
                self._pool.release(key, success=True)
                last_exception = exc
                raise

        raise RuntimeError(
            f"PooledChatModel: exhausted {self._max_retries} retries "
            "due to rate-limit errors"
        ) from last_exception

    async def _ainvoke_with_retry(self, prompt: str) -> object:
        """Async retry loop — non-blocking acquire first, block only if full."""
        import asyncio
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            key = self._pool.try_acquire()
            if key is None:
                key = await asyncio.to_thread(self._pool.acquire)
            llm = self._build_llm(key)
            try:
                result = await llm.ainvoke(prompt)
                self._pool.release(key, success=True)
                if attempt > 0:
                    self._pool.record_retry_success()
                return result
            except Exception as exc:
                if self._is_rate_limit(exc) and attempt < self._max_retries:
                    self._pool.release(key, success=False)
                    logger.debug(
                        "PooledChatModel: rate-limited, retrying "
                        "(attempt %d/%d)",
                        attempt + 1,
                        self._max_retries,
                    )
                    continue
                self._pool.release(key, success=True)
                last_exception = exc
                raise

        raise RuntimeError(
            f"PooledChatModel: exhausted {self._max_retries} retries "
            "due to rate-limit errors"
        ) from last_exception

    def _build_llm(self, key: ApiKey):
        """Build a fresh :class:`~langchain_openai.ChatOpenAI` for *key*."""
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        try:
            import httpx
            _timeout = httpx.Timeout(self._timeout, connect=8.0)
        except ImportError:
            _timeout = self._timeout

        return ChatOpenAI(
            model=key.model,
            base_url=key.base_url,
            api_key=SecretStr(key.key),
            max_completion_tokens=self._max_tokens,
            timeout=_timeout,
        )

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """Detect rate-limit errors from common LLM provider SDKs."""
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                return True
        except ImportError:
            pass

        message = str(exc).lower()
        for marker in ("429", "rate limit", "rate_limit", "too many requests"):
            if marker in message:
                return True

        return False


# ---------------------------------------------------------------------------
# Factory — create pool from environment
# ---------------------------------------------------------------------------


def create_api_key_pool_from_env(
    max_concurrent_per_key: int = _DEFAULT_MAX_CONCURRENT_PER_KEY,
) -> ApiKeyPool | None:
    """Build an :class:`ApiKeyPool` from environment variables.

    Reads ``SKILLSPECTOR_API_KEYS`` — a newline- or semicolon-delimited list
    of ``key|base_url|model`` entries.

    Also supports a fallback format where multiple keys are specified via
    sequentially numbered env vars ``OPENAI_API_KEY``, ``OPENAI_API_KEY_2``,
    etc.

    Parameters
    ----------
    max_concurrent_per_key :
        Maximum simultaneous requests allowed per key (default 5).
        With 10 keys this gives 50 aggregate slots.

    Returns
    -------
    ApiKeyPool or None
        ``None`` when no multi-key configuration is detected.
    """
    keys: list[ApiKey] = []

    raw = os.environ.get(_API_KEYS_ENV, "").strip()
    if raw:
        for line in raw.replace(";", "\n").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 1:
                continue
            key_str = parts[0].strip()
            base_url = parts[1].strip() if len(parts) > 1 else None
            model = parts[2].strip() if len(parts) > 2 else "gpt-5.4"
            keys.append(ApiKey(
                key=key_str, base_url=base_url, model=model,
                max_concurrent=max_concurrent_per_key,
            ))

    if not keys:
        base = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base:
            keys.append(ApiKey(
                key=base, base_url=base_url, model="gpt-5.4",
                max_concurrent=max_concurrent_per_key,
            ))
        for idx in range(2, 10):
            extra = os.environ.get(f"OPENAI_API_KEY_{idx}", "").strip()
            if not extra:
                break
            keys.append(ApiKey(
                key=extra, base_url=base_url, model="gpt-5.4",
                max_concurrent=max_concurrent_per_key,
            ))

    if len(keys) <= 1:
        return None

    total_cap = len(keys) * max_concurrent_per_key
    logger.info(
        "ApiKeyPool: %d keys × %d slots = %d total capacity",
        len(keys), max_concurrent_per_key, total_cap,
    )
    return ApiKeyPool(keys)
