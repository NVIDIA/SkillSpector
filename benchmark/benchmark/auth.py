"""Bedrock bearer-token management, shared across worker processes.

A bedrock bearer token expires (~12h); a long run outlasts it. Workers can't
share an AWS session object (botocore credentials don't survive ``spawn``), but
they CAN share the minted token string through a small 0600 cache file. The
first worker to find it missing/stale re-mints under a file lock (double-
checked) and writes it; everyone else just reads the cached value. botocore
itself already shares/refreshes the underlying AWS credentials cross-process
via its own on-disk cache, so this layer only dedupes the token signing.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta

_TOKEN_TTL = timedelta(hours=12)
_TOKEN_REFRESH_MARGIN = timedelta(minutes=30)
_AUTH_RETRY_INTERVAL = 15  # seconds between re-mint attempts while paused


class AuthAbortedError(Exception):
    """AWS credentials could not be restored within the pause window; give up."""


class BedrockTokenManager:
    """Cross-process cache for a Bedrock bearer token (file + file lock).

    Re-minting near expiry resolves AWS credentials fresh, so the common case --
    short-lived assume-role/STS creds aging out while the SSO *session* is still
    valid -- self-heals (botocore refreshes via STS). When the SSO *session*
    itself has expired, minting needs a human ``aws sso login``; ``token()``
    then pauses (holding the lock so peers queue), retries until creds return,
    and -- if the wait window is exceeded -- drops an abort marker so every
    worker fails fast instead of each blocking for the full window.
    """

    def __init__(
        self,
        region: str,
        cache_path: pathlib.Path | None = None,
        ttl: timedelta = _TOKEN_TTL,
        refresh_margin: timedelta = _TOKEN_REFRESH_MARGIN,
    ) -> None:
        self.region = region
        self.ttl = ttl
        self.refresh_margin = refresh_margin
        self.auth_wait = float(os.environ.get("SKILLSPECTOR_BENCH_AUTH_WAIT", "1800"))
        default = pathlib.Path(tempfile.gettempdir()) / f"skillspector_bench_token_{region}.json"
        self.cache_path = pathlib.Path(
            os.environ.get("SKILLSPECTOR_BENCH_TOKEN_CACHE", cache_path or default)
        )
        self.lock_path = self.cache_path.with_suffix(".lock")
        self.abort_path = self.cache_path.with_suffix(".abort")
        self._mem: tuple[str, datetime] | None = None  # in-process fast path

    def token(self, wait: bool = False) -> str:
        """Return a valid token, refreshing the shared cache if stale.

        ``wait=True`` (workers) pauses and retries on an auth failure so a
        mid-run SSO expiry can be fixed with ``aws sso login`` without losing
        progress. ``wait=False`` (parent startup) fails fast so a bad-creds run
        aborts immediately instead of hanging.
        """
        if self.abort_path.exists():
            raise AuthAbortedError("credential refresh already abandoned this run")
        now = datetime.now(UTC)
        if self._mem and now < self._mem[1]:
            return self._mem[0]
        cached = self._read()
        if cached and now < cached[1]:
            self._mem = cached
            return cached[0]
        # Missing/stale: exactly one process refreshes; others queue on the lock.
        from filelock import FileLock

        with FileLock(str(self.lock_path)):
            if self.abort_path.exists():
                raise AuthAbortedError("credential refresh already abandoned this run")
            cached = self._read()
            if cached and datetime.now(UTC) < cached[1]:
                self._mem = cached
                return cached[0]
            return self._mint_locked(wait)

    def _mint_locked(self, wait: bool) -> str:
        """Mint under the held lock; pause-retry on auth failure when waiting."""
        deadline = time.monotonic() + self.auth_wait
        announced = False
        while True:
            try:
                fresh = self._mint()
            except Exception as e:  # noqa: BLE001 - any mint failure is retryable
                if not wait:
                    raise  # startup: surface immediately to the caller
                if time.monotonic() >= deadline:
                    self.abort_path.touch()  # signal all workers to stop fast
                    raise AuthAbortedError(
                        f"AWS credentials not restored within {int(self.auth_wait)}s ({e})"
                    ) from e
                if not announced:
                    print(
                        f"\n[paused] Bedrock token mint failed: {e}\n"
                        "  Run `aws sso login` to restore credentials; the run will "
                        f"resume automatically.\n  Retrying every {_AUTH_RETRY_INTERVAL}s "
                        f"(giving up in {int(self.auth_wait / 60)}m).",
                        file=sys.stderr,
                        flush=True,
                    )
                    announced = True
                time.sleep(_AUTH_RETRY_INTERVAL)
                continue
            if announced:
                print("[resumed] AWS credentials restored.", file=sys.stderr, flush=True)
            self._write(*fresh)
            self._mem = fresh
            return fresh[0]

    def _mint(self) -> tuple[str, datetime]:
        from aws_bedrock_token_generator import provide_token

        token = provide_token(region=self.region, expiry=self.ttl)
        # Refresh a margin before the real 12h expiry for safety.
        return token, datetime.now(UTC) + self.ttl - self.refresh_margin

    def clear_abort(self) -> None:
        """Remove any stale abort marker from a previous run."""
        self.abort_path.unlink(missing_ok=True)

    def _read(self) -> tuple[str, datetime] | None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data["token"], datetime.fromisoformat(data["expires_at"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def _write(self, token: str, expires_at: datetime) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"token": token, "expires_at": expires_at.isoformat()}),
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.cache_path)  # atomic publish


_TOKEN_MANAGER: BedrockTokenManager | None = None


def token_manager(region: str) -> BedrockTokenManager:
    """Per-process singleton token manager (state is shared on disk)."""
    global _TOKEN_MANAGER
    if _TOKEN_MANAGER is None or _TOKEN_MANAGER.region != region:
        _TOKEN_MANAGER = BedrockTokenManager(region)
    return _TOKEN_MANAGER
