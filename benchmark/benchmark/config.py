"""Provider/run configuration: stable constants and parent-process env wiring."""

from __future__ import annotations

import os

from .auth import token_manager

BEDROCK_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/v1"
BEDROCK_REGION = "us-east-2"
PROVIDER = "openai"
MODEL_ID = "nvidia.nemotron-super-3-120b"
SCAN_TIMEOUT_SECONDS = 300

# SkillSpector's own "do not install" verdict == this recommendation string.
MALICIOUS_RECOMMENDATION = "DO_NOT_INSTALL"


def configure_run(no_llm: bool, timeout: float) -> tuple[dict, str, str, str]:
    """Set stable env in the parent (inherited by spawned workers) and return
    ``(worker_cfg, provider, model, region)``.

    Stable, non-secret config (provider/model/base-url) is placed in the parent
    ``os.environ`` so every spawned worker inherits it. The volatile bedrock
    token is NOT set here -- each worker mints its own (see ``scan_worker``) so
    it can't go stale across a multi-hour run.
    """
    # Inherited by spawned workers (new interpreters read these at startup), so
    # SkillSpector/langchain import-time warnings and logs stay out of the bar.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("SKILLSPECTOR_LOG_LEVEL", "ERROR")
    cfg = {"use_llm": not no_llm, "mint_bedrock": False, "region": "", "timeout": timeout}
    if no_llm:
        return cfg, "", "", ""
    if os.environ.get("SKILLSPECTOR_PROVIDER"):
        # Caller already configured a provider; trust their env (inherited).
        cfg["region"] = os.environ.get("AWS_REGION", "")
        return (
            cfg,
            os.environ["SKILLSPECTOR_PROVIDER"],
            os.environ.get("SKILLSPECTOR_MODEL", ""),
            cfg["region"],
        )
    # Default path: OpenAI-compatible Bedrock endpoint with a minted token.
    model = os.environ.get("MODEL_OVERRIDE") or os.environ.get("AGENTGUARD_MODEL_ID", MODEL_ID)
    region = os.environ.get("AWS_REGION") or BEDROCK_REGION
    manager = token_manager(region)
    manager.clear_abort()  # drop any stale abort marker from a previous run
    try:
        manager.token()  # fail fast (wait=False) + warm the shared cache
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable hint
        raise SystemExit(
            f"Failed to mint a Bedrock token ({type(e).__name__}: {e}).\n"
            "Refresh AWS credentials (e.g. `aws sso login`), set an existing "
            "SKILLSPECTOR_PROVIDER + key in the env, or pass --no-llm."
        ) from e
    os.environ["SKILLSPECTOR_PROVIDER"] = PROVIDER
    os.environ["SKILLSPECTOR_MODEL"] = model
    os.environ["OPENAI_BASE_URL"] = BEDROCK_BASE_URL
    cfg.update(mint_bedrock=True, region=region)
    return cfg, PROVIDER, model, region
