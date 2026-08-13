# SkillSpector v2.9.4

Released: 2026-08-12

## Summary

This patch strengthens SkillSpector’s safe handling of MCP requests and untrusted skill content, while adding broader prompt-injection and supply-chain detection coverage. It also improves permission guidance, ships a companion Skill Inspector guide, and refreshes project documentation.

## Highlights

- HTTP-exposed MCP servers now reject caller-controlled local scan targets and local YARA-rule directories while preserving local scanning for trusted stdio use.
- Detect whitespace-padding prompt-injection attempts and shipped Python bytecode, with improved minimum risk scoring for high-impact findings.

## Added

- Add detection for whitespace padding used to hide prompt-injection instructions.
- Add a HIGH SC8 finding when a skill ships Python bytecode or `__pycache__` content.
- Add the Skill Inspector companion skill guide.

## Changed

- Treat `allowed-tools` as valid least-privilege permission guidance in remediations and documentation.
- Add an OpenSSF Scorecard badge to the project documentation.

## Fixed

- Reject local filesystem scan targets and local YARA-rule directories for HTTP MCP transport, preventing remote callers from selecting scanner-host paths.
- Reject symlinked skill content during discovery and disable Git symlink materialization when cloning input repositories.
- Ensure high-impact findings receive an appropriate minimum risk score.

## Security

- Harden HTTP MCP transport against local-path access and strengthen skill-content handling against symlink traversal.

## Breaking Changes and Migration

- HTTP MCP clients can no longer scan local filesystem paths or provide local YARA-rule directories. Use a remote repository or URL for HTTP requests; use trusted stdio transport for local scans.

## Deprecations

- None.

## Validation

- Internal GitLab merge-request CI passed lint, unit, integration, Docker smoke, and Sonar analysis for the six imported public changes.
- `uv run --locked --extra dev pytest -q tests/unit/test_mcp_server.py` — 26 passed for the HTTP MCP transport remediation.

## Known Limitations

- HTTP MCP transport intentionally rejects local filesystem inputs; this is a security boundary rather than an unsupported scanner capability.

## References

- `CHANGELOG.md`
