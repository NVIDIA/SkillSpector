# Contributor onboarding observation: 6 November 2026

This report records one coding agent following the contributor setup on
[commit `3fb77b4cc9a97724481a6745157efcbfb20e2f32`](https://github.com/NVIDIA/SkillSpector/tree/3fb77b4cc9a97724481a6745157efcbfb20e2f32).
It evaluates that revision's initial setup/check path. It does not establish full
PR compliance or validate later code changes. See the
[repeatable procedure](CONTRIBUTOR_ONBOARDING.md).

The agent found the guides, installed development dependencies, and passed lint
and formatting checks. The full initial unit target **failed**: 20,909 passed,
4 failed, and 51 had setup errors. A focused control outside the execution
sandbox recovered 54 affected cases; one macOS ACL test still failed. The
onboarding observation is therefore complete with an unresolved test failure,
not a green initial-check result.

## Environment and task

| Item | Observed value |
|------|----------------|
| OS | macOS 26.7.2, build 25G332, arm64 |
| Shell | Bash 3.2.57(1)-release, `--noprofile --norc` |
| Python | CPython 3.12.13, fresh project `.venv` |
| Package manager | uv 0.10.9 (`f675560f3`, 2026-03-06) |
| Test/lint tools after install | pytest 9.1.1; Ruff 0.15.19 |
| Git / Make | Git 2.54.0 (Apple Git-157); GNU Make 3.81 |
| Agent | Codex coding subagent, started without the parent conversation; exact model/client build unavailable in this session |
| Source provisioning | Clean local `git clone --no-hardlinks --no-checkout` of fetched public source, then detached checkout of the commit above; HTTPS cloning was not tested |
| Initial state | No `.venv` or `.env`; empty temporary HOME, TMPDIR, and uv cache |
| Unit-selection bound | 900 seconds, declared before execution |

The observed agent's repository reads and commands went through a wrapper that
supplied an explicit subprocess environment instead of inheriting the caller's.
It started Bash with `--noprofile --norc`, captured the exact script, combined
stdout/stderr, exit status, and elapsed time, and terminated the process group
at the command's timeout. The observer independently checked those receipts.

The environment allowlist contained `HOME`, `PATH`, `TMPDIR`, `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `UV_PYTHON_DOWNLOADS=never`, `UV_LINK_MODE=copy`,
`PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`, `GIT_CONFIG_GLOBAL=/dev/null`,
`GIT_CONFIG_NOSYSTEM=1`, `NO_COLOR=1`, and `LC_ALL=en_US.UTF-8`. `PATH` exposed
the existing uv/Python tools and system executables; activation added `.venv/bin`.
No provider credential variables were passed, no `.env` was copied, and the
temporary home had no personal provider configuration. This did not isolate the
filesystem or monitor/block network traffic. Package downloads used the network.

The task given to the agent included this instruction (excerpt):

> Discover the contributor setup guidance, install development dependencies in a
> fresh supported Python virtual environment, run the documented initial local
> checks, and report exactly what passed, failed, was blocked, or was not run.
> Explain formatting versus checking based on what you find. Do not modify
> tracked source, publish anything, or configure/use any LLM provider; integration
> and provider tests are outside this bounded run. Do not replace the full
> documented initial unit selection with a smoke subset. If it exceeds a
> 900-second limit, record timeout rather than passing. A failure is a valid
> observed result; diagnose it briefly without silently changing the checkout or
> claiming full PR compliance.

The agent also received the provisioned checkout/revision, wrapper invocation,
logging constraints, and permission to retry package downloads outside the
restricted execution sandbox. It was not given setup commands or guide paths.
Independent lint/format/version observations were allowed while tests ran.

## Guide discovery

The agent enumerated repository files, read `README.md` and `CONTRIBUTING.md`,
then inspected `Makefile`, `pyproject.toml`, and `docs/DEVELOPMENT.md`. It found
the canonical [README Setup section](https://github.com/NVIDIA/SkillSpector/blob/3fb77b4cc9a97724481a6745157efcbfb20e2f32/README.md#setup),
the links to it from the other two guides, the supported Python range
`>=3.12,<3.16`, and all three intended initial checks. Its first `rg` discovery
command was unavailable on the wrapper's restricted PATH; it recovered with
`find`/`grep`. That was a harness-tool limitation, not a repository prerequisite
failure.

Discovery command excerpts from receipts 002 and 003:

```bash
cat README.md
if [ -f CONTRIBUTING.md ]; then cat CONTRIBUTING.md; fi
nl -ba Makefile
nl -ba pyproject.toml
sed -n '1,150p' docs/DEVELOPMENT.md
```

Reading the Makefile established that `make format` runs `ruff check --fix` and
`ruff format` in place. The agent chose `make format-check`; it did not run the
modifying target. The agent also identified the separate `make test` submission
requirement and its integration/provider prerequisites.

## Check results

Times below are wrapper wall-clock times. Each check ran from the checkout root
in a separate shell; the virtual environment was activated again for each one.
Guide dumps and progress polling are omitted from this transcript. Output shown
below is excerpted; temporary paths and the ACL username are normalized.

| Receipt | Command/step | Result | Seconds |
|---------|--------------|--------|---------|
| 004 | Create `.venv`, activate, `make install-dev` | Venv created; install failed (exit 2): sandbox DNS lookup for `files.pythonhosted.org` | 4.312 |
| 005 | Activate, retry `make install-dev` with permitted package-network access | Passed (exit 0); 158 packages installed; initial cache was empty before the first attempt | 13.384 |
| 006 | Activate, `make test-unit` | Failed (exit 2); no timeout | 588.299 |
| 007 | Activate, `make lint` | Passed (exit 0), `All checks passed!` | 0.624 |
| 008 | Activate, `make format-check` | Passed (exit 0), `365 files already formatted` | 0.125 |
| 020 | Focused 55-case control outside the execution sandbox | 54 passed, 1 failed (exit 1); no timeout | 156.917 |
| 023 | Direct stdlib ACL-write probe outside the execution sandbox | Write-only open succeeded; no SkillSpector call | 0.040 |
| 024 | Final version, Git status/diff, `.env`/`.venv` checks | Passed (exit 0); tracked checkout unchanged, including `uv.lock`; `.env` absent | 0.294 |

The install retry changed the execution permission context, not the recipe or
source. Lint and formatting ran independently while the unit selection ran.

Exact setup and initial-check scripts:

Receipt 004:

```bash
set -e
uv venv --python 3.12 .venv
source .venv/bin/activate
make install-dev
```

Receipt 005:

```bash
set -e
source .venv/bin/activate
make install-dev
```

Receipt 006:

```bash
source .venv/bin/activate
make test-unit
```

Receipt 007:

```bash
source .venv/bin/activate
make lint
```

Receipt 008:

```bash
source .venv/bin/activate
make format-check
```

The full unit command expanded to:

```text
pytest -m "not integration and not provider" tests/
collected 21267 items / 258 deselected / 21009 selected
4 failed, 20909 passed, 41 skipped, 258 deselected, 4 xfailed,
1060 warnings, 51 errors in 582.68s (0:09:42)
make: *** [test-unit] Error 1
```

The warnings included unregistered `pytest.mark.timeout` markers. This run's
900-second outer limit came from the observation wrapper; the markers alone
are not evidence that individual test timeouts were active.

## Failure attribution and control

All 51 setup errors in `tests/unit/test_tp4_openai_http.py` failed while binding
the fixture's HTTP server to `127.0.0.1`, with
`PermissionError: [Errno 1] Operation not permitted`. These fixtures use a local
HTTP stub and synthetic provider configuration. They are not live-provider
acceptance tests.

Three failures in `tests/test_batch_scan_security.py` had the same denied local
socket bind in the multiprocessing manager, followed by `EOFError` or a missing
manager-start assertion:

- `test_worker_timeout_kills_scan_and_releases_pool_capacity`
- `test_process_pool_shares_slots_and_cancels_waiting_owner`
- `test_api_pool_manager_stops_when_supervisor_dies`

The observer requested one focused control with the same source, virtual
environment, allowlisted environment, and a 450-second bound, changing only the
execution sandbox permission context. Exact script:

```bash
source .venv/bin/activate
pytest -q tests/test_batch_scan_security.py::test_worker_timeout_kills_scan_and_releases_pool_capacity tests/test_batch_scan_security.py::test_process_pool_shares_slots_and_cancels_waiting_owner tests/test_batch_scan_security.py::test_api_pool_manager_stops_when_supervisor_dies tests/unit/test_suppression.py::test_dump_baseline_accepts_acl_write_grant_without_mode_write_bits tests/unit/test_tp4_openai_http.py
```

Output excerpt:

```text
FAILED tests/unit/test_suppression.py::test_dump_baseline_accepts_acl_write_grant_without_mode_write_bits
1 failed, 54 passed in 155.75s (0:02:35)
```

All three multiprocessing cases and all 51 loopback HTTP cases recovered. This
supports attributing those original outcomes to execution-sandbox restrictions.
It does not turn the original full unit target into a pass; the complete target
was not rerun outside the sandbox.

The remaining macOS case is reproducible in both runs:
`tests/unit/test_suppression.py::test_dump_baseline_accepts_acl_write_grant_without_mode_write_bits`.
It creates a baseline with mode `0444`, adds a user ACL write grant, then calls
`dump_baseline`. The call fails at
[`suppression.py:801`](https://github.com/NVIDIA/SkillSpector/blob/3fb77b4cc9a97724481a6745157efcbfb20e2f32/src/skillspector/suppression.py#L801):

```text
descriptor = os.open(absolute.name, flags, dir_fd=directory_fd)
PermissionError: [Errno 13] Permission denied: 'baseline.yaml'
```

A separate temporary-file control applied the same mode and ACL grant, then
called `os.open(output, os.O_WRONLY)` directly without SkillSpector. It succeeded:

```text
0: user:CURRENT_USER allow write
Direct stdlib os.open(O_WRONLY): SUCCESS
```

The host can honor this ACL grant for a direct write-only open. The production
path additionally uses `O_NONBLOCK`, `O_NOFOLLOW`, and a directory descriptor;
those differences and the test fixture were not isolated further. Investigate
the macOS baseline-write/test path before claiming the failure is a general host
ACL limitation or a confirmed product root cause. No runtime fix was attempted
in this observation.

## Remaining limits

- **Passed:** guide discovery, fresh uv development installation after the
  network retry, lint, non-modifying format check, and final tracked-file checks.
- **Failed:** the original full unit target and the focused control's one
  persistent macOS ACL case. No check hit its declared timeout.
- **Not run:** `make test`, integration/live-provider targets, coverage, Docker,
  Node extension checks, other CI jobs, the pip fallback, other OS/Python paths,
  and an HTTPS clone. No full submission or CI pass is claimed.
- **Submission limit:** `CONTRIBUTING.md` still requires `make test`, including
  integration tests. This observation supplies no integration evidence and
  establishes no approved replacement/handoff policy. A maintainer must resolve
  missing validation; the initial checks do not authorize a unit-only exemption.
- **Reproducibility limit:** the source and tool versions above are recorded,
  but the exact agent/client build was unavailable. This is one observed task,
  not a model-pinned benchmark or proof of general agent success. No network
  capture was performed; marker exclusions do not prove zero network access.
- **Revision limit:** this observation preceded this PR's sandbox troubleshooting
  note and validation documents. Rerun it after setup, command, dependency, or
  contribution-workflow changes. Broader contribution-type discovery was outside
  this task, and historical release checks were not reused as candidate evidence.
