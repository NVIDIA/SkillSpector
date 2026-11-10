# Checking contributor onboarding with a coding agent

Use this procedure to check whether a coding agent can discover the contributor
guidance, set up a checkout, and perform the documented initial checks. Record
what actually ran, including failures. A documentation checklist alone does not
establish that the workflow works.

The [README setup](../../README.md#setup) remains the canonical installation
recipe. This procedure does not change the submission requirements in
[CONTRIBUTING.md](../../CONTRIBUTING.md#how-to-contribute).

## Prepare the observation

1. Pin the source commit. Start with a clean checkout, no existing virtual
   environment, and no project `.env` file. State whether the repository was
   provisioned locally or cloned by the agent over HTTPS.
2. Record the OS, architecture, shell, Python, package manager, Git, and Make
   versions. Identify the coding agent/client and model build when available;
   explicitly record any unavailable version rather than inventing one.
3. Run project commands with a fresh home directory and an explicit environment
   allowlist. Retain only the paths and settings needed for the chosen tools and
   temporary/cache directories. Do not inherit provider credentials, personal
   configuration, shell startup files, or an activated virtual environment.
   Record whether the package cache starts empty. Do not print secrets.
4. Define a time limit before starting. Give each check its own log and exit
   status so a failed or timed-out test does not hide independent lint and
   formatting results. Package installation needs registry access. Credential
   removal and pytest marker filtering do not establish network isolation.
5. Start an agent session without earlier project conversation. Provide the
   checkout location and execution/logging constraints, but let the agent find
   the guides and choose the setup commands.

Example task prompt:

> Discover the contributor setup guidance in this checkout, install development
> dependencies in a fresh supported Python virtual environment, and run the
> documented initial local checks. Report exactly what passed, failed, was
> blocked, or was not run. Explain formatting versus checking based on what you
> find. Do not modify tracked source, publish anything, or configure/use an LLM
> provider. Integration and live-provider tests are outside this bounded run.
> Do not replace the documented initial unit selection with a smoke subset.
> Record timeouts as incomplete. Diagnose failures without silently changing
> the checkout or claiming full PR compliance.

## Observe and record

Keep the original prompt and a transcript of the agent's guide discovery,
commands, relevant output, exit codes, and elapsed times. Record any operator
intervention or sandbox/network restriction separately from project failures.
For the current guide, check that the agent:

- Finds the canonical setup through the repository documentation and uses a
  supported Python version in a newly created, activated virtual environment.
- Installs development dependencies and records the resulting tool versions
  and any lockfile change.
- Attempts the complete `make test-unit` selection, `make lint`, and
  `make format-check`, with a separate outcome for each.
- Recognizes that `make format` edits files, while `make format-check` checks
  without applying changes. Capture final `git status`/diff to check this.
- Reports unrun integration/provider checks explicitly, without treating a
  skipped or timed-out check as passed.

If a check fails, retain the exact failing command and diagnostic. A focused
diagnostic rerun is additional evidence; it does not replace the original
selection. Attribute missing tools, unavailable registries, collection errors,
test failures, and timeouts separately. Use synthetic configuration only when
needed for a recovery check, and record it. Paid provider inference is not
required for this observation.

Some unit tests need loopback HTTP servers and local multiprocessing sockets.
If a sandbox denies these operations, an authorized rerun in an appropriately
permitted environment can help attribute the failure. Keep the original full
target's outcome alongside any focused control; do not generalize a subset pass.

## Publish the result

Use a dated report beside this procedure. Include:

- Source commit, environment/tool versions, initial state, agent identity,
  original task, and the observation's time limits.
- Guide-discovery path and exact setup/check commands, with exit codes, test
  counts, timings, and diagnostic excerpts. Label excerpts and any redactions.
- Passed, failed, blocked, and not-run checks; actionable defects and the
  evidence supporting their attribution; final tracked-file status.
- Limits: tested OS/Python/package-manager path, source provisioning, agent
  version uncertainty, network observation, and checks outside the run.

The initial checks are a bounded onboarding result. `make test` also includes
integration tests and remains the stated contributor requirement. If that
evidence is missing, record the submission limit and seek a maintainer decision
about who supplies validation; do not grant a unit-only exemption or assume
another pipeline will cover the PR.

Observed runs:

- [6 November 2026](contributor-onboarding-2026-11-06.md)
