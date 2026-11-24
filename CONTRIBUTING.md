# Contributing to SkillSpector

We welcome contributions to SkillSpector! By contributing, you agree to abide
by the [Developer Certificate of Origin](#developer-certificate-of-origin).

> **Security vulnerabilities:** Report suspected vulnerabilities, including in
> SkillSpector itself, privately to NVIDIA PSIRT by following [SECURITY.md](SECURITY.md).
> Do not report them through GitHub issues, pull requests, or Discussions.

## How to Contribute

1. **Open an issue** describing the non-security bug, false positive, or feature you'd like to work on.
2. **Fork** the repository and create a branch for your change.
3. **Make your changes**, ensuring all tests pass (`make test`).
4. **Sign off** every commit (see below).
5. **Open a pull request** referencing the issue.

## Development Setup

Follow the [README development setup](README.md#setup) for supported Python
versions, cloning, virtual environment activation, and dependency installation.
The [Development guide](docs/DEVELOPMENT.md) covers architecture and extending
the analyzers.

Initial local checks:

```bash
make test-unit
make lint
make format-check
```

These initial checks do not replace the `make test` requirement above.
`make test` also runs integration tests, which may call a configured LLM provider
and send test content to it. Review the provider prerequisites in the setup guide
before running the full target. The unit-test marker selection does not cover
every CI job or guarantee network isolation.

### Antivirus alerts on test fixtures

The test corpus deliberately includes malicious-looking samples. In
[#1088](https://github.com/NVIDIA/SkillSpector/issues/1088), Norton 360 on macOS
was reported to quarantine `tests/unit/test_patterns.py` during a Git checkout,
with the detection `Python:Stealer-IG [Pws]`. The pattern tests pass these samples
to static analyzers as text; they do not execute them. Keeping samples in `.txt`
fixtures makes their role explicit, but does not guarantee that antivirus
software will stop flagging them.

The installed CLI wheel does not include the test corpus. However, a Git-based
installation still checks out the source before building, so it can trigger an
alert even when the installed CLI runs normally.

Check the specific alert, file path, and revision against the expected upstream
source before treating a detection as a false positive. Keep antivirus
protection enabled and submit the suspected false positive to the antivirus
vendor for review, including the detection name, file path, revision, and file
hash. An alert on another file or revision needs its own investigation.

## Publishing releases

A release PR must come from a branch in `NVIDIA/SkillSpector` and be merged
into `main` with the `release:publish` label. Update the version in
`pyproject.toml` and add its notes under `docs/release/` in the release PR.
The [release workflow](.github/workflows/release.yml) builds and checks the
wheel and source distribution from the merge commit, attaches both to the
GitHub release, and passes those same files to the package index job.

Before enabling uploads, configure these GitHub Actions environments with
`main` as the deployment branch and release maintainers as required reviewers:

| Index | GitHub environment | Repository variable `SKILLSPECTOR_PUBLISH_INDEX` |
| --- | --- | --- |
| TestPyPI | `pypi-test` | `testpypi` |
| PyPI | `pypi-production` | `pypi` |

Register a GitHub Trusted Publisher for the `skillspector` project in the
chosen index. In each publisher registration use owner `NVIDIA`, repository
`SkillSpector`, workflow filename `release.yml`, and the corresponding
environment. Use the existing project on PyPI and a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
on TestPyPI if it has no project yet. Package uploads remain disabled while
the repository variable is unset. Start with TestPyPI on an approved release
PR to verify the OIDC flow before selecting production for a subsequent
release.

If an upload is interrupted, rerun the failed index job while the build
artifact is available, with the same index selected. The publisher skips
files already present with identical bytes and uploads any missing files.
A published filename with different bytes fails the upload. Ship a new
version if the distributions need correction.

## Coding Standards

- Use `make format` to apply lint fixes and formatting in place before committing.
  Verify with `make lint` and `make format-check` before submitting.
- All new source files must include the SPDX license header (see any existing
  `.py` file for the template).
- New analyzers should include corresponding unit tests and, where applicable,
  test fixtures.

## Commit Sign-Off

All contributions must include a `Signed-off-by` line in the commit message,
certifying that you have the right to submit the work under the project's
open-source license. Use `git commit -s` to add this automatically:

```
feat(analyzer): add new detection rule for X

Signed-off-by: Your Name <your.email@example.com>
```

This sign-off certifies that you agree to the Developer Certificate of Origin
(DCO) below.

## Developer Certificate of Origin

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
