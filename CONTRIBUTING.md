# Contributing to SkillSpector

We welcome contributions to SkillSpector! By contributing, you agree to abide
by the [Developer Certificate of Origin](#developer-certificate-of-origin).

## How to Contribute

1. **Open an issue** describing the bug or feature you'd like to work on.
2. **Fork** the repository and create a branch for your change.
3. **Make your changes**, ensuring all tests pass (`make test`).
4. **Sign off** every commit (see below).
5. **Open a pull request** referencing the issue.

## Coding Standards

- Run `make lint` and `make format` before submitting.
- All new source files must include the SPDX license header (see any existing
  `.py` file for the template).
- New analyzers should include corresponding unit tests and, where applicable,
  test fixtures.

## Provider CLI Validation Expectations

A new or re-verified agent-CLI provider scans untrusted skills, so its
safety boundary must be enforcement, not documentation. Proposals that
only describe a gap without closing it will be sent back. Concretely:

- **Exact-version preflight before stdin, on every completion path.**
  Pin the verified CLI release and re-check it immediately before each
  inference call — not just in the availability probe, which direct
  `complete()` calls never touch. A synthetic future-version binary must
  be rejected before any prompt bytes move (test this).
- **No hook material, no hooks.** User/plugin lifecycle hooks usually
  have no argv off-switch. Where home isolation is usable, redirect the
  CLI's config, plugin, and hook directories to per-invocation temp
  dirs; where the CLI refuses to run isolated (probed and documented),
  refuse inference when hook-capable material such as a non-empty
  `installed-plugins/` tree is present instead. Either way, carry over
  the minimum auth material and never a whole home directory.
- **Adversarial fake-host tests proving zero side effects.** A fake
  binary asserting the exact argv posture plus a marker for any executed
  tool, hook, or weakened env var; empty marker directory or the test
  fails.
- **Synthetic-version gate tests.** A fake binary reporting an
  unverified version must fail closed, including stdin-never-delivered
  where the transport allows asserting it.
- **No silent fallbacks.** Unknown agents, unparseable versions, missing
  auth, and empty output all raise — never degrade to a weaker policy.

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
