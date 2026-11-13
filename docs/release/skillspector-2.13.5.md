# SkillSpector 2.13.5

This maintenance release improves bounded parsing and inspection of large skills,
structured data, and supported generated artifacts. It preserves the provider
defaults, output-file protections, response recovery, and passive Office metadata
handling from 2.13.4.

## Fixes

- Improve source ownership and reconstruction across supported shell, Markdown,
  Python, JavaScript, JSON, and YAML inputs.
- Improve bounded coverage of large structured inputs and nested artifacts,
  with explicit diagnostics for unsupported syntax and exhausted limits.
- Preserve security findings across decoded views and generated shell artifacts.
- Improve report provenance, suppression, and inspection completeness accounting.
- Preserve structured-input output alias checks and bounded metadata proofs.
- Retain the canonical scanner command and provider selection behavior in the
  OpenCode and Pi integrations.

## Scope and upgrading

Parser and archive support remains bounded. Unavailable dependencies, unprovable
runtime behavior, unsupported content, exhausted budgets, and provider failures
continue to have explicit diagnostics. A warning or incomplete assessment is not
proof that a skill is safe.

Existing consumers must continue to handle complete_with_caveats and inspect
the coverage diagnostics. Updating the scanner does not change downstream
warning or gate policy. Review version-bound baselines when upgrading.

GitHub release assets contain a wheel and source distribution. This workflow
does not publish to PyPI.
