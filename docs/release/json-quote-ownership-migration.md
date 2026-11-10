# JSON quote ownership upgrade

Status: unreleased change after the 2.12.0 baseline. The release version will be
assigned during release preparation; this note does not describe shipped 2.12.0 behavior.

JSON quote ownership supports complete candidates through 131,072 raw source
characters. Values above the former 65,536-character ceiling use iterative syntax
validation without decoded-object allocation. Otherwise supported valid examples
can therefore change from incomplete analysis with strict CLI exit 1 to complete
analysis with exit 0. All other analysis must still succeed.

The smaller decoder path, frontmatter boundary, analysis windows, and runtime
gates retain their existing constraints. Structural ownership does not exempt
commands or instructions inside strings from analysis. Unsupported command
parsing or instruction reconstruction remains explicitly incomplete.

Remaining size-limited uncertainty reports `json_quote_ownership_limit` with
source coordinates and the current ceiling. Consumers that enumerate reason
codes must recognize that reason and reject incomplete reports regardless of its
name. Do not add a duplicate generic reason merely to satisfy an old predicate.

Retain historical size-bound expectations and failures as versioned compatibility
evidence. A successor contract can require completion for the previously rejected
65,537-character benign case and move the rejection control to 131,073 characters.
This is an explicit supported-input migration, not a retrospective pass for an old
oracle. Existing expected-complete requirements must not be weakened.

See [JSON quote ownership bounds](../ANALYSIS_RESOURCE_BOUNDS.md#json-quote-ownership)
for counting rules, validation budgets, remaining limitations, and remediation.
