<!--
DRAFT comment for GitHub issue #20 (NVIDIA/SkillSpector).
NOT posted automatically — review and post manually.
-->

Implemented whitespace-padding detection as a new rule **P9 "Whitespace Padding"** under the Prompt Injection category.

**Rule id:** `P9`. P6–P8 were already taken by the System Prompt Leakage patterns, so the next free `P`-series id is P9. A single combined id covers all three signals; per-signal confidence carries the weighting.

**Three signals (all reported as P9):**

1. **Vertical blank-line run** — fires at **>= 20** consecutive blank / whitespace-only lines. Severity **MEDIUM**, escalating to **HIGH** when non-blank content follows a gap of **>= 40** lines (the classic "instructions hidden below the fold" case). Confidence 0.8 when content follows the gap, 0.6 when the gap merely trails the file.
2. **Horizontal whitespace run** — fires at **>= 80** consecutive whitespace characters within a line (including leading indentation), regardless of what follows on the line. Severity **MEDIUM**, confidence 0.7.
3. **Oversized whitespace ratio** — a single contiguous whitespace block **> 2 KB**, or whitespace making up **> 90%** of a file that is **> 4 KB**. Severity **LOW**, confidence 0.4 — it informs rather than dominates the score.

**Whitespace classification:** Unicode-category based, not just ASCII. A character counts as padding if it is an ASCII control (`\t \n \r \v \f`), falls in Unicode categories `Zs`/`Zl`/`Zp` (covers U+00A0, U+2028, U+2029, U+3000, etc.), or is in the zero-width family (U+200B/U+200C/U+200D/U+2060/U+FEFF). The zero-width set is a **single shared definition** (`ZERO_WIDTH_CHARS`) reused by P2's hidden-instructions regex and by MCP `mcp_tool_poisoning`'s zero-width check, so the definitions cannot drift — this also added U+2060/U+FEFF coverage to the MCP check as a strict improvement.

**Reporting:** each finding points at the line where the padding starts and includes a visible-ized snippet of what was hidden, rendered as `U+XXXX xN` segments (e.g. `U+00A0 x82`, `\n x80`) so a reviewer can see the otherwise-invisible content.

**False-positive guards:** Markdown fenced-code regions are skipped for the horizontal signal; vendored/generated files (`*.min.js`, `*.min.css`, `*.lock`, `package-lock.json`, `yarn.lock`, `*.svg`, `*.map`) are skipped entirely; binary-ish content (containing U+FFFD) bails out; the ratio signal stays at LOW confidence. Eval-dataset prose and files > 1 MB are already skipped upstream.

**MCP coverage:** MCP manifest description fields are also covered — the same detector is wired into `mcp_tool_poisoning` for non-identifier description fields (horizontal + contiguous-block signals; the per-file ratio signal is skipped since fields are too short for the 4 KB floor to apply).

Thresholds are module-level named constants for easy tuning against a real-world corpus. Happy to align on the exact signals/thresholds before opening the PR.
