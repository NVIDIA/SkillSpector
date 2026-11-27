# Audit an implemented web interface

Run a read-only technical quality audit of what can be observed in source and the rendered experience. Do not fix the product unless the user also asked for fixes. This is not permission to redesign, invent requirements, replace the incumbent visual identity, or turn personal taste into a defect.

For native iOS/iPadOS, use [native audit](audit.native.md) instead. Browser DOM results do not establish native coverage. For a mixed app, report web and native surfaces separately.

## Establish scope and evidence

Identify the requested screens, primary task, supported browsers, viewports, themes, states, roles, and input methods from the project. Read relevant design guidance and representative components. Preserve the user's scope and established system; do not demand a theme, breakpoint, dependency, or feature the product does not support.

Use the host's available source inspection, running preview, browser accessibility tree, screenshots, interaction tools, and existing project checks. No separate detector, launcher, installation, or background service is required. Treat automated findings as candidates to verify in context, not proof of failure or quality.

Record the URL/build or revision, browser/engine, viewport, states, and input method actually inspected. Separate source-confirmed issues, reproduced runtime defects, suspected risks, and checks that could not run. An unavailable preview, failed scan, empty result, inaccessible frame, unloaded screen, or incomplete traversal means unknown or partial coverage, not a clean audit.

## Inspect actual visible surfaces

Walk the task, not just the initial viewport. Open applicable menus, popovers, dialogs, tabs, and disclosures; inspect scroll regions, sticky/fixed bars, and reachable loading, empty, error, disabled, success, permission, and recovery states.

- **Visibility:** inspect rendered bounds, clipping, overflow, overlap, stacking, and occlusion. A node's presence in the DOM or accessibility tree does not prove that a user can see or activate it. Check labels, validation messages, primary actions, focus rings, and overlay escape from clipping ancestors.
- **Hidden content:** distinguish intentional collapsed/inactive content, responsive alternatives, and off-screen scrollable content from accidentally unreachable information. Reveal a control in the state where it is meant to appear before evaluating it. Hidden content can still be an accessibility defect if it incorrectly remains in the focus or reading order.
- **Contrast:** resolve the actual foreground and effective background at the visible content. Account for inherited colors, opacity and compositing, translucent overlays, dark/light themes, disabled states, gradients, and image backgrounds. Do not assume a white background or infer a ratio from token names. For variable backgrounds, inspect the affected region and worst relevant state; if a reliable ratio cannot be established, request/record manual assessment rather than manufacture a passing number.
- **Reachability:** prove that content can be reached through the intended scroll, keyboard, pointer, or touch path. A screenshot cannot establish focus order or gesture behavior, and a DOM-only scan cannot establish a visible, usable control.

## Five diagnostic dimensions

### 1. Accessibility

- Check semantic headings and landmarks, native control semantics, accessible names, roles, states, meaningful image alternatives, form labels, instructions, errors, and recovery.
- Exercise keyboard navigation, visible and unobscured focus, logical order, dialogs' focus handling, and escape from traps. Inspect screen-reader names and state announcements with the tools actually available.
- Check text contrast against the real background: normally 4.5:1 for text and 3:1 for qualifying large text under WCAG AA. Check applicable control/state indicators and focus visibility requirements separately; do not call every low-contrast decorative or exempt element a failure.
- Check zoom/text enlargement, reflow, color-independent feedback, reduced-motion alternatives, and disruptive flashing. Reduced motion must preserve useful state feedback; an indiscriminate near-zero-duration override can remove essential information.
- Evaluate target size and separation using the applicable standard, including WCAG 2.2's minimum target-size rule and its exceptions. A 44 CSS-pixel target is a useful touch recommendation, not a blanket WCAG AA requirement.

### 2. Performance

- Inspect observed startup/loading delays, layout shift, image loading, interaction latency, scrolling, and animation frame stability. Distinguish measured behavior from source-level risks.
- Look for repeated layout reads/writes, unnecessary work on interaction paths, excessive blur/filter/shadow effects, broad permanent `will-change`, oversized assets, and unnecessary imports.
- Check image dimensions, appropriate responsive sources, and appropriate deferred loading; do not prescribe lazy loading for the critical above-the-fold image. Recommend rendering optimizations only for demonstrated unnecessary work, not memoization everywhere.
- State when no profiler, representative network/device, or runtime evidence was available. Static code alone does not prove smooth performance.

### 3. Theming and system consistency

- Check semantic token use and stable state meanings against the project's own conventions. A justified asset color or local value is not automatically drift.
- Exercise every supported appearance and theme switch; look for stale values, inverted assets, missing variants, and illegible text, controls, overlays, or focus.
- Verify actual rendered combinations, not just token definitions. Do not penalize an unsupported dark theme or demand a new token framework merely to improve an audit score.

### 4. Responsive layout and input

- Inspect relevant narrow, intermediate, and wide sizes, long content, enlarged text, structural reflow, fixed widths, horizontal overflow, and actions hidden by bars or overlays.
- Exercise custom sliders, drags, and scrollable control strips with touch when the host supports it. Check page-scroll cooperation and cleanup after cancel, lost capture, interruption, or blur. Mouse handlers or a mobile-sized screenshot do not prove touch works.
- Record whether evidence came from viewport emulation, synthesized touch (and which engine), or physical hardware, and which inputs remain untested.

### 5. Implementation integrity

- Verify repeated shortcuts, incomplete or misleading states, inconsistent shared components, and departures from the established product system. Tie each finding to an observed implementation and user impact.
- Check that controls and content serve the real task rather than acting as decorative stand-ins. Do not label a user-approved aesthetic, familiar product pattern, or deliberate exception a defect merely for being conventional.
- Group recurring root causes while retaining affected locations. Reject false positives; separate measurable failures from design judgment.

## Report evidence, impact, and limits

Lead with blocking findings and the implementation-integrity assessment, bounded to the inspected scope. Include:

1. **Coverage:** screens/states, environment, source/runtime methods, completed checks, partial checks, and unavailable checks for all five dimensions.
2. **Prioritized findings:** **P0** blocks task completion or risks data loss; **P1** materially impairs use or confirms a relevant accessibility failure; **P2** causes friction with a workaround; **P3** is minor polish. Do not flood the report with preference-only P3s.
3. **Each finding:** category; exact screen/state and file/line when known; reproducible evidence and method; user impact; applicable standard only when verified; smallest concrete correction. Label suspected risks separately from confirmed defects.
4. **Patterns and positives:** recurring root causes, affected locations, intentional exceptions, and practices worth preserving.
5. **Next actions:** specific corrections in priority order, with a final refinement and re-audit after authorized fixes. Recommend ordinary host-native actions or available local playbooks, not invented commands.

Only if the user requests a health score, use 0–4 per dimension: 0 unusable/systemic failure, 1 major gaps, 2 partial with significant issues, 3 isolated minor issues, 4 no material defects in the inspected scope with sufficient evidence. Mark insufficiently assessed dimensions **not assessed**, not zero or four; do not compute a total out of 20 when any dimension lacks coverage. A score is a scoped triage aid, not certification or proof that untested states pass.

If nothing material is found, say “no material findings in the inspected scope” and name that scope and remaining limits. Do not silently fix issues, claim complete WCAG conformance from a scan, or call unavailable/empty evidence clean.
