# Audit a native iOS interface

Run a read-only technical audit of a native iOS/iPadOS app, including SwiftUI, UIKit, React Native, Expo, or Flutter shipping to Apple devices. Review source and the actual native experience; do not fix files, change the design direction, or modify product behavior unless the user also requested fixes. Preserve the project's approved identity and scope.

This playbook is self-contained. Use the project's conventions and Apple's canonical [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/), including [accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility). Platform interaction and accessibility matter more than imposing a particular visual style. Apply the target platform's official guidance for any non-Apple native surface; do not blindly apply iOS units or navigation rules to it.

## Establish native coverage

Identify the screens, primary task, framework, supported OS versions, device classes, orientations, window modes, appearances, roles, and relevant states. Read the corresponding source and established components. Do not invent requirements for an unsupported device, theme, orientation, permission, or service.

Use the host's available native simulator/device controls, native accessibility inspection, screenshots, runtime logs, and existing project checks. No detector, launcher, new dependency, shell permission, download, or background engine is required. Use safe navigation and authorized test data; do not submit consequential actions merely to complete an audit.

Record the build/revision, OS, device or simulator identity, orientation, window size, appearance, text size, screen/state, and inspection method. Identify the exact target when several devices or windows are running. Screenshots must come from the actual native app on a simulator or device; a browser rendering or responsive preview is not native-device evidence.

A browser DOM scan does not cover a native interface. An empty or unavailable DOM result, failed native launch, inaccessible window, empty accessibility tree, or missing screenshot is **not a clean audit**. State what was inspected and what remains unknown. Separate source-confirmed defects, reproduced native defects, suspected risks, and checks that could not run. An accessibility-tree inspection is not a completed VoiceOver session, and a screenshot cannot prove navigation, focus order, keyboard handling, or touch gestures.

## Inspect actual surfaces and states

Walk the supported task through arrival, navigation, input, save/cancel, dismissal, and recovery with the available native tools. Inspect applicable loading, empty, error, success, disabled, offline, permission-denied, and long/missing-content states. Do not invent unreachable states or claim unavailable service paths were exercised.

Check scroll regions, lists, navigation bars, tab bars, sheets, alerts, menus, and keyboards in the states users encounter. Distinguish intentional disclosure, truncation with a usable full-content path, and off-screen scrollable content from required information that is clipped, occluded, or unreachable. A view can exist in source or the accessibility tree while remaining invisible behind another surface. Conversely, hidden/inactive views must not leak into focus or the reading order.

For contrast, use the foreground and actual composited background, including semantic colors, opacity, system materials, translucency, overlays, gradients, and images. Check relevant light/dark and increased-contrast appearances. Never assume white, compare token names as if they were pixels, or report a ratio without a reliable background. Flag unresolved measurement as requiring manual assessment, not passing. Assess content when it is actually visible, including opened sheets and keyboard-present states.

## Five diagnostic dimensions

### 1. Accessibility

- Inspect accessible labels, traits/roles, values, selected/disabled/expanded states, grouping, and meaningful image descriptions. Confirm that icon-only controls have useful names and that changed or error states are communicated without color alone.
- Exercise VoiceOver reading and activation where available. Check order, unreachable or duplicate controls, modal containment, focus after push/pop and dismissal, and announcements of meaningful state changes. If only static labels or a native tree were inspected, say so; do not imply screen-reader behavior was tested.
- Check Dynamic Type, including a large accessibility size: labels, values, lists, buttons, sheets, errors, and controls must remain readable and reachable without overlap or essential truncation. A fixed text size is a risk to investigate, not automatic proof of a defect when scaling is otherwise supported.
- Check at least 44 by 44 point tappable areas on iOS and separation between adjacent targets. Inspect the actual hit region, not just the visible icon. Do not confuse points with screenshot pixels or CSS pixels.
- Check text/control contrast and focus visibility against the real rendered surface and relevant accessibility standard. Honor Reduce Motion with a usable alternative preserving state feedback; also inspect increased-contrast and reduced-transparency behavior where supported.
- Exercise external-keyboard or switch/focus navigation when applicable and available. Mark it untested when only touch or pointer input was used.

### 2. Performance

- Observe startup to usable content, scrolling of representative long lists, image loading, keyboard presentation, navigation, and gesture responsiveness. Use runtime/profiling evidence for frame stability, stalls, or memory claims.
- Inspect source for heavy launch work, synchronous main-thread work in scroll/gesture paths, unbounded list rendering, unstable row identity, unnecessary re-renders, oversized image decoding, absent appropriate caching, and unused dependencies.
- Tie recommendations to demonstrated work and the framework's actual behavior. Do not prescribe memoization or replace a list merely because its name differs from a preferred API.
- Distinguish source-level performance risks from measured failures. A simulator can provide layout and interaction coverage, but does not certify physical-device startup, thermal behavior, 60/120 Hz frame performance, or real touch latency.

### 3. Appearance and theming

- Check semantic system colors or the project's adaptive tokens, state meanings, images, and supported appearance changes. Verify actual text, controls, bars, overlays, materials, and focus, not just token definitions.
- Look for stale theme values, crude inversion, unreadable dark content, and translucency that loses contrast over the content beneath it.
- Evaluate custom materials, icons, typography, and components against the established product system and platform usability. A user-approved brand face or icon set is not automatically a defect; report concrete legibility, consistency, scaling, or interaction failures instead.

### 4. Platform navigation and interaction

- Check that top-level destinations, hierarchy, and focused subtasks have predictable navigation. Tabs represent destinations rather than surprise actions; sheets and alerts have clear completion, cancel, and dismissal behavior.
- Exercise back navigation and the expected edge-back gesture on hierarchical screens, including nested scroll/drag content. Inspect for overlays or gesture recognizers that intercept it. Check focus and state restoration after returning.
- Check safe-area and content insets around the notch/Dynamic Island, status/navigation bars, home indicator, rounded corners, and presented surfaces. Decorative content may extend edge to edge; essential text and interactive targets must stay usable.
- Check standard control behavior, selections, menus, pickers, text editing, and feedback. Flag hover-only or website-shaped affordances only when they impair native use or conflict with binding project/platform conventions, not merely because they are custom.
- Check dismissal and data-preservation behavior without committing destructive actions. Identify unintended loss, blocked recovery, or misleading completion state from authorized evidence.

### 5. Device adaptivity and keyboard

- Inspect representative shipped iPhone sizes and iPad when supported, supported orientations, long content, and large text. Check iPad resizable windows/multitasking where supported rather than assuming a full-screen screenshot covers them.
- Ensure larger screens use the product's intended adaptive structure, with usable navigation, sheets/popovers, and controls; a different layout is not required just to satisfy taste.
- Focus inputs and inspect keyboard presentation, inset adjustment, scrolling to the active field, error visibility, and access to submit/cancel/dismiss controls. Check field-to-field traversal, appropriate input/return behavior, and keyboard dismissal without losing work.
- Reinspect after rotation/window resizing, keyboard show/hide, navigation, and interrupted gestures where applicable. A keyboard-hidden screenshot does not establish keyboard coverage, and simulator taps do not prove physical-device gestures.

## Report evidence, impact, and limits

Lead with blocking findings and a scoped platform-conformance assessment: does the inspected path behave predictably for a native user, and what concrete violations were observed? Do not pass/fail a whole app from one screen or make branding conformity a proxy for quality.

Include:

1. **Coverage by dimension:** source and native runtime methods, devices/OS versions, states, orientations/windows, input methods, accessibility settings, completed checks, partial checks, and unavailable checks.
2. **Prioritized findings:** **P0** blocks the task or risks data loss; **P1** materially impairs use or confirms a relevant accessibility/platform failure; **P2** causes friction with a workaround; **P3** is minor polish. Avoid preference-only noise.
3. **Each finding:** category; exact screen/state and source location when known; reproducible evidence and whether it came from source, simulator, hardware, or accessibility tools; user impact; verified guideline when applicable; smallest concrete correction. Keep suspected risks distinct from confirmed defects.
4. **Patterns and positives:** recurring root causes and affected screens, deliberate exceptions, and working native/accessibility practices to retain.
5. **Next actions:** specific corrections in priority order, then refinement and a native re-audit after authorized fixes. Use ordinary host-native steps, not invented native commands or detector results.

Only if the user requests a health score, use 0–4 for each dimension: 0 unusable/systemic failure, 1 major gaps, 2 partial with significant issues, 3 isolated minor issues, 4 no material defects in the inspected scope with sufficient evidence. Mark insufficiently assessed dimensions **not assessed**, not zero or four. Do not produce a total out of 20 if any dimension lacks coverage. A score is a scoped triage aid, not accessibility certification or proof that untested devices pass.

If nothing material is found, say “no material findings in the inspected scope,” list that scope, and preserve all verification limits. Never treat missing native tools, an empty DOM scan, or a source-only review as a clean native audit. Auditing remains read-only unless fixes were explicitly requested.
