# Refine an existing interface

Polish improves the interface that exists; it does not conceal a redesign. Preserve the incumbent visual language, content, behavior, factual claims, and everything outside the user's scope. If the concept needs replacement, explain why and recommend a separate redesign rather than smuggling one into refinement.

## Establish the system and evidence

Read the project's design guidance, tokens, shared components, and neighboring flows. If no formal system exists, follow coherent project conventions. Identify the intended quality bar, shipping constraints, primary task, supported sizes, and deliberately unfinished work from the brief and implementation; ask only when a binding decision cannot be inferred.

Classify drift before fixing it:

- **Missing token:** a genuinely reusable value is absent from the system.
- **One-off implementation:** an existing shared component or pattern should be reused.
- **Conceptual mismatch:** flow, hierarchy, or information architecture differs from comparable product areas.
- **Local defect:** the implementation is incomplete or inconsistent.

Fix the cause at the narrowest correct level. Do not create a system abstraction for one local exception.

Run the interface and walk the real task with available host tools. Inspect mobile, intermediate, and wide web layouts, or the native device classes and supported orientations in [iOS](ios.md). Use real copy and representative long, missing, localized, slow, offline, or permission-limited content where those states can occur. A source review, automated scan, or attractive screenshot alone does not prove a working path.

If an existing critique is supplied, record the exact artifact and target/revision it concerns. Reproduce relevant high-impact findings against the current interface; a stale or different-target critique is context, not an active defect list. Perform an independent pass either way. Report which findings were actually resolved without claiming to update an unavailable critique store or closing a newer, unrelated review.

## Set a constrained visual target

Follow the [Uizze image-first → 1:1 policy](../references/uizze-image-to-ui.md) without turning refinement into a redesign. Inspect and reuse any supplied/approved target; do not replace it or reopen its direction. For meaningful visual changes without one, retain a valid incumbent screenshot with its actual dimensions, state and configuration, and identify exactly what may change. A constrained baseline plus approved changes can guide local refinement. If a new visual proposal is needed and the host has an authorized image tool, generate that constrained target before writing the changed UI code, using real copy, controls, assets and existing-system rules.

Inspect the proposed target for legibility, truthful capabilities, usable controls, accessibility and native safe areas before implementation. Save the chosen target's actual dimensions and keep it stable. Do not use this step to introduce a new provider, request a fresh approval ritual for settled changes, or demand concept images for audits, backend work, one-line fixes or small technical defects. When generation is unavailable, use the supplied target or incumbent baseline and conventions, disclose the limitation and continue; no invented target or 1:1 claim.

## Triage the whole path

Separate functional defects from cosmetic ones, then work in this order:

1. Broken or blocked tasks, data loss, misleading state, and inaccessible paths.
2. Missing loading, empty, error, success, disabled, permission, and recovery feedback where applicable.
3. Flow, hierarchy, responsive behavior, and design-system drift.
4. Visual, content, and motion inconsistencies.
5. Cleanup made necessary by this refinement.

Do not perfect one corner while leaving the rest of the requested path below the same quality bar. Leave working areas outside scope alone.

## Inspect and refine actual surfaces

- **Flow:** match neighboring terminology, navigation, disclosure, save behavior, and state conventions. Connect arrival, transition, empty, and recovery paths; make the primary task and current state obvious.
- **Layout and type:** align to the project's grid and spacing scale, including optical alignment. Keep same-role typography consistent. Check font loading, zoom, large text, localization expansion, wrapping, image aspect ratios, and layout shift.
- **Visible content:** inspect the surface users actually see, including scroll regions, sticky bars, menus, popovers, dialogs, and their open/closed states. Source presence is not visibility. Look for clipped labels, occluded actions, off-screen errors, unsafe overflow, and content hidden behind overlays. Distinguish intentional truncation or disclosure from required information that cannot be reached.
- **Contrast:** check text, controls, and focus on their actual rendered backgrounds in relevant themes and states. Resolve transparent layers, inherited colors, overlays, gradients, and image backgrounds rather than comparing token names or assuming white. Hidden content is not a visible contrast failure; reveal it in its real state before judging it. If the effective background cannot be established, record the uncertainty instead of inventing a ratio.
- **Icons and media:** preserve a coherent family, stroke/weight, sizing, and optical alignment. Keep meaningful image descriptions and stable image dimensions; do not replace approved assets just to make the pass look different.
- **Interaction:** exercise applicable default, hover, focus, active, selected, expanded, disabled, loading, error, and success states. Keep visible keyboard focus, logical order, accessible names, and platform-appropriate targets. Check touch gestures where relevant; a narrow viewport alone does not test touch.
- **Motion:** keep feedback coherent, interruptible, and responsive. Honor reduced motion with an equally usable path that preserves essential state changes; do not add choreography to advertise polish.
- **Content and implementation:** retain factual claims unless the user approves changes. Keep terminology, capitalization, and punctuation consistent. Reuse shared patterns and remove obsolete local code or duplication caused by the refinement, not unrelated code.

Implement the constrained target at its original viewport first, matching the approved layout, type, text, spacing, colors, icons and crops without changing untouched regions. Use semantic components and real working controls, not a flattened screenshot. Preserve data/auth/backend contracts and native behavior; report unavailable fonts/assets and resolve material target conflicts explicitly instead of silently substituting or sacrificing accessibility.

For technical defects, use [audit](audit.md) on web or [native audit](audit.native.md) for native iOS. Neither a clean scan nor an empty result replaces rendered and interaction inspection.

## Reinspect and finish

Make one coherent correction pass, then run and inspect the affected path again with the applicable mouse, keyboard, and touch inputs. Use one targeted follow-up pass for remaining observed defects rather than an open-ended cosmetic loop.

Recheck changed states and representative supported sizes, themes, zoom/text scaling, focus, semantics, and accessible names. Inspect console/runtime errors, image loading, layout shift, and interaction latency with the host tools actually available. For native work, use native simulator/device evidence; distinguish it from browser previews and physical-device performance checks.

After fonts/images load and motion settles, capture the same screen/state/scroll position at the target's actual dimensions (and recorded native device/scale/chrome where applicable). Compare side by side or with an overlay: check approved changes against the target/change list and untouched regions against the incumbent baseline. Correct layout/proportions before typography, spacing, color and crops; recapture and compare affected states after fixes. Invalid or half-loaded captures are not evidence, and later visual edits invalidate affected comparisons.

Report the changes, target/baseline and actual dimensions, capture environment and comparison method, paths/states actually exercised, resolved and remaining findings, missing assets and unavailable or partial checks. Claim 1:1 fidelity only to a real target in an inspected same-size render with no unresolved visible mismatch in the claimed scope; a baseline/change list alone does not prove a pixel-perfect new target. Without rendering/comparison, the visual match is unverified. Do not claim release readiness, accessibility conformance or a complete device matrix from source inspection alone. Unresolved blockers remain blockers; explain the exact missing prerequisite instead of hiding them behind “polished.”
