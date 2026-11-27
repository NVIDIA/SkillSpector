# Native iOS interfaces

Follow the product's existing iOS conventions first, then Apple platform conventions. A native interface should feel predictable before it feels distinctive.

## Choose a native visual target

Follow the [Uizze image-first → 1:1 policy](../references/uizze-image-to-ui.md). Read the real task, existing screens/components, assets and data/auth contracts before choosing a direction. Reuse and inspect a supplied/approved image instead of regenerating it or reopening settled direction. For new visual work or an approved redesign, use an available authorized image tool before writing UI code to make a straight-on, realistic native screen with real copy, controls and the product's visual language—not a tilted device mockup or web UI in an iPhone frame.

Inspect and correct unreadable labels, false capabilities, impossible controls and unsafe layouts before implementation. Save the stable target and actual pixel dimensions, device class, orientation, scale, safe areas, content viewport and system chrome. Avoid duplicate status/navigation chrome. A scoped refinement can use an incumbent capture plus approved changes; generate a constrained proposal only when needed and available. Audits, backend work and small technical fixes need no new concept image. Without image generation, use a supplied image or disclose the limitation and follow existing conventions; do not install a provider or block on paid resources.

## Structure and controls

- Use a navigation stack for hierarchy, a tab bar for a small set of top-level destinations, and sheets for self-contained tasks.
- Preserve the edge-swipe back gesture and safe-area behavior.
- Prefer native controls, materials, menus, alerts, sheets, pickers, and text behavior unless the product already has a coherent custom system.
- Keep touch targets comfortable, text scalable, labels accessible, and focus order sensible.
- Account for the keyboard, long content, dynamic type, reduced motion, dark mode, and device insets.

## Implement the target with native behavior

At the recorded device configuration, match layout/proportions, typography, text wrapping, spacing, colors, icons and crops. Use real native components and working controls, never a flattened screenshot. Preserve real navigation, data/auth/backend contracts, gestures and accessibility. A picture does not specify unseen interactions or authorize new capabilities. Resolve conflicts with safe areas, Dynamic Type, contrast or usable targets explicitly rather than copying an unsafe target or silently redesigning it. Identify missing fonts/assets and rights limitations; do not substitute and claim exact fidelity. Then adapt to other requested device classes and states without changing the visual identity.

## States

Include the states the task requires: loading, empty, error, success, permission denied, offline, disabled, and recovery where applicable. Feedback should be immediate and should not depend on color alone.

## Render, compare and verify

When a simulator or runnable build is available, capture the intended screen/state at the saved target dimensions and recorded device/scale/chrome configuration after fonts/images load and motion settles. Compare side by side or with an overlay, including full context and detail crops. Fix layout/proportions first, then typography, spacing, colors and assets; recapture and compare after corrections. For refinement, also check unchanged areas against the incumbent baseline.

Inspect relevant device classes and exercise navigation, back gestures, keyboard, controls, Dynamic Type and Reduce Motion. Fix observable clipping, overlap, unsafe-area violations, unreadable text or inaccessible controls, and recheck affected states. A matched still does not verify native interaction or physical-device performance.

Report the target and dimensions, simulator/device configuration, comparison method, states and interactions actually checked, missing assets and remaining mismatches. Claim 1:1 only with a real target and inspected same-size native render with no unresolved visible mismatch in the claimed scope; a browser preview is not native evidence. If the environment cannot run or compare the UI, disclose that visual/device verification remains incomplete instead of requiring unavailable tooling or claiming a pass.

For an explicit Uizze Overdrive request, combine this guidance with [Overdrive](overdrive.md). Use native continuity and responsive feedback, not web-only effects or novelty that interferes with platform navigation. Respect Reduce Motion and preserve an equally usable non-animated path.
