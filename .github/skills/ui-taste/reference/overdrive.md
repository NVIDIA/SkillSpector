# Uizze Overdrive

Adapted from Impeccable's Overdrive, pinned and credited in NOTICE and MODIFICATIONS.md. Use only for an explicit Overdrive or ambitious interaction request. This is a design workflow using the host's existing tools, not a command to download an engine.

## Choose the result before the technique

Make an existing experience unusually good in a way its users will notice. Context decides what that means. A cinematic reveal may suit a portfolio; a settings screen benefits more from clear, immediate state transitions and reliable feedback. Fix hierarchy, readability and usability before adding technical ambition.

Before implementing, compare a few concrete directions internally by visible result, product fit, platform/browser support, performance cost and implementation complexity. Choose one within the requested scope and proceed. Honor an already supplied direction. Present options for a user choice only when the user explicitly requests them or reserves the decision; ordinary Overdrive work does not require an extra approval round.

Keep the chosen enhancement focused. One well-executed interaction is usually stronger than several competing effects. Do not replace an established identity or expand product functionality under the label of polish.

## Freeze the visual result before UI code

Apply the [Uizze image-first → 1:1 policy](../references/uizze-image-to-ui.md) to the chosen visual enhancement. Inspect and reuse a supplied/approved target; do not regenerate it or reopen its direction. For new visual work or an approved redesign, generate a realistic, straight-on UI target before implementation when the host has an available authorized image tool. For a scoped enhancement, retain an incumbent capture and constrain the proposal to approved changes, rather than replacing the identity.

Include real copy, controls, known fonts/assets, task and viewport; inspect legibility, truthful capabilities, feasibility, accessibility and native safe areas. Correct target errors before building, save the actual dimensions and keep the chosen target stable. For native surfaces record device, orientation, scale and chrome. Identify representative key states for a transition; a target still defines composition, not timing, unseen behavior or backend capabilities.

No new concept image is required for a read-only audit, small technical fix, backend work or a purely measured performance change. If generation is unavailable, use supplied images or incumbent captures and existing conventions, disclose the limitation and continue within scope. Do not initialize a provider or fabricate an image, resource or 1:1 result.

## Match ambition to the surface

- **Marketing or expressive work:** a distinctive composition, product-led reveal, or shared-element transition can make the subject tangible. The offer and primary action must remain immediately understandable.
- **Functional UI:** make a dialog maintain continuity with its trigger, filters update without losing focus, or selection/reordering feel direct. Preserve predictable controls, loading, rollback and error feedback.
- **Data-heavy work:** fluid transitions can explain changes; virtualisation can make genuinely large collections usable. Keep labels, keyboard navigation and an accessible representation. Do not render fake data to suggest a capability.
- **Performance-critical work:** move demonstrably expensive computation off the main thread, reduce unnecessary work, or load only what is needed. Measure before claiming an improvement.
- **Native iOS:** use the project's native animation/navigation mechanisms. Preserve back gestures, safe areas, keyboard behavior, Dynamic Type and Reduce Motion. Web APIs are not a native implementation plan.

## Choose the smallest suitable implementation

Prefer the project's existing stack and native platform capabilities. Verify current support for the actual target browsers/devices instead of relying on this document for a compatibility matrix.

- **Continuity and choreography:** CSS transitions, the Web Animations API or the existing motion library handle most UI changes. Use cancellable/reversible sequences; a spring can explain physical continuity, but its mass, damping and settling time must suit the task. Feature-detected `@starting-style` can support entry from hidden states without delaying access.
- **Navigation and scroll:** shared-element/View Transitions can connect a trigger to its dialog or a list item to detail. Scroll-driven animation can explain progress or storytelling; retain normal scrolling, a functional no-enhancement path and visible content when optional APIs fail. Never assume same-document and cross-document support are identical.
- **Custom rendering:** SVG suits geometry, diagrams and precise icons; filter chains can support deliberate distortion. Canvas/WebGL suits genuinely custom, dense or spatial rendering that DOM/CSS cannot express well, not a simple glow. WebGPU is appropriate only for a demonstrated need, with a suitable WebGL/Canvas or simpler functional fallback; unsupported hardware must not lose the task. Preserve semantic labels and an accessible representation outside custom pixels.
- **Data and responsiveness:** virtualize measured large-list costs while preserving focus, selection and keyboard navigation. Transition between data states without inventing history or implying interpolation is actual measurement. Workers suit expensive computation; OffscreenCanvas can move supported custom rendering off the main thread. WASM or a new dependency needs a concrete measured benefit and project approval, not novelty. Do not claim six-figure row counts or frame rates without exercising those workloads.
- **Complex properties:** typed custom properties through supported `@property` can animate values that otherwise do not interpolate. Use them, masks or filters only where the actual rendering cost and fallback are acceptable; syntactic animation support is not proof of smoothness.
- Device sensors, sound and personal data are separate permission decisions. No autoplay sound, surprise permission prompts or unrequested device access.

This changes how existing functionality feels, not what the product promises. Real-time collaboration, offline sync, new backend services, payments and authentication are separate product work. Optimistic feedback is appropriate only when the existing operation supports reliable failure recovery.

Use the host's available tools and the [Uizze reference policy](../references/uizze-reference-policy.md) for optional paid materials. Do not invent a resource, install an engine, initialize an image provider or introduce a service to make a proposed direction appear feasible. Missing preview, asset, device or API capabilities must be named before making claims about the result.

## Implement a resilient enhancement

The baseline must remain usable without the effect, with reduced motion, on touch devices, and when assets or optional capabilities fail. Map the enhanced transition to a reduced-motion state change that preserves its meaning; do not merely remove feedback. Preserve semantics, focus, keyboard access and content order. On native platforms use the platform's Reduce Motion, navigation, rendering and lifecycle APIs, not web feature-detection snippets.

Match the target's composition at its original viewport and agreed key states: layout/proportions, typography, real text, spacing, palette, icons and crops. Keep the target stable instead of quietly redesigning while coding. Build semantic components and working controls, not flattened screenshot frames; use the real data/auth/backend contracts. Identify missing fonts/assets and resolve material departures explicitly. Native behavior, safe areas and accessibility constrain the target, not the other way around. Then adapt requested sizes without changing identity.

Use feature detection where required. Lazy-initialise heavy resources, pause off-screen or background rendering, and clean up observers, animation loops, listeners and GPU resources when the view unmounts. Avoid layout thrashing and unbounded particle/DOM growth. Do not add a dependency or external service without authority.

Target stable frame pacing appropriate to the device (commonly at least 60fps for web interaction); measure responsiveness and representative workloads rather than guessing from the development machine. Investigate sustained drops below roughly 50fps on a 60Hz target, delayed input, battery-heavy rendering and layout shifts, then reduce complexity. A number measured on one machine is not a cross-device guarantee. A simpler implementation that stays responsive is the correct outcome.

## Verify the actual experience

Use browser automation or the native simulator/device when available. Inspect the result in motion, not only its source or a single still frame:

1. Capture the baseline and agreed enhanced key states at the saved target's actual dimensions/configuration after fonts/images load. Compare full context and details side by side or with an overlay. Correct layout/proportions before type, spacing, colors and assets; recapture after corrections and check untouched areas against the incumbent baseline. Then exercise entry, exit, rapid repeated input, interruption, reversal, navigation away and re-entry. Inspect timing, easing, continuity, asset sharpness and typography together: still-image fidelity alone cannot verify the motion.
2. Check desktop/mobile or relevant native classes, keyboard/touch use, reduced motion, unavailable optional APIs and the fallback path. Use representative mid-range hardware when available; identify simulator-only or untested device coverage.
3. Exercise real operations, loading and failure recovery; optimistic polish must not conceal a failed save or change focus unexpectedly. Measure performance for any performance claim.
4. Batch material fixes and repeat the affected checks in a bounded confirmation pass. Continue only for named unresolved findings within scope; stop a no-progress loop and report the limitation. Do not call an unpreviewed effect polished.

Use four final questions: Does it fit the product? Is the experience worse when removed? Does it still work without the enhancement? Is it smooth on the devices actually tested?

Report the target and actual dimensions, captures/environment, comparison method, states/interactions and devices actually exercised, missing assets and remaining mismatches. Claim 1:1 visual fidelity only for a real target and inspected same-size render with no unresolved visible mismatch in the claimed key-state scope; do not imply that this proves temporal fidelity. If preview, comparison or device access is unavailable, keep the implementation bounded and identify exactly what remains unverified. Do not fabricate frame-rate results, user reactions, performance gains or a visual pass. Stop when the chosen result works; propose extra scope separately.
