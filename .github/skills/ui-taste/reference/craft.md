# Uizze craft checks

Adapted from Impeccable's `reference/craft-floor.md`, not its deprecated craft command alias. Apply after choosing the direction, not as a substitute for one. The project's established system and explicit user choices take priority over general aesthetic preferences. Use the host's available inspection tools; this playbook requires no hook or design engine.

## Establish the visual target

Apply the [Uizze image-first → 1:1 policy](../references/uizze-image-to-ui.md) before visual implementation, not as a final cosmetic step. Inspect and reuse a supplied/approved target. For new visual work or an approved redesign, use an available authorized image tool to make a realistic, straight-on UI image before coding; inspect real copy, controls, assets, legibility and platform constraints, then save its actual dimensions and keep it stable. Do not create a provider or regenerate an approved image.

For scoped refinement, use the incumbent screenshot and approved changes as the constraint; generate a constrained proposal only when needed and available, not a new identity. Audits and small technical fixes need no new concept. If generation is unavailable, use a supplied target or disclose the limitation and follow existing conventions; no real target means no 1:1 claim. Craft rules do not authorize aesthetic drift from the approved target or changes to product/data/auth contracts.

## Inspect the built interface

Check these together while exercising the actual screen. A rule written in a plan is not evidence that the render passes it; reuse one batched inspection rather than taking separate screenshots for every item.

- **Hierarchy and coverage:** the primary task or message reads first, supporting information is subordinate, and every brief requirement is present and findable within seconds. Use scale, weight, alignment and grouping before adding containers.
- **Target fidelity:** compare the actual render at the saved target's dimensions and state, including native configuration when relevant. Inspect layout/proportions, line wrapping, typography, spacing, color, icons, borders and image crops against the image, not memory. Report missing fonts/assets and explicit safety/accessibility departures; a visual target never licenses an asset or permits flattened screenshot UI.
- **Typography:** use coherent scale and obvious weight steps, balanced headings, readable line height and real copy at each breakpoint. A web reading measure around 65–75ch is a useful starting point, not a mobile minimum. For unconstrained web display text, about 6rem and tracking no tighter than -0.04em are floor-check guardrails, not ceilings on an approved composition or native Dynamic Type. Inspect computed values and the render for clipped ascenders, awkward wraps, crowded tracking and truncated essential labels.
- **Type character:** a system/workhorse face is appropriate for operational and native UI. For an approved expressive display direction, do not silently replace its lettering with the nearest installed heavy font and call it matched. Use an available, appropriately licensed face and report substitutions or loading gaps.
- **Spacing:** related controls form tight groups; sections have generous separation and normally more space above a heading than below. Optical alignment should agree with the grid. Inspect computed spacing or native layout metrics rather than trusting source declarations.
- **Color and contrast:** body and placeholder text normally need 4.5:1 contrast, large text 3:1. Tint secondary text in harmony with a colored ground or foreground rather than automatically adding neutral gray; measure the resulting contrast. Keep semantic roles, readable error/disabled states and visible focus. Never use color alone to communicate meaning.
- **Depth and surfaces:** borders, radii and depth form one system. Shadows used for elevation normally have an offset and soft blur; a zero-offset colored halo is decoration, not depth. A hard block shadow belongs to a world that deliberately chose it. Check seams, crop/aspect ratio, asset sharpness and duplicate device/status chrome.
- **Controls and states:** use recognizable controls, action-specific labels, sensible focus order, keyboard operation and usable touch targets. Exercise relevant hover, focus, disabled, loading, empty, error, success and recovery states. Loading explains progress; errors name the problem and recovery. Do not add impossible states just to fill a checklist.
- **Layout:** inspect the requested width, a narrow/mobile width and the principal desktop/native device classes. Long names, real copy, empty collections, large text, keyboard insets and overflowing menus must not break the task.
- **Motion:** author one coherent moment or grammar, not scattered effects or the same entrance on every section. Start from visible content, honor reduced motion, and test interruption/reversal. A quick ease-out can fit responsive feedback; choose easing for the actual direction. Blur, masks, clipping or shadows may participate only when smooth and accessible, not as an obligation to animate expensive properties.
- **Browser details:** on web surfaces, inspect selection, caret, focus rings, underline offset, scrollbars and tabular numerals as part of the palette. Theme what benefits the system without losing contrast, forced-colors support, native scroll affordances or platform usability. Use tabular numerals for comparisons; keep visible focus. On native surfaces use the platform's corresponding controls, text behavior and accessibility conventions rather than imitating browser CSS.
- **Content:** use the product's language and credible data. Controls name their action; errors name the problem and recovery. Label fixtures as illustrative. Never invent customers, testimonials, capabilities or performance results.

## Detect habitual design

These are signals to examine, not arbitrary bans on a user-approved aesthetic. When the direction leaves a choice open, replace a habit that adds no information rather than disguising it with another skin.

- Equal icon-title-text cards as the whole page, nested panels without distinct grouping needs, and the big-number/small-label/supporting-metrics hero used instead of product substance.
- Eyebrows that merely repeat a heading and numbered sections that communicate no meaningful sequence.
- A modal when the task needs neither interruption nor protected focus.
- Gradient headlines, glass/blur decoration, thick colored side borders and hard offset shadows imported into an unrelated visual world.
- Sparklines, progress rings, soft rounded rectangles or empty charts standing in for real content.
- Monospace as a technical costume rather than for code, data or measurement; an automatic display-font association instead of an intentional choice.
- Mixed Unicode/emoji stand-ins for a coherent icon system. Use the project's actual library or appropriately authored icons with consistent weight and stroke unless the brief deliberately calls for emoji.
- Geometric cutouts pretending to follow an organic photographic contour. Use a real alpha matte/cutout asset or omit the effect; inspect fine edges against light and dark grounds.
- Light or dark selected from the product category rather than the user's real setting and ambient light.

Ask what the screen actually communicates and enables. “It compiles” and “nothing overlaps” do not establish visual quality. The mechanical floor never selects the direction: compare the rendered hierarchy, detail and commitment with the user's target, not only a defect checklist. Do not add effects to manufacture novelty.

## Close the loop

Capture after fonts/images load and motion settles, at the target's actual dimensions and intended screen/state/scroll position. Compare full context and detail crops side by side or with an overlay. Fix layout/proportions before type, spacing, colors and assets; batch material corrections, recapture affected sizes/states and inspect again. For refinement, check unchanged regions against the incumbent baseline. Check actual controls as well as screenshots. Run relevant project checks when the task calls for them. Keep fix-and-reinspect passes bounded and stop an iteration that resolves nothing; report unresolved material findings rather than self-certifying.

Report the target, its actual dimensions, capture environment, comparison method and scope actually inspected, plus checks exercised and remaining mismatches. A 1:1 claim requires a real target and inspected same-size render with no unresolved visible mismatch in that scope; code review or a successful build is not enough. An unavailable browser, failed service call, skipped device, missing reference or unresolved finding is not a pass; if rendering/comparison is unavailable, visual matching remains unverified. Optional paid references/materials follow the [Uizze reference policy](../references/uizze-reference-policy.md); local craft work does not depend on access.
