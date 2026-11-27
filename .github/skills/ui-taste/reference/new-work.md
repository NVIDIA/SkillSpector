# New interface or major redesign

Adapted by Uizze from the licensed Impeccable new-work workflow; see the archive's NOTICE and MODIFICATIONS.md.

Follow the [Uizze image-first → 1:1 policy](../references/uizze-image-to-ui.md) throughout this workflow. Use an existing approved image first; otherwise, for new visual work or an approved redesign, generate the UI target before writing UI code when the host has an authorized image-generation tool. This applies to operational and reading surfaces too, without overriding product truth, established identity or platform accessibility.

## Establish what must remain true

Read the brief, representative screens, components, tokens, assets, data contracts and existing product/design documentation. Product documentation owns facts and capabilities; design documentation records visual decisions. Neither a missing DESIGN.md nor an incomplete brief erases a coherent identity already in code.

Classify the visual authority before inventing a direction:

- **Preserve:** additions, features and states inside an established surface inherit its identity and composition. A whole new surface in that product inherits the visual system, but may need a different structure.
- **Expand:** an incomplete brand retains its confirmed assets and recognizable traits. Derive missing rules from the brief and existing identity; gaps do not authorize replacing the brand.
- **Replace:** an explicitly approved redesign can replace the old visual world. Preserve product truth, content, working behavior, constraints and explicit brand commitments; the old look is evidence, not a requirement to retain.
- **Create:** a product with no visual authority needs a chosen world, not an automatic dashboard or landing-page template.

Choose the surface's main purpose and keep expression within that purpose:

- **Persuade:** make the real offer understandable and desirable, show credible proof and expose the primary action in its usable form. Do not substitute a decorative link for the search, purchase or booking control the task requires.
- **Operate:** users should recognize the working shell, navigation, work areas, controls and state before noticing the aesthetic. A borrowed visual world contributes type, palette, density and one useful signature move, not a costume of a cockpit, terminal or instrument. State, selection and data colors must do real jobs.
- **Read:** start from the reader's question. Make location, contents and the way forward clear. Express identity in the masthead, navigation, ground, section openers, code and tables while keeping the reading column calm and comfortably measured. Do not turn a document into a poster or photographed book.
- **Experience:** the work itself leads in the first viewport. Exploration and transitions serve it; navigation and actions remain recognizable and accessible.

## Resolve the surface and the direction

Ask only what changes the result: audience, job, primary action, real content/proof, constraints, success, what must not change, and what would make a polished result feel wrong. Resolve material unknowns before implementation; do not re-ask settled facts, request CSS values, or make a precise request pass through a concept tournament.

**Local extension:** resolve the new content, hierarchy, interaction and states in context. Do not reopen identity or change durable design rules without approval.

**Whole surface in an established world:** compare materially different organizations of the real task and content internally, then select one that fits the brief. Resolve navigation, work/reading areas, density, what leads and where the primary action lives. A representative screenshot of the incumbent product is a stronger identity anchor than a prose paraphrase.

**New or replacement world:** name the product's unique mechanism, the audience's real setting, its cultural world and what this surface must prove. Identify both the category's default page and its predictable opposite; merely choosing the opposite is still a rut. Derive a broad set of concrete visual sources from that audience's artifacts, publications, places, graphic traditions or rituals. For an open brief, aim for seven genuinely different candidates across at least three material families; do not spend them all on the product name's literal metaphor. A working product need not look like the tools its audience operates.

Turn the strongest material into two or three complete directions. Each must describe its thesis, palette/materials/type, first viewport, visitor path, signature interaction, reach across other screens and an honest risk. Compare on audience identification and product clarity, not novelty alone. A challenger should improve the selected direction's discipline—such as its grid, density or palette commitment—not donate unrelated decorative motifs. One coherent world owns the result.

Select and implement the direction from the brief and inspected evidence within the authorized scope. Routine design work does not require a concept-selection round. If the user explicitly requests options or reserves a decision, present comparable images or sketches with the visible result and tradeoff in each option, then honor their choice. Supplied direction remains authoritative, and an existing identity changes only when redesign is authorized. A familiar category-standard direction is legitimate when it fits the task: execute it fully, without smuggling in unwanted quirks.

Every offered direction must be feasible with the real facts, assets, tools and performance budget. Neither dice, a catalog, an external decision page nor an Impeccable command is required. Paid references and materials are optional under the [Uizze reference policy](../references/uizze-reference-policy.md). Inspect references actually retrieved and explain the transferable structural lesson; do not copy another product's identity or imply access that failed.

Use the chosen direction to make the target, not another approval tournament. A supplied approved target or settled direction takes precedence. If visual options are needed and image generation is available, show realistic, straight-on UI images at the actual intended viewport with legible product copy and controls; inspect their feasibility before offering them. For an established-system extension, constrain the target to the incumbent identity and requested addition.

## Commit a buildable contract

Choose color by its role and the use scene, not by category: restrained tinted neutrals and an accent, a committed saturated regional color, a small palette with named roles, or a color-drenched surface may each fit. Explain who uses it, where and under what light before choosing dark or light. Strong color belongs at surface scale when the direction calls for it, not only in tiny accents.

Choose type for the approved world and reading task, not an automatic serif-for-books or mono-for-tech association. System/workhorse faces are appropriate for operational and native interfaces; expressive display work can justify a licensed distinctive face. Respect user-approved typography and brand aesthetics. Cream-and-serif, neon-on-black, editorial hairlines and hard-shadowed saturated layouts are not banned; when the brief leaves this axis open, check that the choice came from the product rather than habit.

Record a short development-only direction contract in the working plan or existing surface brief before building:

- **Thesis:** the idea this surface owns, and the habitual arrangement it avoids.
- **Visual world:** palette, type and component language specific enough to remain recognizable without the copy.
- **Story:** what the visitor understands, believes and does, using only true claims.
- **First viewport:** the composition, regions, relative scale, primary action, signature interaction and motion grammar.
- **Authority:** preserve/expand/replace/create, the chosen option, reference or approved comp, and any authorized departures.
- **Target:** saved approved/supplied or generated image, its actual dimensions and state; for native UI include device, orientation, scale and safe-area/system-chrome configuration. For scoped refinement, retain the incumbent baseline and approved change boundaries.
- **Finish:** required sizes/states, interaction and accessibility checks, comparison target and the evidence needed to call it complete.

Keep this contract out of rendered content, hidden DOM, source comments delivered to users, serialized props, client bundles and public assets. Do not create a new document or configuration ritual for a small task. For a new design system or approved durable system change, record reusable decisions in the project's normal design documentation after the built result settles; preserve existing DESIGN.md unless an update is authorized.

## Build the chosen direction

Choose the build path under the [image-first policy](../references/uizze-image-to-ui.md), not by whether the surface is expressive or operational. Inspect a supplied or already approved target and use it without generating a replacement. Otherwise, with an available authorized image tool, generate a straight-on, realistic UI image before implementation: include real copy, controls, fonts/assets, task and viewport. Inspect and correct illegible labels, false claims, impossible interactions/layouts and accessibility or native-safe-area conflicts before saving one stable target with its actual dimensions. Do not install or initialize a provider. If generation is unavailable, use a supplied image; if none exists, disclose the limitation and build from the brief, existing conventions and concrete contract without inventing a target or claiming a 1:1 match. Small technical corrections and read-only audits do not need new concepts.

When presenting visual options, use the actual surface and viewport, real product name and copy, consistent fidelity and aspect, and visibly different compositions—not posters, tilted device mockups or the same template with new colors. Label demonstration data wherever it could be mistaken for real evidence. Never invent prices, customers, benchmarks, endpoints or product capabilities.

An approved comp is a spatial contract, not a mood. Keep the approved target stable. At its actual dimensions, identify every salient image, text, control and chrome region; measure layout boxes, spacing, cap heights/type scale, colors, crop and control placement. Inspect detail crops as well as the whole composition. Identify unavailable fonts or assets rather than claiming an exact match. Resolve material departures explicitly with the user; correct genuine errors, false claims and accessibility conflicts without silently replacing the approved design. An image cannot invent backend behavior or authorize a change to data/auth contracts.

Separate authored artwork from semantic UI. Use real assets for photographs, illustrations and material details; do not approximate an organic subject with a polygon mask or cover missing artwork with decorative chrome. Verify source, license, transparency and sharp edges at the displayed size. A comp crop is reference material, not a shipping image of the interface. Keep text, controls and data semantic, accessible and functional. Do not flatten the page into a screenshot.

Build the first meaningful viewport before extending the page. It must demonstrate the product's task, offer or mechanism with an obvious next action—not merely a mood trapped in a stock hero. Use realistic content and actual assets. Greenfield illustrative content may be authored and clearly labeled, but cannot stand in for factual or commercial claims.

Carry the chosen language through navigation, inputs, buttons, content and secondary states without diluting it into the first familiar template. For expressive surfaces, a metaphor can organize the page only while its parts still work as links, readable text and obvious actions. For working or reading screens, familiar platform controls and task/reading order take priority. Vary density, scale, imagery and quiet through the page within one coherent grammar; end with a real close.

Implement the approved signature interaction with the available platform mechanisms, not a static imitation that claims to work. Keep content visible by default, motion orchestrated rather than scattered, and expensive work bounded. If the promised technique cannot be implemented with available tools, disclose that and resolve the tradeoff rather than silently substituting.

## Complete and verify

Apply [craft](craft.md), and [operate](operate.md) or [iOS](ios.md) where relevant. Implement reachable loading, empty, error, disabled, success and recovery states, not a decorative collection of impossible states. Exercise the actual primary interaction.

Make responsive choices structural: decide what stacks, collapses, scrolls or remains fixed. Use real copy, long values, narrow widths and large text. An approved desktop composition must survive nearby widths, not only its exact capture size; preserve its first-viewport hierarchy rather than pushing the subject below the fold. Capture the relevant native device classes when building native UI.

Run the result and inspect representative sizes/states in a batched round, including the user's actual viewport when known. Wait for fonts/images, settle entrance motion and verify each capture is from the intended screen, state and scroll position, with no blank or half-loaded regions. Capture at the saved target's actual pixel dimensions and native configuration where applicable; compare side by side or with an overlay, opening full-context views and detail crops rather than only thumbnails. Correct layout/proportions first, then type, spacing, color and assets, and recapture after corrections. When a scoped extension uses an incumbent baseline, verify untouched areas remain consistent too.

Separate invalid evidence from design failure: recapture a broken screenshot before judging the build. Re-derive materially wrong structure or missing artwork rather than hiding it with spacing tweaks. Batch named fixes, then run and inspect the affected sizes/states again. Bound ordinary polish to an initial inspection and a confirmation round; after repeated no-progress fixes or unresolved material gaps, report them and resolve the scope/quality decision with the user instead of declaring a pass or looping indefinitely.

Where the host supports an independent review, provide the actual request, contract, approved reference and valid captures; do not imply a review occurred otherwise. A finding-list recheck only establishes whether those findings were resolved, not whole-surface approval. Later edits invalidate affected evidence. Finish with the target and actual dimensions, captures/environment and comparison method, exercised controls/sizes/states, remaining mismatches, source/asset origins and any unavailable capabilities; update authorized durable design decisions from the built result rather than a speculative rulebook. Claim a 1:1 match only for a real target and an inspected same-size render with no unresolved visible mismatch in that scope. Without rendering/comparison, report the visual match as unverified.
