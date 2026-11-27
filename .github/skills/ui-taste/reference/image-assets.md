# Coherent image and illustration assets

Use when the requested interface needs new or revised artwork. Reuse appropriate licensed product assets first. A typography, layout or form fix does not need an illustration. Preserve the existing visual identity and use only available, authorized tools; missing generation access is a limitation, not permission to install a service or spend money.

Adapted from Appllama's MIT-licensed image-assets.md; immutable source is in MANIFEST.json and attribution in NOTICE.

## Define the set before generating it

Record a short art direction: style family, palette from the product tokens, surface color or transparency, lighting/texture, line weight and subject/composition rules. Use the same direction across the set. An approved existing asset is a stronger anchor than a generic style adjective.

For generation, combine that direction with the specific subject and placement. Reserve clear space for real interface text and controls. Keep ordinary labels in the implementation instead of baking them into the image. Empty-state art should support the recovery action; decorative content must not imply data or features the product does not have.

Generate a few candidates when the task and tool budget justify it, inspect them, and choose a consistent direction before expanding the set. Do not regenerate approved work or multiply variants by default. Use an existing symbol set for functional icons; a generated illustration sheet does not automatically provide accessible, legible UI icons.

## Size for the actual placement

Calculate the needed raster size from rendered logical dimensions and device scale: 200 points at 3× needs 600 pixels across before any intentional oversampling. Export suitable sizes using the project's asset pipeline. Do not upscale a tiny source and claim it is sharp. Preserve crop, focal point and consistent padding across variants.

Request true transparency for cutouts when supported, or the exact intended background. Inspect at normal size and enlarged: reject halos, fringe, jagged edges, accidental text, watermarks and inconsistent light or texture. Do not use post-processing to disguise a provenance or licensing problem.

Check the actual composite on every supported appearance/surface used by the task. For a dual-theme product, use transparent assets that work on both grounds, justified theme variants, or a deliberate theme-invariant surface. Verify contrast and meaning without relying on color alone.

## Integrate and inspect

Render the asset in the real layout, including narrow sizes and long copy. Check that the subject is not cropped by safe areas or obscured by an action. Supply alternative text when the image carries information; mark pure decoration appropriately.

Inspect compression artifacts and transfer/decode cost at the displayed size. Select a supported format and an appropriate file budget rather than shipping the largest generation. Keep editable/source assets separate from optimized runtime files using the project's convention.

For an app icon or store asset explicitly in scope, verify the current target platform's export/masking requirements before producing it and inspect the icon at its smallest real placement. This playbook does not add store or marketing work to an interface task.

Finish with inspected in-layout captures and asset paths in [visual verification](../references/uizze-visual-verification.md). “Generated” is not the same as “reviewed in the product.”
