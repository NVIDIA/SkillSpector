# Building a UX Pilot design in the user's code

## Read the design

- `read` with `kind: "design"`, `designId` and `includeHtml: true` returns the screen's HTML. Up to 32,000 characters comes back whole. A bigger screen comes back as an outline of sections with line ranges: read it section by section with `htmlRange` or one element with `htmlSelector`.
- The HTML uses Tailwind classes. Treat it as the spec for layout, spacing, sizes, colors, type and copy, not as code to paste.

## Rebuild it in the project's stack

1. Find what the project already has: its framework, component library, icon set, design tokens and the folder where similar pages live.
2. Map each part of the design to an existing component first (buttons, inputs, cards, nav). Build a new component only when nothing fits, in the project's style.
3. Use the project's tokens when they match the design. When the design adds a new color or size, add it as a token once instead of hard-coding it in many places.
4. Keep the design's structure: section order, grid, gaps, padding, font sizes and weights, and the exact copy.
5. Images: use the project's real assets when it has them. Otherwise download the design's images into the project's assets folder and point to the local copies; never link to UX Pilot's storage from the user's code. Keep the image sizes and crops from the design.
6. Make it work, not just look right: real links, form handling, loading, empty and error states the screen implies.
7. Make it responsive. If the design has only one width, keep its layout at that width and let it stack sensibly on smaller screens.
8. Keep it accessible: real buttons and links, labels on inputs, alt text, visible focus, and the design's contrast.

## Check it

- If you can run the app and take a screenshot (a browser, a preview tool, or a simulator), compare your page with the design's picture side by side and fix the differences that matter: spacing, alignment, font sizes, colors, missing parts.
- On a desktop, headless Chrome works for a static page: `"<chrome path>" --headless=new --screenshot=out.png --window-size=1440,3000 file:///path/index.html`. The design's picture is at the `previewUrl` from `read` with `kind: "design"`.
- Do not import your build onto the user's UX Pilot page to check it unless they ask: it adds screens to their canvas and counts as an action on the free plan.
- If you cannot run it, say so and list what the user should look at.
- Tell the user what you built, which files changed, and anything you could not match and why.
