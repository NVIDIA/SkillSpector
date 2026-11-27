# Evidence for visual completion

Use this for an implemented or changed interface. Scope the checks to the requested work and supported product behavior. Do not invent a dark theme, device target, or new feature to satisfy a checklist. A source-only audit does not need manufactured screenshots.

These checks establish concrete behavior, reference fidelity and reviewable evidence. An AI-authored rating, another agent's preference, or a passed-check count does not establish good taste or that the result is better than a reference. Run the applicable checks and corrections autonomously and deliver the result with its evidence. Do not create a mandatory human taste-review gate; report concrete discrepancies without presenting them as a “less slop” score or claiming user approval.

## Define the checks before the final pass

Name the important states and sizes: the primary path, affected empty/loading/error/recovery states, the reported failing viewport, and relevant keyboard, text-size or theme settings. For a flow, include entry, exit, Back/dismissal, and state carried between steps. For motion, include entry, interruption, exit and reduced-motion behavior. Use real product state; an unavailable backend is a limitation, not successful submission.

For forms, exercise invalid input → correction → blur/submit, then leave and revisit the step when supported. Check the visible message, error styling and accessible invalid state together: corrected input must not retain an unexplained error, and navigation must not hide a message while leaving its error marker behind. Include this recovery cycle in the receipt; a successful retry of a server request does not verify field-validation recovery.

Capture the actual running result after fonts and images load. Inspect its pixels against the approved target or the existing product system. Reject blank, wrong-screen, half-loaded or wrong-size captures. Fix named material issues together, then recapture the affected checks. Stop when the scoped checks pass; record unresolved issues instead of looping indefinitely.

## Save a reviewable receipt

Keep a compact `visual-evidence.json` beside local captures (outside public product assets). Record:

- The scope and affected source/style/asset files with SHA-256 hashes of the bytes used for the final render.
- The comparison mode: an actual saved target, or the existing system. A target image is evidence only after it was inspected and compared with a real render.
- The required checks, actual browser/simulator/device and viewport dimensions, state, observed findings, and saved screenshot/video paths and hashes.
- The interaction steps exercised, and all unresolved or unavailable checks. A blocked check remains blocked; do not remove it to obtain a pass.

Use a small receipt for a small change. Example shape (replace every example value with measured evidence):

```json
{
  "schemaVersion": "uizze.visual-evidence.v1",
  "scope": "Checkout error-state spacing",
  "platform": "web",
  "sources": [{ "path": "../src/checkout.tsx", "sha256": "actual-sha256" }],
  "comparison": "existing-system",
  "requiredChecks": [{ "id": "mobile-error", "kind": "visual" }],
  "checks": [{
    "id": "mobile-error",
    "status": "pass",
    "state": "Payment declined, retry enabled",
    "environment": { "kind": "browser", "name": "Chromium", "width": 390, "height": 844 },
    "observed": "Error wraps beneath the input; retry remains visible with no overlap.",
    "artifacts": [{ "path": "mobile-error.png", "kind": "image", "sha256": "actual-sha256" }]
  }],
  "unresolved": []
}
```

For `comparison: "target"`, add `target: {"path": "target.png", "kind": "image", "sha256": "actual-sha256"}` and record the target's real dimensions in the scope or observations. Compare at matching dimensions. Retain a separate actual render; reusing or copying the target is not runtime evidence. Paths resolve relative to the receipt; absolute local paths also work. Compute hashes from actual files with the host's tools—never invent them. Include every affected input that could invalidate the visual result, not just one convenient file.

Declare `platform: "web"`, `"ios"` or `"android"` according to the implemented surface. A browser imitation cannot verify iOS/Android UI; native checks require simulator/device evidence. A missing native runner leaves those checks blocked even if a browser mockup looks correct.

An `interaction` check needs an image and nonempty `steps`. A `motion` check needs a saved video and `steps`; watch it at normal speed and inspect the relevant transitions. Record `status: "blocked"` or `"fail"` and the reason in `observed` when verification cannot complete.

## Check integrity when execution is available

If the host permits local Node execution, run the bundled helper using its actual installed path:

```sh
node /path/to/installed-skill/scripts/verify-visual-evidence.mjs /path/to/visual-evidence.json
```

It reads local files only. It rejects missing/stale files, incomplete declared checks, unresolved issues, non-media placeholders and motion claims backed only by stills. It does not execute project code, install dependencies, upload evidence, judge aesthetics, or prove that the agent inspected the pixels. If execution is unavailable, retain the same receipt and inspect the links manually; report that integrity validation did not run.

Later source/style/asset or target changes invalidate affected checks. Re-render, reinspect, then update the hashes; changing a hash alone is not revalidation. A passing script is evidence integrity, never a substitute for the visual comparison. In the handoff, link the receipt and relevant captures, state the covered states/sizes, and name remaining limitations. Do not claim a complete visual pass while required evidence is blocked or missing.
