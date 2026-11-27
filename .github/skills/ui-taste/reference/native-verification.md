# Native verification: screens, flows and motion

Use when the task changes native UI. A browser rendering of a mobile layout is web evidence, not a simulator/device result. Use the project's existing runner and test tooling. If no runnable native build or device is available, complete useful code checks and leave native checks explicitly unverified.

Adapted from Appllama's MIT-licensed simulator-loop.md and motion.md; immutable sources are in MANIFEST.json and attribution in NOTICE.

## Choose a task-sized matrix

Use the reported failing configuration first, plus a representative supported device. Check the smallest supported layout when compression matters. Include another platform, tablet, orientation or appearance only if supported and affected; do not add platform support as part of verification. Record OS, device, viewport/scale, build mode, text size, appearance and relevant accessibility settings.

For the affected screens, force loading, empty, failure, success and offline/denied states when applicable. Inspect safe areas, scrolled-under bars, reachable actions, long/localized text, large accessibility text, contrast and accessible names. Check the supported light/dark appearances and Reduce Motion. Open and study actual captures; a saved file alone is not inspection.

## Exercise the whole affected flow

Start from its real entry point. Activate each affected control; include validation, failed submission, retry, rapid double activation, Back, cancellation and reopening where relevant. Verify the state that should survive each transition. After a completed consequential action, Back must not recreate a stale submission or silently repeat the action.

Test keyboard appearance, Next/Return, picker focus, scrolling to a focused field and keyboard dismissal. For sheets, exercise present, drag, inner scrolling, cancel and dismiss. Check background/foreground and restoration only for state the task is meant to retain. Use actual native events where possible; assigning component state directly does not test the control.

## Motion needs a recording and interaction

Record the affected flow from entry through exit, including keyboard and Back behavior. Watch at normal speed for response and continuity, then inspect transition frames for flashes, duplicate content, wrong-theme frames or a jump after settling. Stills cannot establish those properties.

For custom gestures, test slow movement, fast flick, reversal, cancellation and grabbing an animation mid-flight. Position should follow the live gesture; release uses velocity as well as distance, and a new grab starts from the current position. Confirm that cancellation does not commit navigation or a destructive action and that the completion callback fires once. Use installed-version APIs, not copied timing constants as a substitute for behavior.

Check that routine Back/tab operations do not replay decorative entrances. For approved custom continuity, keep the source and destination geometry coherent without blocking navigation. Do not stagger recycled rows or add scroll effects merely because an animation example exists. Follow the frequency/purpose gate in [anti-slop craft](anti-slop-craft.md).

Observe Reduce Motion with the system setting enabled; keep an equally usable path when spatial motion is reduced. A full-flow recording can reveal visual defects but does not measure FPS. Use [state and performance](mobile-state-performance.md) for measured claims on a representative release build.

## Close the loop with evidence

Fix observed issues in a bounded batch and rerun the affected states, plus a neighboring transition when the fix can affect it. Once the scoped checks pass, stop. Keep blocked checks and remaining defects visible instead of substituting a compile result for a visual pass.

Store final screenshots/recordings, interaction steps, tested configurations and current source hashes through [visual verification](../references/uizze-visual-verification.md). Capture after the final source change. The receipt helper checks integrity and declared coverage; it cannot authenticate the device or prove that a person/agent actually watched the recording.
