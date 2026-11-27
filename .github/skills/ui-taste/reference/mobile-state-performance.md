# State and measured responsiveness

Read for data-backed flows, typing, long lists or observed latency. These principles also apply to web UI. Preserve the project's architecture and dependencies; a small form does not need a new query library, global store, persistence engine or animation system.

Adapted from Appllama's MIT-licensed performance.md and design skill; immutable sources are in MANIFEST.json and attribution in NOTICE.

## Put state where its lifetime belongs

- Server-owned data belongs in the project's existing data/cache layer, with its established invalidation and error policy. Avoid parallel caches that disagree about the same record.
- Shared client preferences belong in the existing shared state mechanism with narrow subscriptions. Keep temporary focus, open/closed state and unsaved field edits local unless the flow requires them to survive navigation.
- Decide what survives Back, tab switches, backgrounding and relaunch. Persist only what the product requires, using existing storage/privacy rules; do not persist sensitive form values merely to make restoration convenient.
- Keep interaction feedback immediate, but report completion only when the operation's contract allows it. Optimistic changes suit reversible edits with a rollback/reconciliation path. Payment, permission, account deletion and other consequential outcomes need authoritative confirmation.
- Guard against repeated activation while pending. Reuse existing idempotency support; a disabled button alone is not a server guarantee. Keep valid input on error, give a specific retry path, and prevent a stale response from overwriting a newer choice.
- Retrying should repeat the intended operation, not recreate unrelated steps. Never silently queue consequential offline actions or introduce automatic retries beyond the backend's documented guarantees.

## Loading should preserve orientation

Keep existing content visible during a partial refresh where valid. Use a skeleton only when the final structure is known; otherwise show honest progress in the affected region. Preserve image dimensions and stable list keys to prevent jumps. Prefetch only safe reads when useful, without triggering mutations or fetching private data before access is established.

## Measure → change → measure again

Choose the affected interaction and one useful metric: input-to-visible-response latency, startup-to-interactive time, dropped frames, list blanking or retained memory. Record the build, device, dataset, network conditions and measurement method. Capture a baseline, fix the observed bottleneck, then repeat the same procedure. Report numbers only when actually measured; a video or smooth-looking still cannot establish frame rate.

For React Native, use a release build for performance claims: development overhead changes results. Distinguish JS responsiveness from native/UI-thread motion when diagnosing a stall. See the installed-version [performance guidance](https://reactnative.dev/docs/performance).

Choose fixes from evidence:

- Typing: first narrow the state update and inspect costly subscribers. Change controlled/uncontrolled ownership only if measurement supports it and validation, autofill and draft behavior remain correct.
- Lists: use the project's virtualized/recycled list when growth warrants it. Measure blank cells, memory and touch responsiveness together; larger batches can improve filling while delaying interaction. Use stable identities and appropriately sized images. See [FlatList configuration tradeoffs](https://reactnative.dev/docs/optimizing-flatlist-configuration); do not replace every short list by rule.
- Startup: separate cold and warm runs; defer work unrelated to the first usable screen and inspect actual bundle growth. Keep auth and required data checks intact.
- Motion: profile expensive renders, decoding and per-frame JS work. Use the platform's animation/gesture path; avoid allocating or dispatching application state on each gesture update.
- Memory: reproduce growth across repeated navigation, then inspect retained listeners, subscriptions, timers and images before changing storage technology.

Do not add memoization, a compiler, a library or a numerical performance claim without evidence that it addresses the observed problem. If profiling is unavailable, report the symptom and unverified hypothesis. Save measured results beside the [visual evidence](../references/uizze-visual-verification.md); choose budgets for the supported product/device, not universal benchmark promises.
