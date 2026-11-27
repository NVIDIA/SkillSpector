# Mobile controls and forms

Use this when implementing native inputs, menus, sheets or navigation. Preserve the existing SwiftUI, UIKit, React Native or other framework and its installed versions. Prefer its native controls or established wrappers; this playbook does not require a new package. For mobile web, use semantic browser controls and browser navigation rather than imitating an OS inside a page.

Adapted from Appllama's MIT-licensed native-controls.md and design skill; immutable sources are in MANIFEST.json and attribution in NOTICE.

## Choose the control by the task

| Task | Starting point |
| --- | --- |
| Binary preference | Native switch with a persistent label |
| A few mutually exclusive choices | Segmented control or the platform's equivalent |
| Approximate value within a range | Slider with an accessible value; offer precise entry when needed |
| Date or time | Platform picker with locale, calendar and time-zone semantics |
| Actions on one item | Menu anchored to that item; use destructive roles and confirmation or undo appropriate to the consequence |
| A bounded task | Native sheet with content-driven or standard detents |
| Search | Existing navigation/search control with clear, cancel and empty-result behavior |
| Refresh | Platform refresh control in the scroll container |
| External web content | Existing system-browser integration, preserving the return path |

A custom control must justify its deviation and retain the platform's accessibility value, role, keyboard/assistive activation, focus, disabled state and feedback. Use the product's icon family and semantic colors; haptics supplement visible feedback and fire once per meaningful action.

## Forms must survive the keyboard

- Keep labels visible after entry. Set the correct input purpose, keyboard, autofill and secure-entry behavior; do not disable paste or password-manager support to achieve a visual effect.
- Chain Next through fields and make the last Return action explicit. Keep the focused field, its validation and the relevant action reachable while the keyboard is present.
- Validate on submit or blur when appropriate. Preserve entered values on failure, connect the error to the field, and revalidate a previously reported error as it is corrected. Do not interrupt ordinary typing with a new error on every character.
- Test long values, pasted values, larger text, keyboard appearance and dismissal. Use the framework's keyboard/safe-area mechanisms, not a guessed keyboard height or animation delay.
- Restore focus sensibly after closing a picker or sheet. Pending, disabled and failed submission states must remain understandable with a screen reader.

## Sheets and navigation retain context

Use the sheet implementation's supported scroll integration so scrolling and dragging hand off correctly. Check every relevant detent with the keyboard open. Keep dismissal and Back predictable, preserve drafts that matter, and make a cancelled dismissal settle cleanly. Do not apply tap-outside dismissal to a task that would silently lose important work.

Tabs represent peer destinations; stacks represent drill-down. A multi-step modal needs its own coherent navigation context. Preserve system Back and interactive gestures where valid; after an irreversible completion, prevent returning to a stale submission step. Follow existing deep-link/auth rules rather than declaring every internal step publicly reachable.

Use [anti-slop craft](anti-slop-craft.md) for motion purpose and continuity, [state and performance](mobile-state-performance.md) for pending/recovery behavior, and [native verification](native-verification.md) to test the actual flow. A screenshot of a switch does not verify a working switch.
