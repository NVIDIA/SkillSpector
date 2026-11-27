# Anti-slop craft: consistency, navigation and motion

This playbook carries Appllama's navigation, anti-slop and motion laws, preserving their wording except for the Uizze name and the verification link. Source: design skill v1.3.0 at `dd5caaec3d5d50ad7fc0324da238119c6b7c3707`, MIT; see NOTICE and the package manifest.

Apply these as design guidance within the user's task and existing product system. The visual consistency rules apply to web and native UI; navigation API names below are Expo/React Native examples. Preserve SwiftUI, UIKit, web routing or other existing frameworks and use their equivalent native mechanisms. Verify installed API support before adopting an example; this guidance does not authorize a framework migration or new dependencies.

The user's intentional brand choices and supported platform conventions take precedence. An accent is a primary interaction color, not every color in a chart, photograph, status message or user content. Preserve semantic success/warning/error colors and justified typography systems. Record exceptions that matter to the change; do not flatten an expressive product into a generic single-color template.

Use motion and navigation checks only where the task touches those behaviors. No simulator, recorder or representative device means those checks are unverified, not a reason to invent measurements or block unrelated useful work. Finish with the scoped evidence in [visual verification](../references/uizze-visual-verification.md), not an unbounded perfection loop.

## Navigation laws

Navigation is the part of a screen a screenshot can't show, and users feel
it in ten seconds. Every transition answers three questions: what is the
destination to here, must the user be able to come back, and what does back
(chevron, iOS edge swipe, Android hardware back) do afterwards.

1. **Push goes deeper, replace moves on.** `router.push` when the user will
   want to return here; `router.replace` / `<Redirect>` when coming back
   would land in a state the world has moved past; `router.dismissTo(href)`
   for "finish this flow and land on X". Back undoes *navigation*, never
   *events*.
2. **Presentation is meaning.** A self-contained task with steps →
   `presentation: 'modal'` with its own stack and its own Cancel/Done; a
   short interruption (picker, filters, item options) → `formSheet` with
   detents, drag-to-dismiss; immersive content → `fullScreenModal` with an
   explicit Close; something floating over a still-visible screen (confirm
   card, lightbox, coach mark) → `transparentModal` overlay; destructive
   confirms → action sheet; item actions → native context menu; share /
   web / photo picking → the system controller, never a rebuilt route. A
   sheet that grows a second step was a modal all along; if a link could
   open it, it is a route, not a `useState` sheet.
3. **One-way doors leave the stack.** Sign-in on a wall app, finished
   onboarding (Skip included), a purchase, a completed session: guard with
   `Stack.Protected` and land with `replace`, so back can never re-enter
   the old state — Android back from home exits the app, never shows
   Login; a paid paywall never re-opens. But keep the user's *place*:
   sign-in demanded by one action (save, follow, buy) is a modal over the
   screen that completes the action where it was tapped, and a paywall
   opened from a feature dismisses back onto the feature, unlocked — never
   `replace('/(tabs)')` from there.
4. **Back is blocked in exactly two cases** — an irreversible request in
   flight (seconds, with visible progress) and unsaved work in a modal
   (ask first), both via `usePreventRemove` on the modal's root screen.
   Transient in-screen state (selection mode, an expanded search, an open
   in-screen sheet) consumes the first back, then back leaves. Anything
   else that traps back — a funnel, a rating prompt — is a defect; the
   edge swipe works everywhere else.
5. **Tabs are peers.** No slide between tabs, each tab keeps its own stack,
   re-tapping the active tab pops to its root; full-attention screens
   (composer, player, checkout) live in the root stack *above* the tabs.
   Deep links land with a real stack underneath (`initialRouteName` /
   `withAnchor`); cold start lands by state, splash held until session
   state has resolved — never a Login flash before Home.
6. **Study the grammar, not just the pixels.** Walking a winning flow on
   Uizze, note what each step *is* — push, modal, sheet — and copy that
   consistency.

## Anti-slop laws

AI-built apps share a look, and users file it under "template" within seconds.
Each of these is a *default ban* — there is always an override when the brand
explicitly asks for the thing AND you can articulate why it fits this product.

1. **No AI-default styling.** Purple/indigo gradient CTAs with a glow,
   glassmorphism on every card, mesh-gradient heroes, confetti for minor
   events, sparkles in headings — that is the model's house style, not
   design. Your palette, materials, and layout come from the reference
   screens you studied, never from the priors you'd reach for unprompted.
2. **One accent, locked.** Pick one accent color and it is THE accent on
   every screen — no blue CTA on one screen and teal on the next, no new hue
   appearing in screen seven. Neutrals carry the app; the accent is spent
   where the money is (primary action, active state, progress).
3. **One grey family.** Warm greys or cool greys — never both in one app.
4. **Shape lock.** One corner-radius scale, stated as a rule ("actions are
   pills, cards 16, inputs 8") and never violated. Mixed radii without a
   stated rule read as assembled-from-parts.
5. **No emoji as iconography.** Icons are SF Symbols / Material Symbols
   (fidelity law 3). Emoji appear only when the product's voice is genuinely
   chat-native or playful — sparingly, in content, never in chrome.
6. **One label per intent.** "Get started", "Start now", and "Begin" are the
   same intent — pick one phrasing and use it everywhere it appears.
7. **Emphasis stays in the family.** Emphasize a word with weight or italic
   of the same typeface; injecting a serif word into a sans headline (or vice
   versa) for visual interest is amateur.
8. **Ship full state cycles, not the happy path.** Static-successful-state-
   only is the default failure mode: skeletons must match the final layout's
   shape, empty states are composed (and say how to fill them), errors are
   inline and specific.
9. **The slop pre-flight is mechanical.** Before any flow reaches the
   simulator pass, count: distinct accent hues (must be 1), distinct corner
   radii (all from the stated scale), emoji in UI chrome (0), gradients
   without a brand reason (0), duplicate labels for one intent (0). A failed
   count is a fix, not a judgment call.

## Motion laws

Motion is the highest-leverage polish surface and the easiest to overdo.
Decide in this order:

- **The frequency gate comes first.** Met 100+ times a day (tab switch,
  keyboard, scroll, back) → the platform default and nothing else; tens a
  day (press, row select) → near-imperceptible, under 150 ms; occasional
  (sheets, modals, toasts) → standard motion; delight only on rare,
  first-time moments. Tabs never slide; screen transitions stay native.
  Passing this gate with zero lines of code is a success — when unsure,
  the strongest move is to delete the animation.
- **Name the purpose in one word** — feedback, spatial continuity, state
  change, preventing a jarring cut, explanation, delight — or don't build
  it. Data the user is reading never moves for style.
- **If a finger was involved, it's a spring.** Start from the live value
  (capture it on grab), hand the release velocity into the spring, pick
  the target from projected momentum so a flick commits, rubber-band past
  boundaries, stay grabbable mid-flight. One vocabulary per app —
  `{ duration: 400, dampingRatio: 1 }` to settle, `{ 300, 0.8 }` for
  sheets — and bounce only when the gesture carried momentum.
- **Everything else is timing, under 300 ms, strong ease-out**
  (`Easing.bezier(0.23, 1, 0.32, 1)` — built-in curves are too weak; never
  ease-in on an entrance). Press feedback lands on press-*in*, 100–150 ms:
  scale 0.97 on buttons and cards, a background highlight (never scale) on
  list rows, opacity on bar buttons. Exits are faster than entrances and
  leave the way they came in; enter from `scale(0.95)` + fade, never
  `scale(0)`; menus grow from their trigger (centered modals exempt).
- **Gesture → animation never hops the JS thread.** Worklets + shared
  values (`.get()`/`.set()`; `scheduleOnRN` — Reanimated 4's `runOnJS` —
  only at gesture end), `transform`/`opacity` only, no `entering` on
  recycled list rows, never animate a header's height (translate inside a
  fixed clip), keyboard-tracking UI via `react-native-keyboard-controller`
  — never a keyboard listener plus a guessed duration.
- **Respect Reduce Motion**: your spatial motion collapses to cross-fades;
  native transitions stay the system's.
- The bar: 60 fps through the hero flow, measured on a **release build on
  the slowest device you support** — Expo Go and dev builds hide exactly
  the jank you're hunting
  ([visual verification](../references/uizze-visual-verification.md)). Watch the
  recording once for feel, once frame by frame, and again next day with
  fresh eyes.

