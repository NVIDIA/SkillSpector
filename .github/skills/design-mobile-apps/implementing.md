# Implementing Designs

When the user wants the designs built as code rather than previewed, **always fetch the component HTML code** and build from it; the screenshots confirm the result.

Use `GET /api/v1/projects/:id/components/:componentId` to fetch each screen's code. For a new run, get `componentId` from `result.operations`. For an existing project, list its components, paging through `pagination.total`; each item supplies `id` for HTML/screenshots and a nullable `screenId` for targeted edits. Use a returned screen ID when present; don't substitute the component ID when it is null.

Component code can be large. Fetch it with a shell command that writes the response body straight to disk, so the code goes to a file without passing through your text output.

## Which version to use

Each component carries a `versions[]` array and an `activeVersion`, a **nullable** number. **By default, use the entry where `versions[i].version === activeVersion`**. If `activeVersion` is `null` or has no matching entry, select the **highest numeric `version`** for implementation. Pass that selected entry's `id` in `componentVersionOverrides` for its screenshots too: the renderer can otherwise produce an empty frame. If there are no versions, stop and report that there is no code to implement.

If the user's prompt pins specific versions, follow those instead (see [Pinned versions](#pinned-versions) below).

## Pinned versions

The user's prompt may include a pin block telling you to implement specific historical versions instead of the current ones, like this:

```
... at this exact state instead of the project's current version:
- component V1StGXR8Z5j: version qkP2mN7bXsT
- component 8rJvL4wYhKd: version zC5tRb9WqNm
- theme MfD3xQ6nJyU: version 7hSvA2kLpEr
```

When you see a pin block, implement those exact versions instead of `activeVersion`. Components not named in the pin block continue to use their active version. Theme ids surface only inside pin blocks; this skill exposes no separate endpoint to enumerate them.

**Historical themes are a separate limit.** Component HTML inlines the active project theme, even for historical component versions. Screenshot `themeVersionOverrides` do not change that HTML, and the public API has no theme-version read or HTML theme override. Preserve supplied theme pins for preview and use the screenshot as a visual reference, but do not claim an exact implementation from active-theme HTML. Exact implementation of a pinned historical theme needs its supplied tokens and fonts or an API export that resolves that theme; report this dependency rather than silently substituting the active theme.

### Fetching the right code

For each pinned component, find the entry in `versions[]` whose `versions[i].id` matches the given version id and use its `code`. Match on `id`, the string; `version` is the numeric index, and a pinned component takes its code from that matched entry alone.

If the requested version is absent, report the unavailable pin and stop that export. Do not substitute another version. The screenshot API silently falls back to active versions for invalid overrides, so a successful image response alone does not verify a checkpoint.

### Screenshots of pinned versions

Pass `componentVersionOverrides` and `themeVersionOverrides` to `POST /api/v1/screenshots`:

```json
{
  "componentIds": ["V1StGXR8Z5j"],
  "projectId": "Nq4bZp8wLcT",
  "componentVersionOverrides": { "V1StGXR8Z5j": "qkP2mN7bXsT" },
  "themeVersionOverrides": { "MfD3xQ6nJyU": "7hSvA2kLpEr" }
}
```

Keys are component / theme public ids; values are the corresponding `versions[i].id`. Entities missing from a map fall back to their active version. Include the override maps whenever the prompt specified pinned versions.

## HTML prototypes

The component `code` is a complete HTML document. Save it directly to a `.html` file. No build step needed.

## Native frameworks

Each component document is a **mockup**: a picture of a working screen, drawn in HTML because HTML is what a design tool renders.

- **Structure, layout and styling are the spec.** Element hierarchy, flex direction, spacing, sizing, colors, typography, radii, image URLs and icon names — reproduce these exactly.
- **Content is placeholder.** The names, numbers, copy, avatars, list items and "active" states are stand-ins that make the screen legible. Rebuild the container and feed it real state, props or API data — see [Build the feature, not the mockup](#build-the-feature-not-the-mockup).
- **The mockup draws the app's chrome** — tab bar, header, drawer — inside every screen that shows it, because each screen is a standalone document. Chrome belongs to the app's navigation, so build it there once and let each screen render its own content.
- **Screenshots are the visual target.** Check each built screen against its **review shot**; a user shot crops the page and never proves something is missing.

The HTML tells you _how_ to build it; the screenshot tells you _what_ it should look like; the app supplies the data.

### Icons

Sleek uses [Iconify](https://iconify.design) icons in the format `prefix:name` (e.g. `ph:heart-fill`, `solar:heart-bold`, `lucide:settings`). Sleek's design prompt steers toward four sets — **Phosphor** (`ph:`), **Solar** (`solar:`), **MingCute** (`mingcute:`) and **Hugeicons** (`hugeicons:`) — while allowing any Iconify set, so read the prefixes out of the HTML rather than assuming them.

**Use the exact icons from the HTML code** — same set, same name. Matching icons is what carries design fidelity.

When implementing icons:

1. **Check whether the project already has an icon system** that resolves the same names. On a repo that already has `@expo/vector-icons`, `mdi:*` is a free rename: Iconify's `mdi` and the package's `MaterialCommunityIcons` are the same upstream set with matching glyph names, so `mdi:account-check` is `<MaterialCommunityIcons name="account-check" />`. The rest take the Iconify route below — `solar:*` and `hugeicons:*` have no counterpart in the package at all, and `material-symbols:*` is a different Google generation from its `MaterialIcons`. For any other prefix, check the package's set list before assuming a match.
2. **Otherwise, fetch the SVGs from the Iconify API and embed them in the code:**

   ```
   GET https://api.iconify.design/{prefix}/{name}.svg
   ```

   Example: `https://api.iconify.design/solar/heart-bold.svg`

   Collect all icon names from the HTML, fetch their SVGs, and save them as static assets or string constants in the codebase. For **React Native / Expo**, render them with `react-native-svg`'s `SvgXml` component, which works in Expo Go with no additional native dependencies.

   If the Iconify endpoint is unavailable, obtain the same prefix/name from its official `@iconify-json/<prefix>` data package and embed only the icons used. Preserve the SVG geometry and viewBox; replacing the set changes the design.

### Fonts

For the current theme, the `<head>` `<link>` tags give you the font **families**. For a pinned historical theme, use the supplied theme's families instead. The links request each family's entire published weight range (or a 100–900 fallback for a family Sleek doesn't recognise), so the weights in those URLs describe what Google Fonts can serve, not what the design uses. The weights actually used are in the body's `font-*` Tailwind classes. Read those weights and bundle only the faces used.

Resolve body font classes through the theme's family tokens before bundling. An exported alias such as `--font-font-heading` does not define Tailwind's `font-heading` utility, even when the intended `--font-heading` family is present. Correct malformed aliases when porting and match the family rendered in the Sleek screenshot. If the intended token and screenshot still disagree, record the discrepancy rather than silently choosing one.

### Design tokens

Every component document inlines the project's theme in the `<style type="text/tailwindcss">` block in `<head>`: a `:root` rule holding the raw values (`--background`, `--foreground`, `--primary`, `--border`, `--radius`, `--font-*`, …) plus an `@theme inline` block naming them for Tailwind. The same theme is inlined into every screen of a project, so port it once into a single token module and have every component read from that module.

Two of those values need work on the way across:

- **`--radius` is a base, not a value.** The document derives `--radius-xs … --radius-4xl` from it with `calc()`; resolve the arithmetic to numbers.
- **`--shape` is always present**, valued `round` or `squircle` — test the value, not its presence. `squircle` turns on CSS `corner-shape` and multiplies every radius by `--shape-multiplier` where the browser supports it. React Native has neither, so use plain `borderRadius` and match the rounding the screenshot shows.

Colors can be hex, `rgb()` or `oklch(...)`. Preserve supported values and convert unsupported formats once; the file below covers each styling branch.

---

## React Native (Expo)

Sleek renders web HTML with Tailwind v4 classes (the document loads `@tailwindcss/browser@4` — read the `<head>` to confirm). Several web defaults land differently in React Native, and the failures below are **silent**: the app compiles, runs, and is wrong. Work through them before the first component.

**Styling.** Use the user's requested styling system, otherwise preserve the system used by the feature being edited. Dependencies and source files provide evidence; read the corresponding reference:

| Evidence                                      | Read                                           |
| --------------------------------------------- | ---------------------------------------------- |
| Nativewind classes in the feature, or Nativewind requested | [styling/nativewind.md](styling/nativewind.md) |
| Uniwind classes in the feature, or Uniwind requested | [styling/uniwind.md](styling/uniwind.md) |
| `StyleSheet.create` in the feature, or a new app without a library preference | [styling/stylesheet.md](styling/stylesheet.md) |

Each file carries its own token conversion, class syntax and configuration. For a repo with no styling library and a user with no preference, use StyleSheet: it needs no build configuration, so it cannot be misconfigured into silently doing nothing.

### Flex direction

The most common silent failure. On the web `display: flex` lays children out in a **row**; in React Native the default `flexDirection` is `'column'`. Sleek writes the web idiom, so `class="flex items-center gap-3"` is a horizontal row that renders as a vertical stack until you say `flexDirection: 'row'` (or `flex-row`).

`flex-col` is already the default and needs nothing; `flex-1` maps to `flex: 1`; `inline-flex` is dropped by the style engine with a value warning, which lands on the same layout as `flex` because React Native has no inline mode. Sweep every `flex` in the source and give each one an explicit direction.

### Text

Every string renders inside `<Text>` — including strings hiding in a `&&` branch, a ternary or a stray `{' '}`. A bare string under a `<View>` fails **silently**: React Native's New Architecture routes `Text strings must be rendered within a <Text> component.` through `console.error`, so it appears in a development build's console and nowhere else. A production build carries no such message — the string is simply missing from the screen, and nothing throws.

`<span>` and `<p>` both become `<Text>`, and spans nested inside a paragraph become nested `<Text>`. Text styling (`fontSize`, `color`, `fontWeight`, `lineHeight`) sits on the `<Text>` itself, since it does not cross a `View` boundary.

### Insets

The mockup has no notch, status bar or home indicator: it renders edge to edge inside a fixed frame. On a device, an unguarded screen draws its header under the status bar and its bottom bar under the home indicator or the Android navigation bar — Expo SDK 54 and later make Android edge-to-edge the default, so Android content draws under the system bars just as iOS content does. Inset with **`react-native-safe-area-context`** — React Native core exports a `SafeAreaView` of its own that only insets on iOS, so import from the package — and put a `SafeAreaProvider` at the app root. Native modal or route roots can need their own provider when `react-native-screens` creates a separate view hierarchy; check the presented screen's actual insets. See the [provider documentation](https://appandflow.github.io/react-native-safe-area-context/api/safe-area-provider/).

Inset each edge of each screen exactly once:

- A native-stack navigator header insets the **top only**. A screen under one still owns its bottom edge, so unless a tab bar or other navigator chrome covers it, give the screen `edges={['bottom']}` or bottom padding from `useSafeAreaInsets()`.
- A headerless screen, including a headerless modal, owns its top and bottom insets: `SafeAreaView` around the screen, or `useSafeAreaInsets()` when you need the numbers — a header you drew yourself, a floating or absolutely positioned bottom bar, a sticky CTA, or content that scrolls under the status bar while its padding respects it. On modal routes with navigator chrome, inset only the uncovered edges.
- `edges` narrows a `SafeAreaView` to the sides that still need guarding, which is how a screen keeps the navigator header and insets the bottom alone.

### Shadows

`boxShadow` takes the mockup's CSS shadow across 1:1 on both platforms — `boxShadow: '0px 4px 12px rgba(0,0,0,0.15)'`, or the array-of-objects form. It is New Architecture only, which is the only architecture React Native ships from 0.82, so it is the default choice and the one that keeps Android matching the design.

Legacy fallback, for a project on an older React Native or still on the old architecture: iOS reads `shadowColor`, `shadowOffset`, `shadowOpacity` and `shadowRadius`; Android reads `elevation` and derives its own blur and offset from it, so the design's exact shadow lands as an approximation there. Set both, and give the shadowed view a `backgroundColor` so `elevation` draws at all.

### Gradients and hover

- **Gradients.** Use `expo-linear-gradient` for linear gradients, or preserve an existing native gradient implementation supported by the installed React Native version. Check native API support before porting CSS gradient strings directly.
- **Hover.** `hover:` variants compile but never fire on a touch device. Carry the intent to the press state instead: `Pressable`'s `({ pressed })` style callback, or the `active:` variant.

### Infer the route tree first

Do this before implementing any screen. Each Sleek screen is a standalone document that draws the entire chrome, so the same tab bar is baked into every screen that shows one. Read the screen set as one app, derive the navigator from it, and let each screen render only its own content — the navigator owns the chrome and draws it once.

Preserve the app's existing navigation system. For a new Expo app with no navigator, use **expo-router**; in that branch, import each navigator from the path that actually exports it:

- `Stack` from `expo-router`.
- `Tabs` from `expo-router/js-tabs` on Router 57+. Older Router versions, including Router 6 on SDK 54, export `Tabs` from `expo-router` and may not have `js-tabs`. Check the installed package's exports before choosing the import; preserve the project's SDK unless an upgrade is part of the task.
- `Drawer` from `expo-router/drawer` — the root exports no `Drawer` at all. It also needs `react-native-reanimated`, `react-native-worklets` and `react-native-gesture-handler` (SDK 56+) installed alongside it.

Four signals carry the structure:

1. **The same bottom bar means Tab siblings** — those screens sit under one `_layout` sharing a single tab bar.
2. **A back arrow in the header means a Stack child** of wherever the screen is reached from.
3. **An overlay, sheet or dialog needs a presentation choice.** Use a modal route when it belongs in navigation history, or a screen-local component when it belongs to the current interaction. Match the existing app's behavior.
4. **A hamburger menu means a Drawer**, with its destinations as Drawer screens.

Worked example: Home, Search and Profile all show the same bottom bar → tab siblings. ProductDetail shows a back arrow → Stack child inside the Home tab. Settings is reached from a menu → Stack or Drawer screen.

Two lookalikes are components rather than routes: horizontal tabs inside the content area (`react-native-tab-view`, or a plain segmented control), and a floating action button (an absolutely positioned `Pressable`).

Tab bar and header styling go through `screenOptions` for as long as the design stays inside what those options express; a custom shape, a floating or pill-shaped bar, a raised center button, a custom active indicator or a header carrying a logo and search field is cheaper as a component passed to the `tabBar` prop or `options.header`. Either way, match the icons, labels and active/inactive states from the design.

### Keyboard

Every screen with a `TextInput` needs keyboard avoidance, such as `KeyboardAvoidingView`. Verify the focused fields, errors and form actions remain reachable with the native keyboard open.

### Build the feature, not the mockup

A mockup shipped verbatim is a convincing dead app. For each screen, name the feature the UI represents, then wire it: every value that would change at runtime reads from state, props, context or the API; buttons run actions; forms validate and submit; lists fetch; navigation carries params between screens. The hardcoded name, the "5", the highlighted tab and the pre-filled form are all the same job.

A mockup also shows one moment — full, happy, populated. Decide separately what each screen does with zero items, with one, while loading, and on error.

When editing a persisted record, resolve it from the route parameters and initialize fields after its data loads; test reopening and reloading the route so saved fields do not become blank defaults.

Preserve small icons visually while giving their controls usable touch areas.

If the app also targets web, verify that inactive routes and covered modal content cannot receive keyboard focus; `aria-hidden` alone does not prevent it.

---

## Definition of done

For a native build, apply each relevant check to every screen in the requested implementation scope and any shared navigation it changes. Completion requires those checks to pass; report unavailable platform checks as unverified dependencies.

- [ ] The screen has a route in the navigator, and its own file draws none of the chrome the navigator owns.
- [ ] Every `flex` in the source HTML resolved to an explicit direction, and the built screen's axis matches its screenshot.
- [ ] Review JSX text and conditional branches: every rendered string is inside `<Text>`. Exercise populated and empty branches on native and check the development console for `Text strings must be rendered within a <Text> component.`; grep alone cannot prove the parent element.
- [ ] Headerless screens and modals inset their uncovered edges, every screen's bottom edge is accounted for on Android, the applicable native view hierarchy has a `SafeAreaProvider`, and nothing is inset twice.
- [ ] Colors, radii and fonts come from the one token module — grepping the screen files for a raw hex or `rgb(` returns nothing.
- [ ] Icons are the exact Iconify names from the HTML, set and name.
- [ ] Fonts match the selected theme's families, bundled at the weights the body's `font-*` classes use.
- [ ] Shadows render on Android as well as iOS: `boxShadow`, or `elevation` alongside the `shadow*` props on a legacy-architecture project.
- [ ] Gradients visibly match the design on native through `expo-linear-gradient` or a supported native gradient implementation.
- [ ] Every `hover:` in the source HTML landed on a press state: `Pressable`'s `({ pressed })` callback or `active:`.
- [ ] Every screen holding a `TextInput` uses keyboard avoidance. With the native keyboard open, focus each field and trigger validation: errors and Save/Cancel remain visible or reachable by scrolling. A `KeyboardAvoidingView` in the source alone does not prove this.
- [ ] Grep each screen for display literals — quoted strings and numbers that reach the user — and account for every hit: it reads from state, props, context or the API, or it is genuinely static copy.
- [ ] Exercise applicable populated, empty, single-item, loading and error states. On native, check that filtering to one item does not unexpectedly expand scroll containers or hide the remaining item.
- [ ] When using Nativewind or Uniwind, one throwaway class renders visibly before building screens, including a third-party wrapper if the implementation needs one.
- [ ] The screen has been compared against its own **review shot**, not the user shot.
- [ ] Record which platforms actually ran the app and which only bundled/exported. Verify styling and interactions on an available native target; a web preview or native export alone does not prove native layout, insets or keyboard behavior.
