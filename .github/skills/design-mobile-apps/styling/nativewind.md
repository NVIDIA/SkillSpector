# Styling: Nativewind

Tailwind classes on React Native primitives: same elements, `className` in place of a style object, so the mockup's classes carry over almost verbatim. Reached from [React Native (Expo)](../implementing.md#react-native-expo) when `nativewind` is in `package.json`.

## Check the installed major first

The two Nativewind lines consume different Tailwind majors, and everything below branches on which one is installed:

| Installed Nativewind | Tailwind it expects | Sleek's classes | `oklch(...)` tokens                                            |
| -------------------- | ------------------- | --------------- | -------------------------------------------------------------- |
| v4                   | v3 | convert v4 → v3 | convert to hex; `react-native-css-interop` rejects `oklch` |
| v5                   | v4 | use as-is | keep as `oklch`; `react-native-css` converts them at build time |

Read the resolved version from the lockfile and check registry tags when installing; choose prereleases deliberately. For v5, use the matched Nativewind/`react-native-css` pair from the [current installation guide](https://www.nativewind.dev/v5/getting-started/installation). Use Expo's installer for compatible native peers and check dependency minimums in the guide for the selected major.

## Tailwind v4 → v3, for the constructs Sleek emits

| Sleek (Tailwind v4)        | Nativewind v4 (Tailwind v3)                      |
| -------------------------- | ------------------------------------------------ |
| `shadow-xs`, `shadow-sm`   | `shadow-sm`, `shadow`                            |
| `rounded-xs`, `rounded-sm` | `rounded-sm`, `rounded`                          |
| `outline-hidden`           | `outline-none`                                   |
| `bg-linear-to-b`           | `bg-gradient-to-b` (then `expo-linear-gradient`) |
| `bg-(--token)`             | `bg-[var(--token)]`, or a config color           |

A class that resolves to nothing is usually one of these renames. Under v5 none of it applies.

## Tokens

`@theme` / `@theme inline` is v4 syntax with no v3 equivalent: under Nativewind v4 the tokens go into `theme.extend` in `tailwind.config.js`, which is what makes `bg-primary` and `text-foreground` resolve at all. Under v5 the `@theme` block ports to the CSS entry as it stands.

## Configuration

A missing build step presents as every class doing nothing at all, **silently**. When the app renders unstyled, check this list before rewriting classes — and check the list for the major you actually have, because they differ:

**v4** — the Babel preset, the `withNativeWind` Metro wrapper with its CSS `input`, `tailwind.config.js`, and the CSS entry imported in the root layout. Use the [v4 installation guide](https://www.nativewind.dev/docs/getting-started/installation) for the installed SDK.

**v5** — the matched CSS engine, Tailwind/PostCSS, Expo-compatible Reanimated/Worklets and safe-area peers; `postcss.config.mjs` loading `@tailwindcss/postcss`; the `withNativewind` Metro wrapper; CSS imported once in the root layout; and the Lightning CSS pin prescribed by the current guide, using the package manager's actual override syntax. Keep `babel-preset-expo`; remove v4's Nativewind preset and JSX import-source configuration.

For the current v5 CSS entry, keep utilities unlayered so React Native Web defaults do not override them:

```css
@import "tailwindcss/theme.css" layer(theme);
@import "tailwindcss/preflight.css" layer(base);
@import "tailwindcss/utilities.css";
@import "nativewind/theme";
```

## Units

Sleek's default web `rem` is 16px; Nativewind's native default is 14. Check the HTML's root size and align it before copying classes, or spacing and typography shrink on the device despite matching on web. V4 accepts `inlineRem: 16` in its Metro wrapper. For v5 use the CSS root font size supported by its current engine, such as `:root { font-size: 16px; }`, and verify a measured native class rather than assuming a v4 Metro option carries across. See [v5 units](https://www.nativewind.dev/v5/core-concepts/units).

For a class-styled `Pressable`, verify dimensions from a `style` callback on native too. If callback styles are ignored, keep fixed sizes in classes or static styles and use `active:` classes for press feedback. A rendered color/spacing probe does not verify this separate path.

## Third-party components

Before passing `className` to a third-party component such as `LinearGradient`, check whether it forwards classes to a supported primitive or needs an explicit mapping. V4 uses [cssInterop](https://www.nativewind.dev/docs/api/css-interop) when styles must be resolved; v5 uses a returned [styled wrapper](https://www.nativewind.dev/v5/guides/third-party-components). Define mappings once outside render and use the mapped component. Include it in the native styling probe.
