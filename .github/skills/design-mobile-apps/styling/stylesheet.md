# Styling: StyleSheet

React Native's built-in styling. Reached from [React Native (Expo)](../implementing.md#react-native-expo) when the repo has no styling library.

## Colors

React Native's color parser reads hex, `rgb()` and named colors and carries no `oklch` matcher, so a raw `oklch(...)` token resolves to nothing. Convert each one from the document's `:root` block to hex or `rgb()` once, in the token module, and every consumer stays correct.

## Units

React Native styles use unitless numbers in density-independent pixels. Resolve Tailwind's `0.25rem` spacing steps against the HTML's root font size; the examples below assume the default 16px root:

| Class          | Style                             |
| -------------- | --------------------------------- |
| `p-4`          | `padding: 16`                     |
| `px-5 py-3`    | `paddingHorizontal: 20, paddingVertical: 12` |
| `gap-3`        | `gap: 12`                         |
| `text-sm`      | `fontSize: 14`                    |
| `rounded-xl`   | `borderRadius:` the resolved `--radius-xl` number |
| `w-1/2`        | `width: '50%'` (percentages are strings) |

Resolve `rem` and `em` against their source font sizes. Use `useWindowDimensions()` for viewport-relative sizes, then pass numeric native values.

## Composition

- One `StyleSheet.create({ … })` per component file, holding everything static.
- Conditional styling composes with arrays: `style={[styles.card, isActive && styles.cardActive]}`.
- Values that genuinely change per render — a measured width, an animated height — stay inline; everything else lives in the sheet.
