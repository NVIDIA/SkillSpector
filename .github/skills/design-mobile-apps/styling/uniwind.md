# Styling: Uniwind

Uniwind consumes Tailwind v4, so the mockup's classes carry over unchanged. Reached from [React Native (Expo)](../implementing.md#react-native-expo) when `uniwind` is in `package.json`.

## Theme

Uniwind resolves theme colors from `global.css`. Read the project's existing `global.css` and follow the structure already there. Sleek's own React Native export writes this shape, which is worth confirming against the current docs before relying on it:

```css
@import "tailwindcss";
@import "uniwind";

@layer theme {
  :root {
    @variant dark {
      --color-background: #000000;
      --color-foreground: #ffffff;
    }
    @variant light {
      --color-background: #ffffff;
      --color-foreground: #000000;
    }
  }
}
```

Port the project's `:root` tokens into `--color-*` names in whatever shape that file actually uses. Uniwind documents first-class OKLCH support, so the `oklch(...)` values go across as they are. Uniwind publishes an agent-readable digest at `https://docs.uniwind.dev/llms.txt`; read it for the current theme structure and for any setup detail beyond this file.

## Third-party components

Components from other libraries — `SafeAreaView` from `react-native-safe-area-context`, for instance — accept `className` once wrapped. Wrap each one once, in its own file, and import the wrapped version everywhere:

```tsx
import { SafeAreaView } from "react-native-safe-area-context";
import { withUniwind } from "uniwind";

export const StyledSafeAreaView = withUniwind(SafeAreaView);
```
