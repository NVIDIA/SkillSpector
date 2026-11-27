---
name: video-edit
displayName: "Video Edit — Pro Pack on RunComfy"
description: >
  Edit existing video on RunComfy — this skill is a smart router that
  matches the user's intent to the right edit model in the RunComfy
  catalog. Picks Wan 2.7 Edit-Video (general restyle / background swap
  / packaging swap, identity + motion preservation), Kling 2.6 Pro
  Motion Control (transfer precise motion from a reference video to a
  target character), or Lucy Edit Restyle (lightweight identity-stable
  restyle / outfit swap). Bundles each model's documented prompting
  patterns so the skill gets sharper edits without burning iterations
  on the wrong model. Calls `runcomfy run <vendor>/<model>/<endpoint>`
  through the local RunComfy CLI. Triggers on "video edit", "edit
  video", "restyle video", "swap video background", "motion control",
  "outfit swap video", or any explicit ask to transform a video.
homepage: https://www.runcomfy.com
license: MIT
---

# Video Edit — Pro Pack on RunComfy

[runcomfy.com](https://www.runcomfy.com/?utm_source=skills.sh&utm_medium=skill&utm_campaign=video-edit) · [Wan 2.7 Edit-Video](https://www.runcomfy.com/models/wan-ai/wan-2-7/edit-video?utm_source=skills.sh&utm_medium=skill&utm_campaign=video-edit) · [Kling Motion-Control Pro](https://www.runcomfy.com/models/kling/kling-2-6/motion-control-pro?utm_source=skills.sh&utm_medium=skill&utm_campaign=video-edit) · [Lucy Edit Restyle](https://www.runcomfy.com/models/decart/lucy-edit/restyle?utm_source=skills.sh&utm_medium=skill&utm_campaign=video-edit) · [GitHub](https://github.com/agentspace-so/runcomfy-skills/tree/main/video-edit)

**Video edit, intent-routed.** This skill doesn't lock you to one model — it picks the right video-edit model in the RunComfy catalog based on what the user actually wants: general restyle, motion transfer from a reference clip, or lightweight identity-stable outfit / background swap.

```bash
npx skills add agentspace-so/runcomfy-skills --skill video-edit -g
```

## Pick the right model for the user's intent

| User intent | Model | Why |
|---|---|---|
| Restyle a talking-head video — preserve face / pose / lip movement | **Wan 2.7 Edit-Video** | Strong identity + motion preservation; supports up to 1080p |
| Swap product background, keep camera motion | **Wan 2.7 Edit-Video** | Camera motion preserved; one-direction edit honored |
| Replace packaging design using a reference image | **Wan 2.7 Edit-Video** + `reference_image` | Reference-conditioned design transfer |
| Apply cinematic color grade / commercial polish | **Wan 2.7 Edit-Video** | Good at single-direction global look changes |
| **Transfer precise motion** from a reference video to a target character | **Kling 2.6 Pro Motion Control** | Designed for motion mapping with identity hold |
| Lip-sync motion of a target character to source video's lip movement | **Kling 2.6 Pro Motion Control** | Built for tight temporal coherence |
| **Lightweight outfit / costume swap** with identity preservation | **Lucy Edit Restyle** | Core strength is localized identity-stable edits |
| **Identity-stable restyle** ("astronaut in desert", "warm golden-hour lighting") | **Lucy Edit Restyle** | Specializes in temporal consistency for restyle |
| Default if unspecified | **Wan 2.7 Edit-Video** | Most versatile, highest resolution |

The agent reads this table, classifies the user's intent, and picks the matching subsection below.

## Prerequisites

1. **RunComfy CLI** — `npm i -g @runcomfy/cli`
2. **RunComfy account** — `runcomfy login`.
3. **CI / containers** — set `RUNCOMFY_TOKEN=<token>`.
4. **A source video URL** — formats and limits depend on the chosen route.

---

## Route 1: Wan 2.7 Edit-Video — default for restyle / background / packaging

**Model**: `wan-ai/wan-2-7/edit-video`

### Schema

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `prompt` | string | yes | — | Lead with preservation. One edit direction per call. |
| `video` | string | yes | — | MP4/MOV URL, 2–10s, ≤100MB. |
| `reference_image` | string | no | — | URL — use for direct design / appearance transfer only. |
| `resolution` | enum | no | (input) | `720p` or `1080p`. |
| `aspect_ratio` | enum | no | (input) | W:H. Defaults to input. |
| `duration` | int | no | 0 | `0` = match input; `2–10` = truncate from start. |
| `audio_setting` | enum | no | `auto` | `auto` or `origin` (preserve source audio). |
| `seed` | int | no | — | Reproducibility. |

### Invoke

**Background swap, identity preserved, audio kept:**

```bash
runcomfy run wan-ai/wan-2-7/edit-video \
  --input '{
    "prompt": "Preserve the speaker'\''s face, pose, and lip movement; change the background to a modern office with neutral lighting.",
    "video": "https://.../speaker.mp4",
    "audio_setting": "origin"
  }' \
  --output-dir <absolute/path>
```

**Packaging swap with reference image:**

```bash
runcomfy run wan-ai/wan-2-7/edit-video \
  --input '{
    "prompt": "Maintain the original framing and hand movement; replace the packaging design using the reference image.",
    "video": "https://.../hand-holding-package.mp4",
    "reference_image": "https://.../new-packaging.png",
    "audio_setting": "origin"
  }' \
  --output-dir <absolute/path>
```

### Prompting tips

- **Preservation goals first**: `"Preserve [face / pose / motion / framing / lip movement]; [then state the change]"`.
- **One edit direction per call.** Compound edits drift on motion.
- **`reference_image` only when justified** (packaging swap, costume swap with target visual). Don't pass refs for general restyle.
- **`audio_setting: "origin"`** for talking-head where you don't want soundtrack regenerated.
- **Source video constraints**: 2–10s, ≤100MB.

---

## Route 2: Kling 2.6 Pro Motion Control — when motion FROM a reference clip is the point

**Model**: `kling/kling-2-6/motion-control-pro`

Use when the user wants to **transfer the motion of a reference video** onto a target character (driven by an image OR another video). This isn't restyle — it's motion mapping with identity hold.

### Schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | Describe target motion / style. |
| `image` | string | yes (image orientation) | Reference for character / background consistency. |
| `video` | string | yes | **Motion reference**. 10–30s depending on orientation. |
| `keep_original_sound` | bool | no | Preserve audio from reference video. |
| `character_orientation` | enum | yes | `image` (max 10s output) or `video` (max 30s output). |

### Invoke

```bash
runcomfy run kling/kling-2-6/motion-control-pro \
  --input '{
    "prompt": "A young american woman dancing",
    "image": "https://.../target-character.jpg",
    "video": "https://.../motion-reference-dance.mp4",
    "character_orientation": "image",
    "keep_original_sound": true
  }' \
  --output-dir <absolute/path>
```

### Prompting tips

- **Subject must be > 5% of frame** in the image reference for clean identity hold.
- **Spatial constraints help**: `"character on left side, background motion right"`.
- **Simplify** if results drift between iterations — drop adjectives, keep core motion description.
- **`character_orientation: "image"`** caps output at 10s; `"video"` allows 30s.

---

## Route 3: Lucy Edit Restyle — lightweight identity-stable restyle / outfit swap

**Model**: `decart/lucy-edit/restyle`

Use when the edit is **localized style modification** — outfit swap, scene relight, atmospheric restyle — and identity preservation is critical. Lighter-weight than Wan 2.7 Edit; capped at 720p.

### Schema

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `prompt` | string | yes | — | Natural-language edit instruction. |
| `video_url` | string | yes | — | MP4/MOV/WEBM/GIF. |
| `resolution` | enum | no | `720p` | `720p` only on this tier. |

### Invoke

**Outfit swap:**

```bash
runcomfy run decart/lucy-edit/restyle \
  --input '{
    "prompt": "Change outfit to professional business attire; preserve face and motion.",
    "video_url": "https://.../subject-walking.mp4"
  }' \
  --output-dir <absolute/path>
```

**Atmospheric restyle:**

```bash
runcomfy run decart/lucy-edit/restyle \
  --input '{
    "prompt": "Make lighting warm and golden hour; preserve face, pose, and motion.",
    "video_url": "https://.../subject-portrait.mp4"
  }' \
  --output-dir <absolute/path>
```

### Prompting tips

- **Localized change phrasing wins.** "Outfit", "lighting", "background" — pick one bucket.
- **Preserve identity goals** — `"preserve face and motion"` is enough; don't over-specify.
- **Avoid total replacement** ("astronaut in space" works; "swap subject for a different person" doesn't). Lucy is built for localized style mods, not full character swap.
- **No aspect ratio control** — output matches input. Cropping happens server-side if you don't pre-match.

---

## Limitations

- **Each route inherits its model's limits.** Wan 2.7 Edit: 2–10s, 1080p ceiling. Kling: 10s (image orientation) or 30s (video orientation). Lucy: 720p ceiling, no aspect control.
- **No multi-route blending.** This skill picks one model per call.
- **Brand-specific overrides** — if the user named a specific model, route to the corresponding brand skill (`wan-2-7`) for fuller treatment.

## Exit codes

| code | meaning |
|---|---|
| 0  | success |
| 64 | bad CLI args |
| 65 | bad input JSON / schema mismatch |
| 69 | upstream 5xx |
| 75 | retryable: timeout / 429 |
| 77 | not signed in or token rejected |

Full reference: [docs.runcomfy.com/cli/troubleshooting](https://docs.runcomfy.com/cli/troubleshooting?utm_source=skills.sh&utm_medium=skill&utm_campaign=video-edit).

## How it works

The skill picks one of Wan 2.7 Edit-Video / Kling 2.6 Pro Motion Control / Lucy Edit Restyle based on user intent and invokes `runcomfy run <model_id>` with the matching JSON body. The CLI POSTs to the Model API, polls the request, fetches the result, and downloads any `.runcomfy.net`/`.runcomfy.com` URL into `--output-dir`. `Ctrl-C` cancels the remote request before exit.

## Security & Privacy

- **Token storage**: `runcomfy login` writes the API token to `~/.config/runcomfy/token.json` with mode 0600 (owner-only read/write). Set `RUNCOMFY_TOKEN` env var to bypass the file entirely in CI / containers.
- **Input boundary**: the user prompt is passed as a JSON string to the CLI via `--input`. The CLI does NOT shell-expand the prompt; it transmits the JSON body directly to the Model API over HTTPS. No shell injection surface from prompt content.
- **Third-party content**: image / mask / video URLs you pass are fetched by the RunComfy model server, not by the CLI on your machine. Treat external URLs as untrusted; image-based prompt injection is a known risk for any image-edit / video-edit model.
- **Outbound endpoints**: only `model-api.runcomfy.net` (request submission) and `*.runcomfy.net` / `*.runcomfy.com` (download whitelist for generated outputs). No telemetry, no callbacks.
- **Generated-file size cap**: the CLI aborts any single download > 2 GiB to prevent disk-fill from a malicious or runaway model output.
