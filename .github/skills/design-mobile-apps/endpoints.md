# Endpoints

Full contract for every row in the quick-reference table in [SKILL.md](SKILL.md).

Every list endpoint accepts `limit` (1–100, default 50) and `offset` (≥0), and answers with `pagination.total` so you can page through everything.

## Projects

### List projects

```http
GET /api/v1/projects?limit=50&offset=0
Authorization: Bearer $SLEEK_API_KEY
```

Response `200`:

```json
{
  "data": [
    {
      "id": "Nq4bZp8wLcT",
      "name": "My App",
      "slug": "my-app",
      "createdAt": "2026-01-01T00:00:00Z",
      "updatedAt": "..."
    }
  ],
  "pagination": { "total": 12, "limit": 50, "offset": 0 }
}
```

### Create project

```http
POST /api/v1/projects
Authorization: Bearer $SLEEK_API_KEY
Content-Type: application/json

{ "name": "My New App" }
```

Response `201`: `{ "data": ... }` with a single project.

A plan-capped account gets `403 FORBIDDEN` at `projects_limit` here, and that body carries no `data.url`.

### Get / Delete project

```http
GET    /api/v1/projects/:projectId
DELETE /api/v1/projects/:projectId   → 204 No Content
```

---

## Components

### List components

```http
GET /api/v1/projects/:projectId/components?limit=50&offset=0
Authorization: Bearer $SLEEK_API_KEY
```

Both list and get accept an optional `inlineIcons` query param (default `false`). When omitted, icons render as `<iconify-icon>` web components and the HTML pulls in the Iconify script, so leave it off by default. Pass `?inlineIcons=true` only when the consumer needs self-contained SVGs in the HTML (for example, importing into tools that don't run scripts).

Response `200`:

```json
{
  "data": [
    {
      "id": "V1StGXR8Z5j",
      "screenId": "7hSvA2kLpEr",
      "name": "Hero Section",
      "activeVersion": 3,
      "versions": [
        {
          "id": "qkP2mN7bXsT",
          "version": 1,
          "code": "<!DOCTYPE html>...</html>",
          "createdAt": "..."
        }
      ],
      "createdAt": "...",
      "updatedAt": "..."
    }
  ],
  "pagination": { "total": 5, "limit": 50, "offset": 0 }
}
```

`activeVersion` can be null or unmatched. See [Which version to use](implementing.md#which-version-to-use) for code and screenshot selection. `id` is the component ID; nullable `screenId` addresses targeted chat edits.

### Get component

```http
GET /api/v1/projects/:projectId/components/:componentId
Authorization: Bearer $SLEEK_API_KEY
```

Response `200`: `{ "data": ... }` with a single component in the same shape as a list item.

---

## References

References are curated design styles from featured Sleek projects. They are world-readable: any valid API key can list them, no scope needed.

```http
GET /api/v1/references?limit=50&offset=0
Authorization: Bearer $SLEEK_API_KEY
```

Response `200`:

```json
{
  "data": [
    {
      "id": "3fTkYw6RmQv",
      "name": "Ember Fitness",
      "previewImageUrls": ["https://.../screenshot.png"]
    }
  ],
  "pagination": { "total": 44, "limit": 50, "offset": 0 }
}
```

To use one, pass its `id` as `referenceId` on [Send Message](#chat-send-message).

---

## Chat: Send Message

```http
POST /api/v1/projects/:projectId/chat/messages?wait=false
Authorization: Bearer $SLEEK_API_KEY
Content-Type: application/json
idempotency-key: <optional, max 255 chars>

{
  "message": { "text": "Add a pricing section with three tiers" },
  "source": "claude-code",
  "imageUrls": ["https://example.com/ref.png"],
  "target": { "screenId": "7hSvA2kLpEr" },
  "referenceId": "3fTkYw6RmQv"
}
```

| Field                    | Required | Notes                                                                                    |
| ------------------------ | -------- | ---------------------------------------------------------------------------------------- |
| `message.text`           | Yes      | 1+ chars, trimmed                                                                        |
| `source`                 | Treat as required | Slug of the tool sending the request (see [step 2 of Designing](designing.md#2-send-a-chat-message)) |
| `imageUrls`              | No       | HTTPS URLs only; included as visual context. Sleek's servers fetch these URLs, so pass ones you're willing to have Sleek read |
| `target.screenId`        | No       | Edit a specific screen using a non-null `screenId` from run operations or the components list, not `componentId`; omit to let AI decide |
| `referenceId`            | No       | Seed the design style from a reference (see [References](#references)); invalid id → `400` |
| `?wait=true/false`       | No       | Sync wait mode (default: false)                                                          |
| `idempotency-key` header | No       | Replay-safe re-sends                                                                     |

### Response: async (default, `wait=false`)

Status `202 Accepted`. `result` and `error` are absent until the run reaches a terminal state.

```json
{
  "data": {
    "runId": "kR9dLm2XvBc",
    "status": "queued",
    "statusUrl": "/api/v1/projects/Nq4bZp8wLcT/chat/runs/kR9dLm2XvBc"
  }
}
```

### Response: sync (`wait=true`)

Blocks up to **300 seconds**. Returns `200` when completed, `202` if timed out.

```json
{
  "data": {
    "runId": "kR9dLm2XvBc",
    "status": "completed",
    "statusUrl": "...",
    "result": {
      "assistantText": "I added a pricing section with...",
      "operations": [
        {
          "type": "screen_created",
          "screenId": "wG5xTn1PjHf",
          "screenName": "Pricing",
          "componentId": "V1StGXR8Z5j"
        },
        {
          "type": "screen_updated",
          "screenId": "7hSvA2kLpEr",
          "componentId": "8rJvL4wYhKd"
        },
        { "type": "theme_updated" }
      ]
    }
  }
}
```

`screenId` and `componentId` are distinct ids for the same screen: `screenId` addresses it in `target`, `componentId` addresses it in `GET .../components/:componentId` and in `componentIds` on a screenshot.

---

## Chat: Poll Run Status

```http
GET /api/v1/projects/:projectId/chat/runs/:runId
Authorization: Bearer $SLEEK_API_KEY
```

The response has the same `data` shape as send message: `result` when the run applied operations, `error` when `failed`.

```json
{
  "data": {
    "runId": "kR9dLm2XvBc",
    "status": "failed",
    "statusUrl": "...",
    "error": { "code": "execution_failed", "message": "..." }
  }
}
```

A `failed` run can carry **both** `error` and `result`: the server records the operations that were already applied before the failure. Read `result` on either terminal status.

**Run status lifecycle**: `queued` → `running` → `completed | failed`

---

## Chat: Cancel Run

```http
POST /api/v1/projects/:projectId/chat/runs/:runId/cancel
Authorization: Bearer $SLEEK_API_KEY
```

Marks a `queued` or `running` run as `failed` with error code `cancelled` and returns the updated run; already-finished runs are returned unchanged.

---

## Screenshots

Takes a snapshot of one or more rendered components.

```http
POST /api/v1/screenshots
Authorization: Bearer $SLEEK_API_KEY
Content-Type: application/json

{
  "componentIds": ["V1StGXR8Z5j", "8rJvL4wYhKd"],
  "projectId": "Nq4bZp8wLcT",
  "format": "png",
  "scale": 2,
  "gap": 40,
  "padding": 40,
  "background": "transparent"
}
```

`componentIds` takes `componentId` values, which the `screenId` from a chat operation is not.

| Field                       | Default       | Notes                                                                                                                                      |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`                    | `png`         | `png` or `webp`                                                                                                                            |
| `scale`                     | `2`           | 1–3 (device pixel ratio)                                                                                                                   |
| `gap`                       | `40`          | Pixels between components                                                                                                                  |
| `padding`                   | `40`          | Uniform padding on all sides                                                                                                               |
| `paddingX`                  | _(optional)_  | Horizontal padding; overrides `padding` for left/right when provided                                                                       |
| `paddingY`                  | _(optional)_  | Vertical padding; overrides `padding` for top/bottom when provided                                                                         |
| `paddingTop`                | _(optional)_  | Top padding; overrides `paddingY` when provided                                                                                            |
| `paddingRight`              | _(optional)_  | Right padding; overrides `paddingX` when provided                                                                                          |
| `paddingBottom`             | _(optional)_  | Bottom padding; overrides `paddingY` when provided                                                                                         |
| `paddingLeft`               | _(optional)_  | Left padding; overrides `paddingX` when provided                                                                                           |
| `background`                | `transparent` | Any CSS color (hex, named, `transparent`)                                                                                                  |
| `showDots`                  | `false`       | Overlay a dot grid on the background; light dots on dark backgrounds and dark dots on light ones. No effect when `background` is `transparent` |
| `fullHeight`                | `false`       | Expand each frame to the height of its own content before capturing — this is what makes a review shot                                      |
| `radius`                    | `48`          | Squircle corner radius per component in pixels (integer ≥ 0); pass `0` for sharp corners                                                   |
| `componentVersionOverrides` | _(optional)_  | Map of `componentId` → `versions[i].id` to render at a pinned version instead of `activeVersion` (see [Pinned versions](implementing.md#pinned-versions)) |
| `themeVersionOverrides`     | _(optional)_  | Map of `themeId` → `versions[i].id` to render with a pinned theme version (see [Pinned versions](implementing.md#pinned-versions))                        |

Padding resolves per-side → axis → uniform: `paddingTop` falls back to `paddingY`, which falls back to `padding`.

Frames are capped at **4× the default frame height**, so a screen longer than that is still cut off at the bottom even with `fullHeight: true`. On a very long screen, treat the component HTML as the authority for what's below the cap.

Response: raw binary `image/png` or `image/webp` with `Content-Disposition: attachment`.

---

## Share Cards

Renders one shareable image from up to four screens — the social-post framing, distinct from the plain screenshot.

```http
POST /api/v1/share-cards
Authorization: Bearer $SLEEK_API_KEY
Content-Type: application/json

{
  "componentIds": ["V1StGXR8Z5j", "8rJvL4wYhKd"],
  "projectId": "Nq4bZp8wLcT",
  "layout": "stack",
  "background": "midnight",
  "title": "Ember Fitness"
}
```

| Field                       | Default    | Notes                                                          |
| --------------------------- | ---------- | -------------------------------------------------------------- |
| `componentIds`              | _required_ | 1–4 components                                                 |
| `projectId`                 | _required_ | Project the components belong to                               |
| `layout`                    | `stack`    | `stack` or `row`                                               |
| `background`                | `midnight` | `midnight`, `ember`, `paper`, `violet`, `ocean`                |
| `title`                     | _(optional)_ | Trimmed, max 80 chars                                        |
| `componentVersionOverrides` | _(optional)_ | Same pinning semantics as screenshots                        |
| `themeVersionOverrides`     | _(optional)_ | Same pinning semantics as screenshots                        |

Response: raw binary `image/png`. Share cards and screenshots share one render budget — see the `429` row in [SKILL.md](SKILL.md#error-shapes).

---

## Device: Start

```http
POST /api/v1/device/start
Content-Type: application/json

{ "source": "claude-code" }
```

No auth. `source` is optional (1–64 chars) and drives the `<source> wants to connect` copy the user sees on the approval page.

Response `201`:

```json
{
  "data": {
    "deviceCode": "...",
    "userCode": "ABCD-1234",
    "verificationUrl": "https://sleek.design/...",
    "expiresIn": 900,
    "interval": 5
  }
}
```

## Device: Poll

```http
POST /api/v1/device/poll
Content-Type: application/json

{ "deviceCode": "..." }
```

No auth. `data.status` is one of `pending`, `expired`, or `approved` — and only the `approved` variant carries `key`, `keyId` and `name`. It is handed over exactly once; a poll on an already-claimed code answers `404 NOT_FOUND` / `Unknown device code`, which is indistinguishable from a code that never existed. The full flow is in [SKILL.md](SKILL.md#prerequisites-api-key).
