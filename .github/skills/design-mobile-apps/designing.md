# Designing

Request and response shapes go beyond the fields named here; read the [API reference](endpoints.md) when you need a field this file doesn't name.

## 1. Create a project

Create a project with `POST /api/v1/projects` if one doesn't exist yet.

Each project has its own theme, style, and design system, so a project holds one style direction at a time. Even so, keep design variations in one project by default: project count is plan-capped (free 1, Starter 5), and `POST /api/v1/projects` answers a capped account with `403 FORBIDDEN` at `projects_limit`. That 403 carries no `data.url`, so name the cap and the upgrade to the user yourself. Spread variations across projects only when the user asks to see them side by side — and check the headroom with `GET /api/v1/projects` before you start.

## 2. Send a chat message

Send the request with `POST /api/v1/projects/:id/chat/messages`. Sleek plans screen content and layout from your message, and invents a visual style when you give it none. Send the full intent as a single message in the user's own terms: Sleek does the decomposition into screens itself and produces richer designs when given room to plan. If the user described specific screens, include those.

**Author a style direction**: write one whenever the user has given you anything to ground it in — reference images, apps they like, vibe adjectives, things to avoid — or whenever you're producing variations, one direction per variation. Pass the request through unchanged only when it's bare. A style direction is a single comprehensive paragraph, included in the message, covering mood (2–3 adjectives), color strategy (the logic, not hex codes), typography feel, layout philosophy, component style (radii, borders vs shadows, nav treatment), imagery and illustration style, and one or two distinctive details. Commit to a palette, a type direction, and an overall feel — anything that only sets a mood reads as a hint, not a direction. Put the personality in color, type, and imagery rather than in unusual layout or navigation.

Extend what the user gave you and keep every constraint they set. When they point at reference images or apps they like, study each one and carry what you take into the direction — Sleek only sees images passed as `imageUrls`, so for anything local the direction is how those references reach it. Borrow patterns, and leave the source's branding, content, and name behind.

Use a style direction or a `referenceId`, not both — a reference already carries a full style guide of its own.

**Seed a style with a reference**: Sleek curates a catalog of design references. When the user wants a specific look or asks for style options, list them with `GET /api/v1/references` (each has a `name` and `previewImageUrls` you can show) and pass the chosen id as `referenceId` on the first message to a project, so its style guide seeds the whole design.

**Identify your tool**: always send `source`, the slug of the tool making the request. The Sleek editor uses it to show the user who is designing while the run streams. Recognized values: `claude-code`, `claude`, `codex`, `chatgpt`, `cursor`, `openclaw`, `grok`. If your tool isn't listed, send a short kebab-case slug for it anyway (max 64 chars). Unrecognized values are fine and get a generic label.

**Watch it live**: runs render in the Sleek editor in real time. After sending the first message to a project, tell the user they can watch their screens being designed live in Sleek, and share the editor link: `https://sleek.design/project/:projectId`. Open a browser yourself only when the user asks you to.

**Polling**: chat messages are async by default: you get a `runId` and poll `GET /api/v1/projects/:id/chat/runs/:runId`. Start at a 2s interval, back off to 5s after 10s, and poll for up to **10 minutes**. Exit on `completed` or `failed`; if you can't read the status, stop and report that rather than counting it as "not done yet". After ten minutes, report the last known status and run ID. Elapsed time does not prove completion or release the project's run lock; don't start another attempt while its outcome is unknown. You can also use `?wait=true` for a blocking call (up to 300s; falls back to polling if it times out with `202`).

Save the request body, idempotency key, `runId` and `statusUrl` per project as soon as they are available. A `409` carries no active-run pointer and the API has no run-list endpoint. If the blocking run's ID is unknown, report the conflict and request its run ID or editor context; don't invent a polling URL or cancel an unidentified run.

A `failed` run can still carry a `result`: Sleek records the operations it applied before the failure. Read `result` on both terminal statuses and show the created or updated screens through step 3. Report the failure alongside the applied changes. Continue to implementation only when the user requested code.

**Editing a specific screen**: use `target.screenId` to direct changes to the right screen. Read it from the run's `result.operations` or the component list's `screenId` field, which can be null. It is distinct from the component ID; do not substitute that ID when no screen ID is returned.

**One run at a time**: keep one active run per project. Follow a known run to a terminal status before sending the next message; use the missing-context path above when its ID is unknown. If the user changes direction, cancel the identified run (see [Cancel Run](endpoints.md#chat-cancel-run)). Messages to different projects can run in parallel; use async polling when running them concurrently.

**Safe retries**: choose an `idempotency-key` header (≤255 chars) before sending a message. If the response is lost, resend the same payload with the same key to recover the original run. Reusing a key also returns its terminal failure: after `request_message_persist_failed`, or after the user resolves `out_of_credits`, a new logical attempt needs a new key. Read the existing run first; don't create a new attempt while its outcome is unknown.

## 3. Show the results

After every chat run that produces `screen_created` or `screen_updated` operations, **take screenshots and show them to the user** using `POST /api/v1/screenshots`. The step is done once the user has seen a **user shot** of every screen the run created or updated, and each run ends with the user having seen what it changed.

- **New screens**: one screenshot per screen + one combined screenshot of all screens in the project.
- **Updated screens**: one screenshot per affected screen.

Save screenshots in the project directory (not a temporary folder) so the user can easily view them.

**Review your own work with a review shot.** The user shot's phone-shaped framing is the point when you're showing a screen to the user, and the wrong framing for judging it: everything below the fold is cropped away. Re-shoot with `fullHeight: true`, one screen per request, when you're checking what a run actually produced.

Screenshot requests are independent, so issue them in parallel — the user shot and the review shot go out together, as do the shots for different screens. "One screen per request" governs what goes into each image, not how fast you send them. Back off only if you actually get a `429`.

**Confirm missing content.** Content that looks missing in a user shot can be below the fold. Review the full-height shot first. Before reporting something absent or asking Sleek to add it, check the component HTML from `GET /api/v1/projects/:id/components/:componentId` too: rendering can omit content past its height cap or inside collapsed sections and carousel slides. An element present in HTML can still be invisible to the user.

---

When the user asks for these screens as code, read [implementing.md](implementing.md) before writing the implementation.
