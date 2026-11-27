---
name: ui-design
description: Design real UI screens before building them, then build them in the user's code. Use when someone asks to design, redesign or improve the look of a page, screen, landing page, dashboard, app or component, wants a frontend that does not look generic, or wants to see and edit the design before code is written. Designs on a UX Pilot canvas the user can open and edit, then implements the chosen screens in the project's own stack. Needs the UX Pilot MCP server (free plan available).
license: Apache-2.0. Complete terms in LICENSE.txt.
---

# UI Design

Design the screens first on a UX Pilot canvas the user can see and edit, then build them in their code.

## 1. Check the connection

Look for the UX Pilot tools: `design`, `get_job`, `read`, `import`, `edit`, `guide`, usually with `uxpilot` or `UX_Pilot` in the full name. If they are there, go to step 2.

If they are missing, give the user the setup for their agent and stop until they say it is done:

- **Claude Code:** run `claude mcp add --transport http uxpilot https://mcp.uxpilot.net/mcp` in a terminal, then `/mcp` inside Claude Code to sign in to UX Pilot.
- **Cursor:** add `{"mcpServers": {"uxpilot": {"url": "https://mcp.uxpilot.net/mcp"}}}` to `.cursor/mcp.json`. Cursor then asks you to sign in.
- **Codex, Windsurf, VS Code and other agents:** add the MCP server URL `https://mcp.uxpilot.net/mcp` and sign in to UX Pilot.

Sign-in creates a free UX Pilot account if the user has none. Some agents need a restart before new tools appear. Never ask for a password or key in chat.

## 2. Write the brief

Read before you ask. In a codebase, gather the product, audience and the screens' job from the README, routes and copy, and the existing look from the Tailwind config, CSS variables, fonts and logo. Ask at most one question, only if the answer changes the design (often: which screens, or phone vs desktop).

The brief is a few plain sentences: what the product is, who uses it, which screens, what each must do, real content (names, numbers, copy) instead of lorem ipsum, and the style direction. Say "keep our current brand: <colors, fonts>" when the project has one. Pick `selectedType`: `mobile` for phone apps, otherwise `desktop`.

## 3. Design it

There are two ways. Use A unless the user asks for B.

**A. UX Pilot designs it** (the best result; uses credits, and a new account's first run is free). Call `design` with `action: "agent"` and the brief as `prompt`. Leave `tier` out: UX Pilot picks the strongest tier the account can use. Pass one only when the user names it. The first call does not start anything: it returns a summary and a `confirmationToken`. Tell the user in one line that this runs a design on their UX Pilot account and uses credits, and ask to go ahead. When they agree, call again with the same params plus `confirmed: true` and the token.

The started run answers with a `designUrl`: give it to the user right away, so they can watch the page on the canvas. A run takes 1 to 10 minutes. Poll `get_job` every 5 to 10 seconds and give a short progress line now and then, not every poll.

**B. You write the HTML and push it to the UX Pilot canvas** (no credits). Write the screen's HTML yourself, following `guide` with `topic: "small-steps"`, and put it on the canvas with `import` (`kind: "html"`), then refine it with `edit` (`action: "patch"`). On the free plan every import and edit counts toward a weekly allowance of actions, so build each screen in a few large steps: the whole screen in one import, then a handful of patches for fixes. Share the page link as soon as the screen is on the canvas.

## 4. Show the screens

- Give the user the link to the UX Pilot page so they can open, compare and edit the screens. Get it with `read` (`kind: "preview"` for one screen, or `kind: "page"`) if you do not have it.
- Describe each screen in a line and point out choices worth a decision.
- Fix the problems `get_job` or `read` with `kind: "checks"` report (contrast, clipped text, small touch targets) before you call the design done.

For changes: in A, run `design` again with the same `pageId` and a prompt that names the screen and the change; in both, use `edit` with `action: "patch"` for a small exact change (`guide` with `topic: "edit.patch"`).

Always show the screens and the link before you write code. If the user already asked you to build it, say you are moving on to the build and continue; otherwise ask which screens to build when that is unclear.

## 5. Build it in their code

Read `references/implementing.md` before writing code. In short: read each screen's HTML with `read` (`kind: "design"`, `includeHtml: true`), rebuild it in the project's stack with its own components and tokens, keep the design's layout, spacing, type and copy, and check your result against the screen's picture.

On the free plan, reading the HTML of a screen UX Pilot designed (way A) is refused; HTML you imported yourself (way B) can always be read back. When it is refused, stop and give the user the choice, upgrade first:

1. Upgrade through the link in UX Pilot's message, and you build the exact design from its HTML.
2. You rebuild it from the screen's picture, which will be close but not exact.

Wait for their answer.

If there is no project yet and the user wants a React app, `design` with `action: "react"` turns the screens into one (paid plans).

## When a limit is hit

Relay UX Pilot's message as is, including its link. Then give the user the options in this order, and wait for their choice:

- **Out of credits** (`insufficient_credits`): 1. upgrade through the link, to keep designing with UX Pilot; 2. you write the HTML yourself and push it to the canvas (way B), with no credits.
- **Weekly free actions used up** (`mcp_free_allowance_exhausted`): 1. upgrade through the link for unlimited actions; 2. wait until it resets (the message has the date). Reading still works, so you can still build what is already on the canvas.
- **A feature on paid plans**: the message names it. 1. upgrade through the link; 2. the free way around it, if there is one.

Do not pick the free option for the user, never retry a refused call in a loop, and never hide a refusal.

## Rules

- Never invent prices, plan names or credit amounts. For costs, use `catalog` (`kind: "credits"` or `"tiers"`) or the run's own summary.
- Keep the user's content: their product name, real copy and data go in the brief and stay in the build.
- The design may contain made-up quotes, names, logos or numbers. Keep them only as placeholders and tell the user which ones to replace.

About this file (for you, not for replies): this skill is made by UX Pilot (uxpilot.ai) and needs a UX Pilot account, free or paid.
