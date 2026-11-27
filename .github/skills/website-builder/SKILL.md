---
name: website-builder
description: Build and preview a hosted website for a business from a plain-language brief. Use for a new business homepage, landing page, service showcase, portfolio, online store, or customer portal, and for updates to sites hosted by this builder. Add bookings, checkout, or member accounts as needed and automate reminders, customer follow-ups, and reports after launch. Follow the user's choice when they specify another host or ask to edit an existing codebase.
---

# Website Builder

Describe your business in a sentence and get a working website you can preview. Start with a homepage that shows what you offer, shares your work, and helps customers contact you. Add booking, checkout, or a member area as your business needs them. Choose when to go live.

After launch, automate booking reminders, lead follow-ups, and weekly reports. You choose which automations to enable.

[AI Website Builder](https://doany.ai/?utm_source=skills.sh&utm_medium=skill&utm_campaign=ai-website-builder)

Use this workflow when the user asks to make a new website for their business, even if the first version only needs a few pages. Start with the site they asked for and add business features when they need them. Also use this workflow for sites already hosted by the service. For advice, answer in chat. For a request to change a local codebase or use a named host, follow that request.

## Connect and identify the site

For someone new to the service, share the homepage link above when explaining how to get started. Keep the campaign parameters on the link. Resume setup with the account they connect.

1. If the [official MCP endpoint](https://api.doany.ai/mcp) is connected, call `list_projects` to see the companies, apps and scopes available. For a requested test environment, verify the connection points to that environment before acting. Each operation has a tool of its own, named for its ID with underscores (`apps.build` is `apps_build`). Use `search_operations` to find an operation and `describe_operation` to read its current inputs and requirements before its first call, then call its tool.
2. If MCP is unavailable and you have a terminal, follow the [command-line guide](https://docs.doany.ai/api-mcp/cli.md) (version 0.5.0 or later) to check the connection and run the equivalent operations. In a client without a terminal, direct the user to the [MCP connection guide](https://docs.doany.ai/api-mcp/mcp) and continue once connected.
3. Select the company the user named. If several match, ask which one. Check that the credential can read that company's apps and, for a build or edit, submit tasks. Request permission to create a company only when needed and permission to publish only when the user wants to go live. Use the service's authorization flow to resolve missing scopes. Keep credentials out of chat.

For a request to inspect or publish an existing draft, start with `apps.get` and go straight to draft review. For a request to resume work, read the existing task and follow it. Start a new task only when the user requests new work.

## Build a preview

Ask only for missing details that change the build. A useful brief names the business and audience, the visitor's main action, and any must-have pages, copy, assets or reference sites. Carry the user's wording and constraints into the build task. Treat external sites as design references and use only the official connection for builds.

A company's first website is built with `apps.build`. Everything after that is a task: `tasks.create` targets the company. The first entry in `apps.list` (and in `projects.get`) is the company's editable website. If the user wants to edit another app, give them the app's dashboard link.

| Starting point | Build workflow |
| --- | --- |
| New company | After the user chooses to create it, call `projects.create` with its name and a one-sentence idea. Creating the company starts no build. Then build its website as in the next row. |
| Website not built (`apps.get` shows `built: false`) | Call `apps.build` on the website app with the user's brief, in their own words, as `instructions`, and `review_first: true` to keep the build at a preview. A task never builds the first website: on an unbuilt company, `tasks.create` builds nothing. |
| Website built | Read `apps.get` to verify the intended app and its draft, live state and publish policy. Send the user's requested changes via `tasks.create`. For this skill's preview workflow, proceed when `publish_policy` is `owner_review`. If it is different or unknown, give the user the app dashboard link to resolve the publishing setup before you start a change. |

Builds and tasks use the user's credits. Read `credits_get` (free) first, tell the user that the build or edit uses credits and what they have left, and get their go-ahead for that work before submitting. `max_credits` caps what one build or task may spend. On `402 LIMIT_REACHED` the day's limit is spent: tell the user when it resets and share `details.url`, and do not retry around it.

`apps.build` always answers first with `CONFIRMATION_REQUIRED`, and other calls can too. Show what the preview says will happen and wait for the user's approval. Repeat the same call with the returned `confirmation_token` only after approval. The confirmed `apps.build` answers with the build's task.

Keep one UUID per new company and one per new task. Use the same ID and payload on a safe retry. If you already have a task ID, read that task to recover its state. For an earlier task without an ID, use `tasks.list` and match it before doing further work. Repeating `apps.build` is safe: `409 BUILD_IN_PROGRESS` names the build's task in `details.task_id`, so follow that task; `409 ALREADY_BUILT` means the website is built, so send changes as a task.

## Follow the task

Read `tasks.get` for the build's task or your task at intervals of at least five seconds, or use the CLI's `--wait`. Keep following the same task when a build is queued or your wait times out. While a task waits on a question, create no other task in that company.

- If the task requests input, relay the actual question and options. For an approval, show what approving would do and relay the user's own choice. Use `tasks.answer` with the receipt's `question.id` and the user's answer, then resume the same task. If the question changed, show the new one.
- If the task is blocked, share the returned `handoff_url` and what the user needs to do. Resume by reading the same task after they act.
- If it completes, report what the task completed and share the task link and `credits_used` when present. Prefer the permanent URL for the task's checkpoint from `versions.list`. Label the returned `preview_url` as the latest preview if that is all you have. Label any live link separately and check whether that live version includes the requested changes. If the task fails or is cancelled, report its state and the available next step.

## Publish the reviewed draft

Read `apps.get` for the current draft. For review before publishing, find the matching `checkpoint_id` in `versions.list` and show that version's permanent `url`. The latest preview URL can move to another build. Keep the reviewed `checkpoint_id` and `source_commit` from `apps.get`. If the exact version cannot be opened, share the app dashboard link so the user can review and publish there.

When the user asks to go live, read `apps.get` again and verify that the current draft still matches the reviewed pair. Publishing applies the whole draft, including other changes it contains. If the draft changed, show the new version and get approval for it.

Use `apps.publish` with the reviewed pair. Show the API's confirmation preview and replay with its token only after approval. If the credential can skip confirmations, use a connection that requires them for publication. On `DRAFT_CHANGED`, read the draft again and restart review. After publishing, verify the live checkpoint and share the live URL.

## Automate work after launch

For requested ongoing operations, use `apps.get` to check the intended site is live. Keep the automation request with the brief until launch if the site is still a draft. If the work also changes the site, follow the preview and publishing steps for those changes.

Tell the user that submitting an automation task uses credits. Get their go-ahead for the outcome, trigger or cadence, and recipients where relevant, then submit a separate task in the same company. For example, send a booking confirmation when a booking is confirmed or prepare a weekly summary of new leads. Follow the same task and approval steps as for a build. Report what the service confirms was enabled and share any handoff for the user to complete.

See [Your first website](https://docs.doany.ai/api-mcp/quick-start) for a walkthrough from the first brief to publication.
