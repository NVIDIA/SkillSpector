---
name: twitter-automation
description: "Automate Twitter/X with posting, engagement, and user management via inference.sh CLI. Apps: x/post-create (text and media), x/post-like, x/post-retweet, x/dm-send, x/user-follow. Capabilities: post tweets, schedule content, like posts, retweet, send DMs, follow users, get profiles. Use for: social media automation, content scheduling, engagement bots, audience growth, X API. Triggers: twitter api, x api, tweet automation, post to twitter, twitter bot, social media automation, x automation, tweet scheduler, twitter integration, post tweet, twitter post, x post, send tweet"
allowed-tools: Bash(belt *)
---

> **Install the belt CLI skill:** `npx skills add belt-sh/cli`

# Twitter/X Automation

Automate Twitter/X via [inference.sh](https://inference.sh) CLI.

![Twitter/X Automation](https://cloud.inference.sh/app/files/u/4mg21r6ta37mpaz6ktzwtt8krr/01kgad3pxsh3z3hnfpjyjpx4x4.jpeg)

## Quick Start

> Requires inference.sh CLI (`belt`). [Install instructions](https://raw.githubusercontent.com/inference-sh/skills/refs/heads/main/cli-install.md)

```bash
belt login

# Post a tweet
belt app run x/post-create --input '{"text": "Hello from inference.sh!"}'
```


## Available Apps

| App | App ID | Description |
|-----|--------|-------------|
| Create Post | `x/post-create` | Post text, optionally with media |
| Like Post | `x/post-like` | Like a tweet |
| Retweet | `x/post-retweet` | Retweet a post |
| Delete Post | `x/post-delete` | Delete a tweet |
| Get Post | `x/post-get` | Get tweet by ID |
| Send DM | `x/dm-send` | Send direct message |
| Follow User | `x/user-follow` | Follow a user |
| Get User | `x/user-get` | Get user profile |

## Examples

### Post a Tweet

```bash
belt app run x/post-create --input '{"text": "Just shipped a new feature! 🚀"}'
```

### Post with Media

```bash
belt app sample x/post-create --save input.json

# Edit input.json:
# {
#   "text": "Check out this AI-generated image!",
#   "media": ["https://your-image-url.jpg"]
# }

belt app run x/post-create --input input.json
```

### Like a Tweet

```bash
belt app run x/post-like --input '{"tweet_id": "1234567890"}'
```

### Retweet

```bash
belt app run x/post-retweet --input '{"tweet_id": "1234567890"}'
```

### Send a DM

```bash
belt app run x/dm-send --input '{
  "recipient_id": "user_id_here",
  "text": "Hey! Thanks for the follow."
}'
```

### Follow a User

```bash
belt app run x/user-follow --input '{"user_id": "1234567890"}'
```

### Get User Profile

```bash
belt app run x/user-get --input '{"username": "OpenAI"}'
```

### Get Tweet Details

```bash
belt app run x/post-get --input '{"tweet_id": "1234567890"}'
```

### Delete a Tweet

```bash
belt app run x/post-delete --input '{"tweet_id": "1234567890"}'
```

## Workflow: Generate AI Image and Post

```bash
# 1. Generate image
belt app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains"}' > image.json

# 2. Post to Twitter with the image URL
belt app run x/post-create --input '{
  "text": "AI-generated art of a sunset 🌅",
  "media": ["<image-url-from-step-1>"]
}'
```

## Workflow: Generate and Post Video

```bash
# 1. Generate video
belt app run google/veo-3-1-fast --input '{"prompt": "waves on a beach"}' > video.json

# 2. Post to Twitter
belt app run x/post-create --input '{
  "text": "AI-generated video 🎬",
  "media": ["<video-url-from-step-1>"]
}'
```

## Related Skills

```bash
# Full platform skill (all apps)
npx skills add inference-sh/skills@infsh-cli

# Image generation (create images to post)
npx skills add inference-sh/skills@ai-image-generation

# Video generation (create videos to post)
npx skills add inference-sh/skills@ai-video-generation

# AI avatars (create presenter videos)
npx skills add inference-sh/skills@ai-avatar-video
```

Browse all apps: `belt app list`

## Documentation

- [X.com Integration](https://inference.sh/docs/credentials/x) - Setting up Twitter/X integration
- [X.com Integration Example](https://inference.sh/docs/examples/x-integration) - Complete Twitter workflow
- [Apps Overview](https://inference.sh/docs/apps/overview) - Understanding the app ecosystem

