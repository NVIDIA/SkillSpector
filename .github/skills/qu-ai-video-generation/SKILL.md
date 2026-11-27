---
name: qu-ai-video-generation
description: "Generate AI videos with Google Veo, Seedance 2.5, Kling, HappyHorse, Wan, Grok and 40+ models via inference.sh CLI. Models: Veo 3.1, Seedance 2.5, Kling v4, Runway Gen-4.5, FLUX 3 Video, HappyHorse 1.0, Wan 2.7, Grok Imagine Video 1.5, P-Video 2, OmniHuman, Fabric. Capabilities: text-to-video, image-to-video, reference-to-video, video editing, lipsync, avatar animation, video upscaling, foley sound. Use for: social media videos, marketing content, explainer videos, product demos, AI avatars. Triggers: video generation, ai video, text to video, image to video, veo, animate image, video from image, ai animation, video generator, generate video, t2v, i2v, ai video maker, create video with ai, runway, pika alternative, sora alternative, kling, seedance, happyhorse"
allowed-tools: Bash(belt *)
---

> **Install the belt CLI skill:** `npx skills add belt-sh/cli`

# AI Video Generation

Generate videos with 40+ AI models via [inference.sh](https://inference.sh) CLI.

![AI Video Generation](https://cloud.inference.sh/app/files/u/4mg21r6ta37mpaz6ktzwtt8krr/01kg2c0egyg243mnyth4y6g51q.jpeg)

## Quick Start

> Requires inference.sh CLI (`belt`). [Install instructions](https://raw.githubusercontent.com/inference-sh/skills/refs/heads/main/cli-install.md)

```bash
belt login

# Generate a video with Veo
belt app run google/veo-3-1-fast --input '{"prompt": "drone shot flying over a forest"}'
```


## Available Models

### Text-to-Video

| Model | App ID | Best For |
|-------|--------|----------|
| Veo 3.1 Fast | `google/veo-3-1-fast` | Fast, with optional audio |
| Veo 3.1 | `google/veo-3-1` | Best quality, frame interpolation |
| **P-Video 2** | `pruna/p-video-2` | Fast, economical, with audio support, up to 20s |
| **WAN-T2V** | `pruna/wan-t2v` | Economical 480p/720p |
| Wan 2.7 T2V | `alibaba/wan-2-7-t2v` | 720P/1080P, 2-15s |
| Grok Imagine Video 1.5 | `xai/grok-imagine-video-1-5` | xAI, 1-15s, with audio |
| **Seedance 2.5** | `bytedance/seedance-2-5` | Text/image/ref-to-video with sync audio, up to 1080p, 4-30s |
| Kling v4 | `klingai/video-v4` | Cinematic, up to 1080p, up to 30s |
| Runway Gen-4.5 | `runway/gen-4-5` | Runway's latest, 2-10s |
| FLUX 3 Video | `bfl/flux-3-video` | BFL, keyframes, with audio |
| **HappyHorse T2V** | `alibaba/happyhorse-1-0-t2v` | Physically realistic, up to 15s |

### Image-to-Video

| Model | App ID | Best For |
|-------|--------|----------|
| Wan 2.7 I2V | `alibaba/wan-2-7-i2v` | High quality i2v, first+last frame |
| **WAN-I2V** | `pruna/wan-i2v` | Economical 480p/720p |
| **P-Video 2** | `pruna/p-video-2` | Fast i2v with audio |
| **Seedance 2.5** | `bytedance/seedance-2-5` | Animate images with sync audio, up to 1080p |
| **HappyHorse I2V** | `alibaba/happyhorse-1-0-i2v` | Animate images, up to 1080P/15s |
| **HappyHorse R2V** | `alibaba/happyhorse-1-0-r2v` | Character-preserving from references |

### Avatar / Lipsync

| Model | App ID | Best For |
|-------|--------|----------|
| OmniHuman 1.5 | `bytedance/omnihuman-1-5` | Multi-character |
| OmniHuman 1.0 | `bytedance/omnihuman-1-0` | Single character |
| Fabric 1.0 | `falai/fabric-1-0` | Image talks with lipsync |
| PixVerse Lipsync | `falai/pixverse-lipsync` | Realistic lipsync |

### Video Editing

| Model | App ID | Best For |
|-------|--------|----------|
| HappyHorse Edit | `alibaba/happyhorse-1-0-video-edit` | Natural language video editing |

### Utilities

| Tool | App ID | Description |
|------|--------|-------------|
| MMAudio | `infsh/mmaudio` | Add sound effects to video |
| Topaz Upscaler | `falai/topaz-video-upscaler` | Upscale video quality |
| Media Merger | `infsh/media-merger` | Merge videos with transitions |

## Browse All Video Apps

```bash
belt app list --category video
```

## Examples

### Text-to-Video with Veo

```bash
belt app run google/veo-3-1-fast --input '{
  "prompt": "A timelapse of a flower blooming in a garden"
}'
```

### Grok Video

```bash
belt app run xai/grok-imagine-video-1-5 --input '{
  "prompt": "Waves crashing on a beach at sunset",
  "duration": 5
}'
```

### Image-to-Video with Wan 2.7

```bash
belt app run alibaba/wan-2-7-i2v --input '{
  "first_frame": "https://your-image.jpg",
  "prompt": "slow camera push-in"
}'
```

### AI Avatar / Talking Head

```bash
belt app run bytedance/omnihuman-1-5 --input '{
  "image": "https://portrait.jpg",
  "audio": "https://speech.mp3"
}'
```

### Fabric Lipsync

```bash
belt app run falai/fabric-1-0 --input '{
  "image": "https://face.jpg",
  "audio": "https://audio.mp3"
}'
```

### Seedance 2.5 Text-to-Video with Audio

```bash
belt app run bytedance/seedance-2-5 --input '{
  "prompt": "a jazz band performing in a dimly lit club",
  "generate_audio": true,
  "duration": 10
}'
```

### Seedance 2.5 Image-to-Video

```bash
belt app run bytedance/seedance-2-5 --input '{
  "image": "https://your-image.jpg",
  "prompt": "gentle camera movement, leaves rustling in the wind",
  "generate_audio": true
}'
```

### Seedance 2.5 Reference-to-Video

```bash
belt app run bytedance/seedance-2-5 --input '{
  "prompt": "The person in @Image1 walking through a garden",
  "reference_images": ["https://portrait.jpg"],
  "generate_audio": true
}'
```

### HappyHorse Text-to-Video

```bash
belt app run alibaba/happyhorse-1-0-t2v --input '{
  "prompt": "a golden retriever running through autumn leaves, slow motion",
  "duration": 10,
  "resolution": "1080P"
}'
```

### HappyHorse Video Editing

```bash
belt app run alibaba/happyhorse-1-0-video-edit --input '{
  "video": "https://your-video.mp4",
  "prompt": "change the background to a snowy mountain landscape"
}'
```

### PixVerse Lipsync

```bash
belt app run falai/pixverse-lipsync --input '{
  "video": "https://talking-head.mp4",
  "audio": "https://speech.mp3"
}'
```

Takes a video, not a still image. Omit `audio` and pass `text` (plus optional `voice_id`) to use the built-in TTS.

### Video Upscaling

```bash
belt app run falai/topaz-video-upscaler --input '{"video": "https://..."}'
```

### Add Sound Effects (Foley)

```bash
belt app run infsh/mmaudio --input '{
  "video_input": "https://silent-video.mp4",
  "prompt": "footsteps on gravel, birds chirping"
}'
```

### Merge Videos

```bash
belt app run infsh/media-merger --input '{
  "media_files": [
    {"file": "https://clip1.mp4", "transition_type": "crossfade"},
    {"file": "https://clip2.mp4"}
  ]
}'
```

## Related Skills

```bash
# Full platform skill (all apps)
npx skills add inference-sh/skills@infsh-cli

# Pruna P-Video 2 (fast & economical)
npx skills add inference-sh/skills@p-video

# Google Veo specific
npx skills add inference-sh/skills@google-veo

# Seedance 2.5
npx skills add inference-sh/skills@seedance

# HappyHorse 1.0
npx skills add inference-sh/skills@happyhorse

# AI avatars & lipsync
npx skills add inference-sh/skills@ai-avatar-video

# Text-to-speech (for video narration)
npx skills add inference-sh/skills@text-to-speech

# Image generation (for image-to-video)
npx skills add inference-sh/skills@ai-image-generation

# Twitter (post videos)
npx skills add inference-sh/skills@twitter-automation
```

Browse all apps: `belt app list`

## Documentation

- [Running Apps](https://inference.sh/docs/apps/running) - How to run apps via CLI
- [Streaming Results](https://inference.sh/docs/api/sdk/streaming) - Real-time progress updates
- [Content Pipeline Example](https://inference.sh/docs/examples/content-pipeline) - Building media workflows

