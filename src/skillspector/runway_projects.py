"""Offline Runway video prompt projects.

This module deliberately models prompts without talking to Runway or executing
generated content. Projects are small, deterministic export artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

ProjectFormat = Literal["json", "markdown"]
MAX_CLIPS = 256
MAX_TEXT_LENGTH = 4_000


@dataclass(frozen=True, slots=True)
class RunwayClip:
    """One prompt assembled from the four blocks recommended by the brief."""

    title: str
    camera: str
    subject_action: str
    environment_light: str
    style_rendering: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
            if len(value) > MAX_TEXT_LENGTH:
                raise ValueError(f"{name} exceeds {MAX_TEXT_LENGTH} characters")

    @property
    def prompt(self) -> str:
        return " + ".join(
            (self.camera, self.subject_action, self.environment_light, self.style_rendering)
        )

    def to_dict(self) -> dict[str, str]:
        return {**asdict(self), "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class RunwayProject:
    """A named, exportable collection of Runway clips."""

    name: str
    description: str
    base_style: str
    clips: tuple[RunwayClip, ...]
    source: str = "Attached Runway video-project brief"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip() or not self.base_style.strip():
            raise ValueError("project name, description, and base_style must not be empty")
        if not self.clips:
            raise ValueError("project must contain at least one clip")
        if len(self.clips) > MAX_CLIPS:
            raise ValueError(f"project cannot contain more than {MAX_CLIPS} clips")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "base_style": self.base_style,
            "source": self.source,
            "clips": [clip.to_dict() for clip in self.clips],
        }


_BASE_STYLE = (
    "Dark cinematic fantasy with futuristic cyberpunk technology, volumetric fog, "
    "dramatic lighting, 35mm lens, photorealistic, cinematic textures, no text, no watermark."
)


def _clip(
    title: str,
    camera: str,
    subject_action: str,
    environment_light: str,
) -> RunwayClip:
    return RunwayClip(title, camera, subject_action, environment_light, _BASE_STYLE)


_TEMPLATES: dict[str, RunwayProject] = {
    "automation-legends": RunwayProject(
        name="The Legends of Automation",
        description="A cinematic sequence about automation awakening in a ruined future city.",
        base_style=_BASE_STYLE,
        clips=(
            _clip(
                "Awakening from the darkness",
                "Slow camera push forward from pitch black void",
                "A vast destroyed ancient city emerges, broken columns and ruins revealed",
                "Night, faint glowing digital glyphs flickering between rubble, volumetric fog",
            ),
            _clip(
                "The machine archive",
                "Low tracking shot through a colossal abandoned data hall",
                "Mechanical arms wake and sort luminous memory cores",
                "Cold blue work lights cut through dust and drifting smoke",
            ),
            _clip(
                "The automated procession",
                "Wide crane shot rising above a deserted megastructure",
                "Autonomous machines march in precise formation toward the horizon",
                "Storm-lit skyline, sparks and warm industrial lights beneath dark clouds",
            ),
        ),
    ),
    "cerberus-drone": RunwayProject(
        name="Cerberus Wolf: Drone Runaway",
        description="A tense drone pursuit featuring a cybernetic three-headed wolf escaping containment.",
        base_style=_BASE_STYLE,
        clips=(
            _clip(
                "Containment breach",
                "Handheld camera rushes toward a sealed laboratory gate",
                "A cybernetic three-headed wolf breaks through the collapsing barrier",
                "Red emergency lights, flying sparks, dense white vapor and warning strobes",
            ),
            _clip(
                "Runaway through the wasteland",
                "Fast aerial drone follow shot weaving between ruined towers",
                "Cerberus wolf sprints across the landscape while scanning the horizon",
                "Moonlit industrial wasteland, dust trails, distant fires and blue rim light",
            ),
            _clip(
                "The final leap",
                "Dramatic orbiting camera slows as the subject reaches a rooftop edge",
                "The wolf leaps across a chasm toward freedom, all three heads alert",
                "Dawn breaking through storm clouds, golden backlight, deep atmospheric haze",
            ),
        ),
    ),
}


def template_names() -> tuple[str, ...]:
    """Return stable, alphabetized template identifiers."""
    return tuple(sorted(_TEMPLATES))


def get_template(name: str) -> RunwayProject:
    """Return a built-in project or raise a user-facing validation error."""
    try:
        return _TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(template_names())
        raise ValueError(f"unknown template {name!r}; choose one of: {available}") from exc


def render_project(project: RunwayProject, format: str) -> str:
    """Render a project as stable JSON or Runway-ready Markdown."""
    if format == "json":
        return json.dumps(project.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if format != "markdown":
        raise ValueError(f"unsupported project format: {format}")

    lines = [
        f"# {project.name}",
        "",
        project.description,
        "",
        f"**Source:** {project.source}",
        "",
        "## Base style",
        "",
        project.base_style,
        "",
        "## Clips",
        "",
    ]
    for index, clip in enumerate(project.clips, start=1):
        lines.extend([f"### {index}. {clip.title}", "", clip.prompt, ""])
    return "\n".join(lines)
