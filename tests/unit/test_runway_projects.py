import json

import pytest

from skillspector.runway_projects import (
    RunwayClip,
    get_template,
    render_project,
    template_names,
)


def test_template_names_are_stable() -> None:
    assert template_names() == ("automation-legends", "cerberus-drone")


def test_clip_prompt_uses_brief_order() -> None:
    clip = get_template("automation-legends").clips[0]
    assert clip.prompt == " + ".join(
        (clip.camera, clip.subject_action, clip.environment_light, clip.style_rendering)
    )


def test_json_export_contains_composed_prompts() -> None:
    payload = json.loads(render_project(get_template("cerberus-drone"), "json"))
    assert payload["name"] == "Cerberus Wolf: Drone Runaway"
    assert payload["clips"][0]["prompt"].startswith("Handheld camera rushes")


def test_markdown_export_contains_base_style_and_clip_headings() -> None:
    rendered = render_project(get_template("automation-legends"), "markdown")
    assert "## Base style" in rendered
    assert "### 1. Awakening from the darkness" in rendered
    assert "no text, no watermark." in rendered


def test_unknown_template_is_actionable() -> None:
    with pytest.raises(ValueError, match="unknown template"):
        get_template("missing")


def test_clip_rejects_empty_prompt_block() -> None:
    with pytest.raises(ValueError, match="camera"):
        RunwayClip("title", "", "action", "environment", "style")
