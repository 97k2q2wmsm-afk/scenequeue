"""Scenario-to-storyboard planning and montage export helpers.

The planner turns a free-form scenario into an ordered list of scene prompts that
share one locked character reference (subject, appearance, wardrobe, style).
Every scene prompt references that reference through placeholders, so the same
face, outfit, drawing style, and palette are injected into every frame.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

MIN_SCENES = 2
MAX_SCENES = 12

REFERENCE_FIELDS = ("subject", "appearance", "wardrobe", "style", "scene", "pose")

PLANNER_SYSTEM_PROMPT = """You are a storyboard director for text-to-image generation.
Split the user's scenario into exactly {count} sequential scenes that tell a complete
visual story with a clear opening, development, and a final closing scene.

Lock ONE recurring main character so every frame shows the same face, outfit,
drawing style, and color palette.

Respond with a single JSON object and nothing else, using this shape:
{{
  "title": "short story title in the scenario's language",
  "reference": {{
    "subject": "the main character: who they are, age range, face and hair (English)",
    "appearance": "fixed facial features and expression traits (English)",
    "wardrobe": "exact, fixed outfit and accessories with colors (English)",
    "style": "one drawing style and one explicit color palette used by every frame (English)",
    "scene": "the world's overall setting and era (English)",
    "pose": "default body language (English)"
  }},
  "scenes": [
    {{
      "title": "short scene title in the scenario's language",
      "prompt": "English image prompt for this frame"
    }}
  ]
}}

Rules for every scene prompt:
- Start with "{{style}}, " and mention "the same {{subject}}, {{appearance}}, wearing {{wardrobe}}"
  when the character is visible. Use the literal placeholders in curly braces;
  never restate their values.
- You may use "{{scene}}" for the shared world and "{{pose}}" for default body language.
- Describe only what changes in this frame: location detail, action, camera framing,
  lighting, mood. Vary shot types (wide establishing, medium, close-up, final wide).
- Keep everything suitable for a general audience. No named artists, real people,
  or copyrighted characters.
"""


class StoryboardPlanError(ValueError):
    """Raised when the language model response cannot be used as a storyboard."""


def build_planner_messages(scenario: str, scene_count: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(count=scene_count)},
        {"role": "user", "content": f"Scenario:\n{scenario.strip()}"},
    ]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise StoryboardPlanError("The AI response did not contain a JSON object.") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise StoryboardPlanError("The AI response contained invalid JSON.") from e
    if not isinstance(value, dict):
        raise StoryboardPlanError("The AI response must be a JSON object.")
    return value


def _clean_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip(",").strip()


def lock_character(prompt: str) -> str:
    """Ensure a scene prompt carries every locked character placeholder.

    The result always starts with ``{style}`` and names the subject, appearance,
    and wardrobe placeholders, so every frame receives the same reference.
    """
    text = _clean_line(prompt)
    if text.startswith("{style}"):
        text = text[len("{style}") :].lstrip(" ,")
    else:
        text = text.replace("{style}", "style").strip()

    character = []
    if "{subject}" not in text:
        character.append("the same {subject}")
    if "{appearance}" not in text:
        character.append("{appearance}")
    if "{wardrobe}" not in text:
        character.append("wearing {wardrobe}")
    return ", ".join(part for part in ["{style}", *character, text] if part)


def parse_plan(raw: str, scene_count: int) -> dict[str, Any]:
    data = _extract_json_object(raw)

    reference_raw = data.get("reference") or data.get("character") or {}
    if not isinstance(reference_raw, dict):
        reference_raw = {}
    reference = {field: _clean_line(reference_raw.get(field)) for field in REFERENCE_FIELDS}
    for field in ("subject", "wardrobe", "style"):
        if not reference[field]:
            raise StoryboardPlanError(f"The AI response is missing the character {field}.")

    scenes_raw = data.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise StoryboardPlanError("The AI response did not contain any scenes.")

    scenes = []
    for position, scene in enumerate(scenes_raw[:scene_count], start=1):
        if isinstance(scene, str):
            title, prompt = f"Scene {position}", scene
        elif isinstance(scene, dict):
            title = _clean_line(scene.get("title")) or f"Scene {position}"
            prompt = scene.get("prompt") or scene.get("text") or ""
        else:
            continue
        if not _clean_line(prompt):
            continue
        scenes.append({"title": title, "prompt": lock_character(prompt)})

    if len(scenes) < MIN_SCENES:
        raise StoryboardPlanError("The AI response contained too few usable scenes.")

    return {
        "title": _clean_line(data.get("title")) or "Storyboard",
        "reference": reference,
        "scenes": scenes,
    }


def single_value_db(title: str, text: str) -> dict[str, Any]:
    return {
        "version": 2,
        "title": title,
        "segments": [{"index": 0, "text": text, "history": []}] if text else [],
    }


def scenes_db(title: str, scenario: str, scenes: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "version": 2,
        "title": title,
        "description": scenario.strip(),
        "segments": [
            {"index": i, "title": scene["title"], "text": scene["prompt"], "history": []}
            for i, scene in enumerate(scenes)
        ],
    }


# =====================================================================
# Montage export
# =====================================================================
def select_final_frames(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Pick the newest completed image for each storyboard position.

    Upscaled images win over their originals; otherwise the latest revision wins.
    """
    chosen: dict[int, dict[str, Any]] = {}
    for item in items:
        if item.get("status") != "completed":
            continue
        try:
            position = int(item.get("segment_index", 0))
        except (TypeError, ValueError):
            continue
        rank = (1 if item.get("upscaled") else 0, str(item.get("created_at", "")))
        current = chosen.get(position)
        if current is None or rank > (
            1 if current.get("upscaled") else 0,
            str(current.get("created_at", "")),
        ):
            chosen[position] = item
    return chosen


def build_montage_zip(
    items: list[dict[str, Any]],
    find_image: Callable[[dict[str, Any]], Path | None],
    titles: dict[int, str] | None = None,
    folder: str = "storyboard",
) -> tuple[bytes, int]:
    """Bundle numbered PNG frames and a manifest into an in-memory ZIP archive."""
    titles = titles or {}
    frames = select_final_frames(items)
    positions = sorted(
        {int(i.get("segment_index", 0)) for i in items if i.get("parent_id") is None} | set(frames)
    )
    width = max(3, len(str(len(positions))))

    buffer = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    exported = 0
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for number, position in enumerate(positions, start=1):
            entry: dict[str, Any] = {
                "number": number,
                "segment_index": position,
                "title": titles.get(position, f"Scene {number}"),
                "file": None,
            }
            item = frames.get(position)
            path = find_image(item) if item else None
            if item and path and path.exists():
                name = f"{number:0{width}d}.png"
                archive.write(path, f"{folder}/{name}")
                entry["file"] = name
                entry["item_id"] = item["id"]
                entry["prompt"] = item.get("prompt_resolved", "")
                exported += 1
            manifest.append(entry)

        archive.writestr(
            f"{folder}/storyboard.json",
            json.dumps({"frames": manifest}, indent=2, ensure_ascii=False),
        )
        lines = [f"{e['file'] or 'MISSING':>10}  {e['title']}" for e in manifest]
        archive.writestr(f"{folder}/shotlist.txt", "\n".join(lines) + "\n")
    return buffer.getvalue(), exported
