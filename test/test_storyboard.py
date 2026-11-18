import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storyboard import (
    StoryboardPlanError,
    build_montage_zip,
    lock_character,
    parse_plan,
    select_final_frames,
)
from test.test_api import SyncASGIClient

PLAN = {
    "title": "Osmanlı 2026",
    "reference": {
        "subject": "young Ottoman courier with short black hair",
        "appearance": "green eyes, thin scar on the left eyebrow",
        "wardrobe": "crimson kaftan with brass buttons, white fez",
        "style": "cinematic digital painting, teal and gold palette",
        "scene": "futuristic Istanbul in 2026",
        "pose": "confident stance",
    },
    "scenes": [
        {
            "title": "İstanbul",
            "prompt": "{style}, wide shot of {scene}, flying trams over the Bosphorus",
        },
        {
            "title": "Askerler",
            "prompt": "{style}, the same {subject}, {appearance}, wearing {wardrobe}, "
            "passing Janissary guards in exoskeleton armor",
        },
        {"title": "Saray", "prompt": "the same {subject} inside a glass palace"},
    ],
}


class TestPlanParsing(unittest.TestCase):
    def test_parses_fenced_json_and_limits_scene_count(self):
        raw = "Here you go:\n" + json.dumps(PLAN) + "\nEnjoy"
        plan = parse_plan(raw, scene_count=2)
        self.assertEqual(plan["title"], "Osmanlı 2026")
        self.assertEqual(len(plan["scenes"]), 2)
        self.assertEqual(plan["scenes"][0]["title"], "İstanbul")

    def test_every_scene_carries_locked_character_placeholders(self):
        plan = parse_plan(json.dumps(PLAN), scene_count=5)
        for scene in plan["scenes"]:
            for placeholder in ("{style}", "{subject}", "{appearance}", "{wardrobe}"):
                self.assertIn(placeholder, scene["prompt"])

    def test_lock_character_keeps_complete_prompt_unchanged(self):
        prompt = "{style}, the same {subject}, {appearance}, wearing {wardrobe}, smiling"
        self.assertEqual(lock_character(prompt), prompt)

    def test_rejects_missing_reference(self):
        bad = dict(PLAN, reference={"subject": "someone"})
        with self.assertRaises(StoryboardPlanError):
            parse_plan(json.dumps(bad), scene_count=3)

    def test_rejects_non_json(self):
        with self.assertRaises(StoryboardPlanError):
            parse_plan("no json here", scene_count=3)


class TestMontageExport(unittest.TestCase):
    def test_selects_upscaled_then_latest_revision(self):
        items = [
            {"id": "a", "segment_index": 0, "status": "completed", "created_at": "1"},
            {"id": "b", "segment_index": 0, "status": "completed", "created_at": "2"},
            {"id": "c", "segment_index": 1, "status": "completed", "created_at": "3"},
            {
                "id": "d",
                "segment_index": 1,
                "status": "completed",
                "created_at": "1",
                "upscaled": True,
            },
            {"id": "e", "segment_index": 2, "status": "failed", "created_at": "9"},
        ]
        frames = select_final_frames(items)
        self.assertEqual(frames[0]["id"], "b")
        self.assertEqual(frames[1]["id"], "d")
        self.assertNotIn(2, frames)

    def test_zip_contains_numbered_frames_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for item_id in ("a", "b"):
                p = Path(tmp) / f"{item_id}.png"
                p.write_bytes(b"\x89PNG" + item_id.encode())
                paths[item_id] = p
            items = [
                {"id": "a", "segment_index": 0, "status": "completed", "parent_id": None},
                {"id": "x", "segment_index": 1, "status": "failed", "parent_id": None},
                {"id": "b", "segment_index": 2, "status": "completed", "parent_id": None},
            ]
            data, exported = build_montage_zip(
                items, lambda item: paths.get(item["id"]), {0: "Açılış"}, folder="story"
            )

        self.assertEqual(exported, 2)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            self.assertIn("story/001.png", names)
            self.assertIn("story/003.png", names)
            self.assertNotIn("story/002.png", names)
            manifest = json.loads(archive.read("story/storyboard.json"))
            self.assertEqual(manifest["frames"][0]["title"], "Açılış")
            self.assertIsNone(manifest["frames"][1]["file"])
            self.assertIn("MISSING", archive.read("story/shotlist.txt").decode())


class TestStoryboardRoute(unittest.TestCase):
    def setUp(self):
        self.client = SyncASGIClient()

    def test_rejects_too_many_scenes(self):
        response = self.client.post(
            "/api/ai/storyboard",
            json={
                "scenario": "2026 yılında Osmanlı İmparatorluğu hâlâ yaşasaydı",
                "scene_count": 50,
            },
        )
        self.assertEqual(response.status_code, 422)

    @patch("routes.api.set_active_db_name")
    @patch("routes.api.save_db")
    @patch("routes.api._call_openrouter", new_callable=AsyncMock)
    def test_creates_locked_databases_without_generating(
        self, mock_llm, mock_save_db, mock_set_active
    ):
        mock_llm.return_value = json.dumps(PLAN)
        with patch("routes.api.get_db_lookup", return_value={"NONE": "", "KEY": "value"}):
            response = self.client.post(
                "/api/ai/storyboard",
                json={
                    "scenario": "2026 yılında Osmanlı İmparatorluğu hâlâ yaşasaydı...",
                    "scene_count": 3,
                    "name": "ottoman_2026",
                    "generate": False,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["name"], "ottoman_2026")
        self.assertEqual(len(body["scenes"]), 3)
        self.assertNotIn("batch", body)
        saved_types = {c.kwargs["db_type"] for c in mock_save_db.call_args_list}
        self.assertEqual(
            saved_types,
            {"prompts", "subjects", "appearances", "wardrobes", "styles", "scenes", "poses"},
        )
        self.assertEqual(mock_set_active.call_count, 7)
        prompts_db = next(
            c.args[0] for c in mock_save_db.call_args_list if c.kwargs["db_type"] == "prompts"
        )
        self.assertEqual(prompts_db["segments"][1]["title"], "Askerler")

    def test_export_unknown_session_returns_404(self):
        with patch("routes.api.load_state", return_value=[]):
            response = self.client.get("/api/sessions/missing/export")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
