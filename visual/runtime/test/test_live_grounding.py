"""Real runtime selection/HTTP/streaming paths with isolated credentials and I/O."""
from contextlib import ExitStack
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from visual.runtime import live_grounding as live
from visual.runtime.initialization import detect_initial


def response(boxes=None, status=200, **updates):
    body = dict(model=live.MODEL, service_tier="default", status="completed",
                output=[dict(type="message", content=[dict(type="output_text", text=json.dumps(
                    boxes if boxes is not None else {"peg": [100, 200, 400, 600], "box": None}))])])
    body.update(updates)
    return Mock(status_code=status, json=Mock(return_value=body))


class LiveGroundingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.object(live, "load_env_file", return_value=False))
        self.dino = self.stack.enter_context(patch.object(live, "GDino"))
        self.dino.return_value.side_effect = lambda image, prompt: (
            ([[10, 10, 30, 30]], [.9]) if prompt == "red and white block"
            else ([[60, 60, 80, 80]], [.9]))
        self.post = self.stack.enter_context(patch("requests.post", return_value=response()))
        self.rgb = np.zeros((100, 200, 3), np.uint8)

    def grounder(self, key="test-credential-do-not-log"):
        if key is not None:
            os.environ["OPENAI_API_KEY"] = key
        return live.LiveGrounding(device="cpu", model_id="test-dino")

    def test_terra_is_primary_one_joint_request_and_real_dimension_scaling(self):
        grounder = self.grounder()
        result = detect_initial([self.rgb], grounder)
        self.assertEqual(result["peg"]["box"], [20., 20., 80., 60.])
        self.assertIsNone(result["peg"]["score"])
        self.assertIsNone(result["box"]["box"])
        self.dino.assert_not_called()
        data = self.post.call_args.kwargs
        self.assertFalse(data["allow_redirects"])
        self.assertEqual(data["json"]["model"], live.MODEL)
        self.assertEqual(data["json"]["service_tier"], "default")
        self.assertEqual(data["json"]["reasoning"], {"effort": "high"})
        self.assertFalse(data["json"]["store"])
        self.assertEqual(data["json"]["max_output_tokens"], 5120)
        self.assertEqual(data["json"]["input"][0]["content"][0]["detail"], "original")
        self.assertEqual(grounder.status()["api_requests"], 1)

    def test_missing_or_blank_key_logs_and_runs_legacy_dino(self):
        for key in (None, "   "):
            with self.subTest(key=key), self.assertLogs(live.LOG, level="INFO") as logs:
                grounder = self.grounder(key)
                result = detect_initial([self.rgb], grounder)
            self.assertEqual(grounder.backend, "dino")
            self.assertEqual(grounder.fallback_reason, "missing_key")
            self.assertEqual(result["box"]["frame"], 0)
            self.assertEqual(grounder.dino_calls, 2)
            self.assertTrue(any("FALLBACK Terra -> DINO" in line for line in logs.output))
        self.post.assert_not_called()

    def test_dotenv_discovery_uses_repo_path_independent_of_cwd_and_env_wins(self):
        from utils.experiment_logger import load_env_file
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp)/".env"
            env_file.write_text("OPENAI_API_KEY=dotenv-test-value\n")
            with patch.object(live, "load_env_file", side_effect=load_env_file), \
                    self.assertLogs(live.LOG, level="INFO") as logs:
                grounder = live.LiveGrounding(device="cpu", env_file=env_file)
                self.assertEqual(grounder.backend, "terra")
                self.assertTrue(any("key_source=dotenv" in line for line in logs.output))
                os.environ["OPENAI_API_KEY"] = "process-test-value"
                grounder = live.LiveGrounding(device="cpu", env_file=env_file)
                grounder.initialize([self.rgb])
            self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"], "Bearer process-test-value")
            self.assertNotIn("dotenv-test-value", str(logs.output))
            self.assertNotIn("process-test-value", str(logs.output))
        self.assertEqual(live.ENV_FILE, Path(live.__file__).resolve().parents[2]/".env")

    def test_auth_401_and_403_switch_once_for_all_following_scans(self):
        for code in (401, 403):
            self.post.reset_mock()
            self.post.return_value = response(status=code)
            with self.subTest(code=code), self.assertLogs(live.LOG, level="INFO") as logs:
                grounder = self.grounder()
                first = grounder.initialize([self.rgb])
                second = grounder.initialize([self.rgb]*6)
            self.assertEqual(first["peg"]["frame"], 0)
            self.assertEqual(second["box"]["frame"], 0)
            self.assertEqual(grounder.status()["fallback_reason"], f"http_{code}")
            self.post.assert_called_once()
            self.assertTrue(any("FALLBACK Terra -> DINO" in line for line in logs.output))

    def test_service_and_transport_failures_do_not_silently_switch_backend(self):
        for error in (response(status=429), response(status=500), response(status=404),
                      requests.Timeout("test-credential-do-not-log"),
                      requests.ConnectionError("test-credential-do-not-log")):
            with self.subTest(error=type(error).__name__):
                self.post.side_effect = error if isinstance(error, Exception) else None
                if not isinstance(error, Exception):
                    self.post.return_value = error
                grounder = self.grounder()
                with self.assertLogs(live.LOG, level="ERROR") as logs, \
                        self.assertRaises(live.GroundingError) as raised:
                    grounder.initialize([self.rgb])
                self.assertEqual(grounder.backend, "terra")
                self.dino.assert_not_called()
                self.assertNotIn("test-credential-do-not-log", str(raised.exception) + str(logs.output))

    def test_identical_rgb_is_cached_but_changed_rgb_is_a_new_request(self):
        grounder = self.grounder()
        a = grounder.initialize([self.rgb])
        b = grounder.initialize([self.rgb.copy()])
        self.assertEqual(a, b)
        changed = self.rgb.copy();changed[0, 0, 0] = 1
        grounder.initialize([self.rgb, changed])
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(grounder.cache_hits, 1)

    def test_terra_large_or_overlapping_boxes_bypass_dino_candidate_rules(self):
        self.post.return_value = response({"peg": [0, 0, 1000, 1000], "box": [0, 0, 900, 900]})
        result = self.grounder().initialize([self.rgb])
        self.assertEqual(result["peg"]["box"], [0, 0, 200, 100])
        self.assertIsNotNone(result["box"]["box"])

    def test_invalid_json_boxes_incomplete_model_and_tier_are_visible_errors(self):
        invalid = ['{"peg":[true,0,1,2],"box":null}', '{"peg":[0,0,NaN,2],"box":null}',
                   '{"peg":[-1,0,1,2],"box":null}', '{"peg":[0,0,0,2],"box":null}',
                   '{"peg":null,"box":null,"extra":0}', '{"peg":null,"peg":null,"box":null}',
                   '```json\n{"peg":null,"box":null}\n```']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(live.GroundingError):
                live.parse_boxes(text, 100, 200)
        for kwargs in (dict(status="incomplete"), dict(model="different-model"), dict(service_tier="fast")):
            self.post.return_value = response()
            self.post.return_value.json.return_value.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(live.GroundingError):
                self.grounder().initialize([self.rgb])

    def test_logs_are_visible_without_root_logging_configuration(self):
        stream = StringIO()
        handler = live.LOG.handlers[0]
        with patch.object(handler, "stream", stream):
            grounder = self.grounder()
            grounder.initialize([self.rgb])
        text = stream.getvalue()
        self.assertIn("SELECT", text)
        self.assertIn("REQUEST backend=terra", text)
        self.assertIn("RESULT backend=terra", text)
        self.assertNotIn("test-credential-do-not-log", text)
        self.assertNotIn("test-credential-do-not-log", json.dumps(grounder.status()))

    def test_actual_cache_frontend_selects_terra_without_loading_dino(self):
        from graph.plan_a_bc import cache
        os.environ["OPENAI_API_KEY"] = "test-key"
        with patch.object(cache, "SAM2Tracker") as tracker:
            grounder, actual_tracker = cache.make_vision("cpu", "test-dino", "test-sam")
        self.assertIsInstance(grounder, live.LiveGrounding)
        self.assertEqual(grounder.backend, "terra")
        self.assertIs(actual_tracker, tracker.return_value)
        self.dino.assert_not_called()

    def test_streaming_late_object_and_reset_keep_sam2_memory(self):
        from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter, ObservedFrames
        from graph.integration_baseline.tests.test_online_vision import FakeTracker, append_fixture
        self.post.side_effect = [response({"peg": [0, 0, 250, 250], "box": None}),
                                 response({"peg": None, "box": [500, 500, 750, 750]})]
        grounder = self.grounder();tracker = FakeTracker()
        online = OnlineVisionAdapter(grounder, tracker)
        self.addCleanup(online.close)
        online.reset("episode")
        with patch.object(ObservedFrames, "append_jpeg", append_fixture):
            for t in range(7):
                rgb = np.full((4, 4, 3), t, np.uint8)
                masks, record = online.process(rgb, "episode", t)
                if t == 4:
                    self.assertFalse(masks[1].any())
                if t == 5:
                    self.assertTrue(masks[1].any())
            self.assertEqual(tracker.all_states[0]["calls"], list(range(7)))
            self.assertEqual(tracker.all_states[1]["calls"], [5, 6])
            self.assertEqual(self.post.call_count, 2)
            self.assertEqual(record["grounding"]["backend"], "terra")
            online.reset("new_episode")
            online.process(np.zeros((4, 4, 3), np.uint8), "new_episode", 0)
            self.assertEqual(self.post.call_count, 2)  # byte-identical frame-zero RGB cache
            self.assertEqual(len(tracker.reset_states), 2)

    def test_late_auth_failure_starts_dino_at_current_frame_per_camera(self):
        grounder = self.grounder()
        first, other = grounder.new_stream(), grounder.new_stream()
        first.initialize([self.rgb])
        other.initialize([self.rgb])
        changed = self.rgb.copy();changed[0, 0, 0] = 1
        self.post.return_value = response(status=401)
        result = first.initialize([self.rgb]*5+[changed])
        self.assertEqual(result["box"]["frame"], 5)
        # A camera already using Terra also starts its new prefix now.
        self.assertEqual(other.initialize([self.rgb]*5+[changed])["box"]["frame"], 5)
        self.assertEqual(grounder.new_stream().initialize([self.rgb])["box"]["frame"], 0)
        self.assertEqual(first.initialize([self.rgb]*5+[changed]*6)["box"]["frame"], 5)
        first.reset_initialization()
        self.assertEqual(first.initialize([self.rgb])["box"]["frame"], 0)


if __name__ == "__main__":
    unittest.main()
