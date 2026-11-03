"""Terra Standard grounding for live RGB, with an authentication-only DINO fallback.

The prompt and strict box contract come from the independent Terra experiment.
Historical DINO experiments keep using ``GDino`` directly.
"""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import base64
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from utils.experiment_logger import load_env_file
from .grounding import GDino, GDINO_ID
from .initialization import PROMPTS, SCAN_EVERY, detect_initial


MODEL = "gpt-6.1-terra"
ENDPOINT = "https://api.openai.com/v1/responses"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
PROMPT = '''Locate the following two objects directly in this complete RGB image.
Identify them by their visible appearance:
PEG: A red-and-white elongated rectangular piece. It may be held by the gripper or partly outside the image. A separate object held by the gripper is not a robot part.
BOX: A tan or brown block-shaped fixture. An opening may be visible depending on the view. Locate the fixture itself, not only its opening.
ROBOT: Gray or white robot arm links and dark or black gripper fingers. Do not include robot parts as PEG or BOX.
Use color, shape, and visible parts together. Do not invent an object that cannot be seen. Return null if an object's location cannot be determined reliably.
For each object, tightly enclose its visible part. Do not estimate invisible parts outside the image. Return at most one PEG box and one BOX box.
Coordinates refer to the complete original image, normalized from 0 to 1000: top-left (0,0), bottom-right (1000,1000). Each box is [x1,y1,x2,y2], with x1<x2 and y1<y2.
Return exactly one JSON object with keys "peg" and "box". Each value must be a box array or null. Return only the JSON object, without Markdown, explanation, or additional text.
'''

# Visible even when an entrypoint has not configured Python logging. Propagate
# to configured file/root handlers as well; never print HTTP bodies or secrets.
LOG = logging.getLogger(__name__)
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[live-grounding] %(levelname)s %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


class GroundingError(RuntimeError):
    """Sanitized failure; no authenticated request or provider body is retained."""


class AuthenticationError(GroundingError):
    pass


class GroundingStream:
    """Camera/episode-local DINO prefix; API authentication/cache stay shared."""
    def __init__(self, grounder):
        self.grounder = grounder
        self.reset_initialization()

    def reset_initialization(self):
        self.fallback_start = 0
        self.last_backend = self.grounder.backend

    def status(self):
        return self.grounder.status()

    def initialize(self, frames, prompts=None):
        before = self.grounder.backend
        if self.last_backend == "terra" and before == "dino":
            # A different camera may have triggered the shared auth fallback.
            self.fallback_start = len(frames)-1
        result = self.grounder._initialize(frames, prompts, self.fallback_start)
        if before == "terra" and self.grounder.backend == "dino":
            self.fallback_start = len(frames)-1
        self.last_backend = self.grounder.backend
        return result


def policy_identity():
    """Public settings only, suitable for cache and experiment identities."""
    return dict(policy="terra_standard_auth_fallback_v1", primary=MODEL,
                service_tier="default", reasoning_effort="high", detail="original",
                max_output_tokens=5120, store=False,
                prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                fallback="grounding_dino", fallback_on=["missing_key", "http_401", "http_403"],
                scan_every=SCAN_EVERY)


def parse_boxes(text, height, width):
    """Strict JSON / normalized coordinates, scaled to actual image dimensions."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique)
        if not isinstance(value, dict) or set(value) != {"peg", "box"}:
            raise ValueError("exact keys required")
        result = {}
        for name, box in value.items():
            if box is None:
                result[name] = None
                continue
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           or not 0 <= v <= 1000 for v in box)
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError("invalid box")
            result[name] = [box[0]*width/1000, box[1]*height/1000,
                            box[2]*width/1000, box[3]*height/1000]
        return result
    except (ValueError, TypeError, OverflowError):
        raise GroundingError("Terra INVALID_BOX_JSON; response content suppressed") from None


class LiveGrounding:
    """One joint Terra request per scanned image; DINO loads only on fallback.

    Repeated byte-identical RGB uses a bounded result cache, including the
    recorded prefix-causality probe. The API sees only one current RGB image.
    A rejected key selects DINO for this instance's remaining lifetime.
    """
    def __init__(self, device="cuda", model_id=GDINO_ID, *, env_file=ENV_FILE):
        self.device, self.model_id = device, str(model_id)
        had_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        loaded = load_env_file(env_file)
        self._key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._dino = None
        self._cache = OrderedDict()
        self.backend = "terra" if self._key else "dino"
        self.fallback_reason = None if self._key else "missing_key"
        self.api_requests = self.dino_calls = self.cache_hits = 0
        self._default_stream = self.new_stream()
        source = "environment" if had_key else "dotenv" if self._key else "missing"
        LOG.info("SELECT primary=%s active=%s key_source=%s dotenv_loaded=%s dotenv_file=%s device=%s",
                 MODEL, self.backend, source, loaded, env_file, device)
        if self._key:
            LOG.info("Terra Standard ready: tier=default effort=high detail=original; DINO loading deferred")
        else:
            LOG.warning("FALLBACK Terra -> DINO reason=missing_key; OPENAI_API_KEY was not recognized in environment or dotenv")

    def status(self):
        return dict(primary=MODEL, backend=self.backend, fallback_reason=self.fallback_reason,
                    api_requests=self.api_requests, dino_calls=self.dino_calls,
                    cache_hits=self.cache_hits)

    def new_stream(self):
        return GroundingStream(self)

    def _request(self, rgb):
        # Lazy import allows a missing-key/offline run without HTTP dependencies.
        import requests

        png = BytesIO()
        Image.fromarray(rgb).save(png, format="PNG")
        data = dict(model=MODEL, reasoning={"effort": "high"}, service_tier="default",
                    max_output_tokens=5120, store=False,
                    input=[dict(role="user", content=[
                        dict(type="input_image", detail="original",
                             image_url="data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")),
                        dict(type="input_text", text=PROMPT)])])
        self.api_requests += 1
        started = time.perf_counter()
        LOG.info("REQUEST backend=terra request=%d image=%dx%d tier=default", self.api_requests, rgb.shape[1], rgb.shape[0])
        try:
            response = requests.post(ENDPOINT, json=data,
                                     headers={"Authorization": "Bearer " + self._key},
                                     timeout=(15, 60), allow_redirects=False)
        except requests.RequestException as exc:
            kind = "TIMEOUT" if isinstance(exc, requests.Timeout) else "CONNECTION_ERROR"
            raise GroundingError("Terra " + kind + "; provider exception suppressed") from None
        elapsed = (time.perf_counter() - started)*1000
        if response.status_code in (401, 403):
            raise AuthenticationError("http_" + str(response.status_code))
        if response.status_code != 200:
            raise GroundingError("Terra HTTP_" + str(response.status_code) + "; response body suppressed")
        try:
            body = response.json()
            model = body.get("model", "")
            if not (model == MODEL or model.startswith(MODEL + "-")):
                raise ValueError("model mismatch")
            if body.get("status") != "completed" or body.get("service_tier") != "default":
                raise ValueError("incomplete response or tier mismatch")
            content = [c for m in body.get("output", []) if m.get("type") == "message"
                       for c in m.get("content", [])]
            if any(c.get("type") == "refusal" for c in content):
                raise ValueError("refusal")
            text = "".join(c["text"] for c in content if c.get("type") == "output_text")
        except (ValueError, TypeError, AttributeError, KeyError):
            raise GroundingError("Terra INVALID_RESPONSE; provider content suppressed") from None
        boxes = parse_boxes(text, *rgb.shape[:2])
        LOG.info("RESULT backend=terra elapsed_ms=%.1f peg=%s box=%s",
                 elapsed, "found" if boxes["peg"] else "null", "found" if boxes["box"] else "null")
        return boxes

    def initialize(self, frames, prompts=None):
        """Initializer extension used by both causal streaming and its probe."""
        return self._default_stream.initialize(frames, prompts)

    def _initialize(self, frames, prompts, fallback_start):
        if prompts is not None and prompts != PROMPTS:
            raise ValueError("LiveGrounding uses the fixed joint Terra appearance prompt")
        t = len(frames)-1
        rgb = np.asarray(frames[t])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("uint8 H,W,3 RGB required")
        if self.backend == "terra":
            key = (rgb.shape, hashlib.sha256(rgb.tobytes()).digest())
            try:
                if key in self._cache:
                    boxes = self._cache[key]
                    self._cache.move_to_end(key)
                    self.cache_hits += 1
                    LOG.info("CACHE backend=terra frame=%d identical_rgb=true", t)
                else:
                    boxes = self._request(rgb)
                    self._cache[key] = boxes
                    if len(self._cache) > 64:
                        self._cache.popitem(last=False)
                return {name: dict(box=boxes[name], score=None,
                                   frame=t if boxes[name] is not None else None,
                                   prompt=PROMPT, scans=t//SCAN_EVERY+1)
                        for name in PROMPTS}
            except AuthenticationError as exc:
                self.backend, self.fallback_reason = "dino", str(exc)
                self._key = ""
                self._cache.clear()
                fallback_start = t
                LOG.warning("FALLBACK Terra -> DINO reason=%s frame=%d; switching for this detector's remaining lifetime", self.fallback_reason, t)
            except GroundingError as exc:
                LOG.error("ERROR backend=terra frame=%d %s; DINO fallback is limited to key/auth failures", t, exc)
                raise
        if self._dino is None:
            LOG.info("LOAD backend=dino device=%s reason=%s", self.device, self.fallback_reason)
            self._dino = GDino(device=self.device, model_id=self.model_id)
            LOG.info("READY backend=dino")

        def detect(image, prompt):
            self.dino_calls += 1
            return self._dino(image, prompt)

        # A key can be rejected after Terra already initialized another object.
        # Begin this camera's DINO prefix now, so past t0 detections cannot
        # strand a missing object behind the adapter's current-frame gate.
        result = detect_initial(frames[fallback_start:], detect)
        for value in result.values():
            if value["frame"] is not None:
                value["frame"] += fallback_start
        LOG.info("RESULT backend=dino frame=%d reason=%s peg_frame=%s box_frame=%s calls=%d",
                 t, self.fallback_reason, result["peg"]["frame"], result["box"]["frame"], self.dino_calls)
        return result
