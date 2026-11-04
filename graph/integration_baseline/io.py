"""Local artifact I/O; the existing lossless Vision serializer is reused."""
import hashlib
import json
from pathlib import Path

import numpy as np

from visual.runtime.test.run_vision_feature_edge import read_packet, save_packet
from visual.runtime.test.run_variable_rate_shape_first import packet_digest


from vla.data.legacy_io import save_npz, serial, sha256, write_json
