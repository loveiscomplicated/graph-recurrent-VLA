"""RoArm model data and asset checks without simulator registration."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "envs/maniskill/assets/roarm_m2_s"
URDF_PATH = ASSET_DIR / "roarm_m2_s.urdf"
HOME_QPOS = np.array([0, 0, np.pi / 2, 0.75], dtype=np.float32)
JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_gripper_link",
]
PHYSICAL_LINK_NAMES = ["base_link", "link1", "link2", "link3", "gripper_link"]


def verify_asset_hashes():
    """Check every manifest entry and return its source metadata."""
    manifest = json.loads((ASSET_DIR / "source.json").read_text())
    for name, expected_digest in manifest["sha256"].items():
        actual_digest = hashlib.sha256((ASSET_DIR / name).read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"Asset changed; re-audit before use: {name}")
    return manifest


def load_collision_exclusions():
    """Read the original SRDF's five excluded physical link pairs."""
    root = ET.parse(ASSET_DIR / "upstream.srdf").getroot()
    return {
        frozenset((pair.get("link1"), pair.get("link2")))
        for pair in root.findall("disable_collisions")
    }


def audit_assets():
    """Check source hashes, positive inertia, mesh scale and source revision."""
    import trimesh

    manifest = verify_asset_hashes()
    root = ET.parse(URDF_PATH).getroot()
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            assert name == "hand_tcp", f"Missing inertia for physical link {name}"
            continue

        mass = float(inertial.find("mass").get("value"))
        inertia = inertial.find("inertia")
        # URDF stores the upper triangle: ixx, ixy, ixz, iyy, iyz, izz.
        matrix = np.zeros((3, 3))
        for row, first_axis in enumerate("xyz"):
            for column in range(row, 3):
                second_axis = "xyz"[column]
                value = float(inertia.get(f"i{first_axis}{second_axis}"))
                matrix[row, column] = value
                matrix[column, row] = value
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert mass > 0 and eigenvalues.min() > 0, f"Invalid inertia for {name}"

        mesh_path = ASSET_DIR / link.find("visual/geometry/mesh").get("filename")
        mesh = trimesh.load(mesh_path, force="mesh")
        assert 0.001 < mesh.extents.max() < 1, (name, mesh.extents)
        links[name] = {
            "mass_kg": mass,
            "mesh_bounds_m": mesh.bounds.tolist(),
            "inertia_eigenvalues": eigenvalues.tolist(),
        }

    return {"status": "PASS", "links": links, "source_revision": manifest["revision"]}
