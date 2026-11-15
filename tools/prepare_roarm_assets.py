"""Regenerate the simulation URDF from the vendored, unmodified Waveshare model."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roarm.assets import ASSET_DIR


def prepare_urdf():
    """Rewrite ROS mesh URIs and replace unusable zero drive metadata."""
    tree = ET.parse(ASSET_DIR / "upstream.urdf")
    for mesh in tree.findall(".//mesh"):
        mesh.set(
            "filename", mesh.get("filename").replace("package://roarm_description/", "")
        )
    for joint in tree.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        # Simulation assumptions, NOT manufacturer torque/speed specifications.
        limit = joint.find("limit")
        limit.set("effort", "0.5" if "gripper" in joint.get("name") else "5")
        limit.set("velocity", "1")
    ET.indent(tree, space="  ")
    tree.write(ASSET_DIR / "roarm_m2_s.urdf", encoding="utf-8", xml_declaration=True)


def prepare_srdf():
    """Keep upstream pairs with the reason accepted by SAPIEN 3.0.3."""
    srdf = ET.parse(ASSET_DIR / "upstream.srdf")
    for pair in srdf.findall("disable_collisions"):
        pair.set("reason", "Default")
    ET.indent(srdf, space="  ")
    srdf.write(ASSET_DIR / "roarm_m2_s.srdf", encoding="utf-8", xml_declaration=True)


def main():
    """Regenerate both simulation descriptions from the vendored originals."""
    prepare_urdf()
    prepare_srdf()


if __name__ == "__main__":
    main()
