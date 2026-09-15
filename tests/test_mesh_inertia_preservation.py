"""XML postprocessing must not silently change authored object dynamics."""

import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen


def test_mesh_inertia_preserved():
    xml = """<mujoco>
      <asset>
        <mesh name="default" vertex="0 0 0  .04 0 0  0 .04 0  0 0 .04"/>
        <mesh name="shell" inertia="shell"
              vertex="0 0 0  .04 0 0  0 .04 0  0 0 .04"/>
      </asset>
      <worldbody>
        <body name="solid"><freejoint/><geom type="mesh" mesh="default" density="500"/></body>
        <body name="hollow" pos="1 0 0"><freejoint/><geom type="mesh" mesh="shell" density="500"/></body>
      </worldbody>
    </mujoco>"""
    kitchen = object.__new__(Kitchen)
    kitchen._cam_configs = {}
    kitchen.generative_textures = None
    # Isolate Kitchen's postprocessing from robot initialization and datasets.
    with patch.object(Kitchen.__mro__[1], "edit_model_xml", side_effect=lambda text: text):
        processed = kitchen.edit_model_xml(xml)
    meshes = ET.fromstring(processed).find("asset").findall("mesh")
    assert meshes[0].get("inertia") is None
    assert meshes[1].get("inertia") == "shell"
    before = mujoco.MjModel.from_xml_string(xml)
    after = mujoco.MjModel.from_xml_string(processed)
    for field in ("body_mass", "body_inertia", "body_ipos", "qpos0"):
        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
