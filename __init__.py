import os
import sys

import bpy

bl_info = {
    "name": "Morrowind (.nif)",
    "author": "Greatness7",
    "version": (0, 8, 105),
    "blender": (3, 6, 0),
    "location": "File > Import/Export > Morrowind (.nif)",
    "description": "Import/Export files for Morrowind",
    "wiki_url": "https://blender-morrowind.readthedocs.io/",
    "tracker_url": "https://github.com/Greatness7/io_scene_mw/issues",
    "category": "Import-Export",
}

# Make /lib/ modules accessible to python scripts.
lib = os.path.join(os.path.dirname(__file__), "lib")
if lib not in sys.path:
    sys.path.append(lib)

submodules = (
    "preferences",
    "properties",
    "operators",
    "panels",
)

register, unregister = bpy.utils.register_submodule_factory(__name__, submodules)


if "reload" in locals():
    reload()  # pyright: ignore[reportUndefinedVariable]


def reload() -> None:
    import importlib

    for name in submodules:
        importlib.reload(sys.modules[name])
