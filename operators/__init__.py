import bpy

from .addon_updater import UpdateApply, UpdateCheck, UpdateLimit, UpdateNotes
from .create_shader import CreateShader
from .export_scene import ExportScene
from .import_scene import ImportScene

from .developer_extras import CleanMaterials, ShowRadius  # isort:skip

classes = (
    UpdateApply,
    UpdateCheck,
    UpdateLimit,
    UpdateNotes,
    CleanMaterials,
    CreateShader,
    ExportScene,
    ImportScene,
    ShowRadius,
)

register, unregister = bpy.utils.register_classes_factory(classes)
