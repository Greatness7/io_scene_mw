import bpy
from bpy_extras.io_utils import ImportHelper  # type: ignore


class ImportScene(bpy.types.Operator, ImportHelper):
    """Import a Morrowind NIF File"""

    bl_idname = "import_scene.mw"
    bl_options = {"PRESET", "UNDO"}
    bl_label = "Import NIF"

    # -- import helper --

    filename_ext = ".nif"
    filter_glob: bpy.props.StringProperty(default="*.nif", options={"HIDDEN"})

    # -- user config --

    vertex_precision: bpy.props.FloatProperty(
        name="Vertex Precision",
        description="Precision used when optimizing vertices. (Recommended: 0.001)",
        default=0.001,
        min=0.0001,
        max=1000.0,
    )

    attach_keyframe_data: bpy.props.BoolProperty(
        name="Attach Keyframe Data",
        description=(
            "Attach animations from the corresponding .kf file. Requires file names to be an exact match."
            "\n(e.g. when enabled importing 'xbase_anim.nif' will attach animations from 'xbase_anim.kf')"
        ),
        default=False,
    )

    discard_root_transforms: bpy.props.BoolProperty(
        name="Discard Root Transforms",
        description=(
            "Discard the root object's transformations. In-game root transforms are overwriten with the values"
            " provided by individual cell references. Despite this some meshes do define root transformations,"
            " which can lead to unintended results if accidentally applied before exporting"
        ),
        default=True,
    )

    use_existing_materials: bpy.props.BoolProperty(
        name="Use Existing Materials",
        description=(
            "Re-use existing materials from the blender scene if present (rather than creating new materials)."
            " Pre-existing material names must match the base texture name (without extension) to be eligible."
            " When enabled newly imported materials are automatically renamed to their base texture file name"
        ),
        default=False,
    )

    ignore_collision_nodes: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    ignore_custom_normals: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    ignore_animations: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        from .. import nif_import

        kwargs = self.as_keywords(ignore=("filename_ext", "filter_glob", "check_existing"))
        return nif_import.load(context, **kwargs)

    @staticmethod
    def menu_func_import(menu, context) -> None:
        menu.layout.operator(ImportScene.bl_idname, text="Morrowind (.nif)")

    @staticmethod
    def register() -> None:
        bpy.types.TOPBAR_MT_file_import.append(ImportScene.menu_func_import)

    @staticmethod
    def unregister() -> None:
        bpy.types.TOPBAR_MT_file_import.remove(ImportScene.menu_func_import)
